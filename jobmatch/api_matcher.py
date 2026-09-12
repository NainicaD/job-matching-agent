"""API mode: match a profile against a job description using the Claude API.

This is the centrepiece of the project, so it is written to be read. The four
things worth understanding are kept in four clearly separated places:

  1. REQUEST CONSTRUCTION  -- `build_request()` returns the exact kwargs handed
     to `client.messages.create()`. `--dry-run` prints the output of this very
     function, so what you inspect is guaranteed to be what would be sent.
  2. RESPONSE PARSING      -- `parse_match_json()` turns the model's text into a
     validated dict, and treats malformed JSON as a normal outcome, not a crash.
  3. ERROR HANDLING        -- `MatchClient.match()` catches the SDK's typed
     exceptions most-specific-first, and retries only what is retryable.
  4. COST TRACKING         -- `CostTracker` prices every call from
     `response.usage` and keeps a running session total.

Local mode (`--local`) never imports this module, by design.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic

from .pricing import DEFAULT_MODEL, Usage, format_usd, price_usage
from .result import ENGINE_API, Gap, MatchResult, Strength

# --------------------------------------------------------------------------
# 1. REQUEST CONSTRUCTION
# --------------------------------------------------------------------------

DEFAULT_MAX_TOKENS = 2048

# The JSON contract. It is stated once, here, and referenced by both the prompt
# and the validator below so the two cannot drift apart.
RESPONSE_SCHEMA = {
    "match_score": "integer 0-100",
    "verdict": "one of: strong | moderate | weak",
    "matching_strengths": [
        {
            "requirement": "the requirement from the job description",
            "evidence": "the specific line from the profile that satisfies it",
        }
    ],
    "gaps": [
        {
            "requirement": "the requirement the profile does not evidence",
            "severity": "one of: blocker | major | minor",
            "note": "one short sentence on why it matters or how close the fit is",
        }
    ],
    "rationale": "2-4 sentences explaining the score",
}

SYSTEM_INSTRUCTIONS = """\
You are a technical recruiter assessing how well one candidate's consolidated \
resume profile matches a specific job description.

Ground every claim in the profile text you are given. Do not invent experience, \
do not infer skills that are not evidenced, and prefer quoting the profile line \
that supports a strength. If the profile is thin on something the job requires, \
that is a gap -- say so plainly. A high score must be earned.

Scoring guide:
  80-100  strong   - meets essentially all core requirements with evidence
  50-79   moderate - meets most core requirements, some real gaps
  0-49    weak     - misses several core requirements

Reply with a single JSON object and nothing else -- no prose before or after, no \
markdown code fences. The object must match this shape exactly:

{schema}

List at most 6 strengths and at most 6 gaps, most important first."""


def render_system_prompt() -> str:
    return SYSTEM_INSTRUCTIONS.format(schema=json.dumps(RESPONSE_SCHEMA, indent=2))


def build_request(
    profile: str,
    job_description: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Build the exact keyword arguments for `client.messages.create()`.

    Layout is chosen for prompt caching. Caching is a *prefix* match, so the
    stable content has to come first: instructions and the consolidated profile
    go in `system` (identical on every call of a session), and only the job
    description -- the part that changes per call -- goes in `messages`. With
    `cache_control` on the profile block, the second and later jobs in a run
    re-read those tokens at 10% of the input rate instead of paying full price.

    Note what is *not* here: no `temperature`, no `thinking`, no `effort`.
    Sampling parameters are rejected by the current Opus/Fable models, and
    thinking/effort are rejected by Haiku, so omitting them keeps this one
    request shape valid across every model `--model` accepts.
    """
    profile_block: dict[str, Any] = {
        "type": "text",
        "text": f"<candidate_profile>\n{profile}\n</candidate_profile>",
    }
    if use_cache:
        profile_block["cache_control"] = {"type": "ephemeral"}

    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {"type": "text", "text": render_system_prompt()},
            profile_block,
        ],
        "messages": [
            {
                "role": "user",
                "content": (
                    "<job_description>\n"
                    f"{job_description}\n"
                    "</job_description>\n\n"
                    "Assess the match. Reply with the JSON object only."
                ),
            }
        ],
    }


def describe_request(request: dict[str, Any], max_chars: int = 1200) -> str:
    """Human-readable dump of a request, for --dry-run.

    Long blocks are elided in the middle so the structure stays visible; pass
    --full to the CLI to print the payload verbatim instead.
    """
    lines = [
        "model      : {}".format(request["model"]),
        "max_tokens : {}".format(request["max_tokens"]),
        "endpoint   : client.messages.create()",
        "",
    ]
    for i, block in enumerate(request["system"]):
        cached = " [cache_control: ephemeral]" if "cache_control" in block else ""
        lines.append(f"system[{i}]{cached}")
        lines.append(_indent(_elide(block["text"], max_chars)))
        lines.append("")
    for i, message in enumerate(request["messages"]):
        lines.append(f"messages[{i}] role={message['role']}")
        lines.append(_indent(_elide(message["content"], max_chars)))
        lines.append("")
    return "\n".join(lines)


def _elide(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    hidden = len(text) - len(head) - len(tail)
    return f"{head}\n\n  ... [{hidden:,} characters elided - use --full to see all] ...\n\n{tail}"


def _indent(text: str, prefix: str = "  | ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def estimate_tokens(request: dict[str, Any]) -> int:
    """Rough offline token estimate for --dry-run (~4 characters per token).

    Deliberately local: --dry-run must not touch the network. For an exact
    count, `--count-tokens` calls the free `messages.count_tokens` endpoint.
    """
    chars = sum(len(block["text"]) for block in request["system"])
    chars += sum(len(message["content"]) for message in request["messages"])
    return chars // 4


# --------------------------------------------------------------------------
# 2. RESPONSE PARSING
# --------------------------------------------------------------------------


class MatchError(RuntimeError):
    """Base class for every failure this module reports to the CLI."""


class MalformedResponseError(MatchError):
    """The model replied, but not with usable JSON."""


class ApiCallError(MatchError):
    """The API call itself failed. `hint` explains what to do about it."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_match_json(text: str) -> dict[str, Any]:
    """Extract the JSON object from a model response.

    The prompt asks for bare JSON, but a model can still wrap it in a code
    fence or add a sentence of preamble, and that should not be a crash. Three
    strategies are tried in order of preference before giving up.
    """
    candidates: list[str] = [text.strip()]

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    # Last resort: the outermost {...} span in the response.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    preview = text.strip()[:300] or "<empty response>"
    raise MalformedResponseError(
        f"Model did not return a JSON object. Response began:\n{preview}"
    )


def validate_match_payload(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce a parsed payload into the documented shape.

    Returns the cleaned payload plus a list of human-readable notes about
    anything that had to be corrected -- a score out of range, a missing field,
    a string where a list belonged. The notes are surfaced in the output so a
    silently-repaired response is still visible to you.
    """
    notes: list[str] = []
    clean: dict[str, Any] = {}

    raw_score = data.get("match_score")
    score: float
    try:
        score = float(raw_score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        score = 0.0
        notes.append(f"match_score missing or non-numeric ({raw_score!r}); treated as 0")
    if not 0 <= score <= 100:
        notes.append(f"match_score {score} outside 0-100; clamped")
        score = max(0.0, min(100.0, score))
    clean["match_score"] = score

    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in {"strong", "moderate", "weak"}:
        derived = "strong" if score >= 80 else "moderate" if score >= 50 else "weak"
        if verdict:
            notes.append(f"unrecognised verdict {verdict!r}; derived {derived!r} from score")
        clean["verdict"] = derived
    else:
        clean["verdict"] = verdict

    clean["matching_strengths"] = _coerce_items(
        data.get("matching_strengths"), "matching_strengths", ("requirement", "evidence"), notes
    )
    clean["gaps"] = _coerce_items(
        data.get("gaps"), "gaps", ("requirement", "severity", "note"), notes
    )

    rationale = data.get("rationale")
    if not isinstance(rationale, str):
        rationale = "" if rationale is None else str(rationale)
        if rationale:
            notes.append("rationale was not a string; coerced")
    clean["rationale"] = rationale.strip()

    return clean, notes


def _coerce_items(
    value: Any, field_name: str, keys: tuple[str, ...], notes: list[str]
) -> list[dict[str, str]]:
    """Normalise a list-of-objects field, tolerating a list of bare strings."""
    if value is None:
        notes.append(f"{field_name} missing; treated as empty")
        return []
    if isinstance(value, str):
        notes.append(f"{field_name} was a string, not a list; wrapped")
        value = [value]
    if not isinstance(value, list):
        notes.append(f"{field_name} had unexpected type {type(value).__name__}; ignored")
        return []

    items: list[dict[str, str]] = []
    for entry in value:
        if isinstance(entry, str):
            items.append({keys[0]: entry})
        elif isinstance(entry, dict):
            items.append({k: str(entry.get(k, "") or "") for k in keys})
        else:
            notes.append(f"{field_name} contained a {type(entry).__name__}; skipped")
    return items


def to_match_result(
    payload: dict[str, Any], *, model: str, job_name: str, notes: list[str]
) -> MatchResult:
    """Map a validated payload onto the engine-neutral MatchResult."""
    return MatchResult(
        engine=ENGINE_API,
        score=payload["match_score"],
        verdict=payload["verdict"],
        strengths=[
            Strength(requirement=s.get("requirement", ""), evidence=s.get("evidence", ""))
            for s in payload["matching_strengths"]
            if s.get("requirement")
        ],
        gaps=[
            Gap(
                requirement=g.get("requirement", ""),
                severity=(g.get("severity") or "unknown").lower(),
                note=g.get("note", ""),
            )
            for g in payload["gaps"]
            if g.get("requirement")
        ],
        rationale=payload["rationale"],
        job_name=job_name,
        model=model,
        notes=notes,
    )


# --------------------------------------------------------------------------
# 4. COST TRACKING
# --------------------------------------------------------------------------


@dataclass
class CostTracker:
    """Per-call and running-session token/cost accounting."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    total_cost_usd: float = 0.0
    unpriced_calls: int = 0
    history: list[Usage] = field(default_factory=list)

    def record(self, usage: Usage) -> Usage:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_write_tokens += usage.cache_creation_input_tokens
        self.cache_read_tokens += usage.cache_read_input_tokens
        self.total_cost_usd += usage.cost_usd
        if not usage.priced:
            self.unpriced_calls += 1
        self.history.append(usage)
        return usage

    def format_call(self, usage: Usage, model: str) -> str:
        parts = [
            f"in {usage.input_tokens:,}",
            f"out {usage.output_tokens:,}",
        ]
        if usage.cache_creation_input_tokens:
            parts.append(f"cache-write {usage.cache_creation_input_tokens:,}")
        if usage.cache_read_input_tokens:
            parts.append(f"cache-read {usage.cache_read_input_tokens:,}")
        cost = format_usd(usage.cost_usd) if usage.priced else "unpriced model"
        return f"  tokens: {' | '.join(parts)}   cost: {cost}   [{model}]"

    def format_session(self) -> str:
        if self.calls == 0:
            return "  session: no API calls made"
        lines = [
            f"  session: {self.calls} call(s) | "
            f"in {self.input_tokens:,} | out {self.output_tokens:,}"
        ]
        if self.cache_read_tokens or self.cache_write_tokens:
            lines.append(
                f"           cache: {self.cache_write_tokens:,} written, "
                f"{self.cache_read_tokens:,} read (reads bill at 10% of input)"
            )
        total = format_usd(self.total_cost_usd)
        suffix = f" (+{self.unpriced_calls} unpriced)" if self.unpriced_calls else ""
        lines.append(f"           total cost: {total}{suffix}")
        return "\n".join(lines)

    def append_ledger(self, path: Path, model: str, job_name: str, usage: Usage) -> None:
        """Append one line to a cumulative JSONL spend ledger (best effort)."""
        record = {
            "model": model,
            "job": job_name,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
            "cost_usd": round(usage.cost_usd, 8),
        }
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            pass  # a ledger write must never take down a successful match


def usage_from_response(response: Any, model: str) -> Usage:
    """Read `response.usage` and price it.

    The cache fields are only present on some responses, hence getattr with a
    default rather than direct attribute access.
    """
    raw = response.usage
    return price_usage(
        model,
        input_tokens=getattr(raw, "input_tokens", 0) or 0,
        output_tokens=getattr(raw, "output_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
    )


# --------------------------------------------------------------------------
# 3. ERROR HANDLING + the call itself
# --------------------------------------------------------------------------

RETRYABLE_STATUSES = (408, 409, 429, 500, 502, 503, 504, 529)


class MatchClient:
    """Thin wrapper around `anthropic.Anthropic` for this one task."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        *,
        max_retries: int = 4,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        tracker: CostTracker | None = None,
    ):
        # max_retries=0 on the client: the SDK retries 429/5xx silently by
        # default, which would hide what is happening and double up with the
        # visible loop in `_create_with_retry`. One retry policy, one place.
        self.client = client or anthropic.Anthropic(max_retries=0)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self.tracker = tracker or CostTracker()

    # -- public API --------------------------------------------------------

    def match(
        self,
        profile: str,
        job_description: str,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        job_name: str = "",
        use_cache: bool = True,
        on_retry: Callable[[str], None] | None = None,
    ) -> tuple[MatchResult, Usage]:
        """Run one match. Raises MatchError subclasses; never raises SDK types."""
        request = build_request(
            profile,
            job_description,
            model=model,
            max_tokens=max_tokens,
            use_cache=use_cache,
        )
        response = self._create_with_retry(request, on_retry=on_retry)
        usage = self.tracker.record(usage_from_response(response, model))

        # A refusal is an HTTP 200 with no usable content -- check before
        # reading response.content, or you get a confusing parse error instead.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise MatchError(f"Model declined to answer (category: {category}).")

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

        if response.stop_reason == "max_tokens":
            raise MalformedResponseError(
                f"Response hit the {max_tokens}-token cap and the JSON is truncated. "
                "Re-run with a larger --max-tokens."
            )

        payload, notes = validate_match_payload(parse_match_json(text))
        result = to_match_result(payload, model=model, job_name=job_name, notes=notes)
        return result, usage

    def count_tokens(self, request: dict[str, Any]) -> int:
        """Exact input token count via the free count_tokens endpoint."""
        try:
            response = self.client.messages.count_tokens(
                model=request["model"],
                system=request["system"],
                messages=request["messages"],
            )
        except anthropic.APIError as exc:
            raise ApiCallError(f"count_tokens failed: {exc}") from exc
        return response.input_tokens

    # -- the retry / error-translation loop --------------------------------

    def _create_with_retry(
        self, request: dict[str, Any], on_retry: Callable[[str], None] | None = None
    ) -> Any:
        """Call messages.create, translating SDK errors into MatchError.

        The `except` chain runs most-specific-first. That ordering matters:
        every 4xx/5xx class below inherits from `APIStatusError`, so catching
        the base class first would swallow the specific ones and lose the
        retryable/non-retryable distinction.
        """
        attempt = 0
        while True:
            try:
                return self.client.messages.create(**request)

            # --- not retryable: the request or the credentials are wrong ---
            except anthropic.AuthenticationError as exc:  # 401
                raise ApiCallError(
                    f"Authentication failed: {exc}",
                    hint="ANTHROPIC_API_KEY is set but not valid. Check for a stale or "
                    "revoked key, or a stray quote/space in the exported value.",
                ) from exc
            except anthropic.PermissionDeniedError as exc:  # 403
                raise ApiCallError(
                    f"Permission denied: {exc}",
                    hint="This key cannot use that model. Check your workspace's model "
                    "access in the Anthropic Console.",
                ) from exc
            except anthropic.NotFoundError as exc:  # 404
                raise ApiCallError(
                    f"Model or endpoint not found: {exc}",
                    hint=f"{request['model']!r} is probably not a valid model ID. "
                    "Run `python match.py models` to see the known ones.",
                ) from exc
            except anthropic.BadRequestError as exc:  # 400
                raise ApiCallError(
                    f"Bad request: {exc}",
                    hint="The payload was rejected. Run the same command with "
                    "--dry-run to inspect exactly what is being sent.",
                ) from exc

            # --- retryable ---
            except anthropic.RateLimitError as exc:  # 429
                delay = self._retry_after(exc) or self._backoff(attempt)
                attempt = self._wait_or_fail(
                    attempt, delay, "rate limited (429)", exc, on_retry,
                    hint="Slow down, or request a limit increase in the Console.",
                )
            except anthropic.APIStatusError as exc:  # any other non-2xx
                if exc.status_code not in RETRYABLE_STATUSES:
                    raise ApiCallError(f"API error {exc.status_code}: {exc}") from exc
                attempt = self._wait_or_fail(
                    attempt, self._backoff(attempt),
                    f"server error ({exc.status_code})", exc, on_retry,
                    hint="Anthropic-side issue. See https://status.anthropic.com",
                )
            except anthropic.APIConnectionError as exc:  # network / timeout
                attempt = self._wait_or_fail(
                    attempt, self._backoff(attempt), "network failure", exc, on_retry,
                    hint="No response from the API. Check your internet connection, "
                    "or use --local to match offline at zero cost.",
                )

    def _wait_or_fail(
        self,
        attempt: int,
        delay: float,
        label: str,
        exc: Exception,
        on_retry: Callable[[str], None] | None,
        hint: str = "",
    ) -> int:
        """Sleep and return the next attempt number, or give up."""
        if attempt >= self.max_retries:
            raise ApiCallError(
                f"Giving up after {attempt + 1} attempt(s) - {label}: {exc}", hint=hint
            ) from exc
        if on_retry:
            on_retry(f"{label}; retrying in {delay:.1f}s "
                     f"(attempt {attempt + 2}/{self.max_retries + 1})")
        self._sleep(delay)
        return attempt + 1

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter, so parallel runs do not resynchronise."""
        return min(self.base_delay * (2**attempt) + random.uniform(0, 0.5), self.max_delay)

    @staticmethod
    def _retry_after(exc: anthropic.APIStatusError) -> float | None:
        """Honour the server's `retry-after` header when it sends one."""
        response = getattr(exc, "response", None)
        header = getattr(response, "headers", {}) or {}
        try:
            return float(header.get("retry-after"))
        except (TypeError, ValueError):
            return None
