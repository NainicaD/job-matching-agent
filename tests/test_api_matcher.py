"""Tests for the API path, driven by a fake client so nothing is billed.

Everything here that would cost money is faked: the fake client returns
canned `Message`-shaped objects, and the retry loop is given a `sleep` that
records delays instead of waiting. That makes it possible to exercise the parts
that are otherwise hard to reach on purpose -- a 429 with a `retry-after`
header, a truncated response, a refusal, a model that replies with prose
wrapped around its JSON.

Run directly (no pytest needed):

    python tests/test_api_matcher.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic

from jobmatch.api_matcher import (
    ApiCallError,
    CostTracker,
    MalformedResponseError,
    MatchClient,
    MatchError,
    build_request,
    estimate_tokens,
    parse_match_json,
    validate_match_payload,
)
from jobmatch.pricing import format_usd, price_usage

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


def make_response(
    text: str,
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
    cache_write: int = 0,
    stop_reason: str = "end_turn",
    stop_details: object | None = None,
) -> SimpleNamespace:
    """A stand-in for an `anthropic.types.Message`."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


def make_status_error(cls, status_code: int, headers: dict | None = None):
    """Build a real SDK exception instance (they need a real httpx2 response)."""
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status_code, headers=headers or {}, request=request)
    return cls("simulated", response=response, body=None)


class FakeMessages:
    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script: list):
        self.messages = FakeMessages(script)


VALID_PAYLOAD = {
    "match_score": 78,
    "verdict": "moderate",
    "matching_strengths": [
        {"requirement": "Python", "evidence": "Built pipelines in Python"},
        {"requirement": "PyTorch", "evidence": "Trained a transformer in PyTorch"},
    ],
    "gaps": [
        {"requirement": "Spark", "severity": "major", "note": "No distributed processing shown"}
    ],
    "rationale": "Strong ML fundamentals, no large-scale serving experience.",
}


def client_returning(text: str, **response_kwargs) -> MatchClient:
    delays: list[float] = []
    client = MatchClient(
        client=FakeClient([make_response(text, **response_kwargs)]),
        sleep=delays.append,
    )
    client.recorded_delays = delays  # type: ignore[attr-defined]
    return client


# --------------------------------------------------------------------------
# 1. Request construction
# --------------------------------------------------------------------------


def test_request_shape() -> None:
    request = build_request("PROFILE TEXT", "JOB TEXT", model="claude-haiku-4-5")

    assert request["model"] == "claude-haiku-4-5"
    assert isinstance(request["max_tokens"], int) and request["max_tokens"] > 0

    # Sampling and thinking params must be absent: they are rejected by one
    # model family or the other, and omitting them keeps one shape valid for all.
    for forbidden in ("temperature", "top_p", "top_k", "thinking", "output_config"):
        assert forbidden not in request, f"{forbidden} must not be sent"

    # Stable content in `system`, volatile content in `messages` -- the layout
    # prompt caching depends on.
    assert "PROFILE TEXT" in request["system"][1]["text"]
    assert "JOB TEXT" in request["messages"][0]["content"]
    assert "PROFILE TEXT" not in request["messages"][0]["content"]
    assert request["system"][1]["cache_control"] == {"type": "ephemeral"}

    no_cache = build_request("P", "J", use_cache=False)
    assert "cache_control" not in no_cache["system"][1]
    print("ok  request shape: cacheable prefix, no forbidden params")


def test_dry_run_matches_real_request() -> None:
    """--dry-run must print the request the API call would actually use."""
    args = ("PROFILE", "JOB")
    kwargs = {"model": "claude-sonnet-5", "max_tokens": 999, "use_cache": True}
    dry = build_request(*args, **kwargs)

    client = client_returning(json.dumps(VALID_PAYLOAD))
    client.match(*args, model="claude-sonnet-5", max_tokens=999, use_cache=True)
    sent = client.client.messages.calls[0]

    assert dry == sent, "dry-run payload differs from what match() sends"
    assert estimate_tokens(dry) > 0
    print("ok  dry-run payload is byte-identical to the real request")


# --------------------------------------------------------------------------
# 2. Response parsing
# --------------------------------------------------------------------------


def test_parse_bare_json() -> None:
    assert parse_match_json(json.dumps(VALID_PAYLOAD))["match_score"] == 78
    print("ok  parses bare JSON")


def test_parse_fenced_json() -> None:
    fenced = f"```json\n{json.dumps(VALID_PAYLOAD)}\n```"
    assert parse_match_json(fenced)["match_score"] == 78
    print("ok  parses JSON wrapped in a markdown code fence")


def test_parse_json_with_prose() -> None:
    noisy = f"Sure! Here is the assessment:\n\n{json.dumps(VALID_PAYLOAD)}\n\nHope that helps."
    assert parse_match_json(noisy)["match_score"] == 78
    print("ok  parses JSON surrounded by prose")


def test_parse_malformed_raises_not_crashes() -> None:
    for bad in ("not json at all", "", "{broken: ,}", "[1, 2, 3]"):
        try:
            parse_match_json(bad)
        except MalformedResponseError:
            continue
        raise AssertionError(f"expected MalformedResponseError for {bad!r}")
    print("ok  malformed JSON raises MalformedResponseError, never an uncaught crash")


def test_validation_repairs_and_reports() -> None:
    payload, notes = validate_match_payload(
        {
            "match_score": 140,                     # out of range
            "verdict": "excellent",                 # not in the allowed set
            "matching_strengths": ["Python", "SQL"],  # bare strings, not objects
            # "gaps" missing entirely
            "rationale": 12345,                     # wrong type
        }
    )
    assert payload["match_score"] == 100, payload
    assert payload["verdict"] == "strong"
    assert payload["matching_strengths"][0]["requirement"] == "Python"
    assert payload["gaps"] == []
    assert payload["rationale"] == "12345"
    assert len(notes) >= 4, notes
    print(f"ok  validation repaired 4 defects and reported all of them ({len(notes)} notes)")


def test_missing_score_does_not_crash() -> None:
    payload, notes = validate_match_payload({"verdict": "weak"})
    assert payload["match_score"] == 0.0
    assert any("match_score" in n for n in notes)
    print("ok  missing match_score degrades to 0 with a note")


# --------------------------------------------------------------------------
# 3. Error handling and retries
# --------------------------------------------------------------------------


def test_retries_429_then_succeeds() -> None:
    delays: list[float] = []
    messages_log: list[str] = []
    client = MatchClient(
        client=FakeClient(
            [
                make_status_error(anthropic.RateLimitError, 429, {"retry-after": "7"}),
                make_response(json.dumps(VALID_PAYLOAD)),
            ]
        ),
        sleep=delays.append,
    )
    result, _usage = client.match("P", "J", on_retry=messages_log.append)

    assert result.score == 78
    assert delays == [7.0], f"should honour retry-after, slept {delays}"
    assert "rate limited" in messages_log[0]
    print(f"ok  429 retried after the server's retry-after ({delays[0]:.0f}s), then succeeded")


def test_gives_up_after_max_retries() -> None:
    delays: list[float] = []
    client = MatchClient(
        client=FakeClient([make_status_error(anthropic.InternalServerError, 500)] * 3),
        max_retries=2,
        sleep=delays.append,
    )
    try:
        client.match("P", "J")
    except ApiCallError as exc:
        assert "Giving up after 3 attempt" in str(exc), str(exc)
        assert len(delays) == 2
        print(f"ok  500s retried {len(delays)}x with backoff {[round(d,1) for d in delays]}, "
              "then reported cleanly")
        return
    raise AssertionError("expected ApiCallError")


def test_auth_error_is_not_retried() -> None:
    delays: list[float] = []
    client = MatchClient(
        client=FakeClient([make_status_error(anthropic.AuthenticationError, 401)]),
        sleep=delays.append,
    )
    try:
        client.match("P", "J")
    except ApiCallError as exc:
        assert delays == [], "401 must not be retried"
        assert "ANTHROPIC_API_KEY" in exc.hint
        print("ok  401 fails immediately with an actionable hint (no wasted retries)")
        return
    raise AssertionError("expected ApiCallError")


def test_bad_model_suggests_models_command() -> None:
    client = MatchClient(
        client=FakeClient([make_status_error(anthropic.NotFoundError, 404)]), sleep=lambda _: None
    )
    try:
        client.match("P", "J", model="claude-nonexistent-9")
    except ApiCallError as exc:
        assert "claude-nonexistent-9" in exc.hint
        print("ok  404 names the bad model and points at `match.py models`")
        return
    raise AssertionError("expected ApiCallError")


def test_network_failure_suggests_local_mode() -> None:
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = MatchClient(
        client=FakeClient([anthropic.APIConnectionError(request=request)] * 2),
        max_retries=1,
        sleep=lambda _: None,
    )
    try:
        client.match("P", "J")
    except ApiCallError as exc:
        assert "--local" in exc.hint
        print("ok  network failure points the user at offline mode")
        return
    raise AssertionError("expected ApiCallError")


def test_truncated_response_is_reported() -> None:
    client = client_returning('{"match_score": 78, "verdi', stop_reason="max_tokens")
    try:
        client.match("P", "J", max_tokens=16)
    except MalformedResponseError as exc:
        assert "--max-tokens" in str(exc)
        print("ok  max_tokens truncation is diagnosed, not reported as bad JSON")
        return
    raise AssertionError("expected MalformedResponseError")


def test_refusal_is_handled() -> None:
    client = client_returning(
        "", stop_reason="refusal", stop_details=SimpleNamespace(category="cyber")
    )
    try:
        client.match("P", "J")
    except MatchError as exc:
        assert "declined" in str(exc) and "cyber" in str(exc)
        print("ok  a refusal stop_reason is reported as a refusal")
        return
    raise AssertionError("expected MatchError")


# --------------------------------------------------------------------------
# 4. Cost tracking
# --------------------------------------------------------------------------


def test_cost_arithmetic() -> None:
    # Haiku 4.5: $1.00 / MTok in, $5.00 / MTok out.
    usage = price_usage("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert abs(usage.cost_usd - 6.00) < 1e-9, usage.cost_usd

    # Cache reads bill at 10% of the input rate, writes at 125%.
    cached = price_usage(
        "claude-haiku-4-5", 0, 0,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    assert abs(cached.cost_usd - (1.25 + 0.10)) < 1e-9, cached.cost_usd
    print("ok  cost arithmetic matches published rates (incl. cache multipliers)")


def test_dated_model_alias_is_priced() -> None:
    dated = price_usage("claude-haiku-4-5-20251001", 1_000_000, 0)
    assert dated.priced and abs(dated.cost_usd - 1.00) < 1e-9
    print("ok  date-suffixed model IDs fold onto the base ID for pricing")


def test_unknown_model_reports_tokens_without_inventing_cost() -> None:
    usage = price_usage("some-future-model", 1000, 1000)
    assert usage.priced is False and usage.cost_usd == 0.0
    assert usage.total_tokens == 2000
    print("ok  unknown model: tokens reported, cost not fabricated")


def test_session_total_accumulates() -> None:
    tracker = CostTracker()
    for _ in range(3):
        tracker.record(price_usage("claude-haiku-4-5", 10_000, 1_000))
    assert tracker.calls == 3
    assert tracker.input_tokens == 30_000
    expected = 3 * (10_000 / 1e6 * 1.0 + 1_000 / 1e6 * 5.0)
    assert abs(tracker.total_cost_usd - expected) < 1e-9
    assert "3 call(s)" in tracker.format_session()
    print(f"ok  session total accumulates across calls ({format_usd(tracker.total_cost_usd)})")


def test_usage_read_from_response() -> None:
    client = client_returning(
        json.dumps(VALID_PAYLOAD), input_tokens=20_000, output_tokens=400, cache_read=18_000
    )
    _result, usage = client.match("P", "J")
    assert usage.input_tokens == 20_000
    assert usage.cache_read_input_tokens == 18_000
    assert client.tracker.calls == 1
    print("ok  token usage is read from response.usage, including cache fields")


# --------------------------------------------------------------------------
# 5. End to end
# --------------------------------------------------------------------------


def test_end_to_end_match() -> None:
    client = client_returning(json.dumps(VALID_PAYLOAD))
    result, usage = client.match("PROFILE", "JOB", job_name="acme.txt")
    assert result.engine == "anthropic-api"
    assert result.score == 78
    assert result.verdict == "moderate"
    assert [s.requirement for s in result.strengths] == ["Python", "PyTorch"]
    assert result.gaps[0].severity == "major"
    assert result.job_name == "acme.txt"
    assert not result.notes, f"clean payload should produce no repair notes: {result.notes}"
    assert usage.cost_usd > 0
    print("ok  end-to-end: response -> validated MatchResult + priced usage")


# --------------------------------------------------------------------------

TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = []
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures.append((test.__name__, exc))
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a test harness should report, not die
            failures.append((test.__name__, exc))
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print()
    print(
        f"{len(TESTS) - len(failures)}/{len(TESTS)} passed"
        + ("" if not failures else f" - {len(failures)} FAILED")
    )
    sys.exit(1 if failures else 0)
