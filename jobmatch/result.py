"""The result shape shared by both matching modes, and how to print it.

This module is deliberately dependency-free (stdlib only, no Anthropic import,
no cost or error-handling code). Both `api_matcher` and `local_matcher` build a
`MatchResult`, which is what makes the two modes directly comparable -- and
keeping the shape *here* rather than in `api_matcher` is what lets local mode
stay independent of the API path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

# Engine identifiers, used for the banner and for --json consumers.
ENGINE_API = "anthropic-api"
ENGINE_LOCAL = "local-tfidf"

_BANNERS = {
    ENGINE_API: "LLM JUDGEMENT (Claude API)",
    ENGINE_LOCAL: "OFFLINE LEXICAL SIMILARITY (local TF-IDF) - not an LLM judgement",
}


@dataclass
class Strength:
    """Something the job asks for that the profile demonstrably has."""

    requirement: str
    evidence: str = ""


@dataclass
class Gap:
    """Something the job asks for that the profile does not show."""

    requirement: str
    severity: str = "unknown"  # blocker | major | minor | unknown
    note: str = ""


@dataclass
class MatchResult:
    engine: str
    score: float  # 0-100
    verdict: str
    strengths: list[Strength] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    rationale: str = ""
    job_name: str = ""
    model: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def banner(self) -> str:
        return _BANNERS.get(self.engine, self.engine)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["banner"] = self.banner
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# --------------------------------------------------------------------------
# Terminal rendering
# --------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_COLORS = {"green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m", "cyan": "\033[36m"}


def _c(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_COLORS.get(color, '')}{text}{_RESET}"


def _score_color(score: float) -> str:
    if score >= 70:
        return "green"
    if score >= 40:
        return "yellow"
    return "red"


def _bar(score: float, width: int = 28) -> str:
    filled = max(0, min(width, round(score / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def render(result: MatchResult, color: bool = True) -> str:
    """Format a MatchResult for the terminal. Identical layout for both engines."""
    bold = _BOLD if color else ""
    dim = _DIM if color else ""
    reset = _RESET if color else ""

    lines: list[str] = []
    title = result.job_name or "job description"
    lines.append(f"{bold}Match report - {title}{reset}")
    lines.append(_c(result.banner, "cyan", color))
    if result.model:
        lines.append(f"{dim}model: {result.model}{reset}")
    lines.append("")

    score_text = f"{result.score:.0f}/100"
    lines.append(
        f"  {_c(_bar(result.score), _score_color(result.score), color)} "
        f"{bold}{score_text}{reset}  {result.verdict}"
    )
    lines.append("")

    if result.strengths:
        lines.append(_c("  STRENGTHS", "green", color))
        for item in result.strengths:
            lines.append(f"    + {item.requirement}")
            if item.evidence:
                lines.append(f"{dim}        {item.evidence}{reset}")
        lines.append("")

    if result.gaps:
        lines.append(_c("  GAPS", "yellow", color))
        for gap in result.gaps:
            severity = f" [{gap.severity}]" if gap.severity != "unknown" else ""
            lines.append(f"    - {gap.requirement}{severity}")
            if gap.note:
                lines.append(f"{dim}        {gap.note}{reset}")
        lines.append("")

    if result.rationale:
        lines.append("  RATIONALE")
        for line in _wrap(result.rationale, 74):
            lines.append(f"    {line}")
        lines.append("")

    for note in result.notes:
        lines.append(f"{dim}  note: {note}{reset}")

    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out
