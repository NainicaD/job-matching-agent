"""Consolidate every resume in ./resumes/ into a single profile.txt.

Design notes
------------
*Idempotence* is a hard requirement: running ``build-profile`` twice on an
unchanged ./resumes/ must produce a byte-identical profile.txt. That rules out
anything non-deterministic in the output -- in particular there is no generation
timestamp in the header (a content hash is used instead), files are processed in
sorted order, and de-duplication always keeps the first occurrence.

*Near-duplicates are flagged, never merged*: two different phrasings of the same
project are both real data (one may be tuned for an ML role, the other for a
program-coordinator role), so both stay in profile.txt and the pair is reported
in profile.review.md for a human to decide about.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .extract import SUPPORTED_SUFFIXES, ExtractionError, extract_text
from .repair import build_vocabulary, repair_text, residual_runon_ratio

# --------------------------------------------------------------------------
# Section classification
# --------------------------------------------------------------------------

# Canonical section order in the generated profile.
SECTION_ORDER = [
    "SUMMARY",
    "EDUCATION",
    "EXPERIENCE",
    "PROJECTS",
    "SKILLS",
    "PUBLICATIONS",
    "CERTIFICATIONS",
    "AWARDS",
    "LEADERSHIP",
    "COURSEWORK",
    "OTHER",
]

# Keyword -> canonical section. Matched against a normalized header line.
_SECTION_KEYWORDS: list[tuple[str, str]] = [
    ("summary", "SUMMARY"), ("objective", "SUMMARY"), ("profile", "SUMMARY"),
    ("about", "SUMMARY"),
    ("education", "EDUCATION"), ("academic", "EDUCATION"),
    ("experience", "EXPERIENCE"), ("employment", "EXPERIENCE"),
    ("work history", "EXPERIENCE"), ("internship", "EXPERIENCE"),
    ("professional", "EXPERIENCE"),
    ("project", "PROJECTS"), ("portfolio", "PROJECTS"),
    ("skill", "SKILLS"), ("technical", "SKILLS"), ("technolog", "SKILLS"),
    ("competenc", "SKILLS"), ("tools", "SKILLS"), ("languages", "SKILLS"),
    ("publication", "PUBLICATIONS"), ("research", "PUBLICATIONS"),
    ("paper", "PUBLICATIONS"),
    ("certification", "CERTIFICATIONS"), ("certificate", "CERTIFICATIONS"),
    ("licens", "CERTIFICATIONS"),
    ("award", "AWARDS"), ("honor", "AWARDS"), ("achievement", "AWARDS"),
    ("scholarship", "AWARDS"),
    ("leadership", "LEADERSHIP"), ("activit", "LEADERSHIP"),
    ("volunteer", "LEADERSHIP"), ("extracurricular", "LEADERSHIP"),
    ("involvement", "LEADERSHIP"), ("organization", "LEADERSHIP"),
    ("coursework", "COURSEWORK"), ("courses", "COURSEWORK"),
    ("relevant course", "COURSEWORK"),
]

_BULLET_CHARS = "•‣▪●·⁃-*−–—>"

# A file is only called out as badly extracted above this share of residual
# run-together tokens. Repair recovers most spacing damage, so a clean corpus
# sits at 0.00% and only genuinely broken PDFs clear the bar.
RESIDUAL_DAMAGE_THRESHOLD = 0.0025

# Contact details carry no matching signal but are PII, and profile.txt is what
# gets sent to the API. Redacted by default; use --keep-contact to disable.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
_URL_USER_RE = re.compile(r"(?i)\b(?:linkedin\.com|github\.com)/[\w./-]+")


@dataclass(frozen=True)
class ProfileLine:
    """One content line, tagged with where it came from."""

    text: str
    section: str
    source: str


@dataclass
class BuildReport:
    """Everything the CLI needs to print a useful summary."""

    files_read: list[str]
    files_failed: list[tuple[str, str]]
    raw_lines: int
    kept_lines: int
    duplicates_removed: int
    near_duplicate_clusters: list[list[ProfileLine]]
    profile_path: Path
    review_path: Path
    content_hash: str
    unchanged: bool
    ligatures_fixed: int = 0
    runons_split: int = 0
    low_quality_files: list[tuple[str, float]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build_profile(
    resumes_dir: Path,
    profile_path: Path,
    review_path: Path,
    *,
    keep_contact: bool = False,
    near_dup_threshold: float = 0.72,
) -> BuildReport:
    """Scan `resumes_dir`, consolidate, and write profile.txt + review file."""
    files = sorted(
        (p for p in resumes_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(
            f"No .pdf/.tex/.txt files found in {resumes_dir}. "
            "Put at least one resume there and re-run."
        )

    files_read: list[str] = []
    files_failed: list[tuple[str, str]] = []
    documents: list[str] = []

    # Pass 1: extract raw text from every file.
    for path in files:
        try:
            text = extract_text(path)
        except ExtractionError as exc:
            files_failed.append((path.name, str(exc)))
            continue
        files_read.append(path.name)
        documents.append(text)

    if not files_read:
        raise RuntimeError(
            "Every resume failed to parse:\n  "
            + "\n  ".join(f"{n}: {e}" for n, e in files_failed)
        )

    # Pass 2: repair extraction artifacts. The vocabulary is built from the whole
    # corpus first, because a bullet that garbles in one PDF usually extracts
    # cleanly from another resume version -- that is the repair evidence.
    vocab = build_vocabulary(documents)
    ligatures_fixed = 0
    runons_split = 0
    all_lines: list[ProfileLine] = []
    low_quality: list[tuple[str, float]] = []
    for name, text in zip(files_read, documents):
        text, ligatures, runons = repair_text(text, vocab)
        ligatures_fixed += ligatures
        runons_split += runons
        # Some PDFs are encoded so badly that repair cannot fully recover them.
        # Rather than split more aggressively (which would damage the other 40
        # files), measure the residue and name the file so it can be re-exported.
        damage = residual_runon_ratio(text, vocab)
        if damage > RESIDUAL_DAMAGE_THRESHOLD:
            low_quality.append((name, damage))
        all_lines.extend(_lines_from_document(text, name, keep_contact))
    low_quality.sort(key=lambda item: -item[1])

    kept, dupes_removed = _dedupe_exact(all_lines)
    clusters = _find_near_duplicates(kept, near_dup_threshold)

    body = _render_profile(kept)
    content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    document = _render_header(len(files_read), len(kept), content_hash) + body

    previous = profile_path.read_text(encoding="utf-8") if profile_path.exists() else None
    unchanged = previous == document
    profile_path.write_text(document, encoding="utf-8")
    review_path.write_text(
        _render_review(files_read, files_failed, clusters, content_hash, low_quality),
        encoding="utf-8",
    )

    return BuildReport(
        files_read=files_read,
        files_failed=files_failed,
        raw_lines=len(all_lines),
        kept_lines=len(kept),
        duplicates_removed=dupes_removed,
        near_duplicate_clusters=clusters,
        profile_path=profile_path,
        review_path=review_path,
        content_hash=content_hash,
        unchanged=unchanged,
        ligatures_fixed=ligatures_fixed,
        runons_split=runons_split,
        low_quality_files=low_quality,
    )


def load_profile(profile_path: Path) -> str:
    """Read profile.txt, stripping the generated `#` header comments."""
    if not profile_path.exists():
        raise FileNotFoundError(
            f"{profile_path} not found. Run `python match.py build-profile` first."
        )
    text = profile_path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("# ")]
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------
# Line extraction and classification
# --------------------------------------------------------------------------


# Words that cannot end a sentence -- if a line ends in one, the next line is a
# wrapped continuation even when it starts with a capital ("...interned at\nIIT").
_DANGLING_WORDS = frozenset(
    """a an and as at between by for from in into of on or over the through to
    under using with across within during against about after before while that
    which who whose is are was were has have had been being will would can could
    per via than then both either neither not but so yet""".split()
)

_TERMINAL_PUNCT = (".", "!", "?", ":", ";")


def _is_continuation(prev: str, raw_current: str, current: str) -> bool:
    """True if `current` is the wrapped remainder of `prev`, not a new entry.

    PDF text extraction emits one line per *visual* line, so a three-line bullet
    arrives as three fragments. Re-joining them matters: a fragment like
    "through data-driven solutions. With experience in" is noise on its own, and
    it also defeats duplicate detection across resume versions whose line breaks
    fall in different places.
    """
    if not prev or len(prev) > 600:
        return False
    if raw_current.lstrip()[:1] in _BULLET_CHARS:
        return False  # an explicit bullet always starts a new entry
    if _match_section_header(current) or _match_section_header(prev):
        return False
    if prev.endswith(_TERMINAL_PUNCT):
        return False

    first_alpha = next((c for c in current if c.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return True
    if prev.endswith((",", "-", "–", "/", "&")):
        return True
    tail = prev.rstrip().rstrip(",").split()
    return bool(tail) and tail[-1].lower() in _DANGLING_WORDS


def _reflow(text: str) -> list[str]:
    """Join wrapped visual lines back into logical lines."""
    logical: list[str] = []
    for raw in text.splitlines():
        line = _normalize_line(raw)
        if not line:
            logical.append("")  # a blank line is always a hard break
            continue
        prev = logical[-1] if logical else ""
        if prev and _is_continuation(prev, raw, line):
            if prev.endswith("-") and len(prev) > 1 and prev[-2].isalpha():
                logical[-1] = prev[:-1] + line  # de-hyphenate "docu-\nmentation"
            else:
                logical[-1] = f"{prev} {line}"
        else:
            logical.append(line)
    return [ln for ln in logical if ln]


def _lines_from_document(text: str, source: str, keep_contact: bool) -> list[ProfileLine]:
    out: list[ProfileLine] = []
    section = "SUMMARY"  # content before the first header is usually a summary
    for line in _reflow(text):
        header = _match_section_header(line)
        if header:
            section = header
            continue

        if not keep_contact:
            line = _redact(line)
            if not line:
                continue

        if _is_noise(line):
            continue

        out.append(ProfileLine(text=line, section=section, source=source))
    return out


def _normalize_line(raw: str) -> str:
    line = raw.replace(" ", " ").strip()
    line = line.lstrip(_BULLET_CHARS + " ").strip()
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def _match_section_header(line: str) -> str | None:
    """Return a canonical section if `line` looks like a resume section header."""
    stripped = line.strip().strip(":").strip()
    if not stripped or len(stripped) > 45:
        return None
    words = stripped.split()
    if len(words) > 4:
        return None
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return None
    # Headers are set in caps or title case, and never end in a sentence period.
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    is_titlecase = all(w[0].isupper() for w in words if w and w[0].isalpha())
    if upper_ratio < 0.6 and not is_titlecase:
        return None
    if stripped.endswith("."):
        return None
    low = stripped.lower()
    for keyword, canonical in _SECTION_KEYWORDS:
        if keyword in low:
            return canonical
    return None


def _redact(line: str) -> str:
    line = _EMAIL_RE.sub("[email]", line)
    line = _PHONE_RE.sub("[phone]", line)
    line = _URL_USER_RE.sub("[profile-url]", line)
    # A line that was *only* contact details now carries no information.
    residue = re.sub(r"\[(?:email|phone|profile-url)\]", "", line)
    if len(re.sub(r"[^A-Za-z]", "", residue)) < 3:
        return ""
    return re.sub(r"\s+", " ", line).strip(" |,•-").strip()


def _is_noise(line: str) -> bool:
    if len(line) < 4:
        return True
    if len(re.sub(r"[^A-Za-z]", "", line)) < 3:
        return True  # page numbers, rules, stray punctuation
    if re.fullmatch(r"(?i)page \d+( of \d+)?", line):
        return True
    return False


# --------------------------------------------------------------------------
# De-duplication
# --------------------------------------------------------------------------


def _dedupe_key(text: str) -> str:
    """Key for exact-duplicate detection: case/punctuation/spacing insensitive."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _dedupe_exact(lines: list[ProfileLine]) -> tuple[list[ProfileLine], int]:
    seen: set[tuple[str, str]] = set()
    kept: list[ProfileLine] = []
    removed = 0
    for line in lines:
        key = (line.section, _dedupe_key(line.text))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(line)
    return kept, removed


_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or
    that the to with using used via than then them they this was were will
    which who will your you our we i""".split()
)


def _stem(word: str) -> str:
    """Strip the handful of inflectional suffixes that matter here.

    Not a real stemmer -- just enough that two phrasings of the same bullet
    ("processing 2M rows/day" vs "processed 2M rows per day") are recognised as
    near-duplicates instead of falling under the similarity threshold on a
    verb-tense difference. Only used for near-duplicate clustering; exact
    de-duplication compares the literal text.
    """
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if len(w) >= 3 and w not in _STOPWORDS}


def _find_near_duplicates(
    lines: list[ProfileLine], threshold: float
) -> list[list[ProfileLine]]:
    """Cluster lines that are similar but not identical.

    A full O(n^2) comparison over a few thousand lines is wasteful, so pairs are
    generated from an inverted index built on each line's *rarest* tokens: two
    near-identical bullets are overwhelmingly likely to share an uncommon word.
    """
    token_sets = [_tokens(ln.text) for ln in lines]

    df: dict[str, int] = {}
    for ts in token_sets:
        for tok in ts:
            df[tok] = df.get(tok, 0) + 1

    index: dict[str, list[int]] = {}
    for i, ts in enumerate(token_sets):
        if len(ts) < 3:
            continue
        for tok in sorted(ts, key=lambda t: df[t])[:4]:
            index.setdefault(tok, []).append(i)

    parent = list(range(len(lines)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    checked: set[tuple[int, int]] = set()
    for bucket in index.values():
        if len(bucket) > 200:
            continue  # a token this common is not a useful near-duplicate signal
        for ai in range(len(bucket)):
            for bi in range(ai + 1, len(bucket)):
                a, b = bucket[ai], bucket[bi]
                if (a, b) in checked:
                    continue
                checked.add((a, b))
                ta, tb = token_sets[a], token_sets[b]
                union_size = len(ta | tb)
                if not union_size:
                    continue
                jaccard = len(ta & tb) / union_size
                if threshold <= jaccard < 1.0:
                    union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(len(lines)):
        groups.setdefault(find(i), []).append(i)

    clusters = [
        [lines[i] for i in sorted(members)]
        for members in groups.values()
        if len(members) > 1
    ]
    # Deterministic ordering: biggest clusters first, then by text.
    clusters.sort(key=lambda c: (-len(c), c[0].text))
    return clusters


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _render_header(n_files: int, n_lines: int, content_hash: str) -> str:
    return (
        "# Consolidated resume profile\n"
        "# Generated by job-matching-agent -- do not edit by hand.\n"
        f"# sources: {n_files} resume file(s) | lines: {n_lines} | content-hash: {content_hash}\n"
        "# Regenerate with: python match.py build-profile\n"
        "# (No timestamp on purpose -- the build is idempotent and this file must\n"
        "#  stay byte-identical when ./resumes/ has not changed.)\n"
        "\n"
    )


def _render_profile(lines: list[ProfileLine]) -> str:
    by_section: dict[str, list[ProfileLine]] = {}
    for line in lines:
        by_section.setdefault(line.section, []).append(line)

    parts: list[str] = []
    for section in SECTION_ORDER:
        entries = by_section.get(section)
        if not entries:
            continue
        parts.append(f"== {section} ==")
        parts.extend(f"- {e.text}" for e in entries)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_review(
    files_read: list[str],
    files_failed: list[tuple[str, str]],
    clusters: list[list[ProfileLine]],
    content_hash: str,
    low_quality: list[tuple[str, float]],
) -> str:
    out: list[str] = [
        "# Profile review",
        "",
        f"Content hash: `{content_hash}`",
        "",
        "Near-duplicate bullets are listed below. They are **kept in profile.txt**",
        "-- nothing was auto-merged. Edit ./resumes/ if you want one phrasing gone.",
        "",
        f"## Sources parsed ({len(files_read)})",
        "",
    ]
    out.extend(f"- {name}" for name in files_read)

    if files_failed:
        out += ["", f"## Files that failed to parse ({len(files_failed)})", ""]
        out.extend(f"- `{name}` -- {err}" for name, err in files_failed)

    if low_quality:
        out += [
            "",
            f"## Files that extract poorly ({len(low_quality)})",
            "",
            "These PDFs lost their space characters during extraction and could not be",
            "fully recovered. Their text is still included, but it will match less well.",
            "Re-exporting them from the original document usually fixes it.",
            "",
        ]
        out.extend(
            f"- `{name}` -- {ratio:.1%} of tokens still look run-together"
            for name, ratio in low_quality
        )

    out += ["", f"## Near-duplicate clusters ({len(clusters)})", ""]
    if not clusters:
        out.append("_None found._")
    for n, cluster in enumerate(clusters, 1):
        out.append(f"### {n}. {cluster[0].section} ({len(cluster)} variants)")
        out.append("")
        for line in cluster:
            out.append(f"- {line.text}")
            out.append(f"  <sub>from `{line.source}`</sub>")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
