"""Repair two common PDF text-extraction artifacts.

Both of these show up in real resume PDFs and both silently poison matching --
a garbled token matches nothing, in either API mode or local mode.

1. **Broken ligatures.** Some embedded fonts map the `fi`/`fl` ligature glyphs
   to the wrong code point, so "financial" extracts as "kinancial" and
   "filtering" as "kiltering".

2. **Missing spaces.** Some PDFs position every glyph individually with no space
   characters at all, so a whole bullet extracts as one token:
   "coordinatingcrossfunctionalteams".

Both repairs are **vocabulary-gated**, using two separate tiers:

* the *dictionary* (the system word list) answers "is this a real English word";
* *corpus counts* (every token across all resumes) answer "which spelling does
  this collection actually use", and supply the weights for segmentation.

The two tiers matter because the corpus contains the garbled tokens too --
"kiltering" appears six times, so corpus membership alone would mark it valid.
A token is only rewritten when the repaired form beats the original on one of
those tiers, which is why real words like "workflows", "tracking", "scikit" and
"kind" are never touched.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

_SYSTEM_WORDLISTS = (
    Path("/usr/share/dict/words"),
    Path("/usr/dict/words"),
)

# Short words that are legitimate segmentation results.
_ALLOWED_SHORT = frozenset(
    """a i an as at be by do go he if in is it me my no of on or so to up us we
    and the for you our its not all any can has had was were with""".split()
)

_TOKEN_RE = re.compile(r"[A-Za-z]+")

_MIN_RUNON_LEN = 16
_MAX_WORD_LEN = 20


class Vocabulary:
    """Two-tier vocabulary: a real-word dictionary plus corpus frequencies."""

    def __init__(self, counts: dict[str, int], use_system_wordlist: bool = True):
        self.counts = counts
        self._total = sum(counts.values()) or 1
        self.dictionary: set[str] = set()

        if use_system_wordlist:
            for path in _SYSTEM_WORDLISTS:
                if not path.exists():
                    continue
                try:
                    raw = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                self.dictionary = {
                    w.lower() for w in raw.split() if len(w) >= 2 and w.isalpha()
                }
                break

        # Scoring weights for segmentation: corpus words by observed frequency,
        # dictionary-only words at a floor so corpus usage always wins a tie.
        #
        # Run-on tokens are themselves in `counts`, and leaving them in would let
        # a garbled fragment act as a legal segment -- "coordinatingcross" would
        # score better than "coordinating"+"cross", because a two-part split
        # sums fewer negative log-probabilities than a three-part one. So long
        # tokens that no dictionary recognises are excluded from the weights.
        scoring_counts = {
            word: count
            for word, count in counts.items()
            if len(word) < _MIN_RUNON_LEN or word in self.dictionary
        }
        self._logp: dict[str, float] = {
            word: math.log(count / self._total) for word, count in scoring_counts.items()
        }
        floor = math.log(0.1 / self._total)
        for word in self.dictionary:
            self._logp.setdefault(word, floor)
        # Single-letter words are filtered out of the corpus counts and the word
        # list, but "a" is needed to segment things like "maintainedastructured".
        for word in ("a", "i"):
            self._logp.setdefault(word, floor)

    @property
    def has_dictionary(self) -> bool:
        return bool(self.dictionary)

    def in_dictionary(self, word: str) -> bool:
        return word.lower() in self.dictionary

    def corpus_count(self, word: str) -> int:
        return self.counts.get(word.lower(), 0)

    def logp(self, word: str) -> float | None:
        return self._logp.get(word.lower())


def build_vocabulary(texts: list[str], use_system_wordlist: bool = True) -> Vocabulary:
    """Count every word across all extracted documents."""
    counts: dict[str, int] = {}
    for text in texts:
        for token in _TOKEN_RE.findall(text):
            if 2 <= len(token) <= 24:
                low = token.lower()
                counts[low] = counts.get(low, 0) + 1
    return Vocabulary(counts, use_system_wordlist=use_system_wordlist)


# --------------------------------------------------------------------------
# 1. Ligature repair
# --------------------------------------------------------------------------

# Mis-mapped glyph -> what it should have been.
_LIGATURE_FIXES = (("ki", "fi"), ("kl", "fl"))


def repair_ligatures(text: str, vocab: Vocabulary) -> tuple[str, int]:
    """Rewrite ki->fi / kl->fl where the repaired spelling is demonstrably right.

    Two independent pieces of evidence are accepted, so the repair still works
    on a machine with no system word list:

    * the repaired form is a dictionary word and the original is not; or
    * the repaired form is more common in this very corpus than the original
      (the same bullet usually extracts cleanly from another resume version).
    """
    fixed = 0

    def repair_token(match: re.Match[str]) -> str:
        nonlocal fixed
        token = match.group(0)
        low = token.lower()
        if len(low) < 4 or vocab.in_dictionary(low):
            return token  # a real word: never touch it

        for broken, correct in _LIGATURE_FIXES:
            if broken not in low:
                continue
            candidate = low.replace(broken, correct)
            dictionary_says_yes = vocab.in_dictionary(candidate)
            corpus_says_yes = vocab.corpus_count(candidate) > vocab.corpus_count(low)
            if dictionary_says_yes or corpus_says_yes:
                fixed += 1
                return _apply_case(token, candidate)
        return token

    return _TOKEN_RE.sub(repair_token, text), fixed


def _apply_case(original: str, replacement: str) -> str:
    """Carry the original token's capitalisation onto the repaired spelling."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


# --------------------------------------------------------------------------
# 2. Run-together word splitting
# --------------------------------------------------------------------------

# Where a lost space usually leaves a visible seam in a mixed-case run-on:
#   camelCase          "...builtAnalytics"   -> built | Analytics
#   acronym -> word    "SVDcollaborative"    -> SVD   | collaborative
#   acronym -> Word    "SAPGTSConsultant"    -> SAPGTS | Consultant
_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z][A-Z])(?=[a-z])|(?<=[A-Z])(?=[A-Z][a-z])"
)

# Pieces left over after a seam split are already short, so they get a lower
# run-on threshold than a whole token does.
_MIN_PIECE_LEN = 12


def _accept_segmentation(parts: list[str] | None, vocab: Vocabulary) -> bool:
    """Decide whether a candidate split is real evidence or an over-eager guess.

    Three or more segments is already strong evidence -- a random long word does
    not decompose into three known words by accident. A two-way split is much
    weaker ("hyperparameters" -> "hyper parameters"), so it is only accepted when
    *both* halves are words this corpus actually uses on their own.
    """
    if not parts or len(parts) < 2:
        return False
    if len(parts) == 2:
        return all(vocab.corpus_count(part) >= 2 for part in parts)
    return True


def split_runons(text: str, vocab: Vocabulary) -> tuple[str, int]:
    """Insert spaces into long unknown tokens that segment cleanly into words."""
    split_count = 0

    def repair_token(match: re.Match[str]) -> str:
        nonlocal split_count
        token = match.group(0)
        if len(token) < _MIN_RUNON_LEN or vocab.in_dictionary(token):
            return token

        if token.islower() or token.istitle():
            parts = _segment(token.lower(), vocab)
            if not _accept_segmentation(parts, vocab) or parts is None:
                return token
            split_count += 1
            joined = " ".join(parts)
            return joined.capitalize() if token.istitle() else joined

        # Mixed case: a lost space often survives as a camelCase boundary
        # ("...FormerSAPConsultantatDeloitte"). Split on those boundaries first,
        # then segment any all-lowercase piece that is still a run-on.
        pieces = _CAMEL_BOUNDARY.split(token)
        if len(pieces) < 2:
            return token
        out: list[str] = []
        for piece in pieces:
            if (
                len(piece) >= _MIN_PIECE_LEN
                and (piece.islower() or piece.istitle())
                and not vocab.in_dictionary(piece)
            ):
                segmented = _segment(piece.lower(), vocab)
                if _accept_segmentation(segmented, vocab) and segmented is not None:
                    joined = " ".join(segmented)
                    out.append(joined.capitalize() if piece.istitle() else joined)
                    continue
            out.append(piece)
        split_count += 1
        return " ".join(out)

    return _TOKEN_RE.sub(repair_token, text), split_count


def _segment(token: str, vocab: Vocabulary) -> list[str] | None:
    """Maximum-likelihood word segmentation via dynamic programming.

    best[i] holds the highest-scoring segmentation of token[:i]. Every segment
    must be a known word, so a token that cannot be fully segmented is left
    alone rather than being split on a guess.
    """
    n = len(token)
    best: list[tuple[float, list[str]] | None] = [None] * (n + 1)
    best[0] = (0.0, [])

    for i in range(1, n + 1):
        candidate: tuple[float, list[str]] | None = None
        for j in range(max(0, i - _MAX_WORD_LEN), i):
            prefix = best[j]
            if prefix is None:
                continue
            word = token[j:i]
            if len(word) < 3 and word not in _ALLOWED_SHORT:
                continue
            logp = vocab.logp(word)
            if logp is None:
                continue
            score = prefix[0] + logp
            if candidate is None or score > candidate[0]:
                candidate = (score, prefix[1] + [word])
        best[i] = candidate

    return best[n][1] if best[n] else None


# --------------------------------------------------------------------------


def residual_runon_ratio(text: str, vocab: Vocabulary) -> float:
    """Share of tokens that still look run-together *after* repair.

    Counts only tokens long enough to be a run-on, absent from the dictionary,
    **and** unsegmentable -- the last condition is what keeps genuine long words
    out of the count. Without it the measure just finds words missing from a
    1934-vintage word list ("interpretability", "explainability") and flags
    almost every file.
    """
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return 0.0
    residual = sum(
        1
        for token in tokens
        if len(token) >= _MIN_RUNON_LEN
        and not vocab.in_dictionary(token)
        and _segment(token.lower(), vocab) is None
    )
    return residual / len(tokens)


def repair_text(text: str, vocab: Vocabulary) -> tuple[str, int, int]:
    """Apply both repairs. Returns (text, ligatures_fixed, runons_split)."""
    text, ligatures = repair_ligatures(text, vocab)
    text, runons = split_runons(text, vocab)
    return text, ligatures, runons
