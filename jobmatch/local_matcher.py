"""Local mode: offline, zero-cost matching with TF-IDF + cosine similarity.

This module is completely independent of the API path. It imports nothing from
`api_matcher`, `pricing`, or `anthropic` -- only `result`, which is stdlib-only
shared plumbing. That independence is a hard requirement (local mode must run
with no network, no API key, and no `anthropic` package installed) and it is
enforced by tests/test_local_isolation.py, not just by convention.

Three backends, in decreasing order of quality, selected at runtime:

  * `sentence-transformers` -- semantic embeddings, only when you pass
    --embeddings, because the first run downloads a model (~90 MB).
  * `scikit-learn`          -- TfidfVectorizer, used automatically if installed.
  * pure Python             -- the fallback implementation below, so local mode
    works on a bare interpreter with nothing installed at all.

What the score means
--------------------
The headline number is **keyword coverage**: the share of the job description's
most distinctive terms that your profile evidences. Raw cosine similarity is
reported alongside it, but it is a poor headline -- TF-IDF cosine between two
short documents sits in the 0.1-0.4 range even for an excellent match, which
reads as alarming next to an LLM's 0-100 judgement. Neither number is
numerically comparable to API mode; use local mode to *rank* jobs cheaply, then
spend API tokens on the top of that ranking.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .result import ENGINE_LOCAL, Gap, MatchResult, Strength

# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------

STOPWORDS = frozenset(
    """a about above after again against all am an and any are aren as at be
    because been before being below between both but by can cannot could
    couldn did didn do does doesn doing don down during each few for from
    further had hadn has hasn have haven having he her here hers herself him
    himself his how i if in into is isn it its itself just let ll me more most
    mustn my myself no nor not of off on once only or other ought our ours
    ourselves out over own re same shan she should shouldn so some such than
    that the their theirs them themselves then there these they this those
    through to too under until up ve very was wasn we were weren what when
    where which while who whom why will with won would wouldn you your yours
    yourself yourselves
    ability able across also applicant apply candidate candidates company
    description ensure etc experience including job must new please position
    preferred qualifications related required requirements role strong team
    well work working years
    summary overview responsibilities responsibility duties seeking join
    opportunity benefits salary applicants employer posting hire hiring
    day days week weeks month months full part time based including include
    """.split()
)

_WORD_RE = re.compile(r"[a-z][a-z0-9+#.]*")


# Derivational endings, longest first, mapped to a common stem fragment.
_SUFFIX_RULES = (
    ("ational", "at"),
    ("ization", "iz"),
    ("isation", "iz"),
    ("ation", "at"),
    ("ement", ""),
    ("ment", ""),
    ("ness", ""),
    ("ing", ""),
    ("ed", ""),
)


def _stem(word: str) -> str:
    """Reduce inflected forms of a word to one shared stem.

    Without this, a job asking to "coordinate" programs reports a gap against a
    profile full of "coordination" and "coordinating" -- a false gap that makes
    the whole report untrustworthy.

    The order of operations is what makes it work: strip the plural first, then
    one derivational ending, then a silent trailing "e". That maps coordinate /
    coordinates / coordinated / coordinating / coordination all onto "coordinat".
    Applying the rules in any other order leaves them in separate buckets.

    Not a real stemmer, and deliberately more aggressive than the one in
    `profile_builder` -- here over-merging costs a slightly generous match,
    while under-merging costs a false gap, which is the worse failure.
    """
    stem = word

    if len(stem) > 4 and stem.endswith("ies"):
        stem = stem[:-3] + "y"
    elif len(stem) > 4 and stem.endswith("sses"):
        stem = stem[:-2]
    elif len(stem) > 4 and stem.endswith("es") and not stem.endswith(("oes", "aes")):
        stem = stem[:-2]
    elif len(stem) > 3 and stem.endswith("s") and not stem.endswith("ss"):
        stem = stem[:-1]

    for suffix, replacement in _SUFFIX_RULES:
        if stem.endswith(suffix) and len(stem) - len(suffix) >= 3:
            stem = stem[: -len(suffix)] + replacement
            break

    if len(stem) > 4 and stem.endswith("e"):
        stem = stem[:-1]
    return stem


def _words(text: str) -> list[str]:
    words = [w.strip(".") for w in _WORD_RE.findall(text.lower())]
    return [w for w in words if len(w) >= 2 and w not in STOPWORDS]


def tokenize(text: str, use_bigrams: bool = True) -> list[str]:
    """Stemmed word tokens plus adjacent bigrams.

    Bigrams matter here: "machine learning" and "data pipeline" are single
    concepts, and unigrams alone would credit a profile that mentions "learning"
    and "machine" in unrelated places.
    """
    words = [_stem(w) for w in _words(text)]
    if not use_bigrams:
        return words
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def surface_forms(text: str) -> dict[str, str]:
    """Map each stemmed token back to a readable form as written in `text`.

    Tokens are stemmed for matching but "coordin" is not something to show a
    user, so strengths and gaps are displayed using the job description's own
    wording.
    """
    surfaces: dict[str, str] = {}
    words = _words(text)
    for word in words:
        surfaces.setdefault(_stem(word), word)
    for first, second in zip(words, words[1:]):
        surfaces.setdefault(f"{_stem(first)} {_stem(second)}", f"{first} {second}")
    return surfaces


# --------------------------------------------------------------------------
# Pure-Python TF-IDF
# --------------------------------------------------------------------------


@dataclass
class _Vector:
    weights: dict[str, float]
    norm: float


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> _Vector:
    """Sublinear TF (1 + log tf) times IDF, matching scikit-learn's default."""
    counts = Counter(tokens)
    weights: dict[str, float] = {}
    for term, count in counts.items():
        weight = idf.get(term)
        if weight is None:
            continue
        weights[term] = (1.0 + math.log(count)) * weight
    norm = math.sqrt(sum(w * w for w in weights.values()))
    return _Vector(weights, norm)


def _cosine(a: _Vector, b: _Vector) -> float:
    if not a.norm or not b.norm:
        return 0.0
    # Iterate the smaller vector -- the dot product only needs shared terms.
    small, large = (a, b) if len(a.weights) <= len(b.weights) else (b, a)
    dot = sum(weight * large.weights.get(term, 0.0) for term, weight in small.weights.items())
    return dot / (a.norm * b.norm)


def _build_idf(documents: list[list[str]]) -> dict[str, float]:
    """Smoothed IDF: log((1 + N) / (1 + df)) + 1."""
    n_docs = len(documents)
    df: Counter[str] = Counter()
    for tokens in documents:
        df.update(set(tokens))
    return {
        term: math.log((1 + n_docs) / (1 + count)) + 1.0 for term, count in df.items()
    }


# --------------------------------------------------------------------------
# Profile chunking
# --------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^==\s*(.+?)\s*==$")


@dataclass
class Chunk:
    section: str
    text: str


def chunk_profile(profile: str) -> list[Chunk]:
    """Split profile.txt into one chunk per bullet, tagged with its section.

    Chunks are the corpus the IDF statistics are computed over, and they are
    what "top matching sections" reports back.
    """
    chunks: list[Chunk] = []
    section = "GENERAL"
    for line in profile.splitlines():
        line = line.strip()
        if not line or line.startswith("# "):
            continue
        header = _SECTION_RE.match(line)
        if header:
            section = header.group(1)
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if line:
            chunks.append(Chunk(section=section, text=line))
    return chunks


# --------------------------------------------------------------------------
# The matcher
# --------------------------------------------------------------------------

TOP_JOB_TERMS = 30  # how many of the job's distinctive terms define "coverage"


def match_local(
    profile: str,
    job_description: str,
    *,
    job_name: str = "",
    top_terms: int = 10,
    top_chunks: int = 5,
    use_embeddings: bool = False,
    allow_sklearn: bool = True,
) -> MatchResult:
    """Score a profile against a job description, entirely offline."""
    chunks = chunk_profile(profile)
    if not chunks:
        raise ValueError("Profile is empty - run `python match.py build-profile` first.")

    backend, similarity, chunk_scores = _similarity(
        profile, job_description, chunks, use_embeddings, allow_sklearn
    )

    # Term-level analysis always uses the pure-Python TF-IDF path, so that the
    # strengths/gaps breakdown is identical no matter which backend scored the
    # documents. Only the headline similarity number varies by backend.
    chunk_tokens = [tokenize(c.text) for c in chunks]
    job_tokens = tokenize(job_description)
    idf = _build_idf(chunk_tokens + [job_tokens])

    profile_tokens = [t for tokens in chunk_tokens for t in tokens]
    profile_vec = _tfidf_vector(profile_tokens, idf)
    job_vec = _tfidf_vector(job_tokens, idf)
    profile_term_set = set(profile_vec.weights)

    # The job's most distinctive terms, by TF-IDF weight within the job text.
    ranked_job_terms = sorted(job_vec.weights.items(), key=lambda kv: -kv[1])
    key_terms = ranked_job_terms[:TOP_JOB_TERMS]
    covered = [t for t, _ in key_terms if t in profile_term_set]

    # Coverage is weighted by each term's importance to the job, not a plain
    # count. An unweighted count saturates: a profile built from 40 resume
    # versions has broad enough vocabulary to hit two thirds of *any* job's
    # terms, which made very different roles score identically. Weighting makes
    # missing the job's most distinctive term cost more than missing its 30th.
    total_weight = sum(weight for _, weight in key_terms)
    covered_weight = sum(
        weight for term, weight in key_terms if term in profile_term_set
    )
    coverage = covered_weight / total_weight if total_weight else 0.0

    overlapping = sorted(
        ((term, weight * profile_vec.weights[term])
         for term, weight in job_vec.weights.items()
         if term in profile_term_set),
        key=lambda kv: -kv[1],
    )
    missing = [term for term, _ in ranked_job_terms if term not in profile_term_set]

    # Rank chunks by their own similarity to the job, so the evidence quoted for
    # a term is the most job-relevant line containing it rather than whichever
    # line happens to come first in the profile.
    chunk_rank = dict(chunk_scores)
    used: set[int] = set()
    surfaces = surface_forms(job_description)
    strengths = [
        Strength(
            requirement=surfaces.get(term, term),
            evidence=_best_chunk_for(term, chunks, chunk_tokens, chunk_rank, used),
        )
        for term, _ in overlapping[:top_terms]
    ]
    gaps = [
        Gap(requirement=surfaces.get(term, term), severity="unknown",
            note="no occurrence of this term (or a variant of it) in the profile")
        for term in missing[:top_terms]
    ]

    best_chunks = sorted(chunk_scores, key=lambda kv: -kv[1])[:top_chunks]
    section_summary = ", ".join(
        f"{chunks[i].section.lower()} ({score:.2f})" for i, score in best_chunks if score > 0
    )

    verdict = (
        "strong-overlap" if coverage >= 0.60
        else "moderate-overlap" if coverage >= 0.35
        else "weak-overlap"
    )

    rationale = (
        f"Lexical overlap only: {len(covered)} of the job's {len(key_terms)} most "
        f"distinctive terms appear in the profile, weighted by importance to give "
        f"{coverage:.0%} coverage. Cosine similarity between the "
        f"full profile and the job description is {similarity:.3f}. "
        f"Best-matching profile entries came from: {section_summary or 'no section scored above zero'}."
    )

    return MatchResult(
        engine=ENGINE_LOCAL,
        score=round(coverage * 100, 1),
        verdict=verdict,
        strengths=strengths,
        gaps=gaps,
        rationale=rationale,
        job_name=job_name,
        model=f"offline / {backend}",
        notes=[
            "score = % of the job's key terms evidenced in the profile; "
            f"raw cosine similarity = {similarity:.3f}",
            "this is string overlap, not comprehension - a synonym counts as a gap",
            "not numerically comparable to API-mode scores; use it to rank jobs cheaply",
        ],
    )


def _best_chunk_for(
    term: str,
    chunks: list[Chunk],
    chunk_tokens: list[list[str]],
    chunk_rank: dict[int, float],
    used: set[int],
) -> str:
    """Quote the most job-relevant profile line that evidences `term`.

    One broad summary bullet tends to contain half the job's vocabulary, so
    lines already quoted for an earlier term are skipped while an unused
    alternative exists. That turns a wall of identical quotes into ten distinct
    pieces of evidence.
    """
    candidates = [i for i, tokens in enumerate(chunk_tokens) if term in tokens]
    if not candidates:
        return ""
    unused = [i for i in candidates if i not in used]
    best = max(unused or candidates, key=lambda i: chunk_rank.get(i, 0.0))
    used.add(best)
    text = chunks[best].text
    return text if len(text) <= 110 else text[:107] + "..."


def _similarity(
    profile: str,
    job_description: str,
    chunks: list[Chunk],
    use_embeddings: bool,
    allow_sklearn: bool,
) -> tuple[str, float, list[tuple[int, float]]]:
    """Score with the best available backend.

    Returns (backend name, profile-vs-job similarity, [(chunk index, score)]).
    """
    if use_embeddings:
        scored = _similarity_embeddings(profile, job_description, chunks)
        if scored is not None:
            return scored
        raise RuntimeError(
            "--embeddings requires sentence-transformers, which is not installed.\n"
            "  Install it with:  pip install sentence-transformers\n"
            "  (First run downloads a ~90 MB model. Omit --embeddings to use TF-IDF.)"
        )

    if allow_sklearn:
        scored = _similarity_sklearn(profile, job_description, chunks)
        if scored is not None:
            return scored

    return _similarity_pure_python(profile, job_description, chunks)


def _similarity_pure_python(
    profile: str, job_description: str, chunks: list[Chunk]
) -> tuple[str, float, list[tuple[int, float]]]:
    chunk_tokens = [tokenize(c.text) for c in chunks]
    job_tokens = tokenize(job_description)
    idf = _build_idf(chunk_tokens + [job_tokens])

    job_vec = _tfidf_vector(job_tokens, idf)
    profile_vec = _tfidf_vector([t for tokens in chunk_tokens for t in tokens], idf)
    chunk_scores = [
        (i, _cosine(_tfidf_vector(tokens, idf), job_vec))
        for i, tokens in enumerate(chunk_tokens)
    ]
    return "pure-python tf-idf", _cosine(profile_vec, job_vec), chunk_scores


def _similarity_sklearn(
    profile: str, job_description: str, chunks: list[Chunk]
) -> tuple[str, float, list[tuple[int, float]]] | None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return None

    corpus = [c.text for c in chunks] + [profile, job_description]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=sorted(STOPWORDS),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(corpus)
    job_row = matrix[-1]
    profile_row = matrix[-2]
    overall = float(cosine_similarity(profile_row, job_row)[0][0])
    per_chunk = cosine_similarity(matrix[: len(chunks)], job_row).ravel()
    return "scikit-learn tf-idf", overall, list(enumerate(float(s) for s in per_chunk))


def _similarity_embeddings(
    profile: str, job_description: str, chunks: list[Chunk]
) -> tuple[str, float, list[tuple[int, float]]] | None:
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        return None

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c.text for c in chunks] + [profile, job_description]
    vectors = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    job_vec = vectors[-1]
    overall = float(util.cos_sim(vectors[-2], job_vec))
    per_chunk = util.cos_sim(vectors[: len(chunks)], job_vec).squeeze(-1)
    return (
        "sentence-transformers/all-MiniLM-L6-v2",
        overall,
        list(enumerate(float(s) for s in per_chunk)),
    )
