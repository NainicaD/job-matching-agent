"""Tests for the retrieval stage: loading, chunking, scoring, invalidation.

None of these load the embedding model or touch the network. The parts that
would (Chroma, HuggingFace) are exercised through their inputs and outputs
instead -- `_aggregate` is handed synthetic chunk similarities, and
`_index_key` is pure arithmetic over strings. That keeps this file fast enough
to run on every change.

Run directly (no pytest needed):

    python tests/test_retrieval.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobmatch.postings import (
    Posting,
    PostingsError,
    load_postings,
    split_postings,
    split_profile,
)
from jobmatch.retrieval import (
    CHUNKS_PER_POSTING_SCORE,
    EmbeddingIndex,
    PostingScore,
    _aggregate,
    _similarity_from_distance,
    format_ranking,
    rank_tfidf,
)

POSTING_WITH_HEADER = """\
Title: Machine Learning Engineer
Company: Acme Labs
Location: Boston, MA

Responsibilities
- Build ranking models with PyTorch and evaluate them offline.
- Write SQL to turn raw event logs into training data.

Requirements
- Python, pandas, NumPy, scikit-learn.
- Experience with transformer architectures and retrieval systems.
"""

POSTING_NO_HEADER = """\
Senior Data Analyst, Reporting

Build dashboards in Power BI and write SQL against the warehouse.
Produce recurring reports for leadership stakeholders.
"""

SAMPLE_PROFILE = """\
== SKILLS ==
- Python, SQL, PyTorch, scikit-learn, pandas, NumPy

== PROJECTS ==
- Built a recommender with collaborative filtering and TF-IDF content filtering.
- Trained a transformer classifier in PyTorch with attention explainability.

== EXPERIENCE ==
- Wrote SQL pipelines turning raw event logs into model-ready datasets.
"""


def _write_postings(directory: Path) -> None:
    (directory / "ml_engineer_acme.txt").write_text(POSTING_WITH_HEADER, encoding="utf-8")
    (directory / "data_analyst_reporting.txt").write_text(POSTING_NO_HEADER, encoding="utf-8")


# --------------------------------------------------------------------------
# Loading and header parsing
# --------------------------------------------------------------------------


def test_header_parsing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write_postings(directory)
        postings = {p.slug: p for p in load_postings(directory)}

    acme = postings["ml_engineer_acme"]
    assert acme.title == "Machine Learning Engineer", acme.title
    assert acme.company == "Acme Labs"
    assert acme.location == "Boston, MA"
    assert acme.label == "Machine Learning Engineer - Acme Labs"

    # No header block: the first non-empty line is the title.
    analyst = postings["data_analyst_reporting"]
    assert analyst.title == "Senior Data Analyst, Reporting", analyst.title
    assert analyst.company == ""
    assert analyst.label == "Senior Data Analyst, Reporting"
    print("ok  header parsing: Title:/Company: fields, and first-line fallback")


def test_content_hash_tracks_content() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        path = directory / "role.txt"
        path.write_text(POSTING_WITH_HEADER, encoding="utf-8")
        first = load_postings(directory)[0].content_hash

        # Re-reading identical content must give an identical hash, or every
        # re-index would re-embed everything.
        again = load_postings(directory)[0].content_hash
        assert first == again, "hash is not stable across reads"

        path.write_text(POSTING_WITH_HEADER + "\n- One more requirement.\n", encoding="utf-8")
        changed = load_postings(directory)[0].content_hash

    assert first != changed, "editing a posting did not change its hash"
    print("ok  content hash is stable across reads and changes when edited")


def test_empty_postings_dir_is_a_clean_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_postings(Path(tmp))
        except PostingsError as exc:
            assert "No .txt/.md files" in str(exc)
            print("ok  empty job_postings/ gives a clear error, not a traceback")
            return
    raise AssertionError("expected PostingsError")


def test_missing_postings_dir_is_a_clean_error() -> None:
    try:
        load_postings(Path("/nonexistent/job_postings"))
    except PostingsError as exc:
        assert "does not exist" in str(exc)
        print("ok  missing job_postings/ gives a clear error with a fix")
        return
    raise AssertionError("expected PostingsError")


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def test_chunking_respects_size_and_attaches_metadata() -> None:
    posting = Posting(
        slug="role",
        path=Path("role.txt"),
        title="Machine Learning Engineer",
        company="Acme Labs",
        location="",
        text=POSTING_WITH_HEADER * 4,  # long enough to force several chunks
        content_hash="abc123",
    )
    chunks = split_postings([posting], chunk_size=300, chunk_overlap=60)

    assert len(chunks) > 1, "a long posting should split into multiple chunks"
    # The splitter honours chunk_size as a ceiling except where no separator
    # exists; allow a small margin rather than asserting an exact cut.
    assert all(len(c.page_content) <= 360 for c in chunks), [
        len(c.page_content) for c in chunks
    ]

    # Metadata is the whole reason for going through Document objects: it is how
    # a retrieved chunk is traced back to the posting it came from.
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["slug"] == "role"
        assert chunk.metadata["title"] == "Machine Learning Engineer"
        assert chunk.metadata["company"] == "Acme Labs"
        assert chunk.metadata["content_hash"] == "abc123"
        assert chunk.metadata["chunk_index"] == index
    print(f"ok  chunking: {len(chunks)} chunks, size respected, metadata attached")


def test_chunk_overlap_actually_overlaps() -> None:
    """Consecutive chunks should share text, or boundary phrases are lost."""
    posting = Posting(
        slug="role", path=Path("r.txt"), title="t", company="", location="",
        text=" ".join(f"word{i}" for i in range(400)), content_hash="h",
    )
    no_overlap = split_postings([posting], chunk_size=300, chunk_overlap=0)
    with_overlap = split_postings([posting], chunk_size=300, chunk_overlap=100)

    assert len(with_overlap) > len(no_overlap), "overlap should produce more chunks"

    def shares_text(chunks) -> bool:
        for first, second in zip(chunks, chunks[1:]):
            tail = set(first.page_content.split()[-5:])
            if tail & set(second.page_content.split()[:8]):
                return True
        return False

    assert shares_text(with_overlap), "overlapping chunks share no text"
    assert not shares_text(no_overlap), "zero-overlap chunks unexpectedly share text"
    print(f"ok  overlap works: {len(no_overlap)} chunks at 0 overlap -> "
          f"{len(with_overlap)} at 100, with shared boundary text")


def test_profile_is_chunked_not_truncated() -> None:
    """The profile must be split, not embedded whole - MiniLM truncates at ~256 tokens."""
    profile = SAMPLE_PROFILE * 20
    chunks = split_profile(profile, chunk_size=400, chunk_overlap=80)
    assert len(chunks) > 5, f"expected many chunks, got {len(chunks)}"
    assert all(chunk.strip() for chunk in chunks), "blank chunk emitted"
    # Content from the end of the profile must survive into some chunk. If the
    # profile were embedded whole, this tail is exactly what would be lost.
    assert any("event logs" in chunk for chunk in chunks), "tail content was dropped"
    print(f"ok  profile splits into {len(chunks)} query chunks, tail content preserved")


# --------------------------------------------------------------------------
# Distance -> similarity
# --------------------------------------------------------------------------


def test_similarity_from_cosine_distance() -> None:
    assert _similarity_from_distance(0.0) == 1.0  # identical vectors
    assert _similarity_from_distance(1.0) == 0.0  # orthogonal
    assert abs(_similarity_from_distance(0.25) - 0.75) < 1e-9
    # Opposing vectors give distance > 1; clamped, because for ranking purposes
    # "unrelated" and "actively opposite" are both just "not a match".
    assert _similarity_from_distance(1.7) == 0.0
    assert _similarity_from_distance(-0.1) == 1.0  # float noise below zero
    print("ok  cosine distance converts to a clamped 0-1 similarity")


# --------------------------------------------------------------------------
# Aggregation: the core scoring decision
# --------------------------------------------------------------------------


def _score_one(similarities: list[float]) -> float:
    """Run _aggregate for a single posting with the given chunk similarities."""
    best = {("role", i): s for i, s in enumerate(similarities)}
    titles = {"role": ("Role", "Co")}
    counts = {"role": set(range(len(similarities)))}
    result = _aggregate(best, titles, counts, {}, {"role": len(similarities)})
    return result[0].score


def test_aggregation_is_mean_of_best_n() -> None:
    # Five chunks; only the best three should count.
    score = _score_one([0.9, 0.8, 0.7, 0.1, 0.0])
    expected = (0.9 + 0.8 + 0.7) / 3 * 100
    assert abs(score - expected) < 0.05, (score, expected)
    assert CHUNKS_PER_POSTING_SCORE == 3
    print(f"ok  score is the mean of the best {CHUNKS_PER_POSTING_SCORE} chunks ({score:.1f})")


def test_aggregation_rejects_one_lucky_chunk() -> None:
    """A single boilerplate match must not carry an otherwise-unrelated posting.

    This is the reason the score is not `max`. Nearly every posting contains
    one "strong communication skills" paragraph that matches any profile.
    """
    lucky = _score_one([0.95, 0.05, 0.05, 0.05])
    genuine = _score_one([0.70, 0.65, 0.60, 0.55])
    assert genuine > lucky, (
        f"a posting matching in several places ({genuine:.1f}) must outrank one "
        f"with a single lucky hit ({lucky:.1f})"
    )
    print(f"ok  broad match {genuine:.1f} beats single lucky chunk {lucky:.1f} "
          "(why the score is not max)")


def test_aggregation_does_not_punish_long_postings() -> None:
    """Padding a good posting with filler must not sink it below a mediocre one.

    This is the reason the score is not a mean over *all* chunks.
    """
    good_but_padded = _score_one([0.8, 0.75, 0.7] + [0.02] * 12)
    mediocre_and_short = _score_one([0.45, 0.42])
    assert good_but_padded > mediocre_and_short, (
        f"padded strong posting {good_but_padded:.1f} should still beat "
        f"short mediocre one {mediocre_and_short:.1f}"
    )
    print(f"ok  padded strong posting {good_but_padded:.1f} beats short mediocre "
          f"{mediocre_and_short:.1f} (why the score is not a full mean)")


def test_aggregation_is_deterministic_and_sorted() -> None:
    best = {
        ("alpha", 0): 0.5, ("alpha", 1): 0.5,
        ("beta", 0): 0.9, ("beta", 1): 0.9,
        ("gamma", 0): 0.5, ("gamma", 1): 0.5,
    }
    titles = {s: (s.title(), "") for s in ("alpha", "beta", "gamma")}
    counts = {s: {0, 1} for s in ("alpha", "beta", "gamma")}
    totals = {s: 2 for s in ("alpha", "beta", "gamma")}

    first = _aggregate(best, titles, counts, {}, totals)
    second = _aggregate(dict(reversed(list(best.items()))), titles, counts, {}, totals)

    assert [s.slug for s in first] == [s.slug for s in second], "ordering is not stable"
    assert first[0].slug == "beta", "highest score should sort first"
    # alpha and gamma tie on score; slug breaks the tie so runs are reproducible.
    assert [s.slug for s in first] == ["beta", "alpha", "gamma"], [s.slug for s in first]
    print("ok  ranking is deterministic, ties broken by slug")


# --------------------------------------------------------------------------
# Index invalidation
# --------------------------------------------------------------------------


def test_index_key_covers_content_chunking_and_model() -> None:
    """All three inputs that change the stored vectors must change the key.

    Hashing only the posting text was a real bug: changing --chunk-size or
    --embedding-model left the key identical, so the store said "unchanged" and
    kept answering queries from vectors built under the old settings.
    """
    posting = Posting(
        slug="role", path=Path("r.txt"), title="t", company="", location="",
        text="body", content_hash="hash-a",
    )
    edited = Posting(**{**posting.__dict__, "content_hash": "hash-b"})

    mini = EmbeddingIndex(Path("/tmp/unused"), "model-mini")
    other = EmbeddingIndex(Path("/tmp/unused"), "model-other")

    base = mini._index_key(posting, 400, 80)
    assert base == mini._index_key(posting, 400, 80), "key is not stable"
    assert base != mini._index_key(edited, 400, 80), "content change not detected"
    assert base != mini._index_key(posting, 800, 80), "chunk_size change not detected"
    assert base != mini._index_key(posting, 400, 120), "chunk_overlap change not detected"
    assert base != other._index_key(posting, 400, 80), "model change not detected"
    print("ok  index key invalidates on content, chunk size, overlap, and model")


# --------------------------------------------------------------------------
# TF-IDF backend
# --------------------------------------------------------------------------


def test_tfidf_ranking_orders_sensibly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write_postings(directory)
        (directory / "registered_nurse.txt").write_text(
            "Title: Registered Nurse\n\nAdminister medications, monitor "
            "hemodynamic status, manage ventilated patients in the ICU.\n",
            encoding="utf-8",
        )
        postings = load_postings(directory)

    ranking = rank_tfidf(SAMPLE_PROFILE, postings)
    slugs = [s.slug for s in ranking]

    assert slugs[0] == "ml_engineer_acme", f"expected the ML posting first, got {slugs}"
    assert slugs[-1] == "registered_nurse", f"expected the nurse posting last, got {slugs}"
    assert all(0 <= s.score <= 100 for s in ranking)
    print(f"ok  tfidf ranking: {slugs[0]} first, {slugs[-1]} last")


def test_tfidf_rejects_empty_profile() -> None:
    postings = [
        Posting(slug="a", path=Path("a.txt"), title="A", company="", location="",
                text="python sql", content_hash="h")
    ]
    try:
        rank_tfidf("", postings)
    except Exception as exc:
        assert "build-profile" in str(exc)
        print("ok  empty profile tells you to run build-profile")
        return
    raise AssertionError("expected an error for an empty profile")


# --------------------------------------------------------------------------
# Output formatting
# --------------------------------------------------------------------------


def test_format_ranking_marks_the_cutoff() -> None:
    ranking = [
        PostingScore(slug=f"p{i}", title=f"Posting {i}", company="Co",
                     score=90.0 - i * 10, matched_chunks=3, total_chunks=3)
        for i in range(5)
    ]
    out = format_ranking(ranking, top_k=2, color=False)

    assert "->" in out, "selected rows should be marked"
    assert out.count("->") == 2, f"expected 2 marked rows:\n{out}"
    assert "top 2 selected for reasoning; 3 posting(s) filtered out" in out, out
    for score in ranking:
        assert score.title in out or score.title[:20] in out
    print("ok  ranking output marks the top-k cutoff and counts what was filtered")


def test_format_ranking_handles_empty() -> None:
    assert "no postings" in format_ranking([], top_k=5, color=False)
    print("ok  empty ranking renders a message, not a crash")


# --------------------------------------------------------------------------

TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = []
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures.append(test.__name__)
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(test.__name__)
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print()
    print(
        f"{len(TESTS) - len(failures)}/{len(TESTS)} passed"
        + ("" if not failures else f" - FAILED: {', '.join(failures)}")
    )
    sys.exit(1 if failures else 0)
