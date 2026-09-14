"""The retrieval (RAG) stage: pre-rank every job posting before spending a call.

This is the *vector* half of retrieval -- embeddings, the vector store,
similarity search, and turning chunk-level hits into a posting-level ranking.
`postings.py` is the document half (loading and chunking).

--------------------------------------------------------------------------
The shape of the pipeline
--------------------------------------------------------------------------

    job_postings/*.txt
        |  postings.load_postings()          -> Posting objects
        |  postings.split_postings()         -> Document chunks
        v
    embed each chunk (local model, free)
        |
        v
    Chroma vector store on disk  <-- persisted, so unchanged postings are
        |                            never re-embedded
        |  similarity search, once per profile chunk
        v
    chunk-level hits, aggregated per posting
        |
        v
    ranked postings -> top-k -> the EXISTING reasoning stage

Only the last arrow costs money. Everything above it is local and free, which
is the entire point: an LLM call per posting over 50 postings is real money,
while embedding 50 postings once is a few seconds of CPU.

--------------------------------------------------------------------------
What this module must NOT do
--------------------------------------------------------------------------
Retrieval runs ahead of *both* reasoning stages, including `--local`. So it
imports nothing from `api_matcher`, `pricing`, `preflight`, or `anthropic`, and
never needs an API key. tests/test_local_isolation.py enforces that.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .postings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Posting,
    PostingsError,
    split_postings,
    split_profile,
)

# A small, fast, well-tested sentence embedding model. 384 dimensions, ~90 MB,
# runs on CPU in milliseconds per chunk. Downloaded once from HuggingFace on
# first use and cached in ~/.cache/huggingface thereafter.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_STORE_DIR = "chroma_db"
COLLECTION_NAME = "job_postings"
DEFAULT_TOP_K = 5

# How many store hits to request per profile-chunk query. Larger means a more
# complete picture of which postings the profile touches, at negligible cost on
# a local store of this size.
DEFAULT_FANOUT = 25

# A posting's score is the mean of its best few chunk similarities. See
# _aggregate() for why neither max nor a full mean is the right choice.
CHUNKS_PER_POSTING_SCORE = 3

BACKEND_EMBEDDINGS = "hf"
BACKEND_TFIDF = "tfidf"


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot run, with instructions in the message."""


@dataclass
class PostingScore:
    """One posting's retrieval result."""

    slug: str
    title: str
    company: str
    score: float  # 0-100, comparable only within a single ranking
    matched_chunks: int
    total_chunks: int
    best_chunk: str = ""

    @property
    def label(self) -> str:
        if self.company and self.company not in self.title:
            return f"{self.title} - {self.company}"
        return self.title


@dataclass
class IndexReport:
    """What `index-postings` actually did."""

    backend: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks_embedded: int = 0
    total_chunks: int = 0
    store_dir: Path | None = None
    model: str = ""

    @property
    def did_work(self) -> bool:
        return bool(self.added or self.updated or self.removed)


# ==========================================================================
# Backend 1: LangChain + HuggingFace embeddings + Chroma  (the real RAG path)
# ==========================================================================


class EmbeddingIndex:
    """A persisted Chroma collection of job-posting chunks.

    Wraps LangChain's `Chroma` rather than using chromadb directly, so the
    embedding function, the store, and the search all speak the same `Document`
    interface that `postings.py` produces.
    """

    def __init__(
        self,
        store_dir: Path,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        allow_download: bool = True,
    ):
        self.store_dir = store_dir
        self.model_name = model_name
        self._allow_download = allow_download
        self._store = None  # built lazily: constructing it loads the model

    # -- construction ------------------------------------------------------

    @property
    def store(self):
        if self._store is None:
            self._store = self._open_store()
        return self._store

    def _open_store(self):
        """Open (or create) the on-disk Chroma collection."""
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RetrievalError(
                f"Embedding retrieval needs LangChain and Chroma ({exc}).\n"
                "  pip install -r requirements.txt\n"
                "  Or rank without them:  --retrieval-backend tfidf"
            ) from exc

        embeddings = self._load_embeddings(HuggingFaceEmbeddings)

        # `hnsw:space=cosine` is the important argument. Chroma defaults to
        # squared L2 distance; sentence embeddings are meant to be compared by
        # cosine similarity, and with cosine space the distance Chroma returns
        # is `1 - cosine_similarity`, which converts back to a similarity with
        # one subtraction (see _similarity_from_distance).
        #
        # Note this is fixed when the collection is first created. Changing it
        # later requires --rebuild.
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(self.store_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _load_embeddings(self, embeddings_cls):
        """Instantiate the local embedding model.

        `HuggingFaceEmbeddings` wraps sentence-transformers and adapts it to
        LangChain's `Embeddings` interface: `embed_documents(list[str])` for
        indexing and `embed_query(str)` for searching. The split exists because
        some models encode a query differently from a document; for MiniLM both
        are the same forward pass, but the store calls whichever is right.
        """
        try:
            return embeddings_cls(
                model_name=self.model_name,
                encode_kwargs={
                    # Unit-normalise the vectors so cosine distance and dot
                    # product agree, and so distances land in a predictable
                    # range. Without this, cosine space still works but the
                    # numbers are harder to reason about.
                    "normalize_embeddings": True,
                },
            )
        except Exception as exc:  # network failure, missing cache, bad model id
            hint = (
                "The model is downloaded once (~90 MB) and cached in "
                "~/.cache/huggingface. If you are offline and it was never "
                "downloaded, use --retrieval-backend tfidf, which needs no model."
            )
            raise RetrievalError(
                f"Could not load embedding model {self.model_name!r}: {exc}\n  {hint}"
            ) from exc

    # -- indexing ----------------------------------------------------------

    def sync(
        self,
        postings: list[Posting],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        rebuild: bool = False,
    ) -> IndexReport:
        """Bring the store in line with the postings folder, embedding only deltas.

        Embedding is the slow part, so each stored chunk carries an *index key*
        for its source posting. On a re-run, a posting whose key is unchanged is
        skipped entirely; a changed one has its old chunks deleted and the new
        ones embedded; a deleted file has its chunks removed. Chunk IDs are
        deterministic (`slug:index`), which is what makes delete-then-add safe
        and keeps repeated runs from stacking duplicates.

        The key covers more than the posting text -- see `_index_key`. Hashing
        only the text was a bug: changing `--chunk-size` or `--embedding-model`
        left every key identical, so the store reported "unchanged" and went on
        answering queries from vectors built under the *old* settings.
        """
        report = IndexReport(
            backend=BACKEND_EMBEDDINGS, store_dir=self.store_dir, model=self.model_name
        )

        if rebuild and self.store_dir.exists():
            shutil.rmtree(self.store_dir)
            self._store = None

        indexed = self._indexed_keys()
        on_disk = {posting.slug: posting for posting in postings}

        stale: list[str] = []
        to_embed: list[Posting] = []
        for slug, posting in on_disk.items():
            expected = self._index_key(posting, chunk_size, chunk_overlap)
            known = indexed.get(slug)
            if known is None:
                report.added.append(slug)
                to_embed.append(posting)
            elif known != expected:
                report.updated.append(slug)
                stale.append(slug)
                to_embed.append(posting)
            else:
                report.unchanged.append(slug)

        for slug in indexed:
            if slug not in on_disk:
                report.removed.append(slug)
                stale.append(slug)

        if stale:
            self._delete_slugs(stale)

        if to_embed:
            chunks = split_postings(to_embed, chunk_size, chunk_overlap)
            keys = {
                posting.slug: self._index_key(posting, chunk_size, chunk_overlap)
                for posting in to_embed
            }
            for chunk in chunks:
                chunk.metadata["index_key"] = keys[chunk.metadata["slug"]]
            ids = [f"{c.metadata['slug']}:{c.metadata['chunk_index']}" for c in chunks]
            self.store.add_documents(documents=chunks, ids=ids)
            report.chunks_embedded = len(chunks)

        report.total_chunks = self._count_chunks()
        return report

    def _index_key(self, posting: Posting, chunk_size: int, chunk_overlap: int) -> str:
        """Fingerprint everything that would change a posting's stored vectors.

        Content, chunk geometry, and the embedding model all determine the
        vectors, so all three belong in the key. Change any one of them and the
        posting is correctly treated as stale and re-embedded.
        """
        material = f"{posting.content_hash}|{chunk_size}|{chunk_overlap}|{self.model_name}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def _indexed_keys(self) -> dict[str, str]:
        """slug -> index_key for everything currently in the store."""
        try:
            existing = self.store.get(include=["metadatas"])
        except Exception:
            return {}  # a fresh or unreadable store is simply empty
        keys: dict[str, str] = {}
        for metadata in existing.get("metadatas") or []:
            if metadata and metadata.get("slug"):
                # Missing index_key means the chunk predates this scheme; treat
                # it as stale so it gets rebuilt rather than trusted.
                keys[metadata["slug"]] = metadata.get("index_key", "")
        return keys

    def _delete_slugs(self, slugs: list[str]) -> None:
        existing = self.store.get(include=["metadatas"])
        ids = existing.get("ids") or []
        metadatas = existing.get("metadatas") or []
        doomed = [
            chunk_id
            for chunk_id, metadata in zip(ids, metadatas)
            if metadata and metadata.get("slug") in set(slugs)
        ]
        if doomed:
            self.store.delete(ids=doomed)

    def _count_chunks(self) -> int:
        try:
            return len(self.store.get(include=[]).get("ids") or [])
        except Exception:
            return 0

    # -- search ------------------------------------------------------------

    def rank(
        self,
        profile: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        fanout: int = DEFAULT_FANOUT,
    ) -> list[PostingScore]:
        """Rank every indexed posting by similarity to the profile.

        One similarity search per *profile* chunk, rather than one search for
        the whole profile. That is the point made in `split_profile`: a single
        query vector for a 90 KB profile would encode only its opening lines.
        Querying with each chunk asks "which postings does this specific piece
        of experience speak to?" and lets the aggregation below combine the
        answers.
        """
        queries = split_profile(profile, chunk_size, chunk_overlap)
        if not queries:
            raise RetrievalError("Profile is empty - run `build-profile` first.")

        titles: dict[str, tuple[str, str]] = {}
        chunk_counts: dict[str, set[int]] = {}
        # (slug, chunk_index) -> best similarity seen from any profile chunk
        best_per_chunk: dict[tuple[str, int], float] = {}
        best_text: dict[str, tuple[float, str]] = {}

        for query in queries:
            try:
                hits = self.store.similarity_search_with_score(query, k=fanout)
            except Exception as exc:
                raise RetrievalError(f"Similarity search failed: {exc}") from exc

            for document, distance in hits:
                metadata = document.metadata or {}
                slug = metadata.get("slug")
                if not slug:
                    continue
                index = int(metadata.get("chunk_index", 0))
                similarity = _similarity_from_distance(distance)

                titles.setdefault(slug, (metadata.get("title", slug), metadata.get("company", "")))
                chunk_counts.setdefault(slug, set()).add(index)

                key = (slug, index)
                if similarity > best_per_chunk.get(key, -1.0):
                    best_per_chunk[key] = similarity
                if similarity > best_text.get(slug, (-1.0, ""))[0]:
                    best_text[slug] = (similarity, document.page_content)

        total_chunks = self._chunks_per_slug()
        return _aggregate(best_per_chunk, titles, chunk_counts, best_text, total_chunks)

    def _chunks_per_slug(self) -> dict[str, int]:
        try:
            existing = self.store.get(include=["metadatas"])
        except Exception:
            return {}
        counts: dict[str, int] = {}
        for metadata in existing.get("metadatas") or []:
            if metadata and metadata.get("slug"):
                counts[metadata["slug"]] = counts.get(metadata["slug"], 0) + 1
        return counts


def _similarity_from_distance(distance: float) -> float:
    """Convert a Chroma cosine distance into a 0-1 similarity.

    With `hnsw:space=cosine`, Chroma returns `1 - cosine_similarity`, so a
    perfect match is 0 and an orthogonal one is 1. Doing the conversion here
    rather than calling `similarity_search_with_relevance_scores` keeps the
    arithmetic visible instead of relying on LangChain's internal mapping,
    which differs per distance metric.

    Negative cosine similarity (opposing vectors) is clamped to 0: for ranking
    purposes "unrelated" and "actively opposite" are both simply not a match.
    """
    return max(0.0, min(1.0, 1.0 - float(distance)))


def _aggregate(
    best_per_chunk: dict[tuple[str, int], float],
    titles: dict[str, tuple[str, str]],
    chunk_counts: dict[str, set[int]],
    best_text: dict[str, tuple[float, str]],
    total_chunks: dict[str, int],
) -> list[PostingScore]:
    """Turn chunk-level similarities into one score per posting.

    Search returns *chunks*; the decision is about *postings*, so the scores
    have to be combined. Three obvious options, and why this picks the third:

      * **max** -- one chunk's score decides. Too generous: almost every
        posting contains a "strong communication skills" paragraph that matches
        any profile, so unrelated postings score as highly as real ones.
      * **mean over all chunks** -- penalises long postings. A great match that
        happens to include four paragraphs of benefits boilerplate scores below
        a short mediocre one.
      * **mean of the best N chunks** (N=3, used here) -- requires that
        *several* parts of the posting connect to the profile, without
        punishing a posting for also containing filler. It is the same
        intuition as `recall@k`: one hit is noise, a cluster of hits is signal.

    Scores are scaled to 0-100 for display. They are only meaningful *relative
    to each other within one ranking* -- this is a sort key for deciding where
    to spend API calls, not a percentage match.
    """
    by_slug: dict[str, list[float]] = {}
    for (slug, _index), similarity in best_per_chunk.items():
        by_slug.setdefault(slug, []).append(similarity)

    scores: list[PostingScore] = []
    for slug, similarities in by_slug.items():
        similarities.sort(reverse=True)
        top = similarities[:CHUNKS_PER_POSTING_SCORE]
        score = sum(top) / len(top) if top else 0.0
        title, company = titles.get(slug, (slug, ""))
        scores.append(
            PostingScore(
                slug=slug,
                title=title,
                company=company,
                score=round(score * 100, 1),
                matched_chunks=len(chunk_counts.get(slug, ())),
                total_chunks=total_chunks.get(slug, len(chunk_counts.get(slug, ()))),
                best_chunk=best_text.get(slug, (0.0, ""))[1],
            )
        )

    # Deterministic ordering: score first, slug as the tie-break.
    scores.sort(key=lambda s: (-s.score, s.slug))
    return scores


# ==========================================================================
# Backend 2: TF-IDF  (no LangChain, no model download, no network at all)
# ==========================================================================


def rank_tfidf(profile: str, postings: list[Posting]) -> list[PostingScore]:
    """Rank postings with TF-IDF cosine similarity, entirely in the standard library.

    This is the honest fallback for the "zero network" guarantee. The embedding
    backend needs a one-time model download; this one needs nothing, because it
    reuses the TF-IDF machinery already in `local_matcher`.

    It is a genuinely weaker ranker -- it matches strings, so "recommender
    systems" and "collaborative filtering" look unrelated to it. Use it when
    offline or when you do not want a 2 GB torch install; use the embedding
    backend when you want retrieval that understands paraphrase.

    Note there is no vector store here: with a corpus this size, recomputing
    TF-IDF costs milliseconds, so persistence would be complexity for nothing.
    """
    # Reusing the existing implementation rather than writing a second one.
    from .local_matcher import _build_idf, _cosine, _tfidf_vector, tokenize

    posting_tokens = [tokenize(posting.text) for posting in postings]
    profile_tokens = tokenize(profile)
    if not profile_tokens:
        raise RetrievalError("Profile is empty - run `build-profile` first.")

    # IDF over postings + profile, so terms common to every posting ("team",
    # "experience") carry little weight and distinctive ones dominate.
    idf = _build_idf(posting_tokens + [profile_tokens])
    profile_vector = _tfidf_vector(profile_tokens, idf)

    scores = [
        PostingScore(
            slug=posting.slug,
            title=posting.title,
            company=posting.company,
            score=round(_cosine(_tfidf_vector(tokens, idf), profile_vector) * 100, 1),
            matched_chunks=1,
            total_chunks=1,
        )
        for posting, tokens in zip(postings, posting_tokens)
    ]
    scores.sort(key=lambda s: (-s.score, s.slug))
    return scores


# ==========================================================================
# Public entry points used by the CLI
# ==========================================================================


def build_index(
    postings: list[Posting],
    store_dir: Path,
    *,
    backend: str = BACKEND_EMBEDDINGS,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    rebuild: bool = False,
) -> IndexReport:
    """Create or update the vector store. No-op for the tfidf backend."""
    if backend == BACKEND_TFIDF:
        return IndexReport(
            backend=BACKEND_TFIDF,
            unchanged=[p.slug for p in postings],
            total_chunks=len(postings),
        )
    index = EmbeddingIndex(store_dir, model_name)
    return index.sync(
        postings, chunk_size=chunk_size, chunk_overlap=chunk_overlap, rebuild=rebuild
    )


def rank_postings(
    profile: str,
    postings: list[Posting],
    store_dir: Path,
    *,
    backend: str = BACKEND_EMBEDDINGS,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    fanout: int = DEFAULT_FANOUT,
    auto_index: bool = True,
) -> tuple[list[PostingScore], IndexReport | None]:
    """Rank all postings against the profile. Returns (ranking, index work done)."""
    if backend == BACKEND_TFIDF:
        return rank_tfidf(profile, postings), None

    index = EmbeddingIndex(store_dir, model_name)
    report: IndexReport | None = None
    if auto_index:
        # Ranking against a stale store would silently score the wrong text, so
        # `rank` syncs first. Unchanged postings cost nothing to check.
        report = index.sync(postings, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ranking = index.rank(
        profile, chunk_size=chunk_size, chunk_overlap=chunk_overlap, fanout=fanout
    )

    # A posting in the folder but absent from the ranking matched nothing at
    # all. Report it at zero rather than dropping it, so the printed ranking
    # always accounts for every posting on disk.
    ranked_slugs = {score.slug for score in ranking}
    for posting in postings:
        if posting.slug not in ranked_slugs:
            ranking.append(
                PostingScore(
                    slug=posting.slug,
                    title=posting.title,
                    company=posting.company,
                    score=0.0,
                    matched_chunks=0,
                    total_chunks=0,
                )
            )
    return ranking, report


def format_ranking(
    ranking: list[PostingScore], top_k: int | None = None, color: bool = True
) -> str:
    """Render the ranked list, marking where the top-k cutoff falls."""
    if not ranking:
        return "  (no postings ranked)"

    dim = "\033[2m" if color else ""
    bold = "\033[1m" if color else ""
    reset = "\033[0m" if color else ""

    width = max(len(s.label) for s in ranking)
    width = min(width, 52)

    lines = [
        f"  {'#':>2}  {'score':>5}  {'posting':<{width}}  chunks",
        f"  {'-' * (width + 22)}",
    ]
    for position, score in enumerate(ranking, 1):
        selected = top_k is not None and position <= top_k
        marker = f"{bold}->{reset}" if selected else "  "
        label = score.label if len(score.label) <= width else score.label[: width - 3] + "..."
        row = (
            f"{marker}{position:>2}  {score.score:>5.1f}  {label:<{width}}  "
            f"{score.matched_chunks}/{score.total_chunks}"
        )
        lines.append(row if selected else f"{dim}{row}{reset}")

    if top_k is not None and top_k < len(ranking):
        skipped = len(ranking) - top_k
        lines.append("")
        lines.append(
            f"  top {top_k} selected for reasoning; {skipped} posting(s) filtered out "
            f"before any API call"
        )
    return "\n".join(lines)
