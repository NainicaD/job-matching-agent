"""Load and chunk job postings from ./job_postings/ using LangChain.

This is the *document* half of the retrieval stage. It answers two questions:

  1. How do raw files become `Document` objects with metadata attached?
  2. How does a long posting get cut into pieces small enough to embed?

`retrieval.py` handles the *vector* half (embeddings, store, search).

--------------------------------------------------------------------------
Why LangChain here at all?
--------------------------------------------------------------------------
For a folder of .txt files, `path.read_text()` would genuinely be simpler, and
it is worth being honest about that. What `TextLoader` buys is not convenience
today but two things later:

  * **A uniform Document interface.** Every loader returns the same
    `Document(page_content=..., metadata={...})` shape. That metadata dict is
    what flows through the splitter into the vector store and comes back out of
    a similarity search, which is how a matching chunk is traced back to the
    posting it came from. Rolling that by hand means re-inventing it.
  * **Swappable sources.** Pointing this at PDFs (`PyPDFLoader`), web pages
    (`WebBaseLoader`), or Notion (`NotionDirectoryLoader`) is a one-line change
    because the rest of the pipeline only sees `Document`s.

So: the abstraction is doing nothing impressive on a .txt file. It is doing the
work when the source changes, and the metadata plumbing matters immediately.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # imported lazily at runtime - see _require_langchain()
    from langchain_core.documents import Document

POSTING_SUFFIXES = {".txt", ".md"}

# Chunking defaults. See split_postings() for why these numbers.
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 80

# `Title: ...` / `Company: ...` style header lines at the top of a posting.
_HEADER_RE = re.compile(r"^\s*(title|company|employer|location|role)\s*:\s*(.+?)\s*$", re.I)


@dataclass(frozen=True)
class Posting:
    """One job posting file, with its header metadata parsed out."""

    slug: str  # filename stem: the stable identity used in the vector store
    path: Path
    title: str
    company: str
    location: str
    text: str
    content_hash: str  # drives incremental re-indexing

    @property
    def label(self) -> str:
        if self.company and self.company not in self.title:
            return f"{self.title} - {self.company}"
        return self.title


class PostingsError(RuntimeError):
    """Raised when the postings folder is missing or unusable."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_postings(directory: Path) -> list[Posting]:
    """Load every posting in `directory`, sorted by filename for determinism."""
    if not directory.is_dir():
        raise PostingsError(
            f"{directory} does not exist.\n"
            f"  Create it and add one .txt file per job posting:\n"
            f"      mkdir -p {directory}"
        )

    paths = sorted(
        (p for p in directory.iterdir() if p.suffix.lower() in POSTING_SUFFIXES),
        key=lambda p: p.name,
    )
    if not paths:
        raise PostingsError(
            f"No .txt/.md files found in {directory}. "
            "Add one file per job posting and re-run."
        )

    postings: list[Posting] = []
    for path in paths:
        text = _load_one(path)
        if not text.strip():
            continue
        title, company, location = _parse_header(text, fallback=path.stem)
        postings.append(
            Posting(
                slug=path.stem,
                path=path,
                title=title,
                company=company,
                location=location,
                text=text,
                # Hash the content, not the mtime: touching a file should not
                # force a re-embed, and editing it must.
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            )
        )
    return postings


def _load_one(path: Path) -> str:
    """Read one posting through LangChain's TextLoader.

    `TextLoader.load()` returns a list of Documents (one per file here; other
    loaders split by page or section). We take `.page_content` because the
    metadata we want is richer than the `{"source": path}` it sets by default.
    """
    loader_cls = _require_langchain()
    if loader_cls is None:
        # LangChain absent: fall back to a plain read so the tfidf retrieval
        # backend stays usable with zero third-party dependencies.
        return path.read_text(encoding="utf-8", errors="replace")

    documents = loader_cls(str(path), encoding="utf-8", autodetect_encoding=True).load()
    return "\n".join(doc.page_content for doc in documents)


def _require_langchain():
    """Return TextLoader, or None when LangChain is not installed."""
    try:
        from langchain_community.document_loaders import TextLoader
    except ImportError:
        return None
    return TextLoader


def _parse_header(text: str, fallback: str) -> tuple[str, str, str]:
    """Pull title/company/location from header lines, or guess from the filename.

    Only the first few lines are scanned, so a "Location:" deep inside the body
    is not mistaken for a header field.
    """
    fields: dict[str, str] = {}
    for line in text.splitlines()[:8]:
        match = _HEADER_RE.match(line)
        if match:
            fields[match.group(1).lower()] = match.group(2)

    title = fields.get("title") or fields.get("role") or ""
    company = fields.get("company") or fields.get("employer") or ""

    if not title:
        # No header: the first non-empty line is usually the job title, and
        # failing that the filename stem is a readable last resort.
        for line in text.splitlines():
            if line.strip():
                title = line.strip()
                break
    if not title:
        title = fallback.replace("_", " ").title()

    return title, company, fields.get("location", "")


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def split_postings(
    postings: list[Posting],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split postings into embedding-sized `Document` chunks.

    --------------------------------------------------------------------------
    Why chunk at all, and why these sizes?
    --------------------------------------------------------------------------
    An embedding model compresses its whole input into one fixed-length vector.
    Two consequences drive every number here:

    * **Hard input limit.** `all-MiniLM-L6-v2` truncates at 256 word-pieces
      (~200 words). Embedding a 2,000-word posting as a single vector does not
      fail loudly -- it silently encodes only the opening and throws away the
      requirements section, which is the part that actually matters. Chunking is
      what keeps the tail of the document in the index at all.

    * **Dilution.** Even within the limit, one vector for a long document is an
      average of everything in it. A posting that is 90% company boilerplate and
      10% "PyTorch, recommender systems" produces a vector dominated by the
      boilerplate, and the specific match washes out. Smaller chunks keep
      distinct topics in distinct vectors, so a query can hit the one paragraph
      that matters.

    `chunk_size=400` characters (~60-70 words) was chosen by measurement, not
    taste. At 800 chars these postings produced only 2-3 chunks each, which
    quietly broke the scoring in `retrieval._aggregate`: it averages a posting's
    best 3 chunks, so with 2 chunks it was really averaging *all* of them and
    the dilution problem above came straight back. At 400 chars each posting
    yields 3-5 chunks, so "best 3" is genuinely selective. Measured on this
    corpus, that moved a strongly-matching NLP posting from 6th to 4th and
    pushed an unrelated infrastructure posting from 8th to 10th.

    Going much smaller is not free either: a chunk has to stay large enough to
    carry its own meaning, since "5+ years" is useless detached from what it
    modifies. 400 is the balance point here; re-measure if your postings are
    much longer or shorter.

    `chunk_overlap=80` (20% of chunk size) repeats the tail of each chunk at the
    head of the next.
    Without overlap, a split landing mid-sentence can cut a requirement in half
    -- "experience with transformer" | "architectures and NLP" -- and neither
    half embeds like the whole phrase. Overlap costs ~20% more vectors and makes
    boundary-straddling text retrievable. It is cheap insurance.

    `RecursiveCharacterTextSplitter` tries separators in order -- paragraph
    break, line break, sentence, word -- and only falls back to a blind
    character cut if nothing else fits. That is the "recursive" part, and it is
    why splits tend to land on natural boundaries instead of mid-word.
    """
    splitter_cls = _require_splitter()
    if splitter_cls is None:
        raise PostingsError(
            "LangChain is required to chunk postings.\n"
            "  pip install -r requirements.txt\n"
            "  (or use --retrieval-backend tfidf, which ranks whole postings "
            "without chunking)"
        )
    from langchain_core.documents import Document

    splitter = splitter_cls(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    documents: list[Document] = []
    for posting in postings:
        # split_text returns plain strings; we attach metadata ourselves so the
        # posting a chunk came from survives the round trip through the store.
        for index, chunk in enumerate(splitter.split_text(posting.text)):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "slug": posting.slug,
                        "title": posting.title,
                        "company": posting.company,
                        "chunk_index": index,
                        "content_hash": posting.content_hash,
                        "source": str(posting.path),
                    },
                )
            )
    return documents


def split_profile(
    profile: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split the consolidated profile into query-sized pieces.

    The profile gets chunked for the same truncation reason as the postings, and
    it matters more here: profile.txt is ~90 KB. Embedding it whole would encode
    roughly its first 200 words -- the summary section -- and quietly discard
    every project and skill below that. Each chunk instead becomes its own query
    against the store, and the posting scores are aggregated over all of them.
    """
    splitter_cls = _require_splitter()
    if splitter_cls is None:
        raise PostingsError("LangChain is required to chunk the profile.")

    splitter = splitter_cls(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [chunk for chunk in splitter.split_text(profile) if chunk.strip()]


def _require_splitter():
    """Return RecursiveCharacterTextSplitter, or None if LangChain is absent."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return None
    return RecursiveCharacterTextSplitter
