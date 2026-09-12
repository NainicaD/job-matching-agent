"""Tests for extraction, repair and consolidation.

These use synthetic fixtures written to a temp directory rather than the real
./resumes/ folder, so they are reproducible and contain no personal data.

Run directly (no pytest needed):

    python tests/test_profile_builder.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobmatch.extract import strip_latex
from jobmatch.profile_builder import build_profile, load_profile
from jobmatch.repair import build_vocabulary, repair_ligatures, split_runons

LATEX_RESUME = r"""
\documentclass[letterpaper,11pt]{article}
\usepackage{latexsym}
\usepackage[empty]{fullpage}
% a comment that must not survive
\newcommand{\resumeItem}[1]{\item\small{#1}}

\begin{document}

\textbf{\Large Jane Doe} \\
jane@example.com $|$ 555-123-4567

\section{Experience}
\resumeSubheading{Data Scientist}{Jan 2023 -- Present}{Acme Corp}{Boston, MA}
\begin{itemize}
  \resumeItem{Built an ETL pipeline in \textbf{Python} processing 2M rows/day.}
  \resumeItem{Deployed a model with \href{https://fastapi.tiangolo.com}{FastAPI}.}
\end{itemize}

\section{Skills}
\begin{tabular}{ll}
Languages & Python, SQL, R \\
Tools & PyTorch, Docker \\
\end{tabular}

\end{document}
"""

TXT_RESUME = """\
Jane Doe
jane@example.com | 555-123-4567

EXPERIENCE
Data Scientist, Acme Corp
- Built an ETL pipeline in Python processing 2M rows/day.
- Built an ETL pipeline in Python that processed 2M rows per day.

SKILLS
Python, SQL, R, PyTorch, Docker
"""


def _write_fixtures(directory: Path) -> None:
    (directory / "resume_a.tex").write_text(LATEX_RESUME, encoding="utf-8")
    (directory / "resume_b.txt").write_text(TXT_RESUME, encoding="utf-8")


# --------------------------------------------------------------------------
# LaTeX stripping
# --------------------------------------------------------------------------


def test_latex_stripping() -> None:
    text = strip_latex(LATEX_RESUME)

    assert "a comment that must not survive" not in text, "comments must be stripped"
    assert "documentclass" not in text and "usepackage" not in text, "preamble must go"
    assert "\\textbf" not in text and "\\resumeItem" not in text, "commands must go"

    assert "Jane Doe" in text
    assert "Built an ETL pipeline in Python processing 2M rows/day." in text
    assert "FastAPI" in text, "\\href must keep the label"
    assert "https://fastapi.tiangolo.com" not in text, "\\href must drop the URL"
    assert "Data Scientist" in text and "Acme Corp" in text, "\\resumeSubheading args kept"
    assert "Languages | Python, SQL, R" in text, "tabular content and & separator must survive"
    assert "{ll}" not in text and "tabular" not in text, "column spec must be dropped"
    print("ok  LaTeX: comments/preamble/markup stripped, content and href labels kept")


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------


def test_ligature_repair_is_targeted() -> None:
    corpus = ["financial filtering classification", "workflows tracking skills kind scikit"]
    vocab = build_vocabulary(corpus)

    fixed, count = repair_ligatures("kinancial kiltering classikication", vocab)
    assert fixed == "financial filtering classification", fixed
    assert count == 3

    # Real words containing "ki" must survive untouched.
    untouched, count2 = repair_ligatures("workflows tracking skills kind scikit", vocab)
    assert untouched == "workflows tracking skills kind scikit", untouched
    assert count2 == 0
    print("ok  ligature repair fixes garbled tokens and leaves real 'ki' words alone")


def test_runon_splitting() -> None:
    corpus = [
        "coordinating cross functional teams and translating technical systems",
        "gradient boosted classifiers data transformation cross functional",
        "coordinating cross functional delivery gradient boosted classifiers",
    ]
    vocab = build_vocabulary(corpus)

    split, count = split_runons("coordinatingcrossfunctional", vocab)
    assert split == "coordinating cross functional", split
    assert count == 1

    # A genuine long word must not be split apart.
    intact, _ = split_runons("responsibilities", vocab)
    assert intact == "responsibilities", intact
    print("ok  run-on splitting recovers lost spaces, leaves real long words intact")


# --------------------------------------------------------------------------
# Consolidation
# --------------------------------------------------------------------------


def test_build_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        resumes = root / "resumes"
        resumes.mkdir()
        _write_fixtures(resumes)
        profile, review = root / "profile.txt", root / "review.md"

        first = build_profile(resumes, profile, review)
        content_first = profile.read_text()

        second = build_profile(resumes, profile, review)
        content_second = profile.read_text()

        assert content_first == content_second, "re-running drifted the file"
        assert second.unchanged is True
        assert first.content_hash == second.content_hash
        assert first.kept_lines == second.kept_lines, "re-running changed the line count"
        # The header identifies the build by content hash, not by clock time --
        # a timestamp would make the file differ on every run by construction.
        header = "\n".join(ln for ln in content_first.splitlines() if ln.startswith("#"))
        assert f"content-hash: {first.content_hash}" in header, header
        print(f"ok  build is idempotent (hash {first.content_hash} stable across runs)")


def test_exact_duplicates_removed_near_duplicates_kept() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        resumes = root / "resumes"
        resumes.mkdir()
        _write_fixtures(resumes)
        profile, review = root / "profile.txt", root / "review.md"

        report = build_profile(resumes, profile, review)
        text = profile.read_text()

        # The identical ETL bullet appears in both the .tex and the .txt.
        exact = text.count("Built an ETL pipeline in Python processing 2M rows/day.")
        assert exact == 1, f"exact duplicate not removed (appeared {exact}x)"
        assert report.duplicates_removed >= 1

        # The reworded variant is different data and must be preserved.
        assert "that processed 2M rows per day" in text, "near-duplicate was wrongly merged"

        # ...and it must be flagged for review rather than silently kept.
        assert report.near_duplicate_clusters, "near-duplicate pair was not flagged"
        review_text = review.read_text()
        assert "Near-duplicate clusters" in review_text
        assert "rows per day" in review_text
        print(f"ok  exact dupes removed, {len(report.near_duplicate_clusters)} near-dup "
              "cluster(s) kept and flagged for review")


def test_contact_details_redacted_by_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        resumes = root / "resumes"
        resumes.mkdir()
        _write_fixtures(resumes)
        profile, review = root / "profile.txt", root / "review.md"

        build_profile(resumes, profile, review)
        redacted = profile.read_text()
        assert "jane@example.com" not in redacted
        assert "555-123-4567" not in redacted
        assert "Jane Doe" in redacted, "the name should survive redaction"

        build_profile(resumes, profile, review, keep_contact=True)
        kept = profile.read_text()
        assert "jane@example.com" in kept
        print("ok  contact details redacted by default, kept with --keep-contact")


def test_load_profile_strips_header() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        resumes = root / "resumes"
        resumes.mkdir()
        _write_fixtures(resumes)
        profile, review = root / "profile.txt", root / "review.md"
        build_profile(resumes, profile, review)

        loaded = load_profile(profile)
        assert not loaded.startswith("#")
        assert "Generated by job-matching-agent" not in loaded
        assert "Python" in loaded
        print("ok  load_profile strips generated header comments")


def test_missing_resumes_dir_is_a_clean_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        empty = root / "resumes"
        empty.mkdir()
        try:
            build_profile(empty, root / "p.txt", root / "r.md")
        except FileNotFoundError as exc:
            assert "No .pdf/.tex/.txt files" in str(exc)
            print("ok  an empty resumes/ folder produces a clear error, not a traceback")
            return
    raise AssertionError("expected FileNotFoundError")


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
