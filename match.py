#!/usr/bin/env python3
"""job-matching-agent - match your resume profile against job descriptions.

    python match.py check                        # verify API credentials
    python match.py build-profile                # scan ./resumes/ -> profile.txt
    python match.py index-postings               # scan ./job_postings/ -> vector store
    python match.py rank                         # retrieval only: rank postings, no API calls
    python match.py match --top-k 5              # retrieval + API reasoning on the top 5
    python match.py match job.txt                # API mode on one explicit file
    python match.py match job.txt --dry-run      # print the payload, send nothing
    python match.py match job.txt --local        # offline, free, no key needed
    python match.py match job.txt --model claude-sonnet-5

`anthropic` is imported lazily, inside the API code paths only, so --local and
--dry-run keep working on a machine where the SDK is not installed. The same is
true of LangChain: it is only imported by the retrieval path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jobmatch import __version__

# Retrieval defaults live with the retrieval code; imported here so the CLI help
# text cannot drift from the actual behaviour. Neither module imports LangChain
# or anthropic at import time, so this stays cheap.
from jobmatch.postings import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from jobmatch.retrieval import (
    BACKEND_EMBEDDINGS,
    BACKEND_TFIDF,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FANOUT,
    DEFAULT_STORE_DIR,
    DEFAULT_TOP_K,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = PROJECT_ROOT / "profile.txt"
DEFAULT_REVIEW = PROJECT_ROOT / "profile.review.md"
DEFAULT_RESUMES = PROJECT_ROOT / "resumes"
DEFAULT_LEDGER = PROJECT_ROOT / ".jobmatch_usage.jsonl"
DEFAULT_POSTINGS = PROJECT_ROOT / "job_postings"
DEFAULT_STORE = PROJECT_ROOT / DEFAULT_STORE_DIR

EXIT_OK = 0
EXIT_ERROR = 1


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    """Options shared by every command that runs the retrieval stage."""
    group = parser.add_argument_group("retrieval (local, free, no API key)")
    group.add_argument("--postings", type=Path, default=DEFAULT_POSTINGS)
    group.add_argument(
        "--store", type=Path, default=DEFAULT_STORE, help="vector store directory"
    )
    group.add_argument(
        "--retrieval-backend",
        choices=[BACKEND_EMBEDDINGS, BACKEND_TFIDF],
        default=BACKEND_EMBEDDINGS,
        help=f"{BACKEND_EMBEDDINGS}: sentence-transformer embeddings in Chroma "
        f"(understands paraphrase; one-time model download). "
        f"{BACKEND_TFIDF}: stdlib TF-IDF, no download, works fully offline.",
    )
    group.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    group.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"characters per chunk (default {DEFAULT_CHUNK_SIZE})",
    )
    group.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"characters repeated between chunks (default {DEFAULT_CHUNK_OVERLAP})",
    )
    group.add_argument(
        "--fanout",
        type=int,
        default=DEFAULT_FANOUT,
        help=f"store hits per profile-chunk query (default {DEFAULT_FANOUT})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="match.py",
        description="Match a consolidated resume profile against job descriptions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- check ------------------------------------------------------------
    check = subparsers.add_parser(
        "check", help="verify ANTHROPIC_API_KEY with a 1-token test call"
    )
    check.add_argument("--model", default=None, help="model to test against")
    check.add_argument(
        "--no-call",
        action="store_true",
        help="only check that credentials exist; make no API call",
    )

    # -- build-profile ----------------------------------------------------
    build = subparsers.add_parser(
        "build-profile", help="scan ./resumes/ and write profile.txt"
    )
    build.add_argument("--resumes", type=Path, default=DEFAULT_RESUMES)
    build.add_argument("--out", type=Path, default=DEFAULT_PROFILE)
    build.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    build.add_argument(
        "--keep-contact",
        action="store_true",
        help="keep emails/phones/profile URLs (redacted by default - they add no "
        "matching signal and profile.txt is what gets sent to the API)",
    )
    build.add_argument(
        "--near-dup-threshold",
        type=float,
        default=0.72,
        help="Jaccard similarity above which two bullets are flagged (default: 0.72)",
    )

    # -- index-postings ---------------------------------------------------
    index = subparsers.add_parser(
        "index-postings",
        help="scan ./job_postings/ and build or update the vector store",
    )
    _add_retrieval_args(index)
    index.add_argument(
        "--rebuild",
        action="store_true",
        help="delete the store and re-embed everything from scratch",
    )

    # -- rank -------------------------------------------------------------
    rank = subparsers.add_parser(
        "rank",
        help="retrieval stage only: rank all postings by relevance, no API calls",
    )
    _add_retrieval_args(rank)
    rank.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    rank.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"mark the top-k cutoff in the output (default {DEFAULT_TOP_K})",
    )
    rank.add_argument("--json", action="store_true")
    rank.add_argument("--no-color", action="store_true")

    # -- match ------------------------------------------------------------
    match = subparsers.add_parser("match", help="match the profile against a job")
    match.add_argument(
        "jobs",
        nargs="*",
        metavar="JOB",
        help="job description file(s), or - for stdin. Omit to rank ./job_postings/ "
        "by retrieval and match the top-k instead.",
    )
    match.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"how many retrieval-ranked postings to reason over (default "
        f"{DEFAULT_TOP_K}). Ignored when explicit JOB files are given.",
    )
    _add_retrieval_args(match)
    mode = match.add_mutually_exclusive_group()
    mode.add_argument(
        "--local", action="store_true", help="offline TF-IDF match; no API key, no cost"
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact request that would be sent, without sending it",
    )
    match.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    match.add_argument("--model", default=None, help="override the model (API mode)")
    match.add_argument("--max-tokens", type=int, default=None)
    match.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    match.add_argument("--no-color", action="store_true")
    match.add_argument(
        "--no-cache",
        action="store_true",
        help="disable prompt caching of the profile block (API mode)",
    )
    match.add_argument(
        "--full", action="store_true", help="do not elide long blocks in --dry-run output"
    )
    match.add_argument(
        "--count-tokens",
        action="store_true",
        help="in --dry-run, get an exact input token count from the free "
        "count_tokens endpoint (this does make a network call)",
    )
    match.add_argument(
        "--embeddings",
        action="store_true",
        help="local mode: use sentence-transformers instead of TF-IDF if installed",
    )
    match.add_argument(
        "--no-sklearn",
        action="store_true",
        help="local mode: force the pure-Python TF-IDF implementation",
    )
    match.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    match.add_argument(
        "--no-ledger", action="store_true", help="do not append to the spend ledger"
    )

    # -- models -----------------------------------------------------------
    subparsers.add_parser("models", help="list known model IDs and their prices")

    return parser


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    from jobmatch.preflight import check_credentials, verify_api_key
    from jobmatch.pricing import DEFAULT_MODEL

    model = args.model or DEFAULT_MODEL
    report = check_credentials()

    print("Step 1 - credentials")
    if not report.ok:
        print(f"  FAIL  {report.detail}\n")
        print(report.instructions())
        return EXIT_ERROR
    print(f"  OK    {report.detail}\n")

    if args.no_call:
        print("Step 2 - live verification: skipped (--no-call)")
        return EXIT_OK

    print(f"Step 2 - live verification (1-token call to {model})")
    result = verify_api_key(model)
    if not result.ok:
        print(f"  FAIL  {result.message}")
        print("\n  API mode is unavailable. Offline matching still works:")
        print("      python match.py match job.txt --local")
        return EXIT_ERROR
    print(f"  OK    {result.message}")
    print("\nReady. Next: python match.py build-profile")
    return EXIT_OK


def cmd_build_profile(args: argparse.Namespace) -> int:
    from jobmatch.profile_builder import build_profile

    if not args.resumes.is_dir():
        print(f"error: {args.resumes} does not exist.", file=sys.stderr)
        print(f"Create it and add .pdf/.tex/.txt resumes:\n  mkdir -p {args.resumes}",
              file=sys.stderr)
        return EXIT_ERROR

    try:
        report = build_profile(
            args.resumes,
            args.out,
            args.review,
            keep_contact=args.keep_contact,
            near_dup_threshold=args.near_dup_threshold,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(f"Scanned {args.resumes}/")
    print(f"  parsed          : {len(report.files_read)} file(s)")
    if report.files_failed:
        print(f"  failed          : {len(report.files_failed)} file(s)")
        for name, err in report.files_failed:
            print(f"      - {err}")
    if report.ligatures_fixed or report.runons_split:
        print(
            f"  text repairs    : {report.ligatures_fixed} broken ligature(s), "
            f"{report.runons_split} run-together token(s) split"
        )
    print(f"  lines           : {report.kept_lines:,} kept, "
          f"{report.duplicates_removed:,} exact duplicate(s) removed")
    print(f"  near-duplicates : {len(report.near_duplicate_clusters)} cluster(s) flagged "
          f"for review (kept, not merged)")
    if report.low_quality_files:
        worst = ", ".join(name for name, _ in report.low_quality_files[:3])
        print(f"  poor extraction : {len(report.low_quality_files)} file(s) lost spacing "
              f"({worst}) - see the review file")
    if not args.keep_contact:
        print("  contact details : redacted (use --keep-contact to keep them)")
    print()
    print(f"  -> {args.out}  (content-hash {report.content_hash})")
    print(f"  -> {args.review}")
    if report.unchanged:
        print("\n  Unchanged since the last build - the consolidation is idempotent.")
    return EXIT_OK


def cmd_index_postings(args: argparse.Namespace) -> int:
    """Build or update the vector store from ./job_postings/."""
    from jobmatch.postings import PostingsError, load_postings
    from jobmatch.retrieval import BACKEND_TFIDF, RetrievalError, build_index

    try:
        postings = load_postings(args.postings)
    except PostingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(f"Scanned {args.postings}/")
    print(f"  postings found  : {len(postings)}")

    if args.retrieval_backend == BACKEND_TFIDF:
        print("  backend         : tfidf - no vector store needed, nothing to build")
        print("\n  `rank` and `match` will compute TF-IDF similarity on the fly.")
        return EXIT_OK

    print(f"  embedding model : {args.embedding_model}")
    print(f"  chunking        : {args.chunk_size} chars, {args.chunk_overlap} overlap")
    if args.rebuild:
        print("  mode            : --rebuild (deleting the existing store)")
    print("  embedding...      (first run downloads the model, ~90 MB)")

    try:
        report = build_index(
            postings,
            args.store,
            backend=args.retrieval_backend,
            model_name=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            rebuild=args.rebuild,
        )
    except (RetrievalError, PostingsError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print()
    print(f"  added           : {len(report.added)}")
    print(f"  updated         : {len(report.updated)}")
    print(f"  unchanged       : {len(report.unchanged)} (not re-embedded)")
    if report.removed:
        print(f"  removed         : {len(report.removed)} (file no longer present)")
    print(f"  chunks embedded : {report.chunks_embedded} this run")
    print(f"  chunks in store : {report.total_chunks}")
    print()
    print(f"  -> {report.store_dir}")
    if not report.did_work:
        print("\n  Store already current - nothing was re-embedded.")
    return EXIT_OK


def cmd_rank(args: argparse.Namespace) -> int:
    """Retrieval stage only: rank every posting, make no API calls."""
    from jobmatch.postings import PostingsError, load_postings
    from jobmatch.profile_builder import load_profile
    from jobmatch.retrieval import RetrievalError, format_ranking

    try:
        profile = load_profile(args.profile)
        postings = load_postings(args.postings)
        ranking, _report = _rank(args, profile, postings)
    except (FileNotFoundError, PostingsError, RetrievalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        import json
        from dataclasses import asdict

        print(json.dumps([asdict(score) for score in ranking], indent=2))
        return EXIT_OK

    top_k = args.top_k if args.top_k is not None else DEFAULT_TOP_K
    print(f"Retrieval ranking - {len(ranking)} posting(s), {args.retrieval_backend} backend")
    print()
    print(format_ranking(ranking, top_k=top_k, color=not args.no_color))
    print()
    print("  no API calls were made (retrieval is entirely local and free)")
    return EXIT_OK


def cmd_models(_args: argparse.Namespace) -> int:
    from jobmatch.pricing import DEFAULT_MODEL, PRICING

    print(f"{'model':<22} {'input $/MTok':>13} {'output $/MTok':>14}")
    print("-" * 51)
    for model, (inp, out) in sorted(PRICING.items(), key=lambda kv: kv[1]):
        marker = "  <- default" if model == DEFAULT_MODEL else ""
        print(f"{model:<22} {inp:>13.2f} {out:>14.2f}{marker}")
    print("\nPrices are USD per million tokens, Anthropic first-party API rates.")
    print("Cached input re-reads bill at 10% of the input rate.")
    return EXIT_OK


def cmd_match(args: argparse.Namespace) -> int:
    from jobmatch.profile_builder import load_profile

    try:
        profile = load_profile(args.profile)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    # In API mode, check credentials before running retrieval. Retrieval takes a
    # few seconds of embedding, and there is no reason to spend them only to
    # then discover the key is missing. (_run_api re-checks; this is just for
    # failing fast.)
    if not args.local and not args.dry_run:
        from jobmatch.preflight import check_credentials

        report = check_credentials()
        if not report.ok:
            print(f"error: {report.detail}\n", file=sys.stderr)
            print(report.instructions(), file=sys.stderr)
            return EXIT_ERROR

    jobs: list[tuple[str, str]] = []
    if args.jobs:
        # Explicit files: the original behaviour, unchanged. Retrieval has
        # nothing to decide when you have already named the postings.
        if args.top_k is not None:
            print(
                "note: --top-k applies to retrieval; ignoring it because explicit "
                "JOB files were given.",
                file=sys.stderr,
            )
        for spec in args.jobs:
            try:
                jobs.append(_read_job(spec))
            except OSError as exc:
                print(f"error: cannot read job description {spec!r}: {exc}", file=sys.stderr)
                return EXIT_ERROR
    else:
        # No files named: run the retrieval stage over ./job_postings/ and feed
        # only the top-k into whichever reasoning mode was selected.
        selected = _retrieve_jobs(args, profile)
        if selected is None:
            return EXIT_ERROR
        jobs = selected

    if args.local:
        return _run_local(args, profile, jobs)
    if args.dry_run:
        return _run_dry_run(args, profile, jobs)
    return _run_api(args, profile, jobs)


# --------------------------------------------------------------------------
# Retrieval -> reasoning bridge
# --------------------------------------------------------------------------
#
# The whole extension hinges on one fact: _run_local, _run_dry_run and _run_api
# all take the same `list[tuple[name, text]]`. So retrieval's only job is to
# produce a shorter, better-ordered version of that list. None of the three
# reasoning paths below needed to change.


def _chatter(args: argparse.Namespace):
    """Where human-readable progress output goes.

    With --json, stdout has to stay parseable, so the retrieval table and the
    cost summary go to stderr instead of being dropped. You still see them in a
    terminal; a pipe into `jq` still gets clean JSON.
    """
    return sys.stderr if getattr(args, "json", False) else sys.stdout


def _rank(args: argparse.Namespace, profile: str, postings: list):
    """Run the retrieval stage. Shared by `rank` and `match`."""
    from jobmatch.retrieval import rank_postings

    return rank_postings(
        profile,
        postings,
        args.store,
        backend=args.retrieval_backend,
        model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        fanout=args.fanout,
    )


def _retrieve_jobs(
    args: argparse.Namespace, profile: str
) -> list[tuple[str, str]] | None:
    """Rank ./job_postings/ and return the top-k as (label, text) pairs.

    Prints the full ranking first, including the postings it is about to
    discard, so the pre-filtering decision is visible before anything is spent
    on it. Returns None on error.
    """
    from jobmatch.postings import PostingsError, load_postings
    from jobmatch.retrieval import RetrievalError, format_ranking

    try:
        postings = load_postings(args.postings)
        ranking, report = _rank(args, profile, postings)
    except (PostingsError, RetrievalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None

    top_k = args.top_k if args.top_k is not None else DEFAULT_TOP_K
    out = _chatter(args)
    color = not args.no_color and not getattr(args, "json", False)

    print("=" * 78, file=out)
    print(
        f"RETRIEVAL STAGE - {len(ranking)} posting(s), {args.retrieval_backend} backend, $0.00",
        file=out,
    )
    print("=" * 78, file=out)
    if report is not None and report.did_work:
        print(
            f"  index updated: +{len(report.added)} new, ~{len(report.updated)} changed, "
            f"-{len(report.removed)} removed, {report.chunks_embedded} chunk(s) embedded",
            file=out,
        )
        print(file=out)
    print(format_ranking(ranking, top_k=top_k, color=color), file=out)
    print(file=out)

    by_slug = {posting.slug: posting for posting in postings}
    selected: list[tuple[str, str]] = []
    for score in ranking[:top_k]:
        posting = by_slug.get(score.slug)
        if posting is None:
            continue
        if score.score <= 0:
            # Retrieval found no overlap at all. Reasoning over it would be a
            # paid call with a foregone conclusion.
            print(
                f"  skipping {score.label!r}: retrieval score 0 (no overlap found)",
                file=out,
            )
            continue
        selected.append((posting.label, posting.text))

    if not selected:
        print(
            "error: retrieval selected no postings. Check that ./job_postings/ holds "
            "postings related to your profile, or raise --top-k.",
            file=sys.stderr,
        )
        return None
    return selected


def _read_job(spec: str) -> tuple[str, str]:
    if spec == "-":
        text = sys.stdin.read()
        if not text.strip():
            raise OSError("stdin was empty")
        return "stdin", text
    path = Path(spec)
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise OSError("file is empty")
    return path.name, text


# -- mode: local -----------------------------------------------------------


def _run_local(args: argparse.Namespace, profile: str, jobs: list[tuple[str, str]]) -> int:
    # Imported here, not at module scope, to keep the offline path free of any
    # chance of pulling in the API modules.
    from jobmatch.local_matcher import match_local

    results = []
    for name, text in jobs:
        try:
            results.append(
                match_local(
                    profile,
                    text,
                    job_name=name,
                    use_embeddings=args.embeddings,
                    allow_sklearn=not args.no_sklearn,
                )
            )
        except (ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR

    _emit(results, args)
    print(f"  cost: $0.00 (offline, {len(results)} posting(s))", file=_chatter(args))
    return EXIT_OK


# -- mode: dry-run ---------------------------------------------------------


def _run_dry_run(args: argparse.Namespace, profile: str, jobs: list[tuple[str, str]]) -> int:
    from jobmatch.api_matcher import (
        DEFAULT_MAX_TOKENS,
        build_request,
        describe_request,
        estimate_tokens,
    )
    from jobmatch.pricing import DEFAULT_MODEL, estimate_input_cost, format_usd

    model = args.model or DEFAULT_MODEL
    max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS

    for name, text in jobs:
        request = build_request(
            profile, text, model=model, max_tokens=max_tokens,
            use_cache=not args.no_cache,
        )
        print("=" * 78)
        print(f"DRY RUN - {name} - nothing is sent to the API")
        print("=" * 78)
        print(describe_request(request, max_chars=0 if args.full else 1200))

        if args.count_tokens:
            from jobmatch.api_matcher import ApiCallError, MatchClient

            try:
                tokens = MatchClient().count_tokens(request)
                label = "exact (count_tokens endpoint)"
            except ApiCallError as exc:
                print(f"  count_tokens unavailable: {exc}")
                tokens, label = estimate_tokens(request), "estimated (~4 chars/token)"
        else:
            tokens, label = estimate_tokens(request), "estimated (~4 chars/token)"

        cost = estimate_input_cost(model, tokens)
        cost_text = format_usd(cost) if cost is not None else "unknown (model not priced)"
        print(f"  input tokens : {tokens:,}  [{label}]")
        print(f"  input cost   : {cost_text}")
        print(f"  output cost  : up to {max_tokens:,} tokens, billed at the output rate")
        print()
    return EXIT_OK


# -- mode: api -------------------------------------------------------------


def _run_api(args: argparse.Namespace, profile: str, jobs: list[tuple[str, str]]) -> int:
    from jobmatch.api_matcher import DEFAULT_MAX_TOKENS, MatchClient, MatchError
    from jobmatch.preflight import check_credentials
    from jobmatch.pricing import DEFAULT_MODEL, is_known_model
    from jobmatch.result import render

    # Fail on the credential check rather than deep inside the first API call.
    report = check_credentials()
    if not report.ok:
        print(f"error: {report.detail}\n", file=sys.stderr)
        print(report.instructions(), file=sys.stderr)
        return EXIT_ERROR

    model = args.model or DEFAULT_MODEL
    max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS
    if not is_known_model(model):
        print(
            f"warning: {model!r} is not in the pricing table - tokens will be "
            f"reported but cost will not.",
            file=sys.stderr,
        )

    try:
        client = MatchClient()
    except Exception as exc:  # client construction validates credentials
        print(f"error: could not create the API client: {exc}", file=sys.stderr)
        return EXIT_ERROR

    results = []
    failures = 0
    for name, text in jobs:
        try:
            result, usage = client.match(
                profile,
                text,
                model=model,
                max_tokens=max_tokens,
                job_name=name,
                use_cache=not args.no_cache,
                on_retry=lambda msg: print(f"  ... {msg}", file=sys.stderr),
            )
        except MatchError as exc:
            failures += 1
            print(f"error [{name}]: {exc}", file=sys.stderr)
            hint = getattr(exc, "hint", "")
            if hint:
                print(f"  hint: {hint}", file=sys.stderr)
            continue

        results.append(result)
        if not args.json:
            print(render(result, color=not args.no_color))
            print(client.tracker.format_call(usage, model))
            print()
        if not args.no_ledger:
            client.tracker.append_ledger(args.ledger, model, name, usage)

    if args.json:
        _emit(results, args)

    # The running total across every call in this run. Printed unconditionally:
    # with retrieval driving the job list, "what did this run cost" is the
    # headline number, and it was previously hidden for single-call runs and
    # suppressed entirely under --json.
    if client.tracker.calls:
        print(client.tracker.format_session(), file=_chatter(args))

    if not results:
        return EXIT_ERROR
    return EXIT_OK if failures == 0 else EXIT_ERROR


def _emit(results: list, args: argparse.Namespace) -> None:
    import json

    from jobmatch.result import render

    if args.json:
        payload = [r.to_dict() for r in results]
        print(json.dumps(payload if len(payload) != 1 else payload[0], indent=2))
        return
    for result in results:
        print(render(result, color=not args.no_color))


# --------------------------------------------------------------------------


COMMANDS = {
    "check": cmd_check,
    "build-profile": cmd_build_profile,
    "index-postings": cmd_index_postings,
    "rank": cmd_rank,
    "match": cmd_match,
    "models": cmd_models,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except ImportError as exc:
        # Most likely the anthropic SDK is missing on an API-mode path.
        print(f"error: missing dependency: {exc}", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        print("  (or use --local, which needs no third-party packages)", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
