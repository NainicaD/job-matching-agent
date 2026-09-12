#!/usr/bin/env python3
"""job-matching-agent - match your resume profile against job descriptions.

    python match.py check                        # verify API credentials
    python match.py build-profile                # scan ./resumes/ -> profile.txt
    python match.py match job.txt                # API mode (default)
    python match.py match job.txt --dry-run      # print the payload, send nothing
    python match.py match job.txt --local        # offline, free, no key needed
    python match.py match job.txt --model claude-sonnet-5

`anthropic` is imported lazily, inside the API code paths only, so --local and
--dry-run keep working on a machine where the SDK is not installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jobmatch import __version__

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = PROJECT_ROOT / "profile.txt"
DEFAULT_REVIEW = PROJECT_ROOT / "profile.review.md"
DEFAULT_RESUMES = PROJECT_ROOT / "resumes"
DEFAULT_LEDGER = PROJECT_ROOT / ".jobmatch_usage.jsonl"

EXIT_OK = 0
EXIT_ERROR = 1


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


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

    # -- match ------------------------------------------------------------
    match = subparsers.add_parser("match", help="match the profile against a job")
    match.add_argument(
        "jobs",
        nargs="+",
        metavar="JOB",
        help="job description file(s), or - to read one from stdin",
    )
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

    jobs: list[tuple[str, str]] = []
    for spec in args.jobs:
        try:
            jobs.append(_read_job(spec))
        except OSError as exc:
            print(f"error: cannot read job description {spec!r}: {exc}", file=sys.stderr)
            return EXIT_ERROR

    if args.local:
        return _run_local(args, profile, jobs)
    if args.dry_run:
        return _run_dry_run(args, profile, jobs)
    return _run_api(args, profile, jobs)


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
    if not args.json:
        print("  cost: $0.00 (offline)")
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
    elif client.tracker.calls > 1:
        print(client.tracker.format_session())

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
