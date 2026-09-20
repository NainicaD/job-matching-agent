"""Tests for how the CLI wires retrieval into the existing reasoning modes.

These run the real CLI as a subprocess, using `--local` and the `tfidf`
retrieval backend so nothing needs an API key, a model download, or a network
connection.

The contracts checked here are the ones that are easy to break by accident and
invisible until someone pipes the output somewhere:

  * `--json` must put *only* JSON on stdout. Human-readable progress output
    belongs on stderr. (This was a real regression: the retrieval ranking table
    was printed to stdout ahead of the JSON, so `| jq` failed.)
  * `--top-k` must actually bound the number of postings reasoned over.
  * Naming explicit JOB files must bypass retrieval entirely.
  * A run's total cost must always be reported, including single-call runs.

Run directly (no pytest needed):

    python tests/test_cli_wiring.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_PROFILE = """\
== SKILLS ==
- Python, SQL, PyTorch, scikit-learn, pandas, NumPy

== PROJECTS ==
- Built a recommender with collaborative filtering and TF-IDF content filtering.
- Trained a transformer classifier in PyTorch with attention explainability.

== EXPERIENCE ==
- Wrote SQL pipelines turning raw event logs into model-ready datasets.
- Coordinated cross-functional delivery with clients and stakeholders.
"""


def _run_cli(*extra: str) -> subprocess.CompletedProcess:
    """Run match.py with no API credentials in the environment."""
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    with tempfile.TemporaryDirectory() as tmp:
        profile_path = Path(tmp) / "profile.txt"
        profile_path.write_text(SAMPLE_PROFILE, encoding="utf-8")
        return subprocess.run(
            [sys.executable, "match.py", *extra, "--profile", str(profile_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )


# --------------------------------------------------------------------------
# --json output integrity
# --------------------------------------------------------------------------


def test_json_stdout_is_pure_json() -> None:
    proc = _run_cli(
        "match", "--top-k", "2", "--local", "--retrieval-backend", "tfidf", "--json"
    )
    assert proc.returncode == 0, f"CLI failed:\n{proc.stderr}"

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"stdout is not valid JSON ({exc}). First 200 chars:\n{proc.stdout[:200]}"
        ) from exc

    assert isinstance(payload, list) and len(payload) == 2, payload
    assert all(item["engine"] == "local-tfidf" for item in payload)
    print("ok  --json puts only valid JSON on stdout")


def test_json_shape_is_object_for_one_result_list_for_many() -> None:
    """Pin the existing (shape-varying) --json contract so it cannot drift silently."""
    one = _run_cli("match", "--top-k", "1", "--local", "--retrieval-backend", "tfidf",
                   "--json")
    many = _run_cli("match", "--top-k", "2", "--local", "--retrieval-backend", "tfidf",
                    "--json")
    assert isinstance(json.loads(one.stdout), dict), "one result should be an object"
    assert isinstance(json.loads(many.stdout), list), "several results should be a list"
    print("ok  --json shape: object for one result, list for several "
          "(pre-existing convention; normalise with `jq -s 'flatten'`)")


def test_json_mode_still_shows_the_ranking_on_stderr() -> None:
    """The pre-filtering decision must stay visible, just not on stdout."""
    proc = _run_cli(
        "match", "--top-k", "2", "--local", "--retrieval-backend", "tfidf", "--json"
    )
    assert "RETRIEVAL STAGE" in proc.stderr, proc.stderr[:400]
    assert "top 2 selected for reasoning" in proc.stderr, proc.stderr[:400]
    assert "RETRIEVAL STAGE" not in proc.stdout, "ranking leaked onto stdout"
    print("ok  --json keeps the ranking visible on stderr")


def test_ranking_goes_to_stdout_without_json() -> None:
    proc = _run_cli("match", "--top-k", "2", "--local", "--retrieval-backend", "tfidf",
                    "--no-color")
    assert proc.returncode == 0, proc.stderr
    assert "RETRIEVAL STAGE" in proc.stdout, proc.stdout[:400]
    print("ok  without --json the ranking prints normally on stdout")


def test_rank_json_is_machine_readable() -> None:
    proc = _run_cli("rank", "--retrieval-backend", "tfidf", "--json")
    assert proc.returncode == 0, proc.stderr
    ranking = json.loads(proc.stdout)
    assert len(ranking) >= 3, ranking
    scores = [item["score"] for item in ranking]
    assert scores == sorted(scores, reverse=True), f"not sorted descending: {scores}"
    for key in ("slug", "title", "score", "matched_chunks", "total_chunks"):
        assert key in ranking[0], f"{key} missing from {ranking[0]}"
    print(f"ok  `rank --json` emits {len(ranking)} sorted, machine-readable rows")


# --------------------------------------------------------------------------
# top-k and retrieval bypass
# --------------------------------------------------------------------------


def _results(stdout: str) -> list[dict]:
    """Normalise --json output to a list.

    The CLI emits a bare object for a single result and a list for several --
    the convention the single-file flow has always used. Callers that only care
    about the count should go through here.
    """
    payload = json.loads(stdout)
    return payload if isinstance(payload, list) else [payload]


def test_top_k_bounds_the_reasoning_stage() -> None:
    """--top-k N must produce exactly N reports, not all postings."""
    for k in (1, 3):
        proc = _run_cli(
            "match", "--top-k", str(k), "--local", "--retrieval-backend", "tfidf", "--json"
        )
        assert proc.returncode == 0, proc.stderr
        results = _results(proc.stdout)
        assert len(results) == k, f"--top-k {k} produced {len(results)} result(s)"
        assert all(r["engine"] == "local-tfidf" for r in results)
        assert f"top {k} selected for reasoning" in proc.stderr
    print("ok  --top-k bounds how many postings reach the reasoning stage (1 and 3)")


def test_explicit_files_bypass_retrieval() -> None:
    """Naming a job file must keep the original single-job behaviour."""
    proc = _run_cli("match", "jobs/example_ml_engineer.txt", "--local", "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert isinstance(payload, dict), "one explicit file should emit a single object"
    assert payload["job_name"] == "example_ml_engineer.txt", payload["job_name"]
    assert "RETRIEVAL STAGE" not in proc.stdout + proc.stderr, "retrieval should not run"
    print("ok  explicit JOB files bypass retrieval entirely")


def test_top_k_with_explicit_files_warns() -> None:
    proc = _run_cli(
        "match", "jobs/example_ml_engineer.txt", "--top-k", "3", "--local", "--json"
    )
    assert proc.returncode == 0, proc.stderr
    assert "ignoring it because explicit" in proc.stderr, proc.stderr[:300]
    payload = json.loads(proc.stdout)
    assert isinstance(payload, dict), "still one result, not three"
    print("ok  --top-k alongside explicit files warns instead of silently ignoring")


def test_offline_cost_is_reported() -> None:
    proc = _run_cli("match", "--top-k", "2", "--local", "--retrieval-backend", "tfidf",
                    "--no-color")
    assert "cost: $0.00 (offline" in proc.stdout, proc.stdout[-400:]
    print("ok  local runs report $0.00 explicitly")


# --------------------------------------------------------------------------
# Cost total across a run
# --------------------------------------------------------------------------


def test_session_total_covers_a_single_call() -> None:
    """A one-call run must still report a total.

    match.py used to gate the session total on `calls > 1`, which hid the cost
    of exactly the run you are most likely to make first: `match --top-k 1`.
    """
    from jobmatch.api_matcher import CostTracker
    from jobmatch.pricing import price_usage

    tracker = CostTracker()
    tracker.record(price_usage("claude-haiku-4-5", 20_000, 500))
    summary = tracker.format_session()

    assert "1 call(s)" in summary, summary
    assert "total cost" in summary, summary
    assert "no API calls" not in summary
    print(f"ok  session total reports a single call ({summary.splitlines()[-1].strip()})")


def test_session_total_accumulates_across_top_k() -> None:
    from jobmatch.api_matcher import CostTracker
    from jobmatch.pricing import price_usage

    tracker = CostTracker()
    # First call writes the profile into the prompt cache; the rest read it.
    tracker.record(price_usage("claude-haiku-4-5", 600, 400,
                               cache_creation_input_tokens=23_100))
    for _ in range(4):
        tracker.record(price_usage("claude-haiku-4-5", 600, 400,
                                   cache_read_input_tokens=23_100))

    assert tracker.calls == 5
    summary = tracker.format_session()
    assert "5 call(s)" in summary, summary
    assert "cache" in summary, "cache economics should be visible in the total"
    # Cached reads bill at 10% of input, so five calls must cost far less than
    # five times the first one.
    assert tracker.total_cost_usd < 5 * 0.0315, tracker.total_cost_usd
    print(f"ok  session total accumulates over top-k "
          f"(5 calls = ${tracker.total_cost_usd:.4f}, not 5x the first)")


def test_no_calls_reports_no_calls() -> None:
    from jobmatch.api_matcher import CostTracker

    assert "no API calls made" in CostTracker().format_session()
    print("ok  a run with no API calls says so")


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
