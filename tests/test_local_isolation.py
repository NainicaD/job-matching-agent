"""Enforce the constraint that local mode is genuinely standalone.

The spec requires that `--local` "not import or depend on anything from the API
mode's cost/error-handling code" and run with zero network access. That is easy
to state and easy to break later with one convenience import, so it is checked
mechanically here rather than left to code review.

Run directly (no pytest needed):

    python tests/test_local_isolation.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Modules that must never be reachable from the offline path.
FORBIDDEN = ("anthropic", "jobmatch.api_matcher", "jobmatch.pricing", "jobmatch.preflight")

# The real profile.txt is gitignored (it holds personal data), so these tests
# build their own throwaway one. That keeps them runnable on a fresh clone.
SAMPLE_PROFILE = """\
== SUMMARY ==
- Data science graduate student with machine learning and NLP experience.

== EXPERIENCE ==
- Built ETL pipelines in Python and SQL, processing millions of event records.
- Coordinated cross-functional delivery across engineering and client teams.

== PROJECTS ==
- Built a hybrid recommender combining collaborative filtering and TF-IDF
  content-based filtering, evaluated with RMSE on MovieLens.
- Trained a transformer classifier in PyTorch with attention-based explainability.

== SKILLS ==
- Python, SQL, PyTorch, scikit-learn, pandas, NumPy, Docker, AWS
"""


def _run(code: str, env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    env.update(env_overrides or {})
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def test_import_graph_is_clean() -> None:
    """Importing the local matcher must not drag in any API module."""
    code = f"""
import sys
import jobmatch.local_matcher
leaked = [m for m in {FORBIDDEN!r} if m in sys.modules]
print("LEAKED:" + ",".join(leaked))
"""
    proc = _run(code)
    assert proc.returncode == 0, f"import failed:\n{proc.stderr}"
    leaked = proc.stdout.strip().removeprefix("LEAKED:")
    assert not leaked, f"local_matcher pulled in API modules: {leaked}"
    print("ok  local_matcher imports no API module")


def test_local_match_works_without_anthropic_installed() -> None:
    """Simulate a machine with no `anthropic` package and no API key."""
    code = f"""
import sys

class Blocker:
    def find_module(self, name, path=None):
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError("anthropic is not installed (simulated)")
        return None

sys.meta_path.insert(0, Blocker())

from pathlib import Path
from jobmatch.local_matcher import match_local

profile = {SAMPLE_PROFILE!r}
job = Path("jobs/example_ml_engineer.txt").read_text()
result = match_local(profile, job, job_name="test")
assert result.engine == "local-tfidf"
assert 0 <= result.score <= 100
assert result.strengths, "expected at least one overlapping term"
print("SCORE:%.1f" % result.score)
"""
    proc = _run(code)
    assert proc.returncode == 0, f"local match failed:\n{proc.stdout}\n{proc.stderr}"
    assert "SCORE:" in proc.stdout, proc.stdout
    print(f"ok  local match runs with anthropic blocked and no API key "
          f"({proc.stdout.strip()})")


def test_cli_local_runs_without_key() -> None:
    """The documented CLI invocation must work with ANTHROPIC_API_KEY unset."""
    import os

    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    with tempfile.TemporaryDirectory() as tmp:
        profile_path = Path(tmp) / "profile.txt"
        profile_path.write_text(SAMPLE_PROFILE, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "match.py", "match", "jobs/example_ml_engineer.txt",
             "--local", "--json", "--profile", str(profile_path)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, env=env,
        )
    assert proc.returncode == 0, f"CLI failed:\n{proc.stdout}\n{proc.stderr}"
    assert '"engine": "local-tfidf"' in proc.stdout, proc.stdout[:500]
    print("ok  `match.py match --local` runs with no API key set")


if __name__ == "__main__":
    failures = 0
    for test in (
        test_import_graph_is_clean,
        test_local_match_works_without_anthropic_installed,
        test_cli_local_runs_without_key,
    ):
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print()
    print("all isolation checks passed" if not failures else f"{failures} check(s) failed")
    sys.exit(1 if failures else 0)
