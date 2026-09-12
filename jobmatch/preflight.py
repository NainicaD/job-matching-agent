"""Credential check and a live, one-token verification call.

Spec order of operations: confirm the key exists, then prove it works, and do
both *before* any resume parsing or matching -- so a bad key costs you a second,
not a full profile build followed by a crash on the first real call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import anthropic

from .pricing import DEFAULT_MODEL, format_usd, price_usage

ENV_VAR = "ANTHROPIC_API_KEY"

# The SDK resolves credentials in this order, so an unset ANTHROPIC_API_KEY does
# not necessarily mean "no credentials" -- an `ant auth login` profile works too.
_ALT_ENV_VAR = "ANTHROPIC_AUTH_TOKEN"
_PROFILE_DIRS = (
    Path.home() / ".config" / "anthropic",
    Path.home() / ".anthropic",
)

SETUP_INSTRUCTIONS = f"""\
{ENV_VAR} is not set, so API mode cannot run.

To fix it, get a key from https://console.anthropic.com/settings/keys and then:

  For this shell only (gone when you close the terminal):

      export {ENV_VAR}='sk-ant-...'

  Permanently (zsh, which is your shell):

      echo "export {ENV_VAR}='sk-ant-...'" >> ~/.zshrc
      source ~/.zshrc

  Then confirm it is visible:

      echo ${ENV_VAR}
      python match.py check

Use single quotes so the shell does not touch the key, and do not commit it to
a file that gets pushed anywhere.

Alternatively, run `ant auth login` to store an OAuth profile, which the SDK
picks up with no environment variable at all.

You do not need any of this for offline matching:

      python match.py match job.txt --local
"""


@dataclass
class CredentialReport:
    ok: bool
    source: str
    detail: str

    def instructions(self) -> str:
        return SETUP_INSTRUCTIONS


def check_credentials() -> CredentialReport:
    """Look for usable credentials without making any network call."""
    key = os.environ.get(ENV_VAR, "")
    if key.strip():
        stripped = key.strip()
        detail = f"{ENV_VAR} is set ({_mask(stripped)}, {len(stripped)} chars)"
        warnings = []
        if key != stripped:
            warnings.append("value has leading/trailing whitespace - that will 401")
        if not stripped.startswith("sk-ant-"):
            warnings.append("value does not start with 'sk-ant-' - check it was copied whole")
        if os.environ.get(_ALT_ENV_VAR, "").strip():
            warnings.append(
                f"{_ALT_ENV_VAR} is also set - the SDK sends both headers and the API "
                "rejects that. Unset one."
            )
        if warnings:
            detail += "\n  warning: " + "\n  warning: ".join(warnings)
        return CredentialReport(True, ENV_VAR, detail)

    if os.environ.get(_ALT_ENV_VAR, "").strip():
        return CredentialReport(
            True, _ALT_ENV_VAR, f"{_ALT_ENV_VAR} is set (OAuth token)"
        )

    for directory in _PROFILE_DIRS:
        if directory.is_dir() and any(directory.iterdir()):
            return CredentialReport(
                True,
                "ant profile",
                f"no {ENV_VAR}, but a credential profile exists at {directory} "
                "(run `ant auth status` to confirm it is active)",
            )

    return CredentialReport(False, "none", f"{ENV_VAR} is not set")


def _mask(key: str) -> str:
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:10]}...{key[-4:]}"


@dataclass
class VerificationResult:
    ok: bool
    message: str
    cost_usd: float = 0.0


def verify_api_key(model: str = DEFAULT_MODEL) -> VerificationResult:
    """Make the smallest possible real call to prove the key works.

    `max_tokens=1` means the response is truncated immediately, which is the
    point -- this costs a fraction of a cent and confirms authentication,
    model access, and network reachability in one shot.
    """
    try:
        client = anthropic.Anthropic(max_retries=1, timeout=30.0)
        response = client.messages.create(
            model=model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
    except anthropic.AuthenticationError as exc:
        return VerificationResult(False, f"Key rejected (401): {exc}")
    except anthropic.PermissionDeniedError as exc:
        return VerificationResult(
            False, f"Key valid but not permitted to use {model} (403): {exc}"
        )
    except anthropic.NotFoundError as exc:
        return VerificationResult(False, f"Model {model!r} not found (404): {exc}")
    except anthropic.RateLimitError as exc:
        return VerificationResult(
            False, f"Rate limited (429) before the test call could complete: {exc}"
        )
    except anthropic.APIStatusError as exc:
        return VerificationResult(False, f"API error {exc.status_code}: {exc}")
    except anthropic.APIConnectionError as exc:
        return VerificationResult(False, f"Could not reach the API: {exc}")

    usage = price_usage(
        model,
        input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
    )
    return VerificationResult(
        True,
        f"Key works. Test call to {model} used "
        f"{usage.input_tokens} in / {usage.output_tokens} out tokens "
        f"({format_usd(usage.cost_usd)}).",
        cost_usd=usage.cost_usd,
    )
