"""Model pricing and per-call cost arithmetic.

Prices are USD per million tokens, first-party Anthropic API rates. They are
hard-coded because the API does not expose pricing programmatically -- so treat
this table as a cache that needs updating when rates change, and check
https://anthropic.com/pricing if a number looks wrong.

This module belongs to the API path only: local mode never imports it, because
local mode has no cost to track.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# model id -> (input $/MTok, output $/MTok)
PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-fable-5-1": (10.00, 50.00),
}

# Cheapest current model -- the sensible default when iterating on prompts.
DEFAULT_MODEL = "claude-haiku-4-5"

# Prompt-cache multipliers applied to the input rate.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10

_DATE_SUFFIX = re.compile(r"-\d{8}$")


def normalize_model(model: str) -> str:
    """Map a model string onto a pricing-table key.

    Current model IDs carry no date suffix, but older snapshot-style IDs like
    `claude-haiku-4-5-20251001` still appear in scripts and docs, so they are
    folded onto the base ID rather than being treated as unknown.
    """
    model = model.strip()
    if model in PRICING:
        return model
    stripped = _DATE_SUFFIX.sub("", model)
    return stripped if stripped in PRICING else model


def is_known_model(model: str) -> bool:
    return normalize_model(model) in PRICING


@dataclass
class Usage:
    """Token counts for one API call, plus the dollars they cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    priced: bool = True  # False when the model is not in the pricing table

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


def price_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> Usage:
    """Turn raw token counts into a priced Usage record."""
    key = normalize_model(model)
    rates = PRICING.get(key)
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        priced=rates is not None,
    )
    if rates is None:
        # Unknown model: report tokens honestly and refuse to invent a price.
        return usage

    input_rate, output_rate = rates
    per_token_in = input_rate / 1_000_000
    per_token_out = output_rate / 1_000_000
    usage.cost_usd = (
        input_tokens * per_token_in
        + output_tokens * per_token_out
        + cache_creation_input_tokens * per_token_in * CACHE_WRITE_MULTIPLIER
        + cache_read_input_tokens * per_token_in * CACHE_READ_MULTIPLIER
    )
    return usage


def estimate_input_cost(model: str, input_tokens: int) -> float | None:
    """Input-side cost only -- used by --dry-run, where no output exists yet."""
    rates = PRICING.get(normalize_model(model))
    if rates is None:
        return None
    return input_tokens * rates[0] / 1_000_000


def format_usd(amount: float) -> str:
    """Format a cost without rounding small amounts away to $0.00."""
    if amount == 0:
        return "$0.00"
    if amount < 0.01:
        return f"${amount:.6f}"
    return f"${amount:.4f}"
