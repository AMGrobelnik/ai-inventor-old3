"""Anthropic pricing and cost calculation."""

# Anthropic Pricing (per million tokens)
# Source: https://docs.anthropic.com/en/docs/about-claude/pricing (as of 2025-12)
PRICING = {
    # Claude 4.5 models
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "claude-4.5-opus": {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    # Claude 4.1 models
    "claude-opus-4-1-20250414": {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    "claude-4.1-opus": {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    # Claude 4 models
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    "claude-4-opus": {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    # Claude Sonnet 4.5
    "claude-sonnet-4-5-20250514": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "claude-4.5-sonnet": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    # Claude Sonnet 4
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "claude-4-sonnet": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    # Claude Haiku 4.5
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
    "claude-haiku-4-5-20250514": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
    "claude-4.5-haiku": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
    # Claude Haiku 3.5
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00, "cache_write": 1.00, "cache_read": 0.08},
    "claude-3.5-haiku": {"input": 0.80, "output": 4.00, "cache_write": 1.00, "cache_read": 0.08},
    # Claude Haiku 3
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25, "cache_write": 0.30, "cache_read": 0.03},
    # Default (Sonnet 4.5 rates)
    "default": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
}


def calculate_cost(usage: dict, model: str) -> float:
    """Calculate cost in USD based on usage and model pricing."""
    if not usage:
        return 0.0

    pricing = PRICING.get(model, PRICING["default"])

    input_tokens = usage.get('input_tokens') or 0
    output_tokens = usage.get('output_tokens') or 0
    cache_creation_tokens = usage.get('cache_creation_input_tokens') or 0
    cache_read_tokens = usage.get('cache_read_input_tokens') or 0

    # Regular input (excluding cached)
    regular_input = input_tokens - cache_read_tokens
    input_cost = (regular_input / 1_000_000) * pricing["input"]

    # Cache write cost
    cache_write_cost = (cache_creation_tokens / 1_000_000) * pricing["cache_write"]

    # Cache read cost
    cache_read_cost = (cache_read_tokens / 1_000_000) * pricing["cache_read"]

    # Output cost
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + cache_write_cost + cache_read_cost + output_cost
