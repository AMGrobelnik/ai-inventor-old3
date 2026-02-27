"""Gemini pricing and cost calculation."""

# Gemini Pricing (per million tokens)
# Source: https://ai.google.dev/gemini-api/docs/pricing (as of 2025-12)
PRICING = {
    # Gemini 3 Pro Preview
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00, "cached_input": 0.20},
    "gemini-3-pro": {"input": 2.00, "output": 12.00, "cached_input": 0.20},
    # Gemini 3 Flash Preview
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00, "cached_input": 0.05},
    # Gemini 2.5 Pro
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached_input": 0.125},
    "gemini-2.5-pro-preview-09-2025": {"input": 1.25, "output": 10.00, "cached_input": 0.125},
    # Gemini 2.5 Flash
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cached_input": 0.03},
    "gemini-2.5-flash-preview-09-2025": {"input": 0.30, "output": 2.50, "cached_input": 0.03},
    # Gemini 2.5 Flash-Lite
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "cached_input": 0.01},
    "gemini-2.5-flash-lite-preview-09-2025": {"input": 0.10, "output": 0.40, "cached_input": 0.01},
    # Gemini 2.0 Flash
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "gemini-2.0-flash-001": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    # Gemini 2.0 Flash-Lite
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30, "cached_input": None},
    # Gemini 1.5 models (legacy)
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00, "cached_input": 0.3125},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30, "cached_input": 0.01875},
    # Default (2.5 Flash rates)
    "default": {"input": 0.30, "output": 2.50, "cached_input": 0.03},
}


def calculate_cost(usage: dict, model: str) -> float:
    """Calculate cost in USD based on usage and model pricing."""
    if not usage:
        return 0.0

    pricing = PRICING.get(model, PRICING["default"])

    input_tokens = usage.get('input_tokens', 0) or usage.get('prompt_token_count', 0)
    output_tokens = usage.get('output_tokens', 0) or usage.get('candidates_token_count', 0)
    cached_tokens = usage.get('cached_content_token_count', 0)

    # Regular input (excluding cached)
    regular_input = input_tokens - cached_tokens
    input_cost = (regular_input / 1_000_000) * pricing["input"]

    # Cached input
    cached_cost = 0.0
    if cached_tokens > 0 and pricing.get("cached_input"):
        cached_cost = (cached_tokens / 1_000_000) * pricing["cached_input"]

    # Output
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + cached_cost + output_cost
