"""OpenAI pricing and cost calculation."""

# OpenAI Pricing (per million tokens)
# Source: https://openai.com/api/pricing/ (December 2025)

# Standard tier pricing (default)
PRICING_STANDARD = {
    # Flagship models (GPT-5 family)
    "gpt-5.2": {"input": 1.75, "output": 14.00, "cached_input": 0.175},
    "gpt-5.2-pro": {"input": 21.00, "output": 168.00, "cached_input": None},
    "gpt-5-mini": {"input": 0.25, "output": 2.00, "cached_input": 0.025},
    # Fine-tunable models (GPT-4.1 family)
    "gpt-4.1": {"input": 3.00, "output": 12.00, "cached_input": 0.75},
    "gpt-4.1-mini": {"input": 0.80, "output": 3.20, "cached_input": 0.20},
    "gpt-4.1-nano": {"input": 0.20, "output": 0.80, "cached_input": 0.05},
    # Reasoning models
    "o3": {"input": 1.75, "output": 14.00, "cached_input": 0.175},
    "o4-mini": {"input": 4.00, "output": 16.00, "cached_input": 1.00},
    # Legacy models (GPT-4o)
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 0.625},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.0375},
    # Default fallback (gpt-5-mini rates)
    "default": {"input": 0.25, "output": 2.00, "cached_input": 0.025},
}

# Priority tier pricing (service_tier="priority")
# Source: https://openai.com/api/priority-processing/ (December 2025)
PRICING_PRIORITY = {
    # Flagship models (GPT-5 family)
    "gpt-5.2": {"input": 3.50, "output": 28.00, "cached_input": 0.35},
    "gpt-5.1": {"input": 2.50, "output": 20.00, "cached_input": 0.25},
    "gpt-5": {"input": 2.50, "output": 20.00, "cached_input": 0.25},
    "gpt-5-mini": {"input": 0.45, "output": 3.60, "cached_input": 0.045},
    # Fine-tunable models (GPT-4.1 family)
    "gpt-4.1": {"input": 3.50, "output": 14.00, "cached_input": 0.875},
    "gpt-4.1-mini": {"input": 0.70, "output": 2.80, "cached_input": 0.175},
    "gpt-4.1-nano": {"input": 0.20, "output": 0.80, "cached_input": 0.05},
    # Reasoning models
    "o3": {"input": 3.50, "output": 14.00, "cached_input": 0.875},
    "o4-mini": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    # Legacy models (GPT-4o)
    "gpt-4o": {"input": 4.25, "output": 17.00, "cached_input": 2.125},
    "gpt-4o-mini": {"input": 0.25, "output": 1.00, "cached_input": 0.125},
    # Default fallback (gpt-5-mini priority rates)
    "default": {"input": 0.45, "output": 3.60, "cached_input": 0.045},
}

# Backwards compatibility alias
PRICING = PRICING_STANDARD

# Web search tool pricing: $10.00 per 1K calls + search content tokens at model rates
WEB_SEARCH_PRICING = {"per_1k_calls": 10.00}


def calculate_token_cost(response, model: str, service_tier: str | None = None) -> float:
    """Calculate token cost only (excluding web search) based on usage, model, and service tier.

    Args:
        response: API response with usage data
        model: Model name (e.g., "gpt-5-mini")
        service_tier: "priority" for priority tier pricing, else standard

    Returns:
        Token cost in USD (input + output + cached tokens)
    """
    if not hasattr(response, 'usage') or not response.usage:
        return 0.0

    usage = response.usage

    # Select pricing table based on service tier
    if service_tier == "priority":
        pricing_table = PRICING_PRIORITY
    else:
        pricing_table = PRICING_STANDARD

    pricing = pricing_table.get(model, pricing_table["default"])

    input_tokens = getattr(usage, 'input_tokens', 0)
    output_tokens = getattr(usage, 'output_tokens', 0)
    cached_tokens = 0

    if hasattr(usage, 'input_tokens_details') and usage.input_tokens_details:
        cached_tokens = getattr(usage.input_tokens_details, 'cached_tokens', 0)

    # Non-cached input
    non_cached_input = input_tokens - cached_tokens
    input_cost = (non_cached_input / 1_000_000) * pricing["input"]

    # Cached input
    cached_cost = 0.0
    if cached_tokens > 0 and pricing.get("cached_input"):
        cached_cost = (cached_tokens / 1_000_000) * pricing["cached_input"]

    # Output
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + cached_cost + output_cost


def calculate_web_search_cost(response) -> float:
    """Calculate web search tool cost.

    Args:
        response: API response with usage data

    Returns:
        Web search cost in USD ($10 per 1K calls)
    """
    if not hasattr(response, 'usage') or not response.usage:
        return 0.0

    usage = response.usage
    if hasattr(usage, 'server_tool_use') and usage.server_tool_use:
        web_search_requests = getattr(usage.server_tool_use, 'web_search_requests', 0) or 0
        return (web_search_requests / 1000) * WEB_SEARCH_PRICING["per_1k_calls"]
    return 0.0


def calculate_cost(response, model: str, service_tier: str | None = None) -> float:
    """Calculate total cost in USD based on usage, model, and service tier.

    Args:
        response: API response with usage data
        model: Model name (e.g., "gpt-5-mini")
        service_tier: "priority" for priority tier pricing, else standard

    Returns:
        Total cost including tokens + web search tool calls
    """
    token_cost = calculate_token_cost(response, model, service_tier)
    web_search_cost = calculate_web_search_cost(response)
    return token_cost + web_search_cost
