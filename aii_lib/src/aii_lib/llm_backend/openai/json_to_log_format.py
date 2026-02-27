"""OpenAI response formatters."""

from .pricing import calculate_token_cost, WEB_SEARCH_PRICING


def create_summary(response, model: str, duration: float = 0, **kwargs) -> str:
    """Create summary with cost and metadata."""
    lines = [""]  # Blank line before summary

    # Count web search from output items (not in usage)
    web_search_count = 0
    if hasattr(response, 'output') and response.output:
        for item in response.output:
            if getattr(item, 'type', None) == 'web_search_call':
                web_search_count += 1

    service_tier = kwargs.get('service_tier')
    token_cost = calculate_token_cost(response, model, service_tier)
    web_search_cost = (web_search_count / 1000) * WEB_SEARCH_PRICING["per_1k_calls"]
    total_cost = token_cost + web_search_cost

    resp_model = getattr(response, 'model', model)
    resolved_tier = service_tier or getattr(response, 'service_tier', 'default')

    # Line 1: Total cost with breakdown if web search used
    if web_search_cost > 0:
        lines.append(f"Total cost: ${total_cost:.4f} (Tokens: ${token_cost:.4f} + Search: ${web_search_cost:.4f})")
    else:
        lines.append(f"Total cost: ${total_cost:.4f}")

    # Line 2: Model | Status | Tier
    status = getattr(response, 'status', 'unknown')
    lines.append(f"Model: {resp_model} | Status: {status} | Tier: {resolved_tier}")

    # Line 3: Duration | Reasoning | Verbosity
    reasoning_effort = "unknown"
    verbosity = "unknown"
    if hasattr(response, 'reasoning') and response.reasoning:
        reasoning_effort = getattr(response.reasoning, 'effort', 'unknown')
    if hasattr(response, 'text') and response.text:
        verbosity = getattr(response.text, 'verbosity', 'unknown')
    lines.append(f"Duration: {duration}s | Reasoning: {reasoning_effort} | Verbosity: {verbosity}")

    lines.append("")  # Blank line

    # Token usage with cost
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)

        # Check for cached tokens (OpenAI input_tokens_details.cached_tokens)
        cached_tokens = 0
        if hasattr(usage, 'input_tokens_details') and usage.input_tokens_details:
            cached_tokens = getattr(usage.input_tokens_details, 'cached_tokens', 0) or 0

        # Total context input = input_tokens (already includes cached for OpenAI)
        # OpenAI's input_tokens is the total; cached_tokens is a subset
        token_line = f"Ctx In: {input_tokens:,} | Output: {output_tokens:,}"

        if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
            reasoning_tokens = getattr(usage.output_tokens_details, 'reasoning_tokens', 0)
            if reasoning_tokens > 0:
                token_line += f" | Reasoning: {reasoning_tokens:,}"

        token_line += f" | Cost: ${token_cost:.4f}"
        lines.append(token_line)

        if cached_tokens > 0:
            lines.append(f"Cached: {cached_tokens:,} | Uncached: {input_tokens - cached_tokens:,}")

        # Web search with count and cost
        if web_search_count > 0:
            lines.append(f"Web search: {web_search_count} calls | Cost: ${web_search_cost:.4f}")

    return "\n".join(lines)
