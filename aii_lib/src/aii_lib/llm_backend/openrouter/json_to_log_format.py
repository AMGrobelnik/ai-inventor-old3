"""OpenRouter response to log format conversion."""

from .or_to_json import extract_usage


def create_summary(response, model: str, duration: float = 0, **kwargs) -> str:
    """Create a human-readable summary from response.

    Args:
        response: The OpenRouter API response
        model: Model name used

    Returns:
        Formatted summary string
    """
    lines = [""]  # Blank line before summary

    # Extract usage (includes actual cost from OpenRouter when usage accounting enabled)
    usage = extract_usage(response)
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Use actual cost from OpenRouter (no web search cost for OpenRouter)
    token_cost = usage.get("cost", 0.0)
    total_cost = token_cost

    # Get finish reason
    finish_reason = "unknown"
    if hasattr(response, 'choices') and response.choices:
        finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown') or 'unknown'

    # Get model from response (may differ from requested)
    actual_model = getattr(response, 'model', model) or model

    # Line 1: Total cost
    lines.append(f"Total cost: ${total_cost:.4f}")

    # Line 2: Model | Status
    lines.append(f"Model: {actual_model} | Status: {finish_reason}")

    lines.append("")  # Blank line

    # Token usage with cost
    # OpenRouter's prompt_tokens is total input (includes cached)
    token_line = f"Ctx In: {prompt_tokens:,} | Output: {completion_tokens:,}"
    token_line += f" | Cost: ${token_cost:.4f}"
    lines.append(token_line)

    return "\n".join(lines)
