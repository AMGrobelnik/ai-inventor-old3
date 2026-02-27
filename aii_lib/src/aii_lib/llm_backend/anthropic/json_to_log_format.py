"""Anthropic response formatters."""

from .pricing import calculate_cost


def create_summary(response, model: str, duration: float = 0, **kwargs) -> str:
    """Create summary with cost and metadata."""
    lines = [""]  # Blank line before summary

    # Extract usage
    usage = {}
    if hasattr(response, 'usage') and response.usage:
        usage = {
            "input_tokens": getattr(response.usage, 'input_tokens', 0),
            "output_tokens": getattr(response.usage, 'output_tokens', 0),
            "cache_creation_input_tokens": getattr(response.usage, 'cache_creation_input_tokens', 0),
            "cache_read_input_tokens": getattr(response.usage, 'cache_read_input_tokens', 0),
        }

    total_cost = calculate_cost(usage, model)
    resp_model = getattr(response, 'model', model)

    # Line 1: Cost | Model
    lines.append(f"Total cost: ${total_cost:.4f} | Model: {resp_model}")

    # Line 2: Stop reason
    stop_reason = getattr(response, 'stop_reason', 'unknown')
    lines.append(f"Stop reason: {stop_reason}")

    lines.extend(["", ""])  # Blank lines

    # Token usage
    input_tokens = usage.get('input_tokens', 0)
    output_tokens = usage.get('output_tokens', 0)
    cache_creation = usage.get('cache_creation_input_tokens', 0)
    cache_read = usage.get('cache_read_input_tokens', 0)

    total_ctx_in = input_tokens + cache_creation + cache_read
    lines.append(f"Ctx In: {total_ctx_in:,} | Output: {output_tokens:,}")

    if cache_creation > 0 or cache_read > 0:
        lines.append(f"Uncached: {input_tokens:,} | Cache write: {cache_creation:,} | Cache read: {cache_read:,}")

    total_tokens = total_ctx_in + output_tokens
    lines.append(f"Total tokens: {total_tokens:,}")

    return "\n".join(lines)
