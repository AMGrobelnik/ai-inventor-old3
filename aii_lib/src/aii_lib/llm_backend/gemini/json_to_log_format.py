"""Gemini response formatters."""

from .pricing import calculate_cost


def create_summary(response, model: str, duration: float = 0, **kwargs) -> str:
    """Create summary with cost and metadata."""
    lines = [""]  # Blank line before summary

    usage = kwargs.get('usage', {})
    total_cost = calculate_cost(usage, model)

    # Line 1: Cost | Model
    lines.append(f"Total cost: ${total_cost:.4f} | Model: {model}")

    # Line 2: Finish reason
    finish_reason = "unknown"
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'finish_reason'):
            finish_reason = str(candidate.finish_reason)
    lines.append(f"Finish reason: {finish_reason}")

    lines.extend(["", ""])  # Blank lines

    # Token usage — Gemini's input_tokens already includes cached
    input_tokens = usage.get('input_tokens', 0)
    output_tokens = usage.get('output_tokens', 0)
    total_tokens = usage.get('total_tokens', 0)
    cached_tokens = usage.get('cached_content_token_count', 0)

    lines.append(f"Ctx In: {input_tokens:,} | Output: {output_tokens:,}")

    if cached_tokens > 0:
        lines.append(f"Cached: {cached_tokens:,} | Uncached: {input_tokens - cached_tokens:,}")

    lines.append(f"Total tokens: {total_tokens:,}")

    return "\n".join(lines)
