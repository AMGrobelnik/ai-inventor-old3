"""Anthropic response extractors and JSON serialization."""

from aii_lib.telemetry import logger


# =============================================================================
# Serialization Functions
# =============================================================================


def serialize_response(response) -> dict | None:
    """Serialize response object to dict for JSON logging."""
    if not response:
        return None
    try:
        return response.model_dump() if hasattr(response, 'model_dump') else dict(response)
    except Exception as e:
        raise RuntimeError(f"Anthropic response serialization failed for {type(response).__name__}: {e}") from e


# =============================================================================
# Extractor Functions
# =============================================================================


def extract_thinking(response) -> str:
    """Extract thinking/reasoning from response (extended thinking)."""
    if not response or not hasattr(response, 'content'):
        return ""

    thinking_blocks = []
    for block in response.content:
        if hasattr(block, 'type') and block.type == 'thinking':
            if hasattr(block, 'thinking'):
                thinking_blocks.append(block.thinking)

    return "\n\n".join(thinking_blocks)


def extract_output(response) -> str:
    """Extract text output from response."""
    if not response or not hasattr(response, 'content'):
        return ""

    text_blocks = []
    for block in response.content:
        if hasattr(block, 'type') and block.type == 'text':
            if hasattr(block, 'text'):
                text_blocks.append(block.text)

    return "\n\n".join(text_blocks)


def extract_usage(response) -> dict:
    """Extract usage data from response."""
    if not response or not hasattr(response, 'usage'):
        return {}

    usage = response.usage
    return {
        "input_tokens": getattr(usage, 'input_tokens', 0),
        "output_tokens": getattr(usage, 'output_tokens', 0),
        "cache_creation_input_tokens": getattr(usage, 'cache_creation_input_tokens', 0),
        "cache_read_input_tokens": getattr(usage, 'cache_read_input_tokens', 0),
    }
