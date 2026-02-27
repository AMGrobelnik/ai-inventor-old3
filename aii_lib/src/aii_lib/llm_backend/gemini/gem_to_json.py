"""Gemini response extractors and JSON serialization."""


# =============================================================================
# Serialization Functions
# =============================================================================


def serialize_response(response) -> dict | None:
    """Serialize response object to dict for JSON logging."""
    if not response:
        return None
    try:
        if hasattr(response, 'to_dict'):
            return response.to_dict()
        elif hasattr(response, '__dict__'):
            return {k: str(v) for k, v in response.__dict__.items() if not k.startswith('_')}
        return str(response)
    except Exception as e:
        raise RuntimeError(f"Gemini response serialization failed for {type(response).__name__}: {e}") from e


# =============================================================================
# Extractor Functions
# =============================================================================


def extract_thinking(response) -> str:
    """Extract thinking from response (if thinking is enabled)."""
    if not response:
        return ""

    # Check for thinking in response parts
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and candidate.content:
                content = candidate.content
                if hasattr(content, 'parts'):
                    for part in content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            return part.text if hasattr(part, 'text') else str(part.thought)

    return ""


def extract_output(response) -> str:
    """Extract text output from response."""
    if not response:
        return ""

    # Try .text property first (common shortcut)
    if hasattr(response, 'text'):
        return response.text or ""

    # Try candidates
    if hasattr(response, 'candidates') and response.candidates:
        texts = []
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and candidate.content:
                content = candidate.content
                if hasattr(content, 'parts'):
                    for part in content.parts:
                        if hasattr(part, 'text') and not getattr(part, 'thought', False):
                            texts.append(part.text)
        if texts:
            return "\n\n".join(texts)

    return ""


def extract_usage(response) -> dict:
    """Extract usage data from response including thoughts tokens."""
    if not response:
        return {}

    # Try usage_metadata
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        um = response.usage_metadata
        return {
            "input_tokens": getattr(um, 'prompt_token_count', 0),
            "output_tokens": getattr(um, 'candidates_token_count', 0),
            "total_tokens": getattr(um, 'total_token_count', 0),
            "cached_content_token_count": getattr(um, 'cached_content_token_count', 0),
            "thoughts_token_count": getattr(um, 'thoughts_token_count', 0),
        }

    return {}
