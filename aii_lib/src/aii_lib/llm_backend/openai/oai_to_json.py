"""OpenAI response extractors and JSON serialization."""

import warnings


# =============================================================================
# Serialization Functions
# =============================================================================


def serialize_response(response) -> dict | None:
    """Serialize response object to dict for JSON logging."""
    if not response:
        return None
    try:
        # Suppress Pydantic serialization warnings for OpenAI SDK type mismatches
        # (web_search_call outputs don't perfectly match their Pydantic models)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            return response.model_dump()
    except Exception as e:
        raise RuntimeError(f"OpenAI response serialization failed for {type(response).__name__}: {e}") from e


# =============================================================================
# Extractor Functions
# =============================================================================


def extract_reasoning(response) -> str:
    """Extract reasoning/thinking from response."""
    if not response:
        return ""

    if hasattr(response, 'output') and isinstance(response.output, list):
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'reasoning':
                if hasattr(item, 'summary') and isinstance(item.summary, list):
                    texts = [
                        summary_obj.text
                        for summary_obj in item.summary
                        if hasattr(summary_obj, 'text')
                    ]
                    if texts:
                        return "\n\n".join(texts)
    return ""


def extract_output(response) -> str:
    """Extract output text from response."""
    if not response:
        return ""

    if hasattr(response, 'output_text'):
        return response.output_text or ""

    if hasattr(response, 'output') and isinstance(response.output, list):
        texts = []
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'message':
                if hasattr(item, 'content') and isinstance(item.content, list):
                    for content_part in item.content:
                        if hasattr(content_part, 'text'):
                            texts.append(content_part.text)
        if texts:
            return "\n\n".join(texts)
    return ""
