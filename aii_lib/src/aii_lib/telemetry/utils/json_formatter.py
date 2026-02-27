"""JSON formatting utilities for telemetry output."""

import json
import re


def _clean_string_value(s: str) -> str:
    """
    Clean up string values for display.

    Removes excessive whitespace artifacts like \n\t sequences
    that appear in HuggingFace dataset descriptions.
    """
    # Replace sequences of \n\t with single space
    s = re.sub(r'[\n\t]+', ' ', s)
    # Collapse multiple spaces
    s = re.sub(r' +', ' ', s)
    return s.strip()


def _clean_json_strings(obj: any) -> any:
    """
    Recursively clean string values in a JSON object for display.
    """
    if isinstance(obj, str):
        return _clean_string_value(obj)
    elif isinstance(obj, dict):
        return {k: _clean_json_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_json_strings(item) for item in obj]
    return obj


def _unwrap_mcp_content(parsed: list | dict) -> tuple[any, bool]:
    """
    Unwrap MCP content block format and parse nested JSON strings.

    MCP tools return either:
    - Array: [{"type": "text", "text": "{\"key\": \"value\"}"}]
    - Single: {"type": "text", "text": "{\"key\": \"value\"}"}

    This function extracts and parses the nested JSON for cleaner display.

    Args:
        parsed: Parsed JSON (list or dict)

    Returns:
        Tuple of (unwrapped content, was_unwrapped)
    """
    # Handle single MCP content block: {"type": "text", "text": "..."}
    if isinstance(parsed, dict) and parsed.get("type") == "text" and "text" in parsed:
        text_content = parsed["text"]
        if isinstance(text_content, str):
            try:
                nested = json.loads(text_content)
                nested = _clean_json_strings(nested)
                return nested, True
            except json.JSONDecodeError:
                # Text isn't JSON, just clean the string directly
                return _clean_string_value(text_content), True
        return parsed, False

    # Check for MCP content block array: [{"type": "text", "text": "..."}]
    if isinstance(parsed, list) and len(parsed) >= 1:
        unwrapped_items = []
        any_unwrapped = False

        for item in parsed:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                text_content = item["text"]
                # Try to parse the text field as JSON
                if isinstance(text_content, str):
                    try:
                        nested = json.loads(text_content)
                        # Clean string values for display
                        nested = _clean_json_strings(nested)
                        unwrapped_items.append(nested)
                        any_unwrapped = True
                        continue
                    except json.JSONDecodeError:
                        # Text isn't JSON, just clean the string
                        unwrapped_items.append(_clean_string_value(text_content))
                        any_unwrapped = True
                        continue
            unwrapped_items.append(item)

        if any_unwrapped:
            # If only one item, return it directly instead of array
            if len(unwrapped_items) == 1:
                return unwrapped_items[0], True
            return unwrapped_items, True

    return parsed, False


def format_json_output(text: str, indent: int = 2) -> str:
    """
    Format JSON in text for better readability.

    Handles MCP content blocks by unwrapping nested JSON strings:
    - Input: [{"type": "text", "text": "{\"success\": true}"}]
    - Output: {"success": true}  (pretty-printed)

    Args:
        text: Text that may contain JSON
        indent: Indentation spaces for JSON formatting

    Returns:
        Formatted text with pretty-printed JSON
    """
    # Try to parse the whole text as JSON first
    try:
        parsed = json.loads(text)

        # Unwrap MCP content blocks if present
        parsed, _ = _unwrap_mcp_content(parsed)

        formatted = json.dumps(parsed, indent=indent, ensure_ascii=False)
        lines = formatted.split('\n')
        if len(lines) > 1:
            # Add spacing prefix to continuation lines for aligned display
            formatted = '\n         '.join(lines)
        return formatted
    except json.JSONDecodeError:
        pass

    # Skip regex on large strings to avoid catastrophic backtracking
    if len(text) > 10000:
        return text

    # Try to find JSON objects/arrays within the text
    json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])'

    def format_match(match):
        json_str = match.group(1)
        try:
            parsed = json.loads(json_str)
            formatted = json.dumps(parsed, indent=indent, ensure_ascii=False)
            lines = formatted.split('\n')
            if len(lines) > 1:
                formatted = '\n         '.join(lines)
            return formatted
        except json.JSONDecodeError:
            return json_str

    return re.sub(json_pattern, format_match, text)
