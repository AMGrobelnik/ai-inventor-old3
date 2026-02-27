"""Formatting helper functions for prompt building."""

# Re-export from aii_lib for backwards compatibility
from aii_lib import get_model_short


def format_value(value, indent: int = 0) -> list[str]:
    """Recursively format a value with proper indentation.

    Converts nested dicts and lists into indented text blocks
    suitable for inclusion in LLM prompts.

    Args:
        value: The value to format (dict, list, or scalar)
        indent: Current indentation level

    Returns:
        List of formatted lines
    """
    prefix = "  " * indent
    lines = []

    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}:")
                lines.extend(format_value(v, indent + 1))
            else:
                lines.append(f"{prefix}{k}: {v}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                item_str = ", ".join(f"{k}: {v}" for k, v in item.items())
                lines.append(f"{prefix}- {item_str}")
            else:
                lines.append(f"{prefix}- {item}")
    else:
        lines.append(f"{prefix}{value}")

    return lines
