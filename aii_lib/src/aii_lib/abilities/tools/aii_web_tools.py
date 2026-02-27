"""
Fast web tools wrapper for ToolUniverse.

Uses HTTP ability service for efficient execution.
"""

from typing import Any, Dict

from tooluniverse.tool_registry import register_tool
from tooluniverse.base_tool import BaseTool


@register_tool('WebSearchFast', config={
    "name": "aii_web_search_fast",
    "type": "WebSearchFast",
    "description": "Fast web search using Serper.dev (Google API). Returns titles, URLs, and snippets.",
    "category": "aii_tools",
    "parameter": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default: 10)",
                "default": 10
            }
        },
        "required": ["query"]
    }
})
class WebSearchFast(BaseTool):
    """Fast web search tool using HTTP ability service."""

    def run(self, arguments: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        if arguments is None:
            arguments = kwargs

        from aii_lib.abilities.ability_server import aii_web_search
        return aii_web_search(
            query=arguments.get("query", ""),
            max_results=arguments.get("max_results", 10),
        )


@register_tool('WebFetchDirect', config={
    "name": "aii_web_fetch_direct",
    "type": "WebFetchDirect",
    "description": (
        "Fetch a web page or PDF directly as text. Supports HTML and PDF. "
        "Returns the first max_chars of content starting from char_offset. "
        "Use char_offset to paginate through large documents."
    ),
    "category": "aii_tools",
    "parameter": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch (HTML page or PDF)"
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return (default: 10000, max: 50000)",
                "default": 10000
            },
            "char_offset": {
                "type": "integer",
                "description": "Character offset to start reading from (default: 0). Use with max_chars to paginate through large documents.",
                "default": 0
            }
        },
        "required": ["url"]
    },
    "mcp_annotations": {"readOnlyHint": True, "destructiveHint": False}
})
class WebFetchDirect(BaseTool):
    """Fast web fetch tool using HTTP ability service."""

    def run(self, arguments: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        if arguments is None:
            arguments = kwargs

        from aii_lib.abilities.ability_server import aii_web_fetch
        return aii_web_fetch(
            url=arguments.get("url", ""),
            max_chars=arguments.get("max_chars", 10000),
            char_offset=arguments.get("char_offset", 0),
        )


@register_tool('WebFetchGrep', config={
    "name": "aii_web_fetch_grep",
    "type": "WebFetchGrep",
    "description": (
        "Grep through a web page or PDF for a regex pattern. "
        "Fetches the ENTIRE document (HTML or PDF), extracts all text, "
        "and searches for regex matches — returning each with a character-based "
        "context window around it. Overlapping windows are merged. "
        "Use this to find specific information in research papers, "
        "PDFs, or large web pages without reading the entire content."
    ),
    "category": "aii_tools",
    "parameter": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch and search (HTML page or PDF)"
            },
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for in the document"
            },
            "max_matches": {
                "type": "integer",
                "description": "Maximum number of matches to return (default: 20).",
                "default": 20
            },
            "context_chars": {
                "type": "integer",
                "description": "Characters of context on EACH side of a match (default: 200, so 400 total per match). Use chars_before/chars_after instead for asymmetric windows.",
                "default": 200
            },
            "chars_before": {
                "type": "integer",
                "description": "Characters before each match. If set, overrides context_chars for the before side only."
            },
            "chars_after": {
                "type": "integer",
                "description": "Characters after each match. If set, overrides context_chars for the after side only."
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Case-insensitive matching (default: false).",
                "default": False
            }
        },
        "required": ["url", "pattern"]
    },
    "mcp_annotations": {"readOnlyHint": True, "destructiveHint": False}
})
class WebFetchGrep(BaseTool):
    """Grep through web pages/PDFs using HTTP ability service."""

    def run(self, arguments: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        if arguments is None:
            arguments = kwargs

        from aii_lib.abilities.ability_server import aii_web_fetch_grep
        return aii_web_fetch_grep(
            url=arguments.get("url", ""),
            pattern=arguments.get("pattern", ""),
            max_matches=arguments.get("max_matches", 20),
            context_chars=arguments.get("context_chars", 200),
            chars_before=arguments.get("chars_before"),
            chars_after=arguments.get("chars_after"),
            case_insensitive=arguments.get("case_insensitive", False),
        )


# Async wrappers for compatibility
async def aii_web_search_fast_async(query: str, max_results: int = 10) -> Dict[str, Any]:
    """Async wrapper for web search."""
    from aii_lib.abilities.ability_server import aii_web_search
    return aii_web_search(query=query, max_results=max_results)


async def aii_web_fetch_direct_async(
    url: str,
    max_chars: int = 10000,
    char_offset: int = 0,
) -> Dict[str, Any]:
    """Async wrapper for web fetch."""
    from aii_lib.abilities.ability_server import aii_web_fetch
    return aii_web_fetch(url=url, max_chars=max_chars, char_offset=char_offset)


async def aii_web_fetch_grep_async(
    url: str,
    pattern: str,
    max_matches: int = 20,
    context_chars: int = 200,
    chars_before: int = None,
    chars_after: int = None,
    case_insensitive: bool = False,
) -> Dict[str, Any]:
    """Async wrapper for web grep."""
    from aii_lib.abilities.ability_server import aii_web_fetch_grep
    return aii_web_fetch_grep(
        url=url,
        pattern=pattern,
        max_matches=max_matches,
        context_chars=context_chars,
        chars_before=chars_before,
        chars_after=chars_after,
        case_insensitive=case_insensitive,
    )


__all__ = [
    "WebSearchFast", "WebFetchDirect", "WebFetchGrep",
    "aii_web_search_fast_async", "aii_web_fetch_direct_async", "aii_web_fetch_grep_async",
]
