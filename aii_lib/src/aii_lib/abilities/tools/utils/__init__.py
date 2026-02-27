"""
Tool utilities - shared helpers for ToolUniverse tools.

Includes:
- ToolUniverse schema conversion to OpenRouter format
- Async tool execution with thread pool
- Web cache with TTL support
"""

from .tool_execution import (
    get_openrouter_tools,
    execute_tool_calls,
    convert_schema_to_openrouter,
    TOOL_RENAME_MAP,
    TOOL_REVERSE_MAP,
)

from .._aii_web_tools.web_cache import (
    WebCache,
    search_cache,
    content_cache,
    cache_content,
    cache_search_result,
    get_cached_content,
    get_cached_search,
    has_cached_content,
    clear_all_caches,
)

__all__ = [
    # Tool execution
    "get_openrouter_tools",
    "execute_tool_calls",
    "convert_schema_to_openrouter",
    "TOOL_RENAME_MAP",
    "TOOL_REVERSE_MAP",
    # Web cache
    "WebCache",
    "search_cache",
    "content_cache",
    "cache_content",
    "cache_search_result",
    "get_cached_content",
    "get_cached_search",
    "has_cached_content",
    "clear_all_caches",
]
