"""ToolUniverse schema conversion and execution utilities.

Converts ToolUniverse tool schemas to OpenRouter/OpenAI-compatible format
and provides async execution of tool calls.

Usage:
    from aii_lib.abilities.tools.utils import (
        get_openrouter_tools,
        execute_tool_calls,
    )

    # Get tool definitions for OpenRouter
    tools = get_openrouter_tools(["aii_web_search_fast", "aii_web_fetch_direct"])

    # Execute multiple tool calls from model response
    results = await execute_tool_calls([
        {"id": "call_123", "name": "web_search", "arguments": {"query": "..."}},
    ])
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from tooluniverse import ToolUniverse

from .._aii_web_tools.web_cache import cache_content


# Global ToolUniverse instance (lazy loaded)
_tu: ToolUniverse | None = None

# Custom thread pool for tool execution (default asyncio pool is too small: CPU+4)
# Large pool since semaphore already limits concurrent LLM sessions
_tool_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    """Get or create thread pool executor for tool calls.

    Uses a large pool (1000 workers) since the semaphore already limits
    concurrent LLM sessions. Each session may make ~10 tool calls, so
    with 100 concurrent sessions we need ~1000 thread capacity.
    """
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ThreadPoolExecutor(max_workers=1000, thread_name_prefix="tooluniverse")
    return _tool_executor


def _get_tu() -> ToolUniverse:
    """Get or create ToolUniverse instance with custom tools."""
    global _tu
    if _tu is None:
        # Import custom tools to trigger @register_tool decorators
        from aii_lib.abilities import tools  # noqa: F401 - registers aii_web_search_fast, aii_web_fetch_direct, lean_run_code, etc.

        _tu = ToolUniverse()
        # Only load the specific tools we use (not all 785)
        _tu.load_tools(include_tools=["aii_web_search_fast", "aii_web_fetch_direct", "lean_run_code"])
    return _tu


def convert_schema_to_openrouter(tool_spec: dict, rename_map: dict = None) -> dict:
    """Convert a ToolUniverse tool spec to OpenRouter/OpenAI format.

    ToolUniverse format:
        {"name": "...", "description": "...", "parameter": {"properties": {...}, "required": [...]}}

    OpenRouter format:
        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

    Note: Some models (e.g., gpt-5-mini via OpenRouter) reject schemas with optional
    parameters (properties not in required list). To maximize compatibility, we only
    include required properties in the schema.

    Args:
        tool_spec: ToolUniverse tool specification
        rename_map: Optional dict mapping original names to display names (e.g., {"convert_to_markdown": "web_fetch"})
    """
    original_name = tool_spec.get("name", "")
    # Apply rename if specified
    display_name = rename_map.get(original_name, original_name) if rename_map else original_name
    description = tool_spec.get("description", "")
    parameter = tool_spec.get("parameter", {})

    required_props = set(parameter.get("required", []))

    # Only include required properties in schema - some models reject optional params
    # Also remove internal fields like "required" and "default" from property definitions
    properties = {}
    for prop_name, prop_def in parameter.get("properties", {}).items():
        if prop_name in required_props:
            clean_prop = {k: v for k, v in prop_def.items() if k not in ("required", "default")}
            properties[prop_name] = clean_prop

    return {
        "type": "function",
        "function": {
            "name": display_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required_props),
            },
        },
        "_original_name": original_name,  # Store original for execution lookup
    }


# Default tool name mappings for OpenRouter presentation
# Maps ToolUniverse names → cleaner LLM-facing names
TOOL_RENAME_MAP = {
    "convert_to_markdown": "web_fetch",       # Fetch URL content as markdown (legacy)
    "ArXiv_search_papers": "arxiv_search",    # Search arXiv papers
    # Fast tools (new): aii_web_search_fast, aii_web_fetch_direct stay as-is (already clear)
    # "web_search" stays as-is (already clear)
}

# Reverse mapping: display name -> original ToolUniverse name
TOOL_REVERSE_MAP = {v: k for k, v in TOOL_RENAME_MAP.items()}

# Default web search backend (bing has 100% success rate, others fail frequently)
DEFAULT_WEB_SEARCH_BACKEND = "bing"


def get_openrouter_tools(tool_names: list[str], rename_map: dict = None) -> list[dict]:
    """Get OpenRouter-compatible tool definitions from ToolUniverse.

    Args:
        tool_names: List of ToolUniverse tool names
        rename_map: Optional dict mapping original names to display names.
                   Defaults to TOOL_RENAME_MAP (e.g., convert_to_markdown -> web_fetch)

    Returns:
        List of OpenRouter/OpenAI-compatible tool definitions
    """
    if rename_map is None:
        rename_map = TOOL_RENAME_MAP

    tu = _get_tu()
    specs = tu.get_tool_specification_by_names(tool_names)
    return [convert_schema_to_openrouter(spec, rename_map) for spec in specs]


async def execute_tool_calls(
    tool_calls: list[dict],
    reverse_map: dict = None,
    web_search_backend: str = "auto",
) -> list[dict]:
    """Execute multiple tool calls from model response (async, non-blocking).

    Uses a custom thread pool (1000 workers) to run synchronous ToolUniverse calls,
    allowing multiple LLM sessions to truly run in parallel without blocking the event loop.

    Args:
        tool_calls: List of tool calls with keys: id, name, arguments
        reverse_map: Optional dict mapping display names back to ToolUniverse names.
                    Defaults to TOOL_REVERSE_MAP (e.g., web_fetch -> convert_to_markdown)
        web_search_backend: Backend for web_search tool. "auto" uses default (bing),
                           or specify: google|bing|brave|yahoo|duckduckgo

    Returns:
        List of results with keys: tool_call_id, name, result, error
    """
    if reverse_map is None:
        reverse_map = TOOL_REVERSE_MAP

    # Determine effective backend (auto = use default bing)
    effective_backend = DEFAULT_WEB_SEARCH_BACKEND if web_search_backend == "auto" else web_search_backend

    tu = _get_tu()

    async def execute_single(tc: dict) -> dict:
        """Execute a single tool call asynchronously."""
        tool_call_id = tc.get("id")
        display_name = tc.get("name")
        raw_args = tc.get("arguments", {})

        # Parse arguments if they're a JSON string
        if isinstance(raw_args, str):
            try:
                import json
                arguments = json.loads(raw_args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON in tool call arguments for '{display_name}': {e}") from e
        else:
            arguments = raw_args.copy() if raw_args else {}

        # Map display name back to original ToolUniverse name
        original_name = reverse_map.get(display_name, display_name)

        # Force web_search backend to configured value
        if original_name == "web_search":
            arguments["backend"] = effective_backend

        try:
            # Run synchronous ToolUniverse in custom thread pool (1000 workers)
            # Default asyncio.to_thread uses only CPU+4 workers, causing bottleneck
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                _get_executor(),
                lambda: tu.run({"name": original_name, "arguments": arguments})
            )

            # Cache web content for citation verification reuse (uses global cache)
            if original_name in ("convert_to_markdown", "aii_web_fetch_direct"):
                # convert_to_markdown uses "uri", aii_web_fetch_direct uses "url"
                url = arguments.get("url") or arguments.get("uri")
                content = None
                if isinstance(result, dict):
                    if result.get("error") or not result.get("success", True):
                        # Cache empty string to mark URL as "fetched but failed"
                        if url:
                            cache_content(url, "")
                    else:
                        content = result.get("content") or result.get("markdown_content")
                elif isinstance(result, str):
                    content = result
                if url and content:
                    cache_content(url, content)

            # Check if this was a cache hit (for aii_web_search_fast)
            cache_hit = False
            if isinstance(result, dict) and result.get("cache_hit"):
                cache_hit = True

            return {
                "tool_call_id": tool_call_id,
                "name": display_name,  # Keep display name for logging
                "original_name": original_name,  # Include original for reference
                "result": result,
                "error": None,
                "cache_hit": cache_hit,  # Track if this was a cache hit
            }
        except Exception as e:
            return {
                "tool_call_id": tool_call_id,
                "name": display_name,
                "original_name": original_name,
                "result": None,
                "error": str(e),
                "cache_hit": False,
            }

    # Execute all tool calls concurrently
    results = await asyncio.gather(*[execute_single(tc) for tc in tool_calls])
    return list(results)


__all__ = [
    "get_openrouter_tools",
    "execute_tool_calls",
    "convert_schema_to_openrouter",
    "TOOL_RENAME_MAP",
    "TOOL_REVERSE_MAP",
]
