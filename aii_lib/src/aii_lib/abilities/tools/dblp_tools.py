"""
DBLP bibliography tools for ToolUniverse.

Two tools:
  - dblp_bib_search: Search DBLP for papers (returns metadata, no BibTeX)
  - dblp_bib_fetch: Fetch BibTeX for specific papers by DBLP key

Core functions live in the skill scripts (.claude/skills/dblp_bib/scripts/).
ToolUniverse tools call through the ability server HTTP API.
"""

from typing import Any, Dict

from tooluniverse.base_tool import BaseTool
from tooluniverse.tool_registry import register_tool


# =============================================================================
# ToolUniverse tools (call through ability server HTTP API)
# =============================================================================

@register_tool('DblpBibSearch', config={
    "name": "dblp_bib_search",
    "type": "DblpBibSearch",
    "description": (
        "Search the DBLP computer science bibliography. "
        "Returns paper metadata (title, authors, venue, year, dblp_key). "
        "Use dblp_bib_fetch with the dblp_key to get BibTeX entries. "
        "Best for known-item lookup: include author last name + year "
        "(e.g. 'Vaswani 2017 attention', 'Wei chain of thought 2022'). "
        "Also works for topic searches (e.g. 'multi-agent debate LLM')."
    ),
    "category": "aii_tools",
    "parameter": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query. For best results include author last name + year "
                    "(e.g. 'Vaswani 2017', 'Wei chain of thought 2022'). "
                    "Topic searches also work (e.g. 'multi-agent debate LLM')."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum papers to return (default: 5, max: 20)",
                "default": 5,
            },
            "year_from": {
                "type": "integer",
                "description": "Only include papers from this year onward (optional)",
            },
            "year_to": {
                "type": "integer",
                "description": "Only include papers up to this year (optional)",
            },
        },
        "required": ["query"],
    },
    "mcp_annotations": {"readOnlyHint": True, "destructiveHint": False},
})
class DblpBibSearch(BaseTool):
    """DBLP bibliography search (metadata only)."""

    def run(self, arguments: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        if arguments is None:
            arguments = kwargs

        from aii_lib.abilities.ability_server import dblp_bib_search
        return dblp_bib_search(
            query=arguments.get("query", ""),
            max_results=min(arguments.get("max_results", 5), 20),
            year_from=arguments.get("year_from"),
            year_to=arguments.get("year_to"),
        )


@register_tool('DblpBibFetch', config={
    "name": "dblp_bib_fetch",
    "type": "DblpBibFetch",
    "description": (
        "Fetch BibTeX entries from DBLP by key. "
        "Use dblp_bib_search first to find papers and get their dblp_key, "
        "then call this with the keys to get ready-to-use BibTeX entries. "
        "Accepts one or more keys in a single call."
    ),
    "category": "aii_tools",
    "parameter": {
        "type": "object",
        "properties": {
            "dblp_keys": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "description": (
                    "DBLP key(s) from search results "
                    "(e.g. 'conf/nips/VaswaniSPUJGKP17' or a list of keys)."
                ),
            },
            "years": {
                "oneOf": [
                    {"type": "integer"},
                    {"type": "array", "items": {"type": "integer"}},
                ],
                "description": (
                    "Publication year(s) matching the keys (optional, for cleaner citation keys)."
                ),
            },
        },
        "required": ["dblp_keys"],
    },
    "mcp_annotations": {"readOnlyHint": True, "destructiveHint": False},
})
class DblpBibFetch(BaseTool):
    """Fetch BibTeX entries by DBLP key."""

    def run(self, arguments: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        if arguments is None:
            arguments = kwargs

        from aii_lib.abilities.ability_server import dblp_bib_fetch
        return dblp_bib_fetch(
            dblp_keys=arguments.get("dblp_keys", []),
            years=arguments.get("years", []),
        )


__all__ = ["DblpBibSearch", "DblpBibFetch"]
