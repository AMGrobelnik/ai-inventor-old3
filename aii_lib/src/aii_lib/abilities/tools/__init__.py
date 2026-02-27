"""
aii_lib abilities/tools - ToolUniverse tool wrappers.

Tools import logic from skills/ and add ToolUniverse registration.
This allows the same logic to work as both CLI skills and MCP tools.

Tools:
- openrouter_llms: LLM search and API calls via OpenRouter
- fast_web: Fast web search (Serper) and fetch (aiohttp+html2text+pymupdf)
- lean: Lean 4 proof verification
- dblp: DBLP bibliography search with BibTeX

Usage:
    # Import this package before starting ToolUniverse server
    import aii_lib.abilities.tools

    # Or import specific tool modules
    from aii_lib.abilities.tools import openrouter_llms_tools, aii_web_tools, dblp_tools
"""

# Import tool modules to register them with ToolUniverse
# The @register_tool decorator handles both class AND config registration
from . import openrouter_llms_tools
from . import aii_web_tools
from . import lean_tools
from . import dblp_tools

__all__ = [
    "openrouter_llms_tools",
    "aii_web_tools",
    "lean_tools",
    "dblp_tools",
]
