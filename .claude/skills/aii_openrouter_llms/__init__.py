"""OpenRouter skill - LLM search and API calls.

CLI script: openrouter_llms.py
ToolUniverse tools: aii_lib.abilities.tools.openrouter_tools
"""
from .scripts.openrouter_llms import search_direct, call_direct

__all__ = ["search_direct", "call_direct"]
