"""
OpenRouter tools wrapper for ToolUniverse.

Uses HTTP ability service for efficient execution.
"""

from typing import Any, Dict

from tooluniverse.tool_registry import register_tool
from tooluniverse.base_tool import BaseTool


@register_tool('OpenRouterSearchLLMs', config={
    "name": "openrouter_search_llms",
    "type": "OpenRouterSearchLLMs",
    "description": "Search for AI models on OpenRouter. Returns model names, API IDs, context lengths, and pricing. Use to find models before calling them.",
    "category": "aii_tools",
    "parameter": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to filter models (e.g., 'claude', 'gpt', 'reasoning'). Leave empty for all models."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results. Default: 10.",
                "default": 10
            },
            "series": {
                "type": "string",
                "description": "Filter by model family (e.g., 'GPT', 'Claude', 'Gemini', 'Llama'). Optional."
            }
        },
        "required": []
    }
})
class OpenRouterSearchLLMs(BaseTool):
    """Search OpenRouter models using HTTP ability service."""

    def run(self, arguments: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        if arguments is None:
            arguments = kwargs

        from aii_lib.abilities.ability_server import openrouter_search
        return openrouter_search(
            query=arguments.get("query", ""),
            limit=arguments.get("limit", 10),
            series=arguments.get("series", ""),
        )


@register_tool('OpenRouterCallLLM', config={
    "name": "openrouter_call_llm",
    "type": "OpenRouterCallLLM",
    "description": "Call an LLM model on OpenRouter. Returns model response with token usage. Use search first to find model API names.",
    "category": "aii_tools",
    "parameter": {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "API model name (format: provider/model-name). Examples: 'anthropic/claude-sonnet-4', 'openai/gpt-5', 'google/gemini-2.5-pro'"
            },
            "input_text": {
                "type": "string",
                "description": "Simple string prompt for single-turn conversation."
            },
            "input_json": {
                "type": "string",
                "description": "Full conversation JSON for multi-turn. Mutually exclusive with input_text."
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum output tokens. Default: 9000.",
                "default": 9000
            },
            "reasoning_effort": {
                "type": "string",
                "description": "Reasoning effort for reasoning models: 'minimal', 'low', 'medium', 'high'. Optional."
            },
            "temperature": {
                "type": "number",
                "description": "Randomness (0.0-2.0): 0.0=deterministic, 0.7=balanced, 1.5+=creative. Optional."
            },
            "top_p": {
                "type": "number",
                "description": "Nucleus sampling (0.0-1.0). Optional."
            },
            "instructions": {
                "type": "string",
                "description": "System instructions/prompt. Optional."
            },
            "web_search_max_results": {
                "type": "integer",
                "description": "Enable web search with max results (e.g., 10). Optional."
            }
        },
        "required": ["model"]
    }
})
class OpenRouterCallLLM(BaseTool):
    """Call OpenRouter LLM using HTTP ability service."""

    def run(self, arguments: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        if arguments is None:
            arguments = kwargs

        from aii_lib.abilities.ability_server import openrouter_call
        return openrouter_call(
            model=arguments.get("model", ""),
            input_text=arguments.get("input_text"),
            input_json=arguments.get("input_json"),
            max_tokens=arguments.get("max_tokens", 9000),
            reasoning_effort=arguments.get("reasoning_effort"),
            temperature=arguments.get("temperature"),
            top_p=arguments.get("top_p"),
            instructions=arguments.get("instructions"),
            web_search_max_results=arguments.get("web_search_max_results"),
        )


# Compatibility wrappers
def search_llms(query: str = "", limit: int = 10, series: str = "") -> Dict[str, Any]:
    """Search OpenRouter models via HTTP ability service."""
    from aii_lib.abilities.ability_server import openrouter_search
    return openrouter_search(query=query, limit=limit, series=series)


def call_llm(model: str, input_text: str = None, **kwargs) -> Dict[str, Any]:
    """Call OpenRouter LLM via HTTP ability service."""
    from aii_lib.abilities.ability_server import openrouter_call
    return openrouter_call(model=model, input_text=input_text, **kwargs)


__all__ = ["OpenRouterSearchLLMs", "OpenRouterCallLLM", "search_llms", "call_llm"]
