"""
LLM Clients - Centralized LLM client implementations.

Provides unified interfaces for different LLM providers:
- OpenAI (GPT-5, o-series reasoning) - async
- Anthropic (Claude 4.5) - async
- Gemini (2.5/3.0) - async
- OpenRouter (300+ models) - async

For agent-based structured output, use AgentToLLM from aii_lib.utils.

For available models, see:
- Each client's docstring (e.g., help(OpenAIClient))
- supported_models.py for complete reference
- get_models("openai") / get_recommended("anthropic")

Last updated: January 2026
"""

from .openai import OpenAIClient
from .anthropic import AnthropicClient
from .gemini import GeminiClient
from .openrouter import OpenRouterClient, ConversationStats
from .tool_loop import chat, ToolLoopResult
from .config import (
    load_config,
    get_config,
    get_openai_config,
    get_openrouter_config,
)
from .supported_models import (
    OPENAI_MODELS,
    ANTHROPIC_MODELS,
    GEMINI_MODELS,
    OPENROUTER_POPULAR,
    get_models,
    get_recommended,
    print_models,
)

__all__ = [
    # Clients
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "OpenRouterClient",
    "ConversationStats",
    "chat",
    "ToolLoopResult",
    # Config
    "load_config",
    "get_config",
    "get_openai_config",
    "get_openrouter_config",
    # Model references
    "OPENAI_MODELS",
    "ANTHROPIC_MODELS",
    "GEMINI_MODELS",
    "OPENROUTER_POPULAR",
    "get_models",
    "get_recommended",
    "print_models",
]
