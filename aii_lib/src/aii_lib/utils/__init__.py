"""Utilities module - general utilities for aii_lib."""

# HTTP-based ability client
from aii_lib.abilities.ability_server.ability_client import (
    call_server,
    server_available,
    call_ability,
    ability_available,
)
from .cache_cleanup import cleanup_run_caches
from .start_servers import (
    ensure_server_running,
    stop_server,
    is_server_healthy,
)
from .model_utils import get_model_short
from aii_lib.prompts import LLMPromptModel
from .agent_to_llm import (
    ClaudeAgentToLLMStructOut,
    ClaudeAgentToLLMStructOutResult,
    get_tooluniverse_mcp_config,
)

__all__: list[str] = [
    # Ability client (HTTP-based)
    "call_server",
    "server_available",
    "call_ability",
    "ability_available",
    # Cache cleanup
    "cleanup_run_caches",
    # Server startup/management
    "ensure_server_running",
    "stop_server",
    "is_server_healthy",
    # Model utilities
    "get_model_short",
    # Prompt model
    "LLMPromptModel",
    # ClaudeAgentToLLMStructOut
    "ClaudeAgentToLLMStructOut",
    "ClaudeAgentToLLMStructOutResult",
    "get_tooluniverse_mcp_config",
]
