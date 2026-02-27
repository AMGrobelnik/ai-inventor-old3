"""
Claude Agent SDK implementation.

Sequential agent wrapper for Claude Agent SDK with streaming mode support.
"""

from .models import (
    SessionType,
    SystemPromptPreset,
    AgentOptions,
    ExpectedFile,
    PromptResult,
    AgentResponse,
    TokenUsage,
)
from .agent import Agent
from .core.results import aggregate_summaries
from .utils.execution.sdk_client import AgentProcessError, SubscriptionAccessError

# Backward compatibility alias
SequentialAgent = Agent

# Re-export from utils.init_helpers for convenience
from .utils.init_helpers import (
    # Agent management
    prepare_agents,
    cleanup_agents,
    get_agent,
    list_agents,
    ALL_AGENTS,
    # MCP tool utilities
    load_tools_from_file,
    load_tools_from_files,
    create_custom_tools_server,
    setup_custom_tools,
)

__all__ = [
    # Core types
    "SessionType",
    "SystemPromptPreset",
    "AgentOptions",
    "ExpectedFile",
    "PromptResult",
    "AgentResponse",
    "TokenUsage",
    # Utilities
    "aggregate_summaries",
    # Exceptions
    "AgentProcessError",
    "SubscriptionAccessError",
    # Main agent class
    "Agent",
    "SequentialAgent",
    # Agent management
    "prepare_agents",
    "cleanup_agents",
    "get_agent",
    "list_agents",
    "ALL_AGENTS",
    # MCP tool utilities
    "load_tools_from_file",
    "load_tools_from_files",
    "create_custom_tools_server",
    "setup_custom_tools",
]
