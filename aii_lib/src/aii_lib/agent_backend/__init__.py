"""
Agent Backend - Agent SDK wrappers.

Currently supports:
- claude/: Claude Agent SDK (sequential agent with streaming)

Architecture mirrors llm_backend/ for consistency.
"""

from typing import TYPE_CHECKING

# Re-export everything from claude/ (default implementation)
from .claude import (
    # Core types
    SessionType,
    SystemPromptPreset,
    AgentOptions,
    ExpectedFile,
    PromptResult,
    AgentResponse,
    TokenUsage,
    # Main agent class
    Agent,
    SequentialAgent,
    # Utilities
    aggregate_summaries,
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

# Agent utilities
from .utils import AgentInitializer, AgentFinalizer

# Type stubs for lazy imports
if TYPE_CHECKING:
    from .claude.utils.cli import run_agent, run_agent_sync


# Lazy imports to avoid runpy warning
def __getattr__(name: str):
    if name == "run_agent":
        from .claude.utils.cli import run_agent
        return run_agent
    elif name == "run_agent_sync":
        from .claude.utils.cli import run_agent_sync
        return run_agent_sync
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core types
    "SessionType",
    "SystemPromptPreset",
    "AgentOptions",
    "ExpectedFile",
    "PromptResult",
    "AgentResponse",
    "TokenUsage",
    # Main agent class
    "Agent",
    "SequentialAgent",
    # Run functions
    "run_agent",
    "run_agent_sync",
    # Utilities
    "aggregate_summaries",
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
    # Agent utilities
    "AgentInitializer",
    "AgentFinalizer",
]

__version__ = "0.1.0"
