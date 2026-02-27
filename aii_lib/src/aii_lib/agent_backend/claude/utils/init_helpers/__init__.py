"""
Initialization module for aii_lib agent backend.

Provides registry and loader functionality for agents and MCPs.
All configuration is loaded from .claude/ directory and .mcp.json.
"""

from .agents import (
    AgentDefinition,
    prepare_agents,
    cleanup_agents,
    get_agent,
    list_agents,
    ALL_AGENTS,
    # Individual agent exports
    math_solver,
    quick_calc,
    math_tutor,
    text_analyzer,
    text_transformer,
    palindrome_checker,
    text_master,
)

from .mcp_registry import (
    McpDefinition,
    list_mcps,
    get_mcp,
    ALL_MCPS,
    # Individual MCP exports (if available)
)

# Import individual MCPs if they exist
try:
    from .mcp_registry import context7
except ImportError:
    context7 = None

try:
    from .mcp_registry import hf_mcp_server
except ImportError:
    hf_mcp_server = None

try:
    from .mcp_registry import chrome_devtools
except ImportError:
    chrome_devtools = None

try:
    from .mcp_registry import shadcn
except ImportError:
    shadcn = None

from .mcp_loader import (
    prepare_mcps,
    cleanup_mcps,
)

from .mcp_tools import (
    load_tools_from_file,
    load_tools_from_files,
    create_custom_tools_server,
    setup_custom_tools,
)

__all__ = [
    # Agent management
    "AgentDefinition",
    "prepare_agents",
    "cleanup_agents",
    "get_agent",
    "list_agents",
    "ALL_AGENTS",
    "math_solver",
    "quick_calc",
    "math_tutor",
    "text_analyzer",
    "text_transformer",
    "palindrome_checker",
    "text_master",

    # MCP management
    "McpDefinition",
    "prepare_mcps",
    "cleanup_mcps",
    "list_mcps",
    "get_mcp",
    "ALL_MCPS",
    "context7",
    "hf_mcp_server",
    "chrome_devtools",
    "shadcn",

    # MCP tool utilities
    "load_tools_from_file",
    "load_tools_from_files",
    "create_custom_tools_server",
    "setup_custom_tools",
]
