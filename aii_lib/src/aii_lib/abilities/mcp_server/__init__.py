"""ToolUniverse MCP Server configuration and utilities.

Two HTTP servers available:
- aii_tooluniverse (port 8101): Only aii_web_search_fast, aii_web_fetch_direct
- full_tooluniverse (port 8102): All 775+ tools via execute_tool (compact mode)

Usage:
    # Get MCP config for agent
    from aii_lib.abilities.mcp_server import get_tooluniverse_mcp_config
    mcp_servers = get_tooluniverse_mcp_config(use_aii_server=True)

    # Run server directly
    from aii_lib.abilities.mcp_server import MCPServer
    server = MCPServer.from_config("aii_tooluniverse")
    server.run()
"""

from .config import (
    MCPServerConfig,
    load_mcp_config,
    get_tooluniverse_mcp_config,
    AII_SERVER_NAME,
    FULL_SERVER_NAME,
)
from .server import (
    MCPServer,
    run_aii_server,
    run_full_server,
    run_server,
)

__all__ = [
    # Config
    "load_mcp_config",
    "get_tooluniverse_mcp_config",
    "AII_SERVER_NAME",
    "FULL_SERVER_NAME",
    # Server
    "MCPServer",
    "MCPServerConfig",
    "run_aii_server",
    "run_full_server",
    "run_server",
]
