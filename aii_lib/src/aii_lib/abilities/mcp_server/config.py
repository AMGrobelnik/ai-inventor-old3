"""MCP Server configuration loader.

Loads server config from server_config.yaml and provides helper functions
to get MCP server configs for Claude agents.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

# Server names
AII_SERVER_NAME = "aii_tooluniverse"
FULL_SERVER_NAME = "full_tooluniverse"

# Config file path
CONFIG_FILE = Path(__file__).parent / "server_config.yaml"


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    port: int
    host: str
    max_threads: int
    compact_mode: bool
    search_enabled: bool
    include_tools: list[str] | None


def load_mcp_config() -> dict[str, MCPServerConfig]:
    """Load MCP server configuration from YAML file.

    Returns:
        dict mapping server name to MCPServerConfig
    """
    with open(CONFIG_FILE) as f:
        raw_config = yaml.safe_load(f)

    configs = {}
    for name, cfg in raw_config.items():
        configs[name] = MCPServerConfig(
            name=name,
            port=cfg["port"],
            host=cfg["host"],
            max_threads=cfg["max_threads"],
            compact_mode=cfg["compact_mode"],
            search_enabled=cfg["search_enabled"],
            include_tools=cfg.get("include_tools"),
        )

    return configs


def get_tooluniverse_mcp_config(
    use_aii_server: bool = True,
    use_full_server: bool = False,
) -> dict:
    """Get MCP server config for ToolUniverse.

    Args:
        use_aii_server: If True (default), include AII server (port 8101) with only
                        aii_web_search_fast, aii_web_fetch_direct, and aii_web_fetch_grep tools.
        use_full_server: If True, include full server (port 8102) with all 775+ tools.

    Returns:
        dict: MCP servers config dict to pass to AgentOptions.

    Note:
        Start the servers first:
            ./aii_lib/src/aii_lib/abilities/mcp_server/start_aii_tooluniverse.sh
            ./aii_lib/src/aii_lib/abilities/mcp_server/start_full_tooluniverse.sh

    Example:
        >>> # AII server only - aii_web_search_fast, aii_web_fetch_direct, aii_web_fetch_grep (default)
        >>> mcp_servers = get_tooluniverse_mcp_config()
        >>>
        >>> # Full server only - all 775+ tools via execute_tool
        >>> mcp_servers = get_tooluniverse_mcp_config(use_aii_server=False, use_full_server=True)
        >>>
        >>> # Both servers - AII + full (for artifact executors needing all tools)
        >>> mcp_servers = get_tooluniverse_mcp_config(use_aii_server=True, use_full_server=True)
    """
    configs = load_mcp_config()
    result = {}

    if use_aii_server:
        cfg = configs[AII_SERVER_NAME]
        result[AII_SERVER_NAME] = {
            "type": "http",
            "url": f"http://{cfg.host}:{cfg.port}/mcp",
        }

    if use_full_server:
        cfg = configs[FULL_SERVER_NAME]
        result[FULL_SERVER_NAME] = {
            "type": "http",
            "url": f"http://{cfg.host}:{cfg.port}/mcp",
        }

    return result
