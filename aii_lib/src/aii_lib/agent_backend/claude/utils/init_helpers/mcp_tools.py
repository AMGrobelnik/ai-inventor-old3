"""
MCP Tool Utilities - consolidated from mcp_servers/ module

Provides utility functions for loading tools from Python files and creating
custom MCP servers for use with Claude Agent SDK.
"""

import importlib.util
import inspect
from pathlib import Path
from aii_lib.telemetry import MessageType
from typing import Any


def load_tools_from_file(
    file_path: str | Path,
    telemetry: Any = None,
    run_id: str | None = None,
) -> list[Any]:
    """
    Load @tool decorated functions from a Python file.

    Args:
        file_path: Path to Python file containing @tool decorated functions
        telemetry: AIITelemetry instance for logging
        run_id: Run ID for sequenced logging

    Returns:
        List of tool functions (SdkMcpTool decorated functions)

    Example:
        >>> tools = load_tools_from_file("my_tools.py")
        >>> len(tools)
        2
    """
    t = telemetry  # Shorthand
    file_path = Path(file_path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Custom tool file not found: {file_path}")

    if not file_path.suffix == ".py":
        raise ValueError(f"Custom tool file must be .py file: {file_path}")

    # Load module from file
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find all SdkMcpTool instances (created by @tool decorator)
    from claude_agent_sdk import SdkMcpTool

    tools = []
    for name, obj in inspect.getmembers(module):
        if isinstance(obj, SdkMcpTool):
            tools.append(obj)

    if not tools:
        if t: t.emit(MessageType.WARNING, f"No @tool decorated functions found in {file_path}", run_id=run_id)

    return tools


def load_tools_from_files(
    file_paths: list[str | Path],
    telemetry: Any = None,
    run_id: str | None = None,
) -> list[Any]:
    """
    Load @tool decorated functions from multiple Python files.

    Args:
        file_paths: List of paths to Python files
        telemetry: AIITelemetry instance for logging
        run_id: Run ID for sequenced logging

    Returns:
        List of all tool functions from all files

    Example:
        >>> tools = load_tools_from_files(["tools1.py", "tools2.py"])
        >>> len(tools)
        5
    """
    t = telemetry  # Shorthand
    all_tools = []

    for file_path in file_paths:
        try:
            tools = load_tools_from_file(file_path, telemetry=t, run_id=run_id)
            all_tools.extend(tools)
        except Exception as e:
            if t: t.emit(MessageType.ERROR, f"Failed to load tools from {file_path}: {e}", run_id=run_id)
            raise

    if all_tools:
        if t: t.emit(MessageType.SUCCESS, f"Loaded {len(all_tools)} custom tool(s)", run_id=run_id)
    return all_tools


def create_custom_tools_server(
    tools: list[Any],
    server_name: str = "custom-tools",
    server_version: str = "1.0.0"
) -> Any:
    """
    Create an SDK MCP server with custom tools.

    Args:
        tools: List of @tool decorated functions
        server_name: Name for the MCP server
        server_version: Version string

    Returns:
        SDK MCP server instance

    Example:
        >>> from claude_agent_sdk import tool
        >>> @tool("my_tool", "Description", {})
        ... async def my_tool(args): return {"content": [{"type": "text", "text": "Result"}]}
        >>> server = create_custom_tools_server([my_tool])
    """
    from claude_agent_sdk import create_sdk_mcp_server

    if not tools:
        raise ValueError("No tools provided to create server")

    server = create_sdk_mcp_server(
        name=server_name,
        version=server_version,
        tools=tools
    )

    return server


def setup_custom_tools(
    file_paths: list[str | Path],
    server_name: str = "custom-tools",
    server_version: str = "1.0.0",
    telemetry: Any = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Load tools from files and create MCP server config.

    This is the main function called by AgentOptions to set up custom tools.

    Args:
        file_paths: List of Python file paths with @tool functions
        server_name: Name for the MCP server
        server_version: Version string
        telemetry: AIITelemetry instance for logging
        run_id: Run ID for sequenced logging

    Returns:
        Dict with MCP server config ready for mcp_servers parameter

    Example:
        >>> config = setup_custom_tools(["my_tools.py"])
        >>> # config = {"custom-tools": <server instance>}
    """
    t = telemetry  # Shorthand
    if not file_paths:
        return {}

    # Load all tools from files
    tools = load_tools_from_files(file_paths, telemetry=t, run_id=run_id)

    if not tools:
        if t: t.emit(MessageType.WARNING, "No tools loaded, returning empty config", run_id=run_id)
        return {}

    # Create SDK MCP server
    server = create_custom_tools_server(tools, server_name, server_version)

    # Return MCP server config (direct mapping to server instance per SDK docs)
    config = {
        server_name: server
    }

    return config


__all__ = [
    'load_tools_from_file',
    'load_tools_from_files',
    'create_custom_tools_server',
    'setup_custom_tools',
]
