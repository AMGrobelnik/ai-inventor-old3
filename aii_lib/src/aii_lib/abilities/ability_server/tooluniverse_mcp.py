"""
ToolUniverse MCP Server - Standalone entry point.

Runs ToolUniverse as an MCP server with crash isolation from the main ability server.
Runs in its own zellij session for independent logging and crash isolation.

Usage:
    # Standalone (port 8101)
    python -m aii_lib.abilities.ability_server.tooluniverse_mcp

    # Custom port
    python -m aii_lib.abilities.ability_server.tooluniverse_mcp --port 8102

Endpoints:
    - /mcp          - MCP protocol (JSON-RPC over SSE)
    - /health       - Health check (JSON status)
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger

# =============================================================================
# Loguru configuration (same style as ability_service)
# =============================================================================

_ANSI_COLORS = {
    "magenta": "\033[35m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def _format_mcp_log(record):
    """Custom formatter for MCP server logs (matches ability_service style)."""
    color = _ANSI_COLORS["magenta"]
    reset = _ANSI_COLORS["reset"]
    source = record["extra"].get("source", "tooluniverse_mcp")
    source_padded = f"{source: <22}"
    record["extra"]["colored_source"] = f"{color}{source_padded}{reset}"
    record["extra"]["color"] = color
    record["extra"]["reset"] = reset
    return "<dim>{time:HH:mm:ss}</dim> | <level>{level: <7}</level> | {extra[colored_source]} | {extra[color]}{message}{extra[reset]}\n"


logger.remove()
logger.add(
    sys.stderr,
    format=_format_mcp_log,
    level="INFO",
    colorize=True,
)

# =============================================================================
# Config
# =============================================================================

CONFIG_FILE = Path(__file__).parent / "server_config.yaml"
_server_config: dict = {}
if CONFIG_FILE.exists():
    with open(CONFIG_FILE) as f:
        _server_config = yaml.safe_load(f) or {}

DEFAULT_PORT = 8101
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MAX_WORKERS = _server_config.get("endpoints", {}).get("tooluniverse_mcp", {}).get("max_threads", 30)


# =============================================================================
# Server state (for health endpoint)
# =============================================================================

_server_state = {
    "started_at": None,
    "host": None,
    "port": None,
    "max_workers": None,
    "tool_count": 0,
    "pid": None,
}


# =============================================================================
# Server
# =============================================================================

def run_server(port: int = None, host: str = None, max_workers: int = None) -> None:
    """Run the ToolUniverse MCP server.

    Args:
        port: Port to run on (default: 8101)
        host: Host to bind to (default: 127.0.0.1)
        max_workers: Max concurrent tool executions (default: 30)
    """
    port = port or DEFAULT_PORT
    host = host or DEFAULT_HOST
    max_workers = max_workers or DEFAULT_MAX_WORKERS

    log = logger.bind(source="tooluniverse_mcp")

    log.info("Starting ToolUniverse MCP Server")
    log.info(f"PID: {os.getpid()}")

    # Import and register custom tools
    log.info("Loading ToolUniverse tools...")
    import aii_lib.abilities.tools  # noqa: F401

    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from tooluniverse.smcp import SMCP

    # Create SMCP server
    smcp = SMCP(
        name="ToolUniverse MCP",
        max_workers=max_workers,
        auto_expose_tools=True,
        search_enabled=True,
    )

    tool_count = len(smcp._exposed_tools) if hasattr(smcp, '_exposed_tools') else 0

    # Update server state
    _server_state.update({
        "started_at": datetime.now(),
        "host": host,
        "port": port,
        "max_workers": max_workers,
        "tool_count": tool_count,
        "pid": os.getpid(),
    })

    # Add health endpoint (matches ability server style)
    @smcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        """Health check endpoint."""
        uptime = (datetime.now() - _server_state["started_at"]).total_seconds() if _server_state["started_at"] else 0
        return JSONResponse({
            "status": "ok",
            "server": "tooluniverse_mcp",
            "host": _server_state["host"],
            "port": _server_state["port"],
            "pid": _server_state["pid"],
            "max_workers": _server_state["max_workers"],
            "tool_count": _server_state["tool_count"],
            "uptime_seconds": round(uptime, 1),
            "started_at": _server_state["started_at"].isoformat() if _server_state["started_at"] else None,
        })

    # Log startup info (similar to ability server)
    log.info(f"Loaded {tool_count} tools")
    log.info(f"Max workers: {max_workers}")
    log.info(f"Endpoints: /mcp (MCP protocol), /health (status)")
    log.info(f"Binding to {host}:{port}...")

    # Run server
    smcp.run_simple(transport="http", host=host, port=port)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ToolUniverse MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind to (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind to (default: {DEFAULT_HOST})")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help=f"Max concurrent executions (default: {DEFAULT_MAX_WORKERS})")

    args = parser.parse_args()

    try:
        run_server(port=args.port, host=args.host, max_workers=args.max_workers)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
