"""
ToolUniverse MCP Server - HTTP MCP servers with colorful logging.

Provides two server configurations:
- aii_tooluniverse (port 8101): Filtered tools (aii_web_search_fast, aii_web_fetch_direct)
- full_tooluniverse (port 8102): All 775+ tools via execute_tool (compact mode)

Logging matches ability_service style:
    TIME | LEVEL | SOURCE (colored) | MESSAGE

Usage:
    from aii_lib.abilities.mcp_server import MCPServer

    server = MCPServer.from_config("aii_tooluniverse")
    server.run()
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .config import MCPServerConfig, load_mcp_config

# =============================================================================
# Loguru configuration with colors
# =============================================================================

# ANSI color codes
_ANSI = {
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def _configure_logging():
    """Configure loguru for MCP server logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<dim>{time:HH:mm:ss}</dim>|<level>{level: <5}</level>| {message}",
        level="INFO",
        colorize=True,
    )


# Configure on import
_configure_logging()


# =============================================================================
# Config loading helper
# =============================================================================

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent


def _load_server_config(server_name: str) -> MCPServerConfig:
    """Load server config by name."""
    configs = load_mcp_config()
    if server_name not in configs:
        raise ValueError(f"Unknown server: {server_name}. Available: {list(configs.keys())}")
    return configs[server_name]


# =============================================================================
# MCP Request Logging Middleware
# =============================================================================

# Colors for tool call logging
_TOOL_COLORS = {
    "call": "\033[38;5;33m",   # Blue
    "result": "\033[38;5;34m", # Green
    "error": "\033[38;5;196m", # Red
    "reset": "\033[0m",
    "dim": "\033[2m",
}


class MCPLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log MCP tool calls with colorful output."""

    def __init__(self, app, server_name: str = "mcp"):
        super().__init__(app)
        self.server_name = server_name
        self._call_count = 0

    async def dispatch(self, request: Request, call_next):
        # Only log POST requests to /mcp (tool calls)
        if request.method == "POST" and "/mcp" in request.url.path:
            try:
                # Read and parse the body
                body = await request.body()
                if body:
                    try:
                        data = json.loads(body)
                        method = data.get("method", "")
                        params = data.get("params", {})

                        # Log tool calls
                        if method == "tools/call":
                            tool_name = params.get("name", "unknown")
                            args = params.get("arguments", {})
                            self._call_count += 1

                            # Format args for logging (truncate if long)
                            args_str = json.dumps(args) if args else "{}"
                            if len(args_str) > 100:
                                args_str = args_str[:97] + "..."

                            c = _TOOL_COLORS
                            logger.info(
                                f"{c['call']}CALL{c['reset']} "
                                f"{c['dim']}#{self._call_count}{c['reset']} "
                                f"{tool_name} {c['dim']}{args_str}{c['reset']}"
                            )
                    except json.JSONDecodeError:
                        pass

                # Reconstruct request with body for downstream
                async def receive():
                    return {"type": "http.request", "body": body}
                request = Request(request.scope, receive)

            except Exception as e:
                logger.warning(f"Logging middleware error: {e}")

        response = await call_next(request)
        return response


# =============================================================================
# Memory utilities
# =============================================================================

def _get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # KB to MB on Linux
    except Exception:
        return -1.0


# =============================================================================
# MCP Server class
# =============================================================================

class MCPServer:
    """
    ToolUniverse MCP HTTP server with colorful logging.

    Features:
    - Colorful loguru-based logging (matches ability_service style)
    - Health check logging
    - Tool call tracking
    - Memory usage reporting

    Usage:
        server = MCPServer.from_config("aii_tooluniverse")
        server.run()
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.log = logger
        self._smcp = None
        self._start_time: datetime | None = None
        self._request_count = 0

    @classmethod
    def from_config(cls, server_name: str) -> "MCPServer":
        """Create server from config file."""
        config = _load_server_config(server_name)
        return cls(config)

    def _create_smcp(self):
        """Create and configure SMCP instance."""
        # Import aii_lib tools to register them
        import aii_lib.abilities.tools  # noqa: F401
        from tooluniverse.smcp import SMCP

        self.log.info(f"Creating SMCP instance...")
        self.log.info(f"  Port: {self.config.port}")
        self.log.info(f"  Host: {self.config.host}")
        self.log.info(f"  Max threads: {self.config.max_threads}")
        self.log.info(f"  Compact mode: {self.config.compact_mode}")
        self.log.info(f"  Search enabled: {self.config.search_enabled}")

        if self.config.include_tools:
            self.log.info(f"  Include tools: {self.config.include_tools}")
        else:
            self.log.info(f"  Include tools: ALL")

        self._smcp = SMCP(
            name=self.config.name,
            max_workers=self.config.max_threads,
            auto_expose_tools=True,
            search_enabled=self.config.search_enabled,
            compact_mode=self.config.compact_mode,
            include_tools=self.config.include_tools,
        )

        # Log loaded tools
        tool_count = len(self._smcp._exposed_tools)
        if self.config.compact_mode:
            self.log.info(f"Loaded {tool_count} tools (compact mode - via execute_tool)")
        else:
            self.log.info(f"Loaded {tool_count} tools (direct endpoints)")
            for tool_name in sorted(self._smcp._exposed_tools):
                self.log.debug(f"  - {tool_name}")

        return self._smcp

    def _print_banner(self):
        """Print startup banner."""
        mode = "Compact (execute_tool)" if self.config.compact_mode else "Direct endpoints"
        url = f"http://{self.config.host}:{self.config.port}/mcp"
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ToolUniverse MCP Server                                     ║
║  Server:  {self.config.name: <49}║
║  URL:     {url: <49}║
║  Mode:    {mode: <49}║
║  Threads: {self.config.max_threads: <49}║
╚══════════════════════════════════════════════════════════════╝
""")

    def _get_uvicorn_log_config(self) -> dict:
        """Get uvicorn log config: TIME|LEVEL|MESSAGE (compact)"""
        dim = _ANSI["dim"]
        reset = _ANSI["reset"]

        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": f"{dim}%(asctime)s{reset}|%(levelname)-5s| %(message)s",
                    "datefmt": "%H:%M:%S",
                },
                "access": {
                    "format": f"{dim}%(asctime)s{reset}|%(levelname)-5s| %(message)s",
                    "datefmt": "%H:%M:%S",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
                "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
                "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            },
        }

    def run(self):
        """Run the MCP server with colorful logging throughout."""
        import uvicorn

        self._start_time = datetime.now()
        self._print_banner()

        self.log.info(f"Starting server (pid={os.getpid()}, mem={_get_memory_mb():.1f}MB)")

        # Create SMCP
        smcp = self._create_smcp()

        self.log.info(f"Server ready on http://{self.config.host}:{self.config.port}/mcp")
        self.log.info(f"Health check: http://{self.config.host}:{self.config.port}/health")

        # Get ASGI app and run with uvicorn directly for custom logging
        try:
            app = smcp.http_app()

            # Add logging middleware to show tool calls
            app = MCPLoggingMiddleware(app, server_name=self.config.name)

            uvicorn.run(
                app,
                host=self.config.host,
                port=self.config.port,
                log_config=self._get_uvicorn_log_config(),
            )
        except KeyboardInterrupt:
            self.log.info("Shutting down (KeyboardInterrupt)...")
        except Exception as e:
            self.log.error(f"Server error: {e}")
            raise
        finally:
            uptime = datetime.now() - self._start_time if self._start_time else None
            self.log.info(f"Server stopped (uptime={uptime})")


# =============================================================================
# CLI entry points
# =============================================================================

def run_aii_server():
    """Run the aii_tooluniverse server."""
    server = MCPServer.from_config("aii_tooluniverse")
    server.run()


def run_full_server():
    """Run the full_tooluniverse server."""
    server = MCPServer.from_config("full_tooluniverse")
    server.run()


def run_server(server_name: str):
    """Run a server by name."""
    server = MCPServer.from_config(server_name)
    server.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ToolUniverse MCP Server")
    parser.add_argument(
        "server",
        choices=["aii_tooluniverse", "full_tooluniverse"],
        help="Server to run",
    )
    args = parser.parse_args()

    run_server(args.server)
