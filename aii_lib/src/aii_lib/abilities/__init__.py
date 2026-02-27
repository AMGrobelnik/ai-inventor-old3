"""
aii_lib abilities - Skills and Tools for AI agents.

This package contains:
- skills/: Rich CLI-based workflows with SKILL.md documentation
- tools/: ToolUniverse-registered tools (importable for MCP servers)
- ability_server/: FastAPI server with client, service, and endpoints
"""

from . import ability_server


def __getattr__(name: str):
    """Lazy import for app/run_server to avoid triggering endpoint registration."""
    if name in ("app", "run_server"):
        from .ability_server.endpoints import app, run_server
        return app if name == "app" else run_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ability_server", "app", "run_server"]
