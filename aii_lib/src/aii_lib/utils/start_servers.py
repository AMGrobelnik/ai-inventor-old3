"""Run servers in persistent sessions.

Uses zellij sessions when available, falls back to direct background processes (Docker/CI).

Servers:
- Ability Server (port 8100): REST endpoints for abilities
- AII ToolUniverse (port 8101): aii_web_search_fast, aii_web_fetch_direct, aii_web_fetch_grep
- Full ToolUniverse (port 8102): All tools via execute_tool
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

# Script locations
_UTILS_DIR = Path(__file__).parent
_MCP_SERVER_DIR = _UTILS_DIR.parent / "abilities" / "mcp_server"
_PROJECT_ROOT = _UTILS_DIR.parent.parent.parent.parent  # ai-inventor/
START_ABILITY_SCRIPT = _UTILS_DIR / "start_ability_server.sh"
START_AII_SCRIPT = _MCP_SERVER_DIR / "start_aii_tooluniverse.sh"
START_FULL_SCRIPT = _MCP_SERVER_DIR / "start_full_tooluniverse.sh"

# Default ports
ABILITY_SERVER_PORT = 8100
AII_TOOLUNIVERSE_PORT = 8101
FULL_TOOLUNIVERSE_PORT = 8102

_HAS_ZELLIJ = shutil.which("zellij") is not None


def is_server_healthy(
    port: int,
    host: str = "localhost",
    health_endpoint: str = "/health",
    timeout: float = 2.0,
) -> bool:
    """Check if a server is healthy via HTTP endpoint."""
    try:
        url = f"http://{host}:{port}{health_endpoint}"
        response = httpx.get(url, timeout=timeout)
        return 200 <= response.status_code < 300
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return False


def is_port_open(port: int, host: str = "localhost", timeout: float = 1.0) -> bool:
    """Check if a port is open (for servers without /health endpoint)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def _start_via_zellij(script_path: str | Path) -> subprocess.CompletedProcess:
    """Start server via zellij bash script."""
    return subprocess.run(
        [str(script_path)],
        capture_output=True,
        text=True,
    )


def _start_direct_background(
    module: str,
    port: int,
    args: list[str] | None = None,
    env_extra: dict | None = None,
) -> bool:
    """Start a Python server module directly as a background process.

    Used when zellij is not available (Docker, CI, etc.).
    """
    python = sys.executable
    env = {**os.environ, **(env_extra or {})}

    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{module.split('.')[-1]}.log"

    cmd = [python, "-m", module] + (args or [])

    with open(log_file, "a") as log_f:
        subprocess.Popen(
            cmd,
            cwd=str(_PROJECT_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Wait for server to be ready
    for _ in range(30):
        if is_server_healthy(port) or is_port_open(port):
            return True
        time.sleep(1)

    return False


def _wait_for_port(port: int, timeout: int = 30) -> bool:
    """Wait for a port to become available."""
    for _ in range(timeout):
        if is_server_healthy(port) or is_port_open(port):
            return True
        time.sleep(1)
    return False


def ensure_server_running(
    port: int = ABILITY_SERVER_PORT,
    log_func=None,
) -> bool:
    """Ensure all servers are running.

    Starts:
    - Ability Server (port 8100)
    - AII ToolUniverse (port 8101) - aii_web_search_fast, aii_web_fetch_direct, aii_web_fetch_grep
    - Full ToolUniverse (port 8102) - all tools via execute_tool

    Uses zellij sessions when available, falls back to direct background processes.
    """
    def log(msg: str) -> None:
        if log_func:
            log_func(msg)

    all_ok = True
    use_zellij = _HAS_ZELLIJ

    if not use_zellij:
        log("zellij not available — using direct background processes")

    def _start_server(
        name: str,
        port: int,
        zellij_script: Path,
        module: str,
        module_args: list[str] | None = None,
        env_extra: dict | None = None,
        check_healthy: bool = False,
    ) -> bool:
        """Start a server, trying zellij first then falling back to direct background."""
        check = is_server_healthy if check_healthy else is_port_open
        if check(port):
            log(f"{name} already running")
            return True

        log(f"Starting {name}...")

        # Try zellij first
        if use_zellij:
            # Clean stale sessions before attempting
            session = zellij_script.stem.replace("start_", "")
            subprocess.run(["zellij", "delete-session", session], capture_output=True)
            result = _start_via_zellij(zellij_script)
            if result.returncode == 0 and _wait_for_port(port, timeout=10):
                log(f"{name} started (zellij)")
                return True
            log(f"{name} zellij failed, falling back to direct...")

        # Direct background fallback
        ok = _start_direct_background(
            module=module,
            port=port,
            args=module_args,
            env_extra=env_extra,
        )
        if ok:
            log(f"{name} started (direct)")
            return True

        log(f"Failed to start {name}")
        return False

    # === Ability Server ===
    log(f"Checking ability server on port {port}...")
    if not _start_server(
        name="Ability server",
        port=port,
        zellij_script=START_ABILITY_SCRIPT,
        module="aii_lib.abilities.ability_server.endpoints",
        env_extra={"AII_SKIP_MCP_SUBPROCESS": "1"},
        check_healthy=True,
    ):
        all_ok = False

    # === AII ToolUniverse (port 8101) ===
    log(f"Checking AII ToolUniverse on port {AII_TOOLUNIVERSE_PORT}...")
    if not _start_server(
        name="AII ToolUniverse",
        port=AII_TOOLUNIVERSE_PORT,
        zellij_script=START_AII_SCRIPT,
        module="aii_lib.abilities.mcp_server.server",
        module_args=["aii_tooluniverse"],
    ):
        log("WARN: pipeline steps using MCP tools will use built-in alternatives")

    # === Full ToolUniverse (port 8102) ===
    log(f"Checking Full ToolUniverse on port {FULL_TOOLUNIVERSE_PORT}...")
    if not _start_server(
        name="Full ToolUniverse",
        port=FULL_TOOLUNIVERSE_PORT,
        zellij_script=START_FULL_SCRIPT,
        module="aii_lib.abilities.mcp_server.server",
        module_args=["full_tooluniverse"],
    ):
        log("WARN: invention loop steps may need Full ToolUniverse")

    return all_ok


def stop_server() -> bool:
    """Stop all servers."""
    # Stop ability server
    subprocess.run(["pkill", "-9", "-f", "ability_server.endpoints"], capture_output=True)
    # Stop AII ToolUniverse
    subprocess.run(["pkill", "-9", "-f", "mcp_server.server"], capture_output=True)
    # Stop Full ToolUniverse
    subprocess.run(["pkill", "-9", "-f", "full_tooluniverse"], capture_output=True)

    # Clean up zellij sessions if available
    if _HAS_ZELLIJ:
        for session in ["aii_ability_server", "aii_tooluniverse", "full_tooluniverse"]:
            subprocess.run(["zellij", "kill-session", session], capture_output=True)
            subprocess.run(["zellij", "delete-session", session], capture_output=True)

    return True
