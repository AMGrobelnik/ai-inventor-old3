#!/usr/bin/env python3
"""
Step 10: Visualize Graphs

Launches a web server to visualize the knowledge graphs.

Usage:
    python _10_viz_graphs.py [run_id]
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional
from aii_lib import AIITelemetry, MessageType
import yaml


# Module-level telemetry (set by main)
_telemetry: Optional[AIITelemetry] = None


def _emit(msg_type: MessageType, msg: str):
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")

__all__ = ["main"]

# Base directory (invention_kg root)
BASE_DIR = Path(__file__).parent.parent


def main(run_id: str = None, config: dict = None, telemetry: Optional[AIITelemetry] = None):
    """
    Launch the visualization server.

    Args:
        run_id: Run ID (used to symlink data directory).
        config: Optional config dict with viz_graph settings.
        telemetry: Optional AIITelemetry instance.
    """
    global _telemetry
    _telemetry = telemetry

    # Use config if provided, otherwise use defaults
    viz_config = (config or {}).get("viz_graph", {})
    port = viz_config.get("port", 9020)
    viz_dir = viz_config.get("viz_dir", "steps/_10_viz_graphs")

    _emit(MessageType.INFO, f"Port: {port}")

    # Get viz directory path
    viz_path = BASE_DIR / viz_dir
    if not viz_path.exists():
        _emit(MessageType.ERROR, f"Viz directory not found: {viz_path}")
        return 1

    # Update data symlink if run_id provided
    if run_id:
        graphs_dir = BASE_DIR / "data" / "_9_graphs" / run_id
        data_link = viz_path / "data"

        if graphs_dir.exists():
            # Remove existing symlink or directory
            if data_link.is_symlink():
                data_link.unlink()
            elif data_link.is_dir():
                import shutil
                shutil.rmtree(data_link)
                _emit(MessageType.INFO, "Removed existing data directory")
            # Create symlink
            if not data_link.exists():
                data_link.symlink_to(graphs_dir)
                _emit(MessageType.INFO, f"Linked data -> {graphs_dir.relative_to(BASE_DIR)}")
        else:
            _emit(MessageType.WARNING, f"Graphs dir not found: {graphs_dir}")

    # Launch server
    _emit(MessageType.SUCCESS, f"Starting server at http://localhost:{port}")
    _emit(MessageType.INFO, "Press Ctrl+C to stop")

    try:
        subprocess.run(
            ["python3", "-m", "http.server", str(port)],
            cwd=viz_path,
            check=True
        )
    except KeyboardInterrupt:
        _emit(MessageType.INFO, "Server stopped")
    except Exception as e:
        _emit(MessageType.ERROR, f"Server error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(run_id))
