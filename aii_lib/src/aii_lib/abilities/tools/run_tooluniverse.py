#!/usr/bin/env python3
"""
Wrapper script to run ToolUniverse SMCP server with aii_lib tools registered.

This script:
1. Imports aii_lib.abilities.tools to register our custom tool classes with ToolUniverse's registry
2. Delegates to the standard tooluniverse server (stdio or http mode)

The @register_tool decorator in each tool module registers both the class AND the config
(including parameter schemas) with ToolUniverse's registries. No --tool-config-files needed.

Usage:
    uv run python -m aii_lib.abilities.tools.run_tooluniverse [--http] [tooluniverse-smcp args...]

Examples:
    # Start stdio server (for single agent)
    uv run python -m aii_lib.abilities.tools.run_tooluniverse --compact-mode

    # Start HTTP server on localhost:8000 (for multiple concurrent agents)
    uv run python -m aii_lib.abilities.tools.run_tooluniverse --http

    # Start with only specific tools
    uv run python -m aii_lib.abilities.tools.run_tooluniverse --include-tools hf_dataset_search owid_search_datasets
"""

import sys


def main():
    # Import aii_lib.abilities.tools to register our tool classes with ToolUniverse
    # The @register_tool decorator handles both class AND config registration
    import aii_lib.abilities.tools  # noqa: F401

    # Check for --http flag
    if "--http" in sys.argv:
        sys.argv.remove("--http")
        from tooluniverse.smcp_server import run_http_server
        print("Starting ToolUniverse HTTP server on localhost:8000...")
        sys.exit(run_http_server())
    else:
        # Delegate to ToolUniverse's standard stdio server
        from tooluniverse.smcp_server import run_stdio_server
        sys.exit(run_stdio_server())


if __name__ == "__main__":
    main()
