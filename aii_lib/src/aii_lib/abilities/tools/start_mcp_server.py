#!/usr/bin/env python3
"""
Start ToolUniverse MCP server with aii_lib abilities tools.

This script imports the tools module first to ensure @register_tool
decorators execute and register tools with ToolUniverse's registry,
then runs tooluniverse-smcp-stdio.

Usage:
    python start_mcp_server.py [--name NAME] [--include-tools TOOL1 TOOL2 ...]

Example:
    python start_mcp_server.py --name "AII Tools"
    python start_mcp_server.py --include-tools openrouter_search_llms openrouter_call_llm
"""

import subprocess
import sys
from pathlib import Path

# Import tools module to register all tools via @register_tool decorators
import aii_lib.abilities.tools  # noqa: F401

TOOLS_DIR = Path(__file__).parent


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Start aii_lib ToolUniverse MCP server (stdio)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="aii-tools",
        help="Server display name (default: aii-tools)"
    )
    parser.add_argument(
        "--include-tools",
        nargs="*",
        help="Include only these tool names"
    )
    parser.add_argument(
        "--tools-file",
        type=str,
        help="File with one tool name per line"
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Include only these categories"
    )

    args = parser.parse_args()

    # Build command
    cmd = ["tooluniverse-smcp-stdio", "--name", args.name]

    if args.include_tools:
        cmd.extend(["--include-tools"] + args.include_tools)
    elif args.tools_file:
        cmd.extend(["--tools-file", args.tools_file])
    else:
        # Default: use tools_to_load.txt if it exists
        default_tools_file = TOOLS_DIR / "tools_to_load.txt"
        if default_tools_file.exists():
            cmd.extend(["--tools-file", str(default_tools_file)])

    if args.categories:
        cmd.extend(["--categories"] + args.categories)

    # Run the MCP server
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
