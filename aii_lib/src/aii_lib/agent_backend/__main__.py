"""
CLI entry point for aii_lib agent backend.

Usage:
    python -m aii_lib.agent_backend --help
    python -m aii_lib.agent_backend run "Your prompt" --config config.yaml
    python -m aii_lib.agent_backend validate config.yaml
"""

import sys
import asyncio
from pathlib import Path
import argparse

from .agent import Agent
from .utils.types import AgentOptions
from aii_lib.telemetry.console import CYAN, GREEN, YELLOW, RESET, colorize


async def run_command(args):
    """Run a prompt with the agent"""
    from aii_lib.telemetry import logger
    from .config import load_config_from_yaml, apply_cli_overrides

    # Debug: Show what args we received
    logger.debug(f"Received args: config={repr(args.config)}, model={repr(args.model)}, max_turns={repr(args.max_turns)}")

    # Load config if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        logger.info(f"Loading config from: {config_path.resolve()}")
        options = load_config_from_yaml(config_path, _skip_post_init_log=True)

        # Apply all CLI args as overrides (automatically handles all fields)
        apply_cli_overrides(options, vars(args))
    else:
        logger.debug("No config file specified, using default AgentOptions")
        # No config file - create default options (skip post-init log since apply_cli_overrides will log)
        options = AgentOptions(permission_mode="bypassPermissions", _skip_post_init_log=True)

        # Apply all CLI args as overrides
        apply_cli_overrides(options, vars(args))

    # Create agent
    agent = Agent(options)

    # Run prompts (single or multiple)
    result = await agent.run(args.prompts)

    # Verbose mode: show detailed token usage
    if args.verbose:
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{CYAN}DETAILED TOKEN USAGE{RESET}")
        print(f"{CYAN}{'='*60}{RESET}")
        for prompt_result in result.prompt_results:
            print(f"\n{CYAN}Prompt {prompt_result.prompt_index + 1}:{RESET}")
            print(f"  {CYAN}Input tokens:{RESET} {GREEN}{prompt_result.usage.input_tokens}{RESET}")
            print(f"  {CYAN}Output tokens:{RESET} {GREEN}{prompt_result.usage.output_tokens}{RESET}")
            print(f"  {CYAN}Cache read:{RESET} {GREEN}{prompt_result.usage.cache_read_input_tokens}{RESET}")
        print(f"{CYAN}{'='*60}{RESET}")


def validate_command(args):
    """Validate a config file"""
    from .config import load_config_from_yaml

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"{YELLOW}Error: Config file not found: {args.config}{RESET}", file=sys.stderr)
        sys.exit(1)

    try:
        options = load_config_from_yaml(config_path)
        print(f"{GREEN}✓ Config is valid: {args.config}{RESET}")

        # Show summary
        print(f"\n{CYAN}Configuration Summary:{RESET}")
        print(f"  {CYAN}Working directory:{RESET} {GREEN}{options.cwd or 'current'}{RESET}")
        print(f"  {CYAN}Max turns:{RESET} {GREEN}{options.max_turns}{RESET}")
        print(f"  {CYAN}Permission mode:{RESET} {GREEN}{options.permission_mode}{RESET}")
        print(f"  {CYAN}Session type:{RESET} {GREEN}{options.session_type.value}{RESET}")
        print(f"  {CYAN}Model:{RESET} {GREEN}{options.model}{RESET}")

        if options.custom_tool_files:
            print(f"  {CYAN}Custom tools:{RESET} {GREEN}{len(options.custom_tool_files)} files{RESET}")
        if options.custom_agent_files:
            print(f"  {CYAN}Custom agents:{RESET} {GREEN}{len(options.custom_agent_files)} files{RESET}")

        sys.exit(0)
    except Exception as e:
        print(f"{YELLOW}✗ Config validation failed: {e}{RESET}", file=sys.stderr)
        sys.exit(1)


def info_command(args):
    """Show package information"""
    from . import __version__

    print(f"""
{CYAN}aii_lib.agent_backend{RESET} {GREEN}v{__version__}{RESET}

A Python wrapper around Claude Agent SDK with streaming mode support.

{CYAN}Features:{RESET}
  {GREEN}•{RESET} YAML configuration
  {GREEN}•{RESET} Custom tools from Python files
  {GREEN}•{RESET} Custom agents from YAML files
  {GREEN}•{RESET} Session management (NEW/RESUME/FORK)
  {GREEN}•{RESET} Usage tracking (tokens & costs)
  {GREEN}•{RESET} Sequential execution

{CYAN}Documentation:{RESET} README.md
{CYAN}Examples:{RESET} examples/
""")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="aii_agent",
        description="AII Agent - Run Claude Agent SDK with streaming support",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run one or more prompts")
    run_parser.add_argument("prompts", nargs='+', help="One or more prompts to run sequentially")
    run_parser.add_argument(
        "-c", "--config",
        help="Path to YAML config file"
    )
    run_parser.add_argument(
        "-t", "--max-turns",
        type=int,
        default=None,
        help="Maximum conversation turns (default: 1000)"
    )
    run_parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Claude model to use (e.g., haiku, sonnet, opus)"
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output with detailed token usage"
    )
    run_parser.add_argument(
        "--session-type",
        choices=["new", "resume", "fork"],
        help="Session type: new (default), resume, or fork"
    )
    run_parser.add_argument(
        "--session-id",
        help="Session ID to resume or fork from (required for resume/fork)"
    )
    run_parser.add_argument(
        "--continue-seq-item",
        type=lambda x: x.lower() == 'true',
        default=None,
        help="Continue conversation for 2nd+ prompts in sequence (true/false)"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a config file")
    validate_parser.add_argument("config", help="Path to YAML config file")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show package information")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Execute command
    if args.command == "run":
        asyncio.run(run_command(args))
    elif args.command == "validate":
        validate_command(args)
    elif args.command == "info":
        info_command(args)


if __name__ == "__main__":
    main()
