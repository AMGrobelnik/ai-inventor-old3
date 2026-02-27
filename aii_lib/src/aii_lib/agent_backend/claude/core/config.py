"""
Configuration building and initialization.
"""

import sys
from pathlib import Path
from aii_lib.telemetry import MessageType
from datetime import datetime
from claude_agent_sdk import ClaudeAgentOptions

from ..models import AgentOptions, SessionType
from ..utils.constants import TimestampFormat
from aii_lib.telemetry import logger


def initialize_agent(options: AgentOptions):
    """
    Initialize agent (one-time setup when Agent is created).

    Builds SDK options by:
    - Parsing agent markdown files to AgentDefinition dataclasses
    - Preparing workspace (.claude/ directories)
    - Setting up MCP configurations
    - Converting AgentOptions to ClaudeAgentOptions

    Args:
        options: Agent configuration options

    Returns:
        ClaudeAgentOptions ready for SDK execution
    """
    # Prepare workspace and parse agents/skills/MCPs
    cwd_path = Path(options.cwd).resolve() if options.cwd else Path.cwd()
    t = options.telemetry  # Shorthand for telemetry
    rid = options.run_id   # Shorthand for run_id

    # Convert selected agents to programmatic definitions (no copying to workspace)
    # This prevents SDK from discovering unwanted agents from parent directories
    if options.selected_agents:
        from ..utils.init_helpers import get_agent
        from ..utils.init_helpers.agent_parser import parse_agent_markdown

        if t: t.emit(MessageType.DEBUG, f"Processing selected_agents: {options.selected_agents}", run_id=rid)
        if t: t.emit(MessageType.DEBUG, f"options.agents before: {options.agents}", run_id=rid)

        if not options.agents:
            options.agents = {}

            for agent in options.selected_agents:
                if isinstance(agent, str):
                    # Resolve string name to AgentDefinition from registry
                    agent_def_obj = get_agent(agent)
                    if t: t.emit(MessageType.DEBUG, f"get_agent('{agent}') returned: {agent_def_obj}", run_id=rid)

                    if agent_def_obj:
                        # Parse the source .md file to get SDK AgentDefinition
                        agent_def = parse_agent_markdown(agent_def_obj.path)
                        if t: t.emit(MessageType.DEBUG, f"Parsed agent '{agent}': {agent_def}", run_id=rid)
                        options.agents[agent] = agent_def
                        if t: t.emit(MessageType.SUCCESS, f"Added programmatic agent '{agent}' to options.agents", run_id=rid)
                    else:
                        if t: t.emit(MessageType.WARNING, f"Agent '{agent}' not found in registry, skipping", run_id=rid)
                else:
                    # Already an SDK AgentDefinition object
                    options.agents[agent.name] = agent
                    if t: t.emit(MessageType.SUCCESS, f"Added programmatic agent '{agent.name}' (already SDK object)", run_id=rid)

        if t: t.emit(MessageType.INFO, f"Final options.agents: {len(options.agents) if options.agents else 0} agents", run_id=rid)

    # Prepare MCPs to workspace
    mcp_config_path = None
    if options.selected_mcps:
        from ..utils.init_helpers import prepare_mcps
        mcp_config_path = prepare_mcps(
            options.selected_mcps,
            cwd=cwd_path,
            telemetry=options.telemetry,
            run_id=options.run_id,
        )

    # Auto-set mcp_servers if MCPs were prepared
    if mcp_config_path and not options.mcp_servers:
        options.mcp_servers = str(mcp_config_path)

    # Determine session parameters based on session_type enum
    if options.session_type == SessionType.NEW:
        resume_id = None
        fork = False
    elif options.session_type == SessionType.RESUME:
        resume_id = options.resume_session_id
        fork = False
    elif options.session_type == SessionType.FORK:
        resume_id = options.resume_session_id
        fork = True
    else:
        raise ValueError(f"Invalid session_type: {options.session_type}")

    # Handle custom tools
    mcp_servers = options.mcp_servers
    if options.custom_tool_files:
        from ..utils.init_helpers.mcp_tools import setup_custom_tools

        # Load custom tools and create SDK MCP server
        custom_server_config = setup_custom_tools(
            options.custom_tool_files,
            telemetry=t,
            run_id=rid,
        )

        # Merge with existing mcp_servers
        if isinstance(mcp_servers, dict):
            mcp_servers = {**mcp_servers, **custom_server_config}
        else:
            # If mcp_servers is a path string, keep it and log warning
            if t: t.emit(
                MessageType.WARNING,
                "custom_tool_files specified but mcp_servers is a path string. "
                "Custom tools will be ignored. Use dict format for mcp_servers.",
                run_id=rid
            )

    # Handle custom agents (TODO: not yet implemented)
    agents = options.agents

    # Build SDK options dict
    # Convert cwd to absolute path for Skills to work
    cwd_path = Path(options.cwd).resolve() if options.cwd else Path.cwd()

    # First prompt: don't set continue_conversation (SDK default is False)
    # For RESUME/FORK, it's set via resume/fork_session parameters
    # NOTE: system_prompt is NOT passed to SDK - it's emitted as S_PROMPT in initialize_execution
    options_dict = {
        "allowed_tools": options.allowed_tools,
        # system_prompt intentionally omitted - emitted as S_PROMPT before PROMPT
        "permission_mode": options.permission_mode,
        "continue_conversation": None,  # Will be filtered out
        "max_turns": options.max_turns,
        "model": options.model,
        "cwd": str(cwd_path),
        "resume": resume_id,
        "fork_session": fork,
        "disallowed_tools": options.disallowed_tools,
        "mcp_servers": mcp_servers,
        "permission_prompt_tool_name": options.permission_prompt_tool_name,
        "settings": options.settings,
        "add_dirs": [str(d) for d in options.add_dirs],
        "env": options.env,
        "extra_args": {**options.extra_args, **({"effort": options.effort} if options.effort else {})},
        "max_buffer_size": options.max_buffer_size,
        "include_partial_messages": options.include_partial_messages,
        "agents": agents,
        "setting_sources": options.setting_sources,
        "output_format": options.output_format,
    }

    # Filter out None values
    options_dict = {key: value for key, value in options_dict.items() if value is not None}

    # Create SDK options
    sdk_options = ClaudeAgentOptions(**options_dict)

    # Debug: Log SDK options (through telemetry if available)
    from dataclasses import fields as dataclass_fields

    config_lines = ["SDK ClaudeAgentOptions config:"]
    for field in dataclass_fields(sdk_options):
        # Skip internal fields
        if field.name.startswith('_'):
            continue

        value = getattr(sdk_options, field.name, None)
        # Format complex types nicely
        if isinstance(value, list):
            display_value = f"{len(value)} items" if value else "[]"
        elif isinstance(value, dict):
            display_value = f"{len(value)} entries" if value else "{}"
        else:
            display_value = value
        config_lines.append(f"  {field.name}: {display_value}")

    config_msg = "\n".join(config_lines)
    if t: t.emit(MessageType.DEBUG, config_msg, run_id=rid)

    # Flush to ensure proper sequencing with telemetry output
    sys.stdout.flush()

    return sdk_options


def initialize_execution(
    options: AgentOptions,
    prompt: str,
    prompt_index: int,
    telemetry=None,
):
    """
    Initialize execution for a single prompt.

    Emits S_PROMPT (system prompt) followed by PROMPT (user prompt),
    matching the pattern used by other LLM backends (OpenRouter, etc.).

    Args:
        options: Agent configuration options
        prompt: The prompt text to execute
        prompt_index: Index of this prompt in the sequence
        telemetry: Optional AIITelemetry instance. If provided, overrides
            ``options.telemetry``. If both are None, a new AIITelemetry
            instance is created with default sinks.

    Returns:
        Tuple of (telemetry, execution_state dict)
    """
    from aii_lib.telemetry import AIITelemetry, ConsoleSink, JSONSink, load_telemetry_config

    # Use telemetry from options if provided, then parameter, then create new
    if telemetry is None:
        telemetry = options.telemetry

    # Create telemetry if still not provided
    if telemetry is None:
        telemetry = AIITelemetry()

        # Load telemetry config from config.yaml
        telem_config = load_telemetry_config()
        config_truncation = telem_config.get("console_msg_truncate", 150)
        config_log_messages = telem_config.get("log_messages", True)

        # Parse truncation (can be int, False, or null)
        if config_truncation is False or config_truncation is None:
            truncation = None
        else:
            truncation = int(config_truncation)

        # log_mode controls console output
        # None = use config (default - respects console_msg_truncate from config.yaml)
        # "none" = no console output
        # "truncated" = use config truncation value (same as None)
        # "full" = explicitly no truncation (overrides config)
        if options.log_mode == "none":
            pass  # No console sink
        elif options.log_mode == "full":
            # Explicitly disable truncation (override config)
            telemetry.add_sink(ConsoleSink(truncation=None, log_messages=config_log_messages))
        else:  # None or "truncated" - use config
            telemetry.add_sink(ConsoleSink(truncation=truncation, log_messages=config_log_messages))

        # Add JSON sink
        if options.json_log_path:
            json_log_path = Path(options.json_log_path)
        else:
            cwd = Path(options.cwd) if options.cwd else Path.cwd()
            json_log_path = cwd / "all_messages.jsonl"
        telemetry.add_sink(JSONSink(json_log_path))

    # Execution state (tracks model, timing, etc.)
    # Use options.model as initial model (fallback if SDK doesn't provide it)
    execution_state = {
        "prompt_index": prompt_index,
        "current_model": options.model,  # Initialize with options.model as fallback
        "module_start_time": datetime.now().isoformat(),
        "message_count": 0,
        "custom_metadata": options.custom_metadata or {},
        "run_id": options.run_id,  # For sequenced parallel execution
        "agent_context": options.agent_context,  # Display name for logs (e.g., "data-0")
    }

    # Emit S_PROMPT (system prompt) before PROMPT - matches OpenRouter pattern
    if options.system_prompt and prompt_index == 0:
        # Only emit system prompt on first prompt of a sequence
        system_prompt_text = options.system_prompt
        if isinstance(system_prompt_text, dict):
            # Handle preset dict format
            system_prompt_text = str(system_prompt_text)

        s_prompt_message = {
            'type': MessageType.S_PROMPT.value,
            'message_text': system_prompt_text,
            'iso_timestamp': datetime.now().isoformat(),
            'tool_name': '',
            'tool_id': '',
            'agent_context': options.agent_context or '',
            'subagent_id': None,
            'is_error': False,
            'message_metadata': {'model': options.model},
            'prompt_index': prompt_index,
            'run_id': options.run_id,  # For sequenced parallel execution
        }
        telemetry.emit_dict(s_prompt_message)
        execution_state["message_count"] += 1

    # Log user prompt message
    prompt_message = {
        'type': MessageType.PROMPT.value,
        'message_text': prompt,
        'iso_timestamp': datetime.now().isoformat(),
        'tool_name': '',
        'tool_id': '',
        'agent_context': options.agent_context or '',
        'subagent_id': None,
        'is_error': False,
        'message_metadata': {'model': options.model},
        'prompt_index': prompt_index,
        'run_id': options.run_id,  # For sequenced parallel execution
    }
    telemetry.emit_dict(prompt_message)
    execution_state["message_count"] += 1

    return telemetry, execution_state
