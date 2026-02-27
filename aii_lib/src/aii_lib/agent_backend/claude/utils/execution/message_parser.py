"""
Message parsing utilities
"""

import re
from typing import Optional, Callable
from datetime import datetime
from pathlib import Path

from loguru import logger

from claude_agent_sdk import (
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
    UserMessage,
    SystemMessage,
)
from ...models import TokenUsage


def _clean_string_value(s: str) -> str:
    """Clean up string values for display (remove \\n\\t artifacts)."""
    s = re.sub(r'[\n\t]+', ' ', s)
    s = re.sub(r' +', ' ', s)
    return s.strip()


def _clean_json_strings(obj):
    """Recursively clean string values in a JSON object."""
    if isinstance(obj, str):
        return _clean_string_value(obj)
    elif isinstance(obj, dict):
        return {k: _clean_json_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_json_strings(item) for item in obj]
    return obj


def get_short_timestamp() -> str:
    """Get timestamp in short format MM-DD HH:MM:SS"""
    now = datetime.now()
    return now.strftime("%m-%d %H:%M:%S")



def get_tool_abbrev(tool_name: str, suffix: str) -> str:
    """
    Get abbreviated tool name for display.

    Args:
        tool_name: Full tool name (e.g., "mcp__custom-tools__calculator_basic")
        suffix: "_IN" or "_OUT"

    Returns:
        Abbreviated name (e.g., "CALC_IN")
    """
    # For MCP tools, extract the actual tool name after the last "__"
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__")
        if len(parts) >= 3:
            # Last part is the actual tool name
            actual_tool_name = parts[-1]
        else:
            actual_tool_name = tool_name
    else:
        actual_tool_name = tool_name

    # Special case: openrouter tools all show as "OPER"
    if actual_tool_name.lower().startswith("openrouter"):
        return f"OPER{suffix}"

    # Special case: all HuggingFace MCP tools show as "HF"
    if "hf-mcp-server" in tool_name:
        return f"HF{suffix}"

    # Special case: chrome-devtools MCP tools show as "CHRM"
    if "chrome-devtools" in tool_name:
        return f"CHRM{suffix}"

    # Special case: context7 MCP tools show as "CTX7"
    if "context7" in tool_name:
        return f"CTX7{suffix}"

    # Special case: shadcn MCP tools show as "SHAD"
    if "shadcn" in tool_name:
        return f"SHAD{suffix}"

    # Special case: calculator MCP tools show as "CALC"
    if "calc" in tool_name or "calculator" in actual_tool_name.lower():
        return f"CALC{suffix}"

    # Special case: arxiv MCP tools - distinguish Search vs Fetch
    if "arxiv" in tool_name:
        if "search" in actual_tool_name.lower():
            return f"ARXS{suffix}"
        elif "fetch" in actual_tool_name.lower():
            return f"ARXF{suffix}"
        else:
            return f"ARXV{suffix}"

    # Special case: wikipedia MCP tools show as "WIKS"
    if "wikipedia" in tool_name or "wikisearch" in actual_tool_name.lower():
        return f"WIKS{suffix}"

    # Try to create a smart 4-char abbreviation
    # Strategy 1: For camelCase names with multiple capitals (e.g., ArxivSearch, ArxivFetch)
    # Use first 3 chars + last capital letter
    capitals = [c for c in actual_tool_name if c.isupper()]
    if len(capitals) >= 2:
        # Get first 3 chars + last capital
        # ArxivSearch → "Arx" + "S" = "ARXS"
        # ArxivFetch → "Arx" + "F" = "ARXF"
        first_three = actual_tool_name[:3].upper()
        last_capital = capitals[-1]
        return f"{first_three}{last_capital}{suffix}"

    # Strategy 2: Default to first 4 chars
    return f"{actual_tool_name[:4].upper()}{suffix}"


# Map tool names to abbreviated format (aii_pipeline style)
TOOL_NAME_MAP_INPUT = {
    "Read": "READ_IN",
    "Write": "WRIT_IN",
    "Edit": "EDIT_IN",
    "Bash": "BASH_IN",
    "Grep": "GREP_IN",
    "Glob": "GLOB_IN",
    "Task": "TASK_IN",
    "TodoWrite": "TODO_IN",
    "WebSearch": "SRCH_IN",
    "WebFetch": "FTCH_IN",
    "Skill": "SKIL_IN",
    "OpenRouter": "OPER_IN",
}

TOOL_NAME_MAP_OUTPUT = {
    "Read": "READ_OUT",
    "Write": "WRIT_OUT",
    "Edit": "EDIT_OUT",
    "Bash": "BASH_OUT",
    "Grep": "GREP_OUT",
    "Glob": "GLOB_OUT",
    "Task": "TASK_OUT",
    "TodoWrite": "TODO_OUT",
    "WebSearch": "SRCH_OUT",
    "WebFetch": "FTCH_OUT",
    "Skill": "SKIL_OUT",
    "OpenRouter": "OPER_OUT",
}


def _pretty_print_value(value, indent: int = 2, max_chars: int | None = None) -> str:
    """
    Pretty-print a value, detecting and formatting JSON strings.

    Args:
        value: Any value to format
        indent: Indentation for JSON formatting
        max_chars: Optional max chars for truncation (from telemetry config)

    Returns:
        Formatted string
    """
    import json

    value_str = str(value) if not isinstance(value, str) else value

    # Try to parse as JSON if it looks like JSON
    if value_str.strip().startswith('{') or value_str.strip().startswith('['):
        try:
            parsed = json.loads(value_str)
            formatted = json.dumps(parsed, indent=indent, ensure_ascii=False)
            # Apply truncation if configured
            if max_chars and len(formatted) > max_chars:
                formatted = formatted[:max_chars] + f"\n... (+{len(value_str) - max_chars} chars truncated)"
            return formatted
        except json.JSONDecodeError:
            pass

    # For non-JSON, just apply truncation if needed
    if max_chars and len(value_str) > max_chars:
        return value_str[:max_chars] + f"... (+{len(value_str) - max_chars} chars)"

    return value_str


def _get_telemetry_truncation() -> int | None:
    """Get truncation value from telemetry config."""
    try:
        from aii_lib.telemetry import load_telemetry_config
        config = load_telemetry_config()
        truncation = config.get("console_msg_truncate")
        if truncation is False or truncation is None:
            return None
        return int(truncation)
    except Exception as e:
        logger.debug(f"Failed to load telemetry truncation config: {e}")
        return None


def format_mcp_data(data: dict | list | str, max_chars: int | None = None) -> str:
    """
    Format MCP tool data (input or output) for display.

    For simple values: key: value on one line
    For complex/long values: pretty-printed JSON with optional truncation

    Args:
        data: Dictionary, list, or string data from MCP tool
        max_chars: Optional max chars for truncation. If None, uses telemetry config.

    Returns:
        Formatted string
    """
    import json

    # Get truncation from config if not provided
    if max_chars is None:
        max_chars = _get_telemetry_truncation()

    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            # Pretty-print JSON values
            value_str = _pretty_print_value(value, indent=2, max_chars=max_chars)

            # For multiline values, format nicely
            if '\n' in value_str:
                lines.append(f"{key}:")
                # Indent each line of the value
                for line in value_str.split('\n'):
                    lines.append(f"  {line}")
            else:
                lines.append(f"{key}: {value_str}")

        return "\n".join(lines) if lines else str(data)

    elif isinstance(data, list):
        # For lists (like MCP output), format each item
        # First, try to unwrap MCP content blocks: [{"type": "text", "text": "..."}]
        unwrapped_items = []
        for item in data:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                text_content = item["text"]
                if isinstance(text_content, str):
                    # Try to parse nested JSON
                    try:
                        nested = json.loads(text_content)
                        # Clean string values (remove \n\t artifacts from HF descriptions)
                        nested = _clean_json_strings(nested)
                        unwrapped_items.append(nested)
                        continue
                    except json.JSONDecodeError:
                        # Not JSON, use text directly (cleaned)
                        unwrapped_items.append(_clean_string_value(text_content))
                        continue
            unwrapped_items.append(item)

        # If we unwrapped to a single item, just format that
        if len(unwrapped_items) == 1:
            item = unwrapped_items[0]
            if isinstance(item, (dict, list)):
                formatted = json.dumps(item, indent=2, ensure_ascii=False)
                if max_chars and len(formatted) > max_chars:
                    formatted = formatted[:max_chars] + f"\n... (+{max(0, len(str(item)) - max_chars)} chars truncated)"
                return formatted
            return str(item)

        # Multiple items - format with indices
        items = []
        for i, item in enumerate(unwrapped_items):
            if isinstance(item, dict):
                # Pretty-print dict items
                try:
                    formatted = json.dumps(item, indent=2, ensure_ascii=False)
                    if max_chars and len(formatted) > max_chars:
                        formatted = formatted[:max_chars] + f"\n... (+{max(0, len(str(item)) - max_chars)} chars)"
                    items.append(f"[{i}]:\n{formatted}")
                except (TypeError, ValueError):
                    items.append(f"[{i}]: {str(item)}")
            else:
                items.append(f"[{i}]: {str(item)}")
        return "\n\n".join(items) if items else str(data)

    else:
        # For raw strings, try to parse as JSON
        return _pretty_print_value(data, indent=2, max_chars=max_chars)


def format_tool_input(tool_name: str, tool_input: dict, agent_name: str = "") -> str:
    """Format tool input for display (aii_pipeline style)"""
    if tool_name == "Task":
        # Format: subagent_name:\nprompt (newline after colon)
        description = tool_input.get('description', '')
        prompt = tool_input.get('prompt', '')
        subagent_type = tool_input.get('subagent_type', agent_name or 'task')

        # Put agent name and colon on first line, then newline, then prompt
        return f"{subagent_type}:\n{prompt}"

    elif tool_name == "Bash":
        command = tool_input.get('command', '')
        description = tool_input.get('description', '')
        if description:
            return f"{description}:\n{command}"
        return command

    elif tool_name == "TodoWrite":
        todos = tool_input.get('todos', [])
        todo_lines = []
        for i, todo in enumerate(todos, 1):
            status = todo.get('status', 'pending')
            content_text = todo.get('content', '')
            # Add 1 newline before first todo, 2 newlines before others for better readability
            prefix = "\n" if i == 1 else "\n\n"
            todo_lines.append(f"{prefix}{i}. [{status}] {content_text}")
        return "".join(todo_lines) if todo_lines else "No todos"

    elif tool_name == "Write":
        file_path = tool_input.get('file_path', 'unknown')
        content = tool_input.get('content', '')
        return f"File: {file_path}\n\n{content}"

    elif tool_name == "Edit":
        file_path = tool_input.get('file_path', 'unknown')
        old_string = tool_input.get('old_string', '')
        new_string = tool_input.get('new_string', '')
        return f"File: {file_path}\nOLD: {old_string}\nNEW: {new_string}"

    elif tool_name == "Read":
        # Just show file path
        file_path = tool_input.get('file_path', 'unknown')
        return file_path

    elif tool_name == "Glob":
        # Just show pattern in quotes
        pattern = tool_input.get('pattern', '')
        return f'Pattern: "{pattern}"'

    elif tool_name == "Grep":
        # Show pattern in quotes
        pattern = tool_input.get('pattern', '')
        return f'Pattern: "{pattern}"'

    elif tool_name == "WebSearch":
        # Show query and allowed_domains if present
        query = tool_input.get('query', '')
        allowed_domains = tool_input.get('allowed_domains', [])
        if allowed_domains:
            return f"{query} | allowed_domains: {allowed_domains}"
        return query

    elif tool_name == "WebFetch":
        # Show URL and prompt
        url = tool_input.get('url', '')
        prompt = tool_input.get('prompt', '')
        if prompt:
            return f"URL: {url}\nPrompt: {prompt}"
        return f"URL: {url}"

    elif tool_name == "Skill":
        # Show skill name and command
        skill = tool_input.get('skill', 'unknown')
        command = tool_input.get('command', '')
        if command:
            return f"{skill}:\n{command}"
        return f"{skill}"

    elif tool_name.startswith("mcp__"):
        # Format MCP tool inputs with each field on its own line
        return format_mcp_data(tool_input)

    else:
        return str(tool_input)


def serialize_message_for_debug(message) -> dict:
    """
    Serialize a message object for full debug output.
    Captures all fields from the Anthropic API response.
    """
    try:
        # Try to get all attributes from the message object
        if hasattr(message, '__dict__'):
            debug_data = {}
            for key, value in message.__dict__.items():
                # Handle complex objects
                if hasattr(value, '__dict__'):
                    debug_data[key] = serialize_message_for_debug(value)
                elif isinstance(value, list):
                    debug_data[key] = [serialize_message_for_debug(item) if hasattr(item, '__dict__') else str(item) for item in value]
                else:
                    # Convert to string for non-serializable types
                    try:
                        import json
                        json.dumps(value)  # Test if serializable
                        debug_data[key] = value
                    except (TypeError, ValueError):
                        debug_data[key] = str(value)
            return debug_data
        else:
            return {"raw": str(message)}
    except Exception as e:
        return {"error": f"Failed to serialize message: {e}", "raw": str(message)}


def parse_assistant_message(
    message: AssistantMessage,
    prompt_index: int,
    on_message_logged: Optional[Callable] = None,
    last_tool_id: Optional[str] = None,
    last_tool_name: Optional[str] = None,
    tool_id_to_agent_name: Optional[dict] = None,
    tool_id_to_tool_name: Optional[dict] = None,
    model: Optional[str] = None,
    tool_calls_count: Optional[dict] = None,
    seen_tool_result_ids: Optional[set] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Parse an AssistantMessage and extract text/tool blocks.

    Args:
        message: The AssistantMessage to parse
        prompt_index: Index of the current prompt
        on_message_logged: Optional callback for message events
        last_tool_id: Previous tool ID for tracking
        last_tool_name: Previous tool name for tracking
        tool_id_to_agent_name: Mapping of Task tool IDs to agent names (for subagent tracking)
        tool_id_to_tool_name: Mapping of all tool IDs to tool names (for result matching)
        model: Model name for accurate cost calculation
        tool_calls_count: Dict to track tool call counts {tool_name: count}

    Returns:
        Tuple of (new_last_tool_id, new_last_tool_name)
    """
    current_tool_id = last_tool_id
    current_tool_name = last_tool_name

    # Initialize tool_id_to_agent_name if not provided
    if tool_id_to_agent_name is None:
        tool_id_to_agent_name = {}

    # Initialize tool_id_to_tool_name if not provided
    if tool_id_to_tool_name is None:
        tool_id_to_tool_name = {}

    # Initialize tool_calls_count if not provided
    if tool_calls_count is None:
        tool_calls_count = {}

    # Initialize seen_tool_result_ids if not provided
    if seen_tool_result_ids is None:
        seen_tool_result_ids = set()

    # Capture complete API response for debugging
    raw_api_message = serialize_message_for_debug(message)

    # Check if this message belongs to a subagent (Task tool)
    parent_tool_use_id = getattr(message, 'parent_tool_use_id', None)
    agent_context = ""
    subagent_id = None

    if parent_tool_use_id and parent_tool_use_id in tool_id_to_agent_name:
        # This message is from a subagent
        agent_name = tool_id_to_agent_name[parent_tool_use_id]
        # Format: agent_name + last 2 chars of tool ID
        tool_id_short = parent_tool_use_id[-2:] if len(parent_tool_use_id) >= 2 else parent_tool_use_id
        agent_context = f"{agent_name}:{tool_id_short}"
        subagent_id = parent_tool_use_id

    for block in message.content:
        if isinstance(block, TextBlock):

            # Log text blocks (Claude's responses)
            if on_message_logged:
                on_message_logged({
                    "type": "claude_msg",
                    "message_text": block.text,
                    "timestamp": get_short_timestamp(),
                    "tool_name": "",
                    "tool_id": "",
                    "agent_context": agent_context,
                    "subagent_id": subagent_id,
                    "parent_tool_use_id": parent_tool_use_id,
                    "is_error": False,
                    "message_metadata": {
                        "text_block": block.text,  # Full text
                    },
                    "raw_api_message": raw_api_message
                })

        elif isinstance(block, ThinkingBlock):
            # Log thinking blocks (Claude's internal reasoning)
            if on_message_logged:
                on_message_logged({
                    "type": "thinking",
                    "message_text": block.thinking,
                    "timestamp": get_short_timestamp(),
                    "tool_name": "",
                    "tool_id": "",
                    "agent_context": agent_context,
                    "subagent_id": subagent_id,
                    "parent_tool_use_id": parent_tool_use_id,
                    "is_error": False,
                    "message_metadata": {
                        "thinking": block.thinking,  # Full thinking content
                        "signature": getattr(block, 'signature', None),  # Optional signature field
                    },
                    "raw_api_message": raw_api_message
                })

        elif isinstance(block, ToolUseBlock):
            # Save tool info for matching with results
            current_tool_id = block.id
            current_tool_name = block.name

            # Track tool ID to name mapping for later result matching
            tool_id_to_tool_name[block.id] = block.name

            # Track tool call counts for summary metrics
            tool_calls_count[block.name] = tool_calls_count.get(block.name, 0) + 1

            # Check if this ToolUseBlock has parent_tool_use_id (for nested tool calls from subagents)
            block_parent_id = getattr(block, 'parent_tool_use_id', None)
            if block_parent_id and not parent_tool_use_id:
                # Update agent_context for this specific tool use
                if block_parent_id in tool_id_to_agent_name:
                    agent_name = tool_id_to_agent_name[block_parent_id]
                    tool_id_short = block_parent_id[-2:] if len(block_parent_id) >= 2 else block_parent_id
                    agent_context = f"{agent_name}:{tool_id_short}"
                    subagent_id = block_parent_id

            # Track Task tool invocations for subagent mapping
            if block.name == "Task":
                # Get fields for agent name extraction
                task_description = block.input.get("description", "")
                subagent_type = block.input.get("subagent_type", "")
                task_prompt = block.input.get("prompt", "")

                # Extract agent name with correct priority
                agent_name = "task"

                # PRIORITY 1: Use subagent_type (this is the actual agent name!)
                if subagent_type:
                    agent_name = subagent_type.replace("-", "_")

                # PRIORITY 2: Extract from prompt (fallback if no subagent_type)
                elif task_prompt and ":" in task_prompt[:100]:
                    # Check if prompt starts with "agent_name:" pattern
                    potential_agent_name = task_prompt.split(":", 1)[0].strip()
                    # Validate it looks like an agent name (short, no spaces/special chars)
                    if len(potential_agent_name) < 30 and not any(c in potential_agent_name for c in ['\n', '\t', '  ']):
                        agent_name = potential_agent_name

                # PRIORITY 3: Use description (last resort)
                elif task_description and len(task_description) < 30:
                    agent_name = task_description.replace(" ", "_")

                tool_id_to_agent_name[block.id] = agent_name

            # Format tool input
            message_text = format_tool_input(block.name, block.input, tool_id_to_agent_name.get(block.id, ""))
            tool_name_abbrev = TOOL_NAME_MAP_INPUT.get(block.name, get_tool_abbrev(block.name, "_IN"))

            # Log tool use
            if on_message_logged:
                on_message_logged({
                    "type": "tool_input",
                    "message_text": message_text,
                    "timestamp": get_short_timestamp(),
                    "tool_name": tool_name_abbrev,
                    "tool_id": block.id,
                    "agent_context": agent_context,
                    "subagent_id": subagent_id,
                    "parent_tool_use_id": parent_tool_use_id,
                    "is_error": False,
                    "message_metadata": {
                        "raw_tool_input": block.input,
                        "tool_name_full": block.name,  # Full tool name (not abbreviated)
                        "block_id": block.id,
                    },
                    "raw_api_message": raw_api_message
                })

        elif isinstance(block, ToolResultBlock):
            # Get tool name from last tracked tool
            tool_use_id_for_result = block.tool_use_id if hasattr(block, 'tool_use_id') else current_tool_id

            # Deduplication: skip if we've already logged this tool result
            if tool_use_id_for_result and tool_use_id_for_result in seen_tool_result_ids:
                continue
            if tool_use_id_for_result:
                seen_tool_result_ids.add(tool_use_id_for_result)

            tool_name_abbrev = TOOL_NAME_MAP_OUTPUT.get(current_tool_name or "", get_tool_abbrev(current_tool_name or "", "_OUT"))

            # Format tool output based on tool type
            output_text = str(block.content) if block.content else ""
            if current_tool_name and current_tool_name.startswith("mcp__") and block.content is not None:
                # Format MCP tool outputs with each field on its own line
                output_text = format_mcp_data(block.content)

            # Format message text with tool name prefix (consistent with OpenRouter format)
            display_text = f"Tool: {current_tool_name or 'unknown'}\nResult:\n{output_text}"

            # Log tool results
            if on_message_logged:
                on_message_logged({
                    "type": "tool_output",
                    "message_text": display_text,
                    "timestamp": get_short_timestamp(),
                    "tool_name": tool_name_abbrev,
                    "tool_id": tool_use_id_for_result,
                    "agent_context": agent_context,
                    "subagent_id": subagent_id,
                    "parent_tool_use_id": parent_tool_use_id,
                    "is_error": block.is_error,
                    "message_metadata": {
                        "raw_tool_output": block.content,  # Raw output (not stringified)
                        "tool_name_full": current_tool_name,  # Full tool name
                        "tool_use_id": tool_use_id_for_result,
                    },
                    "raw_api_message": raw_api_message
                })

    return current_tool_id, current_tool_name


def parse_result_message(
    message: ResultMessage,
    prompt_index: int,
    on_message_logged: Optional[Callable] = None,
    module_start_time: Optional[str] = None,
    message_count: int = 0,
    model: Optional[str] = None,
    tool_calls_count: Optional[dict] = None,
    emit_summary: bool = True,
) -> tuple[str, str, float, TokenUsage, dict, int, dict | None]:
    """
    Parse a ResultMessage and extract response, session ID, cost, and token usage.

    Emits standardized SummaryMetrics format compatible with all LLM backends.

    Args:
        message: The ResultMessage to parse
        prompt_index: Index of the current prompt
        on_message_logged: Optional callback for message events
        module_start_time: ISO timestamp of first message in module (for runtime calculation)
        message_count: Total number of messages in this module
        model: Model name (e.g., "claude-sonnet-4-5")
        tool_calls_count: Dict of tool call counts {tool_name: count}
        emit_summary: Whether to emit the summary immediately (False for multi-prompt aggregation)

    Returns:
        Tuple of (response_text, session_id, cost, token_usage, summary_data, num_turns, structured_output)
    """
    response_text = message.result or ""
    session_id = message.session_id or ""
    cost = message.total_cost_usd or 0.0

    # Capture complete API response for debugging
    raw_api_message = serialize_message_for_debug(message)

    # Extract usage data (token counts)
    # The usage dict is available on the message and contains all token counts
    usage_dict = getattr(message, "usage", {}) or {}
    usage = TokenUsage(
        input_tokens=usage_dict.get("input_tokens", 0),
        output_tokens=usage_dict.get("output_tokens", 0),
        cache_creation_input_tokens=usage_dict.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage_dict.get("cache_read_input_tokens", 0),
        total_cost=cost,
        raw_usage=usage_dict,
    )

    # Extract additional ResultMessage fields for comprehensive tracking
    duration_ms = getattr(message, 'duration_ms', None) or 0
    duration_api_ms = getattr(message, 'duration_api_ms', None) or 0
    num_turns = getattr(message, 'num_turns', None) or 1
    is_error = getattr(message, 'is_error', False)
    subtype = getattr(message, 'subtype', None)  # "success" or error subtype
    structured_output = getattr(message, 'structured_output', None)  # SDK native structured output

    # Calculate runtime_seconds (wall clock time)
    runtime_seconds = duration_ms / 1000.0 if duration_ms else 0.0

    # Calculate llm_time_seconds (API call time)
    llm_time_seconds = duration_api_ms / 1000.0 if duration_api_ms else 0.0

    # Determine status from subtype and is_error
    if is_error:
        status = subtype or "failed"
    else:
        status = subtype or "completed"

    # Tool calls count (default to empty dict)
    tool_calls = tool_calls_count or {}

    # Claude Agent SDK doesn't have separate tool costs (included in API pricing)
    # but we include the field for consistency with SummaryMetrics schema
    tool_cost = 0.0
    tool_costs = {}

    # Build summary data dict (always, for aggregation)
    summary_data = {
        # === Standardized SummaryMetrics fields ===
        "type": "summary",
        "total_cost": cost,              # token_cost + tool_cost
        "token_cost": cost,              # All cost is token cost for Claude SDK
        "tool_cost": tool_cost,          # No separate tool costs
        "model": model or "",
        "status": status,
        "is_aggregated": False,
        "num_calls": num_turns,          # Number of conversation turns
        "runtime_seconds": runtime_seconds,
        "llm_time_seconds": llm_time_seconds,
        "input_tokens": usage_dict.get("input_tokens", 0),
        "output_tokens": usage_dict.get("output_tokens", 0),
        "reasoning_tokens": 0,           # Claude doesn't report reasoning tokens separately
        "cache_write_tokens": usage_dict.get("cache_creation_input_tokens", 0),
        "cache_read_tokens": usage_dict.get("cache_read_input_tokens", 0),
        "tool_calls": tool_calls,
        "tool_costs": tool_costs,

        # === Additional agent-specific fields for backward compatibility ===
        "message_text": f"Total cost: ${cost:.4f}",
        "timestamp": get_short_timestamp(),
        "tool_name": "",
        "tool_id": "",
        "agent_context": "",
        "subagent_id": None,
        "parent_tool_use_id": None,
        "is_error": is_error,
        "message_metadata": {
            "session_id": session_id,
            "final_result": response_text,
            "message_count": message_count,
            "raw_usage": usage_dict,
        },
        "raw_api_message": raw_api_message
    }

    # Only emit summary if requested (False for multi-prompt aggregation)
    if emit_summary and on_message_logged:
        on_message_logged(summary_data)

    return response_text, session_id, cost, usage, summary_data, num_turns, structured_output


def parse_user_message(
    message: UserMessage,
    prompt_index: int,
    on_message_logged: Optional[Callable] = None,
    last_tool_id: Optional[str] = None,
    last_tool_name: Optional[str] = None,
    tool_id_to_agent_name: Optional[dict] = None,
    tool_id_to_tool_name: Optional[dict] = None,
    seen_tool_result_ids: Optional[set] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Parse a UserMessage and extract tool results (especially Task tool results).

    Args:
        message: The UserMessage to parse
        prompt_index: Index of the current prompt
        on_message_logged: Optional callback for message events
        last_tool_id: Previous tool ID for tracking
        last_tool_name: Previous tool name for tracking
        tool_id_to_agent_name: Mapping of Task tool IDs to agent names
        tool_id_to_tool_name: Mapping of all tool IDs to tool names (for result matching)

    Returns:
        Tuple of (new_last_tool_id, new_last_tool_name)
    """
    current_tool_id = last_tool_id or ""
    current_tool_name = last_tool_name or ""

    # Initialize tool_id_to_agent_name if not provided
    if tool_id_to_agent_name is None:
        tool_id_to_agent_name = {}

    # Initialize tool_id_to_tool_name if not provided
    if tool_id_to_tool_name is None:
        tool_id_to_tool_name = {}

    # Initialize seen_tool_result_ids if not provided
    if seen_tool_result_ids is None:
        seen_tool_result_ids = set()

    # Capture complete API response for debugging
    raw_api_message = serialize_message_for_debug(message)

    # Check if message content is a list and contains ToolResultBlocks
    if isinstance(message.content, list):
        for content_block in message.content:
            if isinstance(content_block, ToolResultBlock):
                # Extract tool result info
                tool_use_id = content_block.tool_use_id if hasattr(content_block, 'tool_use_id') else current_tool_id

                # Deduplication: skip if we've already logged this tool result
                if tool_use_id and tool_use_id in seen_tool_result_ids:
                    continue
                if tool_use_id:
                    seen_tool_result_ids.add(tool_use_id)
                is_error = content_block.is_error if hasattr(content_block, 'is_error') else False

                # Find which tool this result belongs to
                # Try tool_id_to_tool_name first, fallback to last_tool_name
                tool_name = tool_id_to_tool_name.get(tool_use_id, current_tool_name)

                # Format the result based on tool type
                display_content = ""

                # Handle Task tool results specially - extract just the text
                if tool_name == "Task":
                    # content_block.content can be:
                    # 1. A list of dicts: [{'type': 'text', 'text': '...'}]
                    # 2. A string representation: "[{'type': 'text', 'text': '...'}]"
                    # 3. Just a string: "result text"

                    if isinstance(content_block.content, list):
                        # Case 1: Already a list
                        if len(content_block.content) > 0:
                            first_element = content_block.content[0]
                            if isinstance(first_element, dict) and 'text' in first_element:
                                display_content = first_element['text']
                            else:
                                display_content = str(first_element)
                    elif isinstance(content_block.content, str):
                        # Case 2 or 3: String (might be JSON or plain text)
                        if content_block.content.startswith('['):
                            try:
                                import json
                                parsed = json.loads(content_block.content)
                                if isinstance(parsed, list) and len(parsed) > 0:
                                    first_element = parsed[0]
                                    if isinstance(first_element, dict) and 'text' in first_element:
                                        display_content = first_element['text']
                                    else:
                                        display_content = str(first_element)
                            except (json.JSONDecodeError, ValueError, TypeError, IndexError, KeyError):
                                # If JSON parsing fails, use the whole string
                                display_content = content_block.content
                        else:
                            # Plain text
                            display_content = content_block.content
                    else:
                        # Unknown type, convert to string
                        display_content = str(content_block.content) if content_block.content else ''
                else:
                    # For other tools, use raw content as string
                    display_content = str(content_block.content) if content_block.content else ''

                # Format MCP tool outputs with each field on its own line
                if tool_name and tool_name.startswith("mcp__") and content_block.content is not None:
                    display_content = format_mcp_data(content_block.content)

                # Get abbreviated tool name for output
                tool_name_abbrev = TOOL_NAME_MAP_OUTPUT.get(tool_name or "", get_tool_abbrev(tool_name or "", "_OUT"))

                # Check if this message belongs to a subagent
                parent_tool_use_id = getattr(message, 'parent_tool_use_id', None)
                agent_context = ""
                subagent_id = None

                # For TASK_OUT, subagent_id is the Task tool's OWN ID (tool_use_id), not its parent
                if tool_name == "Task" and tool_use_id and tool_use_id in tool_id_to_agent_name:
                    agent_name = tool_id_to_agent_name[tool_use_id]
                    tool_id_short = tool_use_id[-2:] if len(tool_use_id) >= 2 else tool_use_id
                    agent_context = f"{agent_name}:{tool_id_short}"
                    subagent_id = tool_use_id  # Use Task tool's own ID
                # For other tools, use parent_tool_use_id as before
                elif parent_tool_use_id and parent_tool_use_id in tool_id_to_agent_name:
                    agent_name = tool_id_to_agent_name[parent_tool_use_id]
                    tool_id_short = parent_tool_use_id[-2:] if len(parent_tool_use_id) >= 2 else parent_tool_use_id
                    agent_context = f"{agent_name}:{tool_id_short}"
                    subagent_id = parent_tool_use_id

                # Format message text with tool name prefix (consistent with OpenRouter format)
                display_text = f"Tool: {tool_name or 'unknown'}\nResult:\n{display_content if display_content else ''}"

                # Log tool result
                if on_message_logged:
                    on_message_logged({
                        "type": "tool_output",
                        "message_text": display_text,
                        "timestamp": get_short_timestamp(),
                        "tool_name": tool_name_abbrev,
                        "tool_id": tool_use_id,
                        "agent_context": agent_context,
                        "subagent_id": subagent_id,
                        "parent_tool_use_id": parent_tool_use_id,
                        "is_error": is_error,
                        "message_metadata": {
                            "raw_tool_output": content_block.content,  # Raw output (not formatted)
                            "tool_name_full": tool_name,  # Full tool name
                            "tool_use_id": tool_use_id,
                            "display_content": display_content,  # Formatted version
                        },
                        "raw_api_message": raw_api_message
                    })

    return current_tool_id, current_tool_name


def parse_system_message(
    message: SystemMessage,
    prompt_index: int,
    on_message_logged: Optional[Callable] = None,
    system_prompt = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Parse a SystemMessage and extract early session ID and initialization data.

    Args:
        message: The SystemMessage to parse
        prompt_index: Index of the current prompt
        on_message_logged: Optional callback for message events
        system_prompt: Optional system prompt from AgentOptions to include in metadata

    Returns:
        Tuple of (session_id, model) if available, else (None, None)
    """
    # Capture complete API response for debugging
    raw_api_message = serialize_message_for_debug(message)

    # Extract data from SystemMessage
    subtype = getattr(message, 'subtype', None)
    data = getattr(message, 'data', {})

    # Extract key fields from data
    session_id = None
    cwd = None
    model = None

    if isinstance(data, dict):
        session_id = data.get('session_id')
        cwd = data.get('cwd')
        model = data.get('model')

    # Use full model name (no shortening)

    # Log system message (skip if no model - e.g., continued conversations)
    if on_message_logged and model:

        # Build details dict with all system info
        details = {}
        if model:
            details["model"] = model
        if session_id:
            details["Session ID"] = session_id
        if cwd:
            # Show last part of path
            cwd_short = Path(cwd).name if cwd else ""
            if cwd_short:
                details["Working Directory"] = cwd_short

        # Add counts for tools, skills, MCP servers
        if isinstance(data, dict):
            tools = data.get('tools', [])
            skills = data.get('skills', [])
            mcp_servers = data.get('mcp_servers', [])

            if tools and len(tools) > 0:
                details["Tools"] = len(tools)
            if skills and len(skills) > 0:
                details["Skills"] = len(skills)
            if mcp_servers and len(mcp_servers) > 0:
                details["MCP Servers"] = len(mcp_servers)

            # Add permission mode if not default
            permission_mode = data.get('permissionMode')
            if permission_mode and permission_mode != 'askForPermission':
                details["Permission"] = permission_mode

        # Prepare metadata with full data object for verbose logging
        metadata = {
            "subtype": subtype,
            "session_id": session_id,
            "cwd": cwd,
            "model": model,
        }

        # Include full data object if it's a dict (for verbose logging)
        if isinstance(data, dict):
            # Add system_prompt to data if provided (SDK doesn't include it)
            data_with_prompt = data.copy()
            if system_prompt is not None:
                data_with_prompt["system_prompt"] = system_prompt
            metadata["data"] = data_with_prompt
        else:
            # If data is not a dict, create one with system_prompt
            metadata["data"] = {"system_prompt": system_prompt} if system_prompt is not None else {}

        on_message_logged({
            "type": "system",
            "model": model,
            "details": details,
            "message_text": "",  # Leave empty, will be built by preprocessor
            "timestamp": get_short_timestamp(),
            "tool_name": "",
            "tool_id": "",
            "agent_context": "",
            "subagent_id": None,
            "parent_tool_use_id": None,
            "is_error": False,
            "message_metadata": metadata,
            "raw_api_message": raw_api_message
        })

    # Return full model name for consistent display across SYSTEM and SUMMARY
    return session_id, model
