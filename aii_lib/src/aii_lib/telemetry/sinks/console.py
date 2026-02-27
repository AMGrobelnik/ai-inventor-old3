"""
Console sink - colorized console output with formatting.
"""

import re
from typing import TYPE_CHECKING

from .base import Sink
from ..utils import format_json_output


if TYPE_CHECKING:
    from ..message import TelemetryMessage, MessageType


# ANSI color codes
class Colors:
    # Basic colors
    RED = "\033[31m"
    BRIGHT_RED = "\033[91m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    RESET = "\033[0m"

    # Semantic colors
    ERROR = "\033[31m"
    WARNING = "\033[33m"
    SUCCESS = "\033[32m"
    INFO = "\033[34m"
    DEBUG = "\033[36m"

    # Message type colors
    SYSTEM = "\033[38;5;240m"         # Grey for system
    ASSISTANT = "\033[38;5;214m"      # Orange for assistant (CLAUDE_M)
    THINKING = "\033[38;5;201m"       # Pink/magenta for thinking
    TOOL_CALL = "\033[38;5;245m"      # Grey for tool input
    TOOL_RESULT = "\033[38;5;28m"     # Dark green for tool output
    SUMMARY = "\033[38;5;81m"         # Light blue for summary
    TASK_IN = "\033[38;5;208m"        # Orange for task start
    TASK_OUT = "\033[38;5;28m"        # Dark green for task end
    TODO_IN = "\033[38;5;82m"         # Bright lime for todo input
    TODO_OUT = "\033[38;5;82m"        # Bright lime for todo output
    READ_OUT = "\033[38;5;130m"       # Brown/tan for read output
    PROMPT = "\033[38;5;156m"         # Light green for prompts
    S_PROMPT = "\033[38;5;140m"       # Purple for system prompts

    # OpenRouter specific colors
    OR_MSG = "\033[38;5;153m"         # Light blue for OR message
    OR_REASONING = "\033[38;5;135m"   # Purple for OR reasoning/thinking
    OR_TOOL_IN = "\033[38;5;245m"     # Grey for OR tool input
    OR_TOOL_OUT = "\033[38;5;28m"     # Dark green for OR tool output

    # Context tracking
    CONTEXT_USAGE = "\033[38;5;177m"  # Lavender/purple for context usage

    # Module/aggregation summaries
    MOD_SUM = "\033[38;5;81m"         # Light blue for module summary
    AGG_SUM = "\033[38;5;81m"         # Light blue for aggregated summary
    RUN_SUM = "\033[38;5;81m"         # Light blue for run summary

    # For JSON key highlighting
    JSON_KEY = "\033[38;5;81m"        # Light blue for JSON keys


# Export colors for backward compatibility
COLORS = Colors
RED = Colors.RED
BRIGHT_RED = Colors.BRIGHT_RED
YELLOW = Colors.YELLOW
GREEN = Colors.GREEN
BLUE = Colors.BLUE
CYAN = Colors.CYAN
RESET = Colors.RESET


def get_color(message_type: str) -> str:
    """Get color for a message type."""
    # Define color map with lowercase keys for case-insensitive lookup
    color_map = {
        # Logging levels (with short aliases)
        "info": Colors.INFO,
        "warning": Colors.WARNING,
        "warn": Colors.WARNING,  # Short alias
        "error": Colors.ERROR,
        "success": Colors.SUCCESS,
        "debug": Colors.DEBUG,
        # Schema validation / retry
        "retry": Colors.WARNING,
        "schema_error": Colors.ERROR,
        # LLM messages
        "system": Colors.SYSTEM,
        "s_prompt": Colors.S_PROMPT,
        "prompt": Colors.PROMPT,
        "user": Colors.PROMPT,
        "assistant": Colors.ASSISTANT,
        "thinking": Colors.THINKING,
        "claude_msg": Colors.ASSISTANT,
        # Tool messages (generic types)
        "tool_call": Colors.TOOL_CALL,
        "tool_result": Colors.TOOL_RESULT,
        "tool_input": Colors.TOOL_CALL,
        "tool_output": Colors.TOOL_RESULT,
        # Task lifecycle
        "task_in": Colors.TASK_IN,
        "task_out": Colors.TASK_OUT,
        "run_start": Colors.INFO,
        "run_end": Colors.SUCCESS,
        # Context tracking
        "context_usage": Colors.CONTEXT_USAGE,
        "ctx_use": Colors.CONTEXT_USAGE,
        # Summaries
        "summary": Colors.SUMMARY,
        "group_summary": Colors.SUMMARY,
        "module_summary": Colors.SUMMARY,
        "module_group_summary": Colors.SUMMARY,
        "pipeline_summary": Colors.SUMMARY,
        # Tool names (specific tools)
        "todo_in": Colors.TODO_IN,
        "todo_out": Colors.TODO_OUT,
        "bash_in": "\033[38;5;39m",   # Light blue
        "bash_out": "\033[38;5;202m", # Orange-red
        # File tools
        "read_in": Colors.TOOL_CALL,
        "read_out": Colors.READ_OUT,
        "writ_in": Colors.TOOL_CALL,
        "writ_out": Colors.TOOL_RESULT,
        "edit_in": Colors.TOOL_CALL,
        "edit_out": Colors.TOOL_RESULT,
        # Search tools
        "grep_in": Colors.TOOL_CALL,
        "grep_out": Colors.TOOL_RESULT,
        "glob_in": Colors.TOOL_CALL,
        "glob_out": Colors.TOOL_RESULT,
        # Web tools
        "srch_in": Colors.TOOL_CALL,
        "srch_out": Colors.TOOL_RESULT,
        "ftch_in": Colors.TOOL_CALL,
        "ftch_out": Colors.TOOL_RESULT,
        # MCP/HuggingFace tools (various abbreviations)
        "hf_in": "\033[38;5;220m",      # Gold
        "hf_out": Colors.TOOL_RESULT,
        "hf_d_in": "\033[38;5;220m",    # Gold (for hf_dataset tools)
        "hf_d_out": Colors.TOOL_RESULT,
        "hf_s_in": "\033[38;5;220m",    # Gold (for hf_search tools)
        "hf_s_out": Colors.TOOL_RESULT,
        # OWID tools
        "owid_in": "\033[38;5;39m",     # Light blue
        "owid_out": Colors.TOOL_RESULT,
        # Model message types
        "claude_m": Colors.ASSISTANT,
        "openai_m": "\033[38;5;153m",
        # Summary variants
        "mod_sum": Colors.MOD_SUM,
        "run_sum": Colors.RUN_SUM,
        "agg_sum": Colors.AGG_SUM,
        "grp_sum": Colors.AGG_SUM,
        # OpenRouter message types
        "or_msg": Colors.OR_MSG,
        "or_reasoning": Colors.OR_REASONING,
        "or_tool_in": Colors.OR_TOOL_IN,
        "or_tool_out": Colors.OR_TOOL_OUT,
        "or_think": Colors.OR_REASONING,
        "or_tl_in": Colors.OR_TOOL_IN,
        "or_tl_ou": Colors.OR_TOOL_OUT,
    }
    # Case-insensitive lookup
    return color_map.get(message_type.lower(), Colors.RESET)


def colorize(text: str, color: str) -> str:
    """Wrap text in color codes."""
    return f"{color}{text}{Colors.RESET}"


def colorize_json_keys(text: str, content_color: str) -> str:
    """Colorize JSON keys in text with a distinct color."""
    pattern = r'("[\w_-]+")(\s*:\s*)'

    def replace_key(match):
        key = match.group(1)
        colon_space = match.group(2)
        return f"{Colors.JSON_KEY}{key}{Colors.RESET}{content_color}{colon_space}"

    return re.sub(pattern, replace_key, text)


class ConsoleSink(Sink):
    """Colorized console output with formatting."""

    # Console sink uses sequencing for clean parallel task display
    bypass_sequencer: bool = False

    # Message types/tools that are always shown (even when log_messages=False)
    ALWAYS_SHOW_TYPES = {
        "group_summary", "module_summary", "module_group_summary", "pipeline_summary",
        "info", "success", "warning", "error", "debug",
        "prompt", "s_prompt",  # Always show prompts
        "retry", "schema_error",  # Always show retry and schema errors
        "context_usage",  # Always show live context tracking
    }
    ALWAYS_SHOW_TOOLS = {
        "TASK_IN", "TASK_OUT", "MOD_SUM", "MODGRP_SUM", "RUN_SUM",
        "NOVELTY", "FEASIB", "INFO", "STATUS", "VERIFY", "RETRY",
        "PROMPT", "S_PROMPT",  # Always show prompts
    }

    def __init__(
        self,
        truncation: int | None = 150,
        log_messages: bool = True,
    ):
        """
        Args:
            truncation: Max chars for content, None = no truncation
            log_messages: If False, only show summaries/task lifecycle/log levels
        """
        self.truncation = truncation
        self.log_messages = log_messages

    def emit(self, message: "TelemetryMessage") -> None:
        """Emit message to console with formatting."""
        # Convert to dict for compatibility with existing formatting
        msg_dict = message.to_dict()
        self._format_and_print(msg_dict)

    def emit_dict(self, message_dict: dict) -> None:
        """Emit from legacy dict format (for backward compatibility)."""
        self._format_and_print(message_dict)

    def _should_show(self, message_dict: dict) -> bool:
        """Check if message should be shown based on log_messages setting."""
        if self.log_messages:
            return True
        msg_type = message_dict.get("type", "")
        tool_name = message_dict.get("tool_name", "")
        return msg_type in self.ALWAYS_SHOW_TYPES or tool_name in self.ALWAYS_SHOW_TOOLS

    def _format_and_print(self, message_dict: dict) -> None:
        """Format a message dict for console output with colors."""
        if not self._should_show(message_dict):
            return

        message_type = message_dict.get("type", "unknown")
        message_text = message_dict.get("message_text", "")
        tool_name = message_dict.get("tool_name", "")
        tool_id = message_dict.get("tool_id", "")
        agent_context = message_dict.get("agent_context", "")
        is_error = message_dict.get("is_error", False)

        # Special handling for SYSTEM messages with details
        if message_type == "system" and message_dict.get("details"):
            message_text = self._format_system_message(message_dict)

        # Special handling for SUMMARY messages (including aggregated summaries)
        if message_type in ("summary", "group_summary", "module_summary", "module_group_summary", "pipeline_summary"):
            message_text = self._format_summary_message(message_dict)

        # Types/tools that should never be truncated
        # Log levels and summaries: never truncate (short, important status messages)
        # prompts/s_prompt: DO truncate (can be very long)
        no_truncate_types = [
            "info", "success", "warning", "error", "debug",
            "summary", "group_summary", "module_summary", "module_group_summary", "pipeline_summary",
            "context_usage",
        ]
        # Tool outputs that should never be truncated
        no_truncate_tools = ["TODO_IN", "TODO_OUT", "TASK_IN", "TASK_OUT"]
        should_truncate = (
            self.truncation
            and message_type not in no_truncate_types
            and tool_name not in no_truncate_tools
        )

        # Format JSON FIRST (before truncation) for tool outputs
        if (tool_name and tool_name.endswith("_OUT")) or message_type in ("tool_output", "tool_result"):
            message_text = format_json_output(message_text)

        # Then apply truncation
        if should_truncate and len(message_text) > self.truncation:
            truncated_amount = len(message_text) - self.truncation
            message_text = message_text[:self.truncation] + f"\n... (+{truncated_amount} chars truncated in console)"

        # Get display label and color
        display_label = tool_name if tool_name else message_type.upper()
        display_name_map = {
            "CLAUDE_MSG": "CLAUDE_M",
            "OPENAI_MSG": "OPENAI_M",
            "OR_TOOL_IN": "OR_TL_IN",
            "OR_TOOL_OUT": "OR_TL_OU",
            "OR_REASONING": "OR_THINK",
            "CONTEXT_USAGE": "CTX_USE",
        }
        display_name = display_name_map.get(display_label, display_label) or "UNKNOWN"

        # Get color for tool name, with fallback to message type or suffix-based default
        content_color = get_color(display_label)
        if content_color == Colors.RESET:
            # Try message type
            content_color = get_color(message_type)
        if content_color == Colors.RESET:
            # Fallback based on suffix (_IN = tool call, _OUT = tool result)
            if display_label.endswith("_IN"):
                content_color = Colors.TOOL_CALL
            elif display_label.endswith("_OUT"):
                content_color = Colors.TOOL_RESULT

        if is_error:
            message_text = f"Error: {message_text}"

        # Handle TASK_HEADER format - for TASK_IN, content is just the task name (already in agent_context)
        is_task_header = "TASK_HEADER:" in message_text
        if is_task_header:
            clean_content = message_text[12:]
            if "\n" in clean_content:
                header, prompt = clean_content.split("\n", 1)
                colorized_prompt = colorize_json_keys(prompt, content_color)
                formatted_content = f"{Colors.YELLOW}{header}{Colors.RESET}\n{content_color}{colorized_prompt}{Colors.RESET}"
            else:
                # Just the task name - will be shown in agent_context column, no need to repeat
                formatted_content = ""
        else:
            # JSON formatting already done earlier (before truncation)
            colorized_text = colorize_json_keys(message_text, content_color)
            formatted_content = f"{content_color}{colorized_text}{Colors.RESET}"

        # Build output line: TYPE|AGENT| content
        # For WARN/ERROR, use same color for entire line (not bright red for agent_context)
        is_log_level = display_label.upper() in ("WARN", "WARNING", "ERROR", "SUCCESS", "INFO", "DEBUG")
        agent_color = content_color if is_log_level else Colors.BRIGHT_RED

        if agent_context:
            line = f"{content_color}{display_name: <8}{Colors.RESET}|{agent_color}{agent_context}{Colors.RESET}| {formatted_content}\n"
        else:
            line = f"{content_color}{display_name: <8}{Colors.RESET}| {formatted_content}\n"

        print(line, end="", flush=True)

    def _format_system_message(self, message_dict: dict) -> str:
        """Format system message with model details."""
        # Model is at top level, not inside details
        model = message_dict.get("model") or "unknown"
        return f"Model: {model}"

    def _format_summary_message(self, message_dict: dict) -> str:
        """
        Format summary message with metrics using standardized SummaryMetrics format.
        """
        # Merge message_metadata into top level (for emit() via TelemetryMessage)
        metadata = message_dict.get("message_metadata") or {}
        data = {**message_dict, **metadata}

        # Costs
        total_cost = data.get("total_cost", 0.0) or 0.0
        token_cost = data.get("token_cost", total_cost) or total_cost
        tool_cost = data.get("tool_cost", 0.0) or 0.0

        # Model and status
        model = data.get("model", "")
        status = data.get("status", "")
        is_aggregated = data.get("is_aggregated", False)
        is_error = data.get("is_error", False)

        # Timing
        runtime_seconds = data.get("runtime_seconds", 0.0) or 0.0
        llm_time_seconds = data.get("llm_time_seconds", 0.0) or 0.0
        num_calls = data.get("num_calls", 1) or 1

        # Tokens
        input_tokens = data.get("input_tokens", 0) or 0
        output_tokens = data.get("output_tokens", 0) or 0
        reasoning_tokens = data.get("reasoning_tokens", 0) or 0
        cache_write = data.get("cache_write_tokens")
        cache_read = data.get("cache_read_tokens")

        # Tool calls
        tool_calls = data.get("tool_calls", {}) or {}
        tool_costs_dict = data.get("tool_costs", {}) or {}

        # Build output lines
        lines = [""]  # Blank line before summary

        # Line 1: Total cost with breakdown
        if tool_cost > 0:
            lines.append(f"Total: ${total_cost:.4f} (Tokens: ${token_cost:.4f} + Tools: ${tool_cost:.4f})")
        else:
            lines.append(f"Total: ${total_cost:.4f}")

        # Line 2: Model | Status or Calls
        status_str = f"{status} (error)" if is_error else status
        if is_aggregated:
            lines.append(f"Model: {model} | Calls: {num_calls}")
        else:
            lines.append(f"Model: {model} | Status: {status_str}")

        # Line 3: Runtime
        runtime_str = self._format_time(runtime_seconds)
        llm_time_str = self._format_time(llm_time_seconds)
        if is_aggregated:
            avg_time = llm_time_seconds / num_calls if num_calls > 0 else 0
            avg_str = self._format_time(avg_time)
            lines.append(f"Runtime: {runtime_str} | LLM Time: {llm_time_str} | Avg/Call: {avg_str}")
        else:
            lines.append(f"Runtime: {runtime_str} | Turns: {num_calls}")
        lines.append("")

        # Line 4: Tokens
        # For individual summaries: context_window_used (last turn's context window size)
        # For aggregated summaries: total input = uncached + cache_write + cache_read
        total_in = data.get("context_window_used", 0) or 0
        if not total_in:
            total_in = input_tokens + (cache_write or 0) + (cache_read or 0)
        token_line = f"Ctx In: {total_in:,} | Out: {output_tokens:,}"
        if reasoning_tokens > 0:
            token_line += f" | Reasoning: {reasoning_tokens:,}"
        if token_cost > 0:
            token_line += f" | Cost: ${token_cost:.4f}"
        lines.append(token_line)

        # Line 5: Cache breakdown
        cache_write_str = f"{cache_write:,}" if cache_write is not None else "N/A"
        cache_read_str = f"{cache_read:,}" if cache_read is not None else "N/A"
        uncached_str = f"{input_tokens:,}"
        lines.append(f"Uncached: {uncached_str} | Cache write: {cache_write_str} | Cache read: {cache_read_str}")

        # Line 6: Tool calls
        active_tools = {k: v for k, v in tool_calls.items() if v > 0}
        if active_tools:
            tool_strs = [f"{name}: {count}" for name, count in sorted(active_tools.items())]
            lines.append(f"Tools: {', '.join(tool_strs)}")
        else:
            lines.append("Tools: N/A")

        # Line 7: Tool costs (if any)
        if tool_costs_dict:
            cost_parts = []
            for tool_name, cost_info in tool_costs_dict.items():
                if isinstance(cost_info, dict):
                    count = cost_info.get("count", 0)
                    unit = cost_info.get("unit", 0)
                    total = cost_info.get("total", 0)
                    cost_parts.append(f"{tool_name}: {count} × ${unit:.4f} = ${total:.4f}")
            if cost_parts:
                lines.append(f"Tool Cost: {', '.join(cost_parts)}")

        return "\n".join(lines)

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds <= 0:
            return "N/A"
        if seconds >= 60:
            return f"{seconds / 60:.1f}m"
        return f"{seconds:.0f}s"


__all__ = [
    "ConsoleSink",
    "Colors",
    "COLORS",
    "RED",
    "BRIGHT_RED",
    "YELLOW",
    "GREEN",
    "BLUE",
    "CYAN",
    "RESET",
    "get_color",
    "colorize",
]
