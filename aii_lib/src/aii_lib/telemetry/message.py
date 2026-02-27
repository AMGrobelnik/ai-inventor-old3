"""
Message types and dataclasses for telemetry.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MessageType(Enum):
    """Standardized message types."""

    # Logging levels
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    DEBUG = "debug"

    # LLM Messages
    SYSTEM = "system"
    S_PROMPT = "s_prompt"  # System prompt (for logging full prompts)
    USER = "user"
    PROMPT = "prompt"
    ASSISTANT = "assistant"
    THINKING = "thinking"
    CLAUDE_MSG = "claude_msg"

    # Tool messages
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"

    # Task/Agent lifecycle
    TASK_IN = "task_in"
    TASK_OUT = "task_out"
    RUN_START = "run_start"
    RUN_END = "run_end"

    # Summaries
    SUMMARY = "summary"
    GROUP_SUMMARY = "group_summary"
    MODULE_SUMMARY = "module_summary"
    MODULE_GROUP_SUMMARY = "module_group_summary"
    PIPELINE_SUMMARY = "pipeline_summary"

    # Context tracking
    CONTEXT_USAGE = "context_usage"

    # Module output (final results)
    MODULE_OUTPUT = "module_output"


@dataclass
class TelemetryMessage:
    """Standard message format for telemetry."""

    type: MessageType | str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Context
    module: str | None = None
    group: str | None = None
    run_id: str | None = None
    agent_context: str | None = None

    # Tool info (for tool messages)
    tool_name: str | None = None
    tool_id: str | None = None

    # Metadata
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        type_val = self.type.value if isinstance(self.type, MessageType) else self.type
        return {
            "type": type_val,
            "message_text": self.content,
            "iso_timestamp": self.timestamp.isoformat(),
            "module": self.module,
            "group": self.group,
            "run_id": self.run_id,
            "agent_context": self.agent_context,
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "message_metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TelemetryMessage":
        """Create from dictionary (e.g., legacy message format)."""
        # Handle type conversion
        type_val = data.get("type", "info")
        try:
            msg_type = MessageType(type_val)
        except ValueError:
            msg_type = type_val  # Keep as string if not in enum

        # Handle timestamp
        ts = data.get("iso_timestamp")
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts)
        else:
            timestamp = datetime.now()

        return cls(
            type=msg_type,
            content=data.get("message_text", ""),
            timestamp=timestamp,
            module=data.get("module"),
            group=data.get("group"),
            run_id=data.get("run_id"),
            agent_context=data.get("agent_context"),
            tool_name=data.get("tool_name"),
            tool_id=data.get("tool_id"),
            metadata=data.get("message_metadata"),
        )


@dataclass
class SummaryMetrics:
    """
    Standardized summary metrics - ALL LLM clients must emit this format.

    Each client (OpenRouter, Anthropic, Agent SDK) calculates these differently
    but emits the same standardized fields via callback.
    """

    # Costs (USD)
    total_cost: float = 0.0          # token_cost + tool_cost
    token_cost: float = 0.0          # cost from LLM tokens
    tool_cost: float = 0.0           # cost from tool calls (web search, etc.)

    # Model/Status
    model: str = ""                  # e.g., "openai/gpt-5-mini"
    status: str = "completed"        # "completed", "failed", "stop", "aggregated"
    is_aggregated: bool = False      # True for group/module/pipeline summaries
    num_calls: int = 1               # number of LLM calls (>1 for aggregated)

    # Timing (seconds)
    runtime_seconds: float = 0.0     # wall clock time
    llm_time_seconds: float = 0.0    # actual LLM processing time

    # Tokens
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0        # for reasoning models (o1, etc.)
    cache_write_tokens: int = 0      # Anthropic cache creation
    cache_read_tokens: int = 0       # Anthropic cache read

    # Tools
    tool_calls: dict = field(default_factory=dict)   # {tool_name: count}
    tool_costs: dict = field(default_factory=dict)   # {tool_name: {"count": N, "unit": $, "total": $}}

    def to_dict(self) -> dict:
        """Convert to dict for message_callback emission."""
        return {
            "type": "summary",
            "total_cost": self.total_cost,
            "token_cost": self.token_cost,
            "tool_cost": self.tool_cost,
            "model": self.model,
            "status": self.status,
            "is_aggregated": self.is_aggregated,
            "num_calls": self.num_calls,
            "runtime_seconds": self.runtime_seconds,
            "llm_time_seconds": self.llm_time_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "tool_calls": self.tool_calls,
            "tool_costs": self.tool_costs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SummaryMetrics":
        """Create from dict (e.g., from message_callback)."""
        return cls(
            total_cost=data.get("total_cost", 0.0),
            token_cost=data.get("token_cost", 0.0),
            tool_cost=data.get("tool_cost", 0.0),
            model=data.get("model", ""),
            status=data.get("status", "completed"),
            is_aggregated=data.get("is_aggregated", False),
            num_calls=data.get("num_calls", 1),
            runtime_seconds=data.get("runtime_seconds", 0.0),
            llm_time_seconds=data.get("llm_time_seconds", 0.0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            tool_calls=data.get("tool_calls", {}),
            tool_costs=data.get("tool_costs", {}),
        )


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for summaries."""

    total_runs: int = 0
    completed: int = 0
    failed: int = 0
    total_cost: float = 0.0
    total_tokens: dict = field(default_factory=lambda: {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cached": 0,
    })
    tool_calls: dict[str, int] = field(default_factory=dict)
    wall_time_ms: float = 0.0
    llm_time_ms: float = 0.0


__all__ = [
    "MessageType",
    "TelemetryMessage",
    "SummaryMetrics",
    "AggregatedMetrics",
]
