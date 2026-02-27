"""
Live context window tracking from StreamEvent usage data.

Extracts per-turn token usage from Anthropic API stream events
(message_start for input, message_delta for output/stop_reason)
and calculates context window consumption + cumulative cost.
"""

from claude_agent_sdk.types import StreamEvent

# All current Claude models have 200K context window
_MODEL_CONTEXT_WINDOWS = {
    "claude-opus-4-6": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-5-20250514": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
}
_DEFAULT_CONTEXT_WINDOW = 200_000

# Pricing per million tokens (USD)
# From: https://platform.claude.com/docs/en/build-with-claude/prompt-caching#pricing
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {
        "input": 5.0, "cache_write": 6.25, "cache_read": 0.50, "output": 25.0,
    },
    "claude-opus-4-20250514": {
        "input": 5.0, "cache_write": 6.25, "cache_read": 0.50, "output": 25.0,
    },
    "claude-sonnet-4-5-20250514": {
        "input": 3.0, "cache_write": 3.75, "cache_read": 0.30, "output": 15.0,
    },
    "claude-sonnet-4-5-20250929": {
        "input": 3.0, "cache_write": 3.75, "cache_read": 0.30, "output": 15.0,
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.0, "cache_write": 1.25, "cache_read": 0.10, "output": 5.0,
    },
}
_DEFAULT_PRICING = {"input": 3.0, "cache_write": 3.75, "cache_read": 0.30, "output": 15.0}


def get_context_window_size(model: str) -> int:
    """Get context window size for a model."""
    return _MODEL_CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)


def _get_pricing(model: str) -> dict[str, float]:
    """Get pricing per MTok for a model."""
    return _MODEL_PRICING.get(model, _DEFAULT_PRICING)


class ContextTracker:
    """Tracks live context window usage from StreamEvent objects.

    Processes two event types:
    - `message_start`: Full input token breakdown. Adds input cost to cumulative.
      Emits context_usage message.
    - `message_delta`: Output token count + stop_reason. Adds output cost to
      cumulative. Stored for display on next turn's line.

    Cost is accumulated correctly: input cost at message_start, output cost at
    message_delta, so each turn's cost uses its own tokens (not mixed turns).
    """

    def __init__(self, model: str = "", initial_turn: int = 0, initial_cost: float = 0.0, initial_context_used: int = 0):
        self._model = model
        self._context_window = get_context_window_size(model)
        self._pricing = _get_pricing(model)
        self._turn = initial_turn
        self._prev_context_used = initial_context_used
        # Context from last fully completed turn (message_delta received).
        # Used for fork carry-over: if timeout interrupts mid-turn, the fork
        # only has context up to the last completed turn, not the interrupted one.
        self._last_completed_context_used = initial_context_used
        self._cumulative_cost = initial_cost
        # From previous turn's message_delta
        self._prev_output_tokens = 0
        self._prev_stop_reason = ""
        # Cumulative token counts across all turns (for timeout carry-over)
        self._cum_input_tokens = 0
        self._cum_output_tokens = 0
        self._cum_cache_write_tokens = 0
        self._cum_cache_read_tokens = 0
        # Mutable dict ref for persisting state across timeouts
        self._execution_state: dict | None = None

    def bind_execution_state(self, execution_state: dict) -> None:
        """Bind to execution_state dict to persist tracker state across timeouts."""
        self._execution_state = execution_state

    def _persist_state(self) -> None:
        """Save current state to execution_state (survives coroutine cancellation)."""
        if self._execution_state is not None:
            self._execution_state["_ctx_turn"] = self._turn
            self._execution_state["_ctx_cost"] = self._cumulative_cost
            self._execution_state["_ctx_input_tokens"] = self._cum_input_tokens
            self._execution_state["_ctx_output_tokens"] = self._cum_output_tokens
            self._execution_state["_ctx_cache_write_tokens"] = self._cum_cache_write_tokens
            self._execution_state["_ctx_cache_read_tokens"] = self._cum_cache_read_tokens
            # Use last COMPLETED turn's context for fork carry-over.
            # If timeout interrupts mid-turn (after message_start but before
            # message_delta), _prev_context_used reflects the interrupted turn's
            # input — but the fork won't include that turn, so using it causes
            # a false ~8k context drop in the fork's first CTX_USE line.
            self._execution_state["_ctx_window_used"] = self._last_completed_context_used

    def set_model(self, model: str) -> None:
        """Update model (and context window size) when discovered from SystemMessage."""
        self._model = model
        self._context_window = get_context_window_size(model)
        self._pricing = _get_pricing(model)

    def process_stream_event(self, event: StreamEvent) -> dict | None:
        """Process a StreamEvent and return a telemetry dict if it has usage data.

        - `message_start`: Adds input cost, emits context_usage with stats.
        - `message_delta`: Adds output cost, captures output_tokens + stop_reason.

        Returns:
            Dict ready for message_callback() emission, or None.
        """
        raw = event.event
        event_type = raw.get("type")

        # End of turn: add output cost + capture for display
        if event_type == "message_delta":
            usage = raw.get("usage", {})
            if usage:
                output_tokens = usage.get("output_tokens", 0)
                self._prev_output_tokens = output_tokens
                self._cum_output_tokens += output_tokens
                # Add output cost for THIS turn
                self._cumulative_cost += output_tokens * self._pricing["output"] / 1_000_000
                # Mark this turn's context as "completed" — safe baseline for fork
                self._last_completed_context_used = self._prev_context_used
                self._persist_state()
            delta = raw.get("delta", {})
            if delta:
                self._prev_stop_reason = delta.get("stop_reason", "")
            return None

        if event_type != "message_start":
            return None

        # Start of turn: extract input token breakdown
        usage = raw.get("message", {}).get("usage", {})
        if not usage:
            return None

        input_tokens = usage.get("input_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)

        # Accumulate per-turn token counts (for timeout carry-over)
        self._cum_input_tokens += input_tokens
        self._cum_cache_write_tokens += cache_creation
        self._cum_cache_read_tokens += cache_read

        # Total context = all input tokens (uncached + cache write + cache read)
        # Per Anthropic docs: total_input = input_tokens + cache_creation + cache_read
        # input_tokens = tokens AFTER last cache breakpoint (not all input)
        context_used = input_tokens + cache_creation + cache_read
        growth = context_used - self._prev_context_used if self._prev_context_used > 0 else 0

        self._turn += 1
        prev_output = self._prev_output_tokens
        prev_stop = self._prev_stop_reason
        self._prev_context_used = context_used
        self._prev_output_tokens = 0
        self._prev_stop_reason = ""

        # Add input cost for THIS turn
        input_cost = (
            input_tokens * self._pricing["input"]
            + cache_creation * self._pricing["cache_write"]
            + cache_read * self._pricing["cache_read"]
        ) / 1_000_000
        self._cumulative_cost += input_cost
        self._persist_state()

        pct = (context_used / self._context_window * 100) if self._context_window > 0 else 0

        return {
            "type": "context_usage",
            "message_text": _format_context_line(
                turn=self._turn,
                context_used=context_used,
                context_window=self._context_window,
                pct=pct,
                growth=growth,
                prev_output=prev_output,
                cumulative_cost=self._cumulative_cost,
            ),
            "tool_name": "",
            "tool_id": "",
            "agent_context": "",
            "subagent_id": None,
            "parent_tool_use_id": getattr(event, "parent_tool_use_id", None),
            "is_error": False,
            "message_metadata": {
                "turn": self._turn,
                "input_tokens": input_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "prev_output_tokens": prev_output,
                "prev_stop_reason": prev_stop,
                "cumulative_cost_usd": round(self._cumulative_cost, 6),
                "context_used": context_used,
                "context_window": self._context_window,
                "context_pct": round(pct, 1),
                "context_growth": growth,
                "model": self._model,
            },
        }


def _format_context_line(
    turn: int,
    context_used: int,
    context_window: int,
    pct: float,
    growth: int,
    prev_output: int = 0,
    cumulative_cost: float = 0.0,
) -> str:
    """Format a compact context usage line for console display."""
    ctx = f"T{turn}: {context_used:,}/200K ({pct:.1f}%)"
    if growth > 0:
        ctx += f" [+{growth:,}]"
    elif growth < 0:
        ctx += f" [{growth:,}]"
    parts = [ctx]
    if prev_output > 0:
        parts.append(f"Out: {prev_output:,}")
    if cumulative_cost > 0:
        parts.append(f"Tok: ${cumulative_cost:.4f}")
    return " | ".join(parts)
