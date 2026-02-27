"""
Execution - main SDK streaming loop.
"""

import asyncio
from typing import Optional

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    UserMessage,
    SystemMessage,
    TextBlock,
)
from claude_agent_sdk.types import StreamEvent
from claude_agent_sdk._errors import ProcessError

from ..models import TokenUsage
from ..utils.execution.sdk_client import StreamingExecutor, AgentProcessError, SubscriptionAccessError
from ..utils.execution.message_parser import (
    parse_assistant_message,
    parse_result_message,
    parse_user_message,
    parse_system_message,
)
from ..utils.context_stats import ContextTracker
from aii_lib.telemetry import AIITelemetry, MessageType


class MessageTimeoutError(Exception):
    """Raised when a single SDK message exceeds message_timeout.

    Distinct from asyncio.TimeoutError so agent.py can handle it separately
    with its own retry budget (message_retries) before escalating to
    seq_prompt_retries.
    """
    pass


# Patterns that indicate subscription/access is unavailable (checked case-insensitive)
_SUBSCRIPTION_ERROR_PATTERNS = [
    "does not have access to claude",
    "organization does not have access",
]


def _is_subscription_error_message(message: AssistantMessage) -> bool:
    """Check if an AssistantMessage indicates a subscription/access error.

    Detects via the SDK error field (authentication_failed) or by matching
    known error text patterns in TextBlock content.
    """
    # Check error field (SDK sets this for authentication errors)
    if getattr(message, 'error', None) == "authentication_failed":
        return True

    # Check text content for known patterns
    for block in message.content:
        if isinstance(block, TextBlock):
            text_lower = block.text.lower()
            for pattern in _SUBSCRIPTION_ERROR_PATTERNS:
                if pattern in text_lower:
                    return True
    return False


async def execute_prompt_streaming(
    prompt: str,
    sdk_options,  # ClaudeAgentOptions (pre-built in agent.py)
    telemetry: AIITelemetry,
    execution_state: dict,
    emit_summary: bool = True,
    message_timeout: int | None = None,
) -> tuple[str, str, float, TokenUsage, dict, int, dict | None]:
    """
    Execute a single prompt using SDK streaming.

    Args:
        prompt: The prompt to execute
        sdk_options: Pre-built ClaudeAgentOptions from config_builder
        telemetry: AIITelemetry instance for logging
        execution_state: Dict with prompt_index, current_model, etc.
        emit_summary: Whether to emit summary immediately (False for multi-prompt aggregation)

    Returns:
        Tuple of (response_text, session_id, cost, usage, summary_data, num_turns, structured_output)
    """
    prompt_index = execution_state["prompt_index"]

    # Message callback - emit to telemetry
    # Inject run_id and agent_context for sequenced parallel execution
    run_id = execution_state.get("run_id")
    agent_context = execution_state.get("agent_context")

    def message_callback(msg_dict: dict):
        execution_state["message_count"] += 1
        if run_id:
            msg_dict["run_id"] = run_id
        if agent_context:
            msg_dict["agent_context"] = agent_context
        telemetry.emit_dict(msg_dict)

    # Create executor
    executor = StreamingExecutor(sdk_options)

    # Initialize return values
    response_text = ""
    session_id = ""
    cost = 0.0
    usage = TokenUsage()
    summary_data = {}
    num_turns = 0
    structured_output = None

    # Track tool IDs across messages
    last_tool_id: Optional[str] = None
    last_tool_name: Optional[str] = None

    # Track Task tool invocations for subagent identification
    tool_id_to_agent_name: dict[str, str] = {}

    # Track all tool use IDs to names (for matching results to tools)
    tool_id_to_tool_name: dict[str, str] = {}

    # Track tool calls for summary metrics {tool_name: count}
    # Stored in execution_state so it survives timeout (dict is mutable)
    tool_calls_count: dict[str, int] = {}
    execution_state["_tool_calls_count"] = tool_calls_count

    # Deduplication: track seen tool result IDs to prevent duplicates
    # The SDK sometimes sends the same tool result in both AssistantMessage and UserMessage
    seen_tool_result_ids: set[str] = set()

    # Live context window tracking (from StreamEvent message_start events)
    # Carry over turn/cost from previous session (e.g., fork after timeout)
    context_tracker = ContextTracker(
        model=execution_state.get("current_model", ""),
        initial_turn=execution_state.get("_ctx_turn", 0),
        initial_cost=execution_state.get("_ctx_cost", 0.0),
        initial_context_used=execution_state.get("_ctx_window_used", 0),
    )
    context_tracker.bind_execution_state(execution_state)

    # Main execution loop - stream messages from SDK
    # When message_timeout is set, the SDK iterator runs in a background task
    # communicating via asyncio.Queue. This avoids cancelling anyio-managed
    # coroutines directly (which breaks anyio's cancel scope tracking).
    # The queue.get() timeout is pure asyncio and safe to interrupt.

    _SENTINEL_DONE = object()

    if message_timeout is not None:
        # --- Queue-based iteration with per-message timeout ---
        msg_queue: asyncio.Queue = asyncio.Queue()
        iter_error: list = []  # Mutable container to capture iterator exceptions

        async def _run_sdk_iterator():
            """Background task: pump SDK messages into the queue."""
            try:
                async for msg in executor.execute(prompt):
                    await msg_queue.put(msg)
                await msg_queue.put(_SENTINEL_DONE)
            except Exception as exc:
                iter_error.append(exc)
                await msg_queue.put(_SENTINEL_DONE)

        iter_task = asyncio.create_task(_run_sdk_iterator())
        try:
            while True:
                try:
                    message = await asyncio.wait_for(msg_queue.get(), timeout=message_timeout)
                except asyncio.TimeoutError:
                    # No SDK message within message_timeout — trigger fork+resume
                    _run_id = execution_state.get("run_id")
                    _agent_ctx = execution_state.get("agent_context")
                    if _run_id and _agent_ctx:
                        telemetry.emit_message("WARNING", f"Message-level timeout ({message_timeout}s) — triggering fork+resume", _agent_ctx, _run_id)
                    else:
                        telemetry.emit(MessageType.WARNING,f"Message-level timeout ({message_timeout}s) — triggering fork+resume")
                    raise MessageTimeoutError(f"No SDK message received for {message_timeout}s")

                if message is _SENTINEL_DONE:
                    # Check if iterator ended with an error
                    if iter_error:
                        exc = iter_error[0]
                        if isinstance(exc, ProcessError):
                            error_msg = f"Agent subprocess terminated (will retry): {exc}"
                            _run_id = execution_state.get("run_id")
                            _agent_ctx = execution_state.get("agent_context")
                            if _run_id and _agent_ctx:
                                telemetry.emit_message("WARNING", error_msg, _agent_ctx, _run_id)
                            else:
                                telemetry.emit(MessageType.WARNING,error_msg)
                            raise AgentProcessError(error_msg) from exc
                        raise exc
                    break

                # --- Process message (same logic as non-timeout path) ---
                if isinstance(message, SystemMessage):
                    early_session_id, model = parse_system_message(
                        message, prompt_index, message_callback,
                        system_prompt=sdk_options.system_prompt,
                    )
                    if early_session_id and not session_id:
                        session_id = early_session_id
                        execution_state["session_id"] = early_session_id
                    if model:
                        execution_state["current_model"] = model
                        context_tracker.set_model(model)
                elif isinstance(message, AssistantMessage):
                    # Check for subscription/access error BEFORE parsing
                    # (parse logs 1 message, then we raise to stop the spam)
                    is_sub_error = _is_subscription_error_message(message)
                    last_tool_id, last_tool_name = parse_assistant_message(
                        message, prompt_index, message_callback,
                        last_tool_id, last_tool_name,
                        tool_id_to_agent_name, tool_id_to_tool_name,
                        model=execution_state["current_model"],
                        tool_calls_count=tool_calls_count,
                        seen_tool_result_ids=seen_tool_result_ids,
                    )
                    if is_sub_error:
                        error_text = " ".join(
                            block.text for block in message.content
                            if isinstance(block, TextBlock)
                        )
                        raise SubscriptionAccessError(error_text)
                elif isinstance(message, UserMessage):
                    last_tool_id, last_tool_name = parse_user_message(
                        message, prompt_index, message_callback,
                        last_tool_id, last_tool_name,
                        tool_id_to_agent_name, tool_id_to_tool_name,
                        seen_tool_result_ids=seen_tool_result_ids,
                    )
                elif isinstance(message, ResultMessage):
                    response_text, session_id, cost, usage, summary_data, num_turns, structured_output = parse_result_message(
                        message, prompt_index, message_callback,
                        module_start_time=execution_state["module_start_time"],
                        message_count=execution_state["message_count"],
                        model=execution_state["current_model"],
                        tool_calls_count=tool_calls_count,
                        emit_summary=False,
                    )
                    if summary_data:
                        summary_data["context_window_used"] = execution_state.get("_ctx_window_used", 0)
                    if emit_summary and summary_data:
                        message_callback(summary_data)
                elif isinstance(message, StreamEvent):
                    ctx_dict = context_tracker.process_stream_event(message)
                    if ctx_dict:
                        message_callback(ctx_dict)

        finally:
            # Clean up: cancel the background iterator task.
            # The SDK uses anyio internally — asyncio.Task.cancel() doesn't
            # reliably interrupt anyio coroutines, so use asyncio.wait with
            # a timeout to prevent the finally block from hanging indefinitely.
            if not iter_task.done():
                iter_task.cancel()
                try:
                    await asyncio.wait({iter_task}, timeout=10.0)
                except Exception:
                    pass

    else:
        # --- Standard iteration (no message_timeout) ---
        try:
            async for message in executor.execute(prompt):
                if isinstance(message, SystemMessage):
                    early_session_id, model = parse_system_message(
                        message, prompt_index, message_callback,
                        system_prompt=sdk_options.system_prompt,
                    )
                    if early_session_id and not session_id:
                        session_id = early_session_id
                        execution_state["session_id"] = early_session_id
                    if model:
                        execution_state["current_model"] = model
                        context_tracker.set_model(model)
                elif isinstance(message, AssistantMessage):
                    # Check for subscription/access error BEFORE parsing
                    is_sub_error = _is_subscription_error_message(message)
                    last_tool_id, last_tool_name = parse_assistant_message(
                        message, prompt_index, message_callback,
                        last_tool_id, last_tool_name,
                        tool_id_to_agent_name, tool_id_to_tool_name,
                        model=execution_state["current_model"],
                        tool_calls_count=tool_calls_count,
                        seen_tool_result_ids=seen_tool_result_ids,
                    )
                    if is_sub_error:
                        error_text = " ".join(
                            block.text for block in message.content
                            if isinstance(block, TextBlock)
                        )
                        raise SubscriptionAccessError(error_text)
                elif isinstance(message, UserMessage):
                    last_tool_id, last_tool_name = parse_user_message(
                        message, prompt_index, message_callback,
                        last_tool_id, last_tool_name,
                        tool_id_to_agent_name, tool_id_to_tool_name,
                        seen_tool_result_ids=seen_tool_result_ids,
                    )
                elif isinstance(message, ResultMessage):
                    response_text, session_id, cost, usage, summary_data, num_turns, structured_output = parse_result_message(
                        message, prompt_index, message_callback,
                        module_start_time=execution_state["module_start_time"],
                        message_count=execution_state["message_count"],
                        model=execution_state["current_model"],
                        tool_calls_count=tool_calls_count,
                        emit_summary=False,
                    )
                    if summary_data:
                        summary_data["context_window_used"] = execution_state.get("_ctx_window_used", 0)
                    if emit_summary and summary_data:
                        message_callback(summary_data)
                elif isinstance(message, StreamEvent):
                    ctx_dict = context_tracker.process_stream_event(message)
                    if ctx_dict:
                        message_callback(ctx_dict)

        except ProcessError as e:
            error_msg = f"Agent subprocess terminated (will retry): {e}"
            _run_id = execution_state.get("run_id")
            _agent_ctx = execution_state.get("agent_context")
            if _run_id and _agent_ctx:
                telemetry.emit_message("WARNING", error_msg, _agent_ctx, _run_id)
            else:
                telemetry.emit(MessageType.WARNING,error_msg)
            raise AgentProcessError(error_msg) from e

    return response_text, session_id, cost, usage, summary_data, num_turns, structured_output
