"""
Main AIITelemetry class - central observability system.

Thread-safe and async-safe for parallel task execution.
Uses sequence numbers for deterministic task ordering even when
multiple tasks register simultaneously via asyncio.gather().
"""

import contextvars
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from .message import MessageType, TelemetryMessage, AggregatedMetrics
from .sinks import Sink, ConsoleSink, JSONSink

# Context variable to track current buffered module per async task
# This allows parallel tasks to each have their own buffered module context
_current_buffered_module: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_buffered_module", default=None
)


def load_telemetry_config() -> dict:
    """Load telemetry config from aii_pipeline/config.yaml."""
    try:
        import yaml
        # Path: aii_lib/src/aii_lib/telemetry -> ../../../../aii_pipeline/config.yaml
        aii_lib_root = Path(__file__).parent.parent.parent.parent
        project_root = aii_lib_root.parent  # ai-inventor/
        config_path = project_root / "aii_pipeline" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return config.get("telemetry", {})
    except Exception as e:
        import sys
        print(f"WARNING: Failed to load telemetry config: {e}", file=sys.stderr)
    return {}


def create_telemetry(
    output_dir: Path | str,
    module_name: str,
    sequenced: bool = True,
) -> "AIITelemetry":
    """
    Create AIITelemetry with sinks configured from config.yaml.

    Args:
        output_dir: Directory for JSON log file
        module_name: Module name (used for log filename and context)
        sequenced: Buffer parallel output until task completes

    Returns:
        Configured AIITelemetry instance with ConsoleSink and JSONSink
    """
    config = load_telemetry_config()

    # Parse console truncation (can be int, False, or null)
    truncation_val = config.get("console_msg_truncate", 150)
    if truncation_val is False or truncation_val is None:
        truncation = None
    else:
        truncation = int(truncation_val)

    log_messages = config.get("log_messages", True)

    # Create telemetry with sinks
    output_dir = Path(output_dir)
    json_log_path = output_dir / f"{module_name.lower()}_messages.jsonl"

    telemetry = AIITelemetry(sequenced=sequenced, output_dir=output_dir)
    telemetry.add_sink(ConsoleSink(truncation=truncation, log_messages=log_messages))
    telemetry.add_sink(JSONSink(json_log_path))
    telemetry.start_module(module_name.upper())

    return telemetry


class AIITelemetry:
    """
    Central telemetry - single instance for pipeline.

    Handles:
    - Emitting messages to all sinks
    - Convenience methods (info, success, warning, error)
    - Run lifecycle (emit_run_start, emit_run_end)
    - Summary aggregation (group, module, pipeline levels)
    - Callback creation for streaming output
    - Sequencing parallel output (buffer per task, flush on complete)
    """

    def __init__(self, sequenced: bool = False, output_dir: Path | str | None = None):
        """
        Args:
            sequenced: If True, buffer parallel task output and flush on task end.
                      If False, emit immediately (default for backward compat).
            output_dir: Directory for module output files (optional).
        """
        self.sinks: list[Sink] = []
        self._context: dict = {}
        self._sequenced = sequenced
        self._output_dir = Path(output_dir) if output_dir else None

        # Sequencer: buffer messages per task_id, stream one task at a time
        # Uses sequence numbers for deterministic ordering even when tasks register
        # simultaneously via asyncio.gather()
        self._task_buffers: dict[str, list] = defaultdict(list)  # task_id -> [messages]
        self._active_tasks: set[str] = set()  # Tasks that are still running
        self._completed_tasks: set[str] = set()  # Tasks completed but not yet flushed
        self._task_sequence: dict[str, int] = {}  # task_id -> sequence number
        self._next_sequence: int = 0  # Monotonic counter for task ordering
        self._current_task_id: str | None = None  # Currently streaming task
        self._current_has_emitted: bool = False  # True once current task emits first message
        self._buffer_lock = threading.RLock()  # Reentrant lock for nested calls

        # Summary aggregation storage
        # _pending_summaries: individual run summaries waiting to be aggregated into module summary
        self._pending_summaries: list[dict] = []
        self._module_summaries: list[dict] = []
        self._module_group_summaries: dict[str, list[dict]] = defaultdict(list)  # parent_group -> [child summaries]

        # Timing
        self._start_time = datetime.now()
        self._module_start_times: dict[str, datetime] = {}
        self._module_group_start_times: dict[str, datetime] = {}

        # Module group stack for nesting (e.g., ["INVENTION_LOOP", "iter_1"])
        self._module_group_stack: list[str] = []

        # Module-level buffering for parallel modules
        # When multiple modules run in parallel (e.g., Track A and Track B),
        # each module's output is buffered separately and flushed as a block
        self._module_buffers: dict[str, list] = defaultdict(list)  # module_name -> [messages]
        self._buffered_modules: set[str] = set()  # Modules currently being buffered
        self._module_buffer_sequence: dict[str, int] = {}  # module_name -> sequence number
        self._next_module_sequence: int = 0  # Monotonic counter for module ordering
        self._current_streaming_module: str | None = None  # Module currently allowed to stream
        self._module_buffer_lock = threading.RLock()  # Lock for module-level buffering

    # === Sink management ===

    def add_sink(self, sink: Sink) -> "AIITelemetry":
        """Add a sink. Returns self for chaining."""
        self.sinks.append(sink)
        return self

    def remove_sink(self, sink: Sink) -> None:
        """Remove a sink."""
        if sink in self.sinks:
            self.sinks.remove(sink)

    # === Context management ===

    def set_context(self, **kwargs) -> None:
        """Set context fields (module, group, run_id, etc.)."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context."""
        self._context.clear()

    # === Core emit ===

    def _enrich_with_traceback(self, msg_type: MessageType | str, content: str) -> str:
        """Add traceback context to WARNING/ERROR messages if there's an active exception.

        Args:
            msg_type: Message type (MessageType enum or string)
            content: Original message content

        Returns:
            Enriched content with traceback if applicable, otherwise original content
        """
        import sys
        import traceback

        # Check if this is a WARNING or ERROR
        type_str = msg_type.value if isinstance(msg_type, MessageType) else str(msg_type).upper()
        if type_str not in ("WARNING", "ERROR", "WARN"):
            return content

        # Check if there's an active exception
        exc_info = sys.exc_info()
        if exc_info[0] is None:
            return content

        # Get short traceback (last 3 lines for context)
        tb_lines = traceback.format_exception(*exc_info)
        tb_short = "".join(tb_lines[-3:]).strip() if len(tb_lines) > 2 else "".join(tb_lines).strip()

        # Append traceback if not already in message
        if tb_short and tb_short not in content:
            return f"{content}\n{tb_short}"
        return content

    def emit(
        self,
        msg_type: MessageType | str,
        content: str,
        **metadata,
    ) -> None:
        """Emit a message to all sinks (or buffer if sequenced)."""
        # Enrich WARNING/ERROR with traceback context
        content = self._enrich_with_traceback(msg_type, content)

        message = TelemetryMessage(
            type=msg_type,
            content=content,
            timestamp=datetime.now(),
            module=self._context.get("module"),
            group=self._context.get("group"),
            run_id=metadata.pop("run_id", self._context.get("run_id")),
            agent_context=metadata.pop("agent_context", None),
            tool_name=metadata.pop("tool_name", None),
            tool_id=metadata.pop("tool_id", None),
            metadata=metadata if metadata else None,
        )

        self._emit_or_buffer(message)

    def _emit_or_buffer(self, message: TelemetryMessage) -> None:
        """Emit immediately or buffer based on sequenced mode and sink settings.

        Bypass sinks (bypass_sequencer=True, e.g., JSONSink): Always emit immediately.
        Sequenced sinks (bypass_sequencer=False, e.g., ConsoleSink): Use sequencing logic.

        For sequenced sinks (two levels of buffering):
        1. Module-level: When parallel modules run, non-current modules buffer.
        2. Task-level: Within a module, non-current tasks buffer.
        - ERROR messages always emit immediately.

        Thread-safe: uses RLock to protect shared state.
        """
        task_id = message.run_id

        # Always emit to bypass sinks immediately (e.g., JSON for crash recovery)
        self._emit_to_sinks(message, bypass_only=True)

        # Errors always emit immediately to sequenced sinks too
        msg_type = message.type.value if isinstance(message.type, MessageType) else str(message.type)
        if msg_type.lower() == "error":
            self._emit_to_sinks(message, sequenced_only=True)
            return

        # Module-level buffering: if message belongs to a non-current buffered module, buffer it
        if self._module_buffer_message(message):
            return  # Message was buffered at module level

        if self._sequenced and task_id:
            with self._buffer_lock:
                # Check membership inside lock to avoid race with emit_task_end
                if task_id not in self._active_tasks:
                    # Task not active (finished or never started), emit immediately
                    self._emit_to_sinks(message, sequenced_only=True)
                    return

                # Lazy selection: if no task has emitted yet, select lowest sequence as current
                if not self._current_has_emitted:
                    self._select_lowest_sequence_task()

                # Check if this is the current streaming task
                is_current = self._current_task_id == task_id
                if is_current:
                    # Mark that current task has started emitting
                    # (prevents swapping to a lower-sequence task mid-stream)
                    self._current_has_emitted = True
                    # Current task - emit immediately for real-time streaming
                    self._emit_to_sinks(message, sequenced_only=True)
                else:
                    # Not current task - buffer until it becomes current
                    self._task_buffers[task_id].append(("message", message))
        else:
            # Emit immediately to sequenced sinks (no sequencing or no task_id)
            self._emit_to_sinks(message, sequenced_only=True)

    def _emit_to_sinks(self, message: TelemetryMessage, sequenced_only: bool = False, bypass_only: bool = False) -> None:
        """Emit message to sinks.

        Args:
            message: Message to emit
            sequenced_only: If True, only emit to sinks with bypass_sequencer=False
            bypass_only: If True, only emit to sinks with bypass_sequencer=True
        """
        for sink in self.sinks:
            if sequenced_only and getattr(sink, 'bypass_sequencer', True):
                continue  # Skip bypass sinks
            if bypass_only and not getattr(sink, 'bypass_sequencer', True):
                continue  # Skip sequenced sinks
            sink.emit(message)

    def _emit_dict_to_sinks(self, message_dict: dict, sequenced_only: bool = False, bypass_only: bool = False) -> None:
        """Emit dict to sinks.

        Args:
            message_dict: Message dict to emit
            sequenced_only: If True, only emit to sinks with bypass_sequencer=False
            bypass_only: If True, only emit to sinks with bypass_sequencer=True
        """
        for sink in self.sinks:
            if sequenced_only and getattr(sink, 'bypass_sequencer', True):
                continue  # Skip bypass sinks
            if bypass_only and not getattr(sink, 'bypass_sequencer', True):
                continue  # Skip sequenced sinks
            if hasattr(sink, "emit_dict"):
                sink.emit_dict(message_dict)
            else:
                message = TelemetryMessage.from_dict(message_dict)
                sink.emit(message)

    def emit_dict(self, message_dict: dict) -> None:
        """Emit from legacy dict format (or buffer if sequenced).

        Bypass sinks (bypass_sequencer=True, e.g., JSONSink): Always emit immediately.
        Sequenced sinks (bypass_sequencer=False, e.g., ConsoleSink): Use sequencing logic.

        For sequenced sinks (two levels of buffering):
        1. Module-level: When parallel modules run, non-current modules buffer.
        2. Task-level: Within a module, non-current tasks buffer.
        - ERROR messages always emit immediately.

        Thread-safe: uses RLock to protect shared state.
        """
        # Inject root_dir from context into message_metadata (for relative path display)
        root_dir = self._context.get("root_dir")
        if root_dir:
            if "message_metadata" not in message_dict or message_dict["message_metadata"] is None:
                message_dict["message_metadata"] = {}
            message_dict["message_metadata"]["agent_cwd"] = root_dir

        task_id = message_dict.get("run_id")
        level = message_dict.get("level", "")
        msg_type = message_dict.get("type", "")

        # Always emit to bypass sinks immediately (e.g., JSON for crash recovery)
        self._emit_dict_to_sinks(message_dict, bypass_only=True)

        # Errors always emit immediately to sequenced sinks too
        if level.lower() == "error":
            self._emit_dict_to_sinks(message_dict, sequenced_only=True)
            return

        # Module-level buffering: if message belongs to a non-current buffered module, buffer it
        if self._module_buffer_dict(message_dict):
            return  # Message was buffered at module level

        if self._sequenced and task_id:
            with self._buffer_lock:
                # Check membership inside lock to avoid race with emit_task_end
                if task_id not in self._active_tasks:
                    # Task not active (finished or never started), emit immediately
                    self._emit_dict_to_sinks(message_dict, sequenced_only=True)
                    return

                # Lazy selection: if no task has emitted yet, select lowest sequence as current
                if not self._current_has_emitted:
                    self._select_lowest_sequence_task()

                # Check if this is the current streaming task
                is_current = self._current_task_id == task_id
                if is_current:
                    # Mark that current task has started emitting
                    self._current_has_emitted = True
                    # Current task - emit immediately for real-time streaming
                    self._emit_dict_to_sinks(message_dict, sequenced_only=True)
                else:
                    # Not current task - buffer until it becomes current
                    self._task_buffers[task_id].append(("dict", message_dict))
        else:
            self._emit_dict_to_sinks(message_dict, sequenced_only=True)

    def _select_lowest_sequence_task(self) -> None:
        """Select the task with lowest sequence number as current.

        Called lazily at first emit time to ensure deterministic ordering
        even when tasks register in non-deterministic order.
        Must be called while holding _buffer_lock.
        """
        if not self._active_tasks:
            self._current_task_id = None
            return

        # Find task with lowest sequence number
        best_task_id = None
        min_seq = float('inf')
        for tid in self._active_tasks:
            seq = self._task_sequence.get(tid, float('inf'))
            if seq < min_seq:
                min_seq = seq
                best_task_id = tid

        self._current_task_id = best_task_id

    # === Task lifecycle (for parallel task sequencing) ===

    def emit_task_start(self, task_id: str, task_name: str, sequence: int | None = None, **metadata) -> None:
        """Signal start of a task (registers for sequencing if enabled).

        First task to start (by sequence number) becomes the 'current' task and streams immediately.
        Subsequent tasks are queued and their messages buffered until they become current.

        Args:
            task_id: Unique identifier for this task
            task_name: Display name for the task
            sequence: Optional explicit sequence number for deterministic ordering.
                     If not provided, uses internal monotonic counter.
                     Lower sequence = higher priority (processed first).
            **metadata: Additional metadata to include in the TASK_IN message
        """
        # Register task for sequencing (lock protects _active_tasks and _current_task_id atomically)
        with self._buffer_lock:
            if self._sequenced:
                self._active_tasks.add(task_id)
                # Assign sequence number (explicit or auto-increment)
                if sequence is not None:
                    self._task_sequence[task_id] = sequence
                else:
                    self._task_sequence[task_id] = self._next_sequence
                    self._next_sequence += 1

                # Always select the task with lowest sequence number as current
                # This handles the case where tasks register in non-deterministic order
                self._update_current_task()

        # Emit task start immediately (not buffered)
        # Use TASK_HEADER: prefix for ConsoleSink to display just the task name
        self._emit_to_sinks(TelemetryMessage(
            type=MessageType.TASK_IN,
            content=f"TASK_HEADER:{task_name}",
            timestamp=datetime.now(),
            run_id=task_id,
            tool_name="TASK_IN",
            tool_id=task_id,  # For tool_id column display
            agent_context=task_name,
            module=self._context.get("module"),
            group=self._context.get("group"),
            metadata=metadata if metadata else None,
        ))

    def emit_task_end(self, task_id: str, task_name: str, status: str, **metadata) -> None:
        """Signal end of a task.

        If sequenced:
        1. For current task: flush buffer, emit TASK_OUT, promote next
        2. For non-current task: buffer TASK_OUT with messages, mark as completed
        """
        task_out_msg = TelemetryMessage(
            type=MessageType.TASK_OUT,
            content=status,
            timestamp=datetime.now(),
            run_id=task_id,
            tool_name="TASK_OUT",
            tool_id=task_id,  # For tool_id column display
            agent_context=task_name,
            module=self._context.get("module"),
            group=self._context.get("group"),
            metadata=metadata if metadata else None,
        )

        # Always emit TASK_OUT to bypass sinks immediately
        self._emit_to_sinks(task_out_msg, bypass_only=True)

        if self._sequenced:
            with self._buffer_lock:
                is_current = self._current_task_id == task_id

                if is_current:
                    # Current task: flush buffer to sequenced sinks, emit TASK_OUT, promote next
                    if task_id in self._task_buffers:
                        for msg_type, msg in self._task_buffers[task_id]:
                            if msg_type == "message":
                                self._emit_to_sinks(msg, sequenced_only=True)
                            else:  # dict
                                self._emit_dict_to_sinks(msg, sequenced_only=True)
                        del self._task_buffers[task_id]
                    self._active_tasks.discard(task_id)
                    self._task_sequence.pop(task_id, None)
                    self._current_has_emitted = False

                    # Emit TASK_OUT for current task to sequenced sinks
                    self._emit_to_sinks(task_out_msg, sequenced_only=True)
                else:
                    # Non-current task: buffer TASK_OUT for sequenced sinks, mark completed
                    self._task_buffers[task_id].append(("message", task_out_msg))
                    self._active_tasks.discard(task_id)
                    self._completed_tasks.add(task_id)
                    # Keep task_sequence for ordering when we flush later
                    return  # Don't promote yet

            # Promote next task (only for current task completion)
            self._promote_next_task()
        else:
            # Non-sequenced: emit immediately to sequenced sinks
            self._emit_to_sinks(task_out_msg, sequenced_only=True)

    def emit_message(
        self,
        label: MessageType | str,
        message: str,
        task_name: str,
        task_id: str,
    ) -> None:
        """Emit a message for a task, buffered with the task if sequenced mode is enabled.

        Args:
            label: MessageType or string label. Standard labels (INFO, WARNING, ERROR, SUCCESS, DEBUG)
                   map to their MessageType. Custom string labels (VERIFY, RETRY, PROMPT, etc.)
                   use MessageType.TOOL_OUTPUT with label as tool_name.
            message: Message content
            task_name: Display name for the task
            task_id: Task identifier for sequencer routing
        """
        # Handle MessageType directly
        if isinstance(label, MessageType):
            msg_type = label
            tool_name = None
        else:
            # Map standard string labels to MessageType, custom labels use TOOL_OUTPUT
            label_to_type = {
                "INFO": MessageType.INFO,
                "WARNING": MessageType.WARNING,
                "ERROR": MessageType.ERROR,
                "SUCCESS": MessageType.SUCCESS,
                "DEBUG": MessageType.DEBUG,
            }
            msg_type = label_to_type.get(label.upper(), MessageType.TOOL_OUTPUT)
            tool_name = label if msg_type == MessageType.TOOL_OUTPUT else None

        # Enrich WARNING/ERROR with traceback context
        message = self._enrich_with_traceback(label, message)

        msg = TelemetryMessage(
            type=msg_type,
            content=message,
            timestamp=datetime.now(),
            run_id=task_id,
            tool_name=tool_name,
            tool_id=task_id,
            agent_context=task_name,
            module=self._context.get("module"),
            group=self._context.get("group"),
        )
        self._emit_or_buffer(msg)

    def _update_current_task(self) -> None:
        """Update current task to be the one with lowest sequence number.

        Called when tasks register. Since we use lazy selection at first emit,
        this only sets current if no task has started emitting yet.
        Must be called while holding _buffer_lock.
        """
        # Only update if no task has started emitting
        # The actual selection happens lazily in _emit_or_buffer/_emit_dict
        if not self._current_has_emitted:
            self._select_lowest_sequence_task()

    def _promote_next_task(self) -> None:
        """Promote next task by sequence number: flush its buffered messages so it starts streaming.

        Called after TASK_OUT is emitted for the previous task.
        First processes any completed tasks in sequence order (they already finished
        but were waiting for their turn), then selects from active tasks.
        Thread-safe: uses RLock to protect shared state.
        """
        with self._buffer_lock:
            # First, flush any completed tasks in sequence order
            # These tasks finished early but their buffers were held
            while self._completed_tasks:
                # Find completed task with lowest sequence number
                next_completed = None
                min_seq = float('inf')
                for tid in self._completed_tasks:
                    seq = self._task_sequence.get(tid, float('inf'))
                    if seq < min_seq:
                        min_seq = seq
                        next_completed = tid

                if next_completed is None:
                    break

                # Check if there's an active task with lower sequence
                # If so, we need to wait for it to emit first
                has_lower_active = False
                for tid in self._active_tasks:
                    seq = self._task_sequence.get(tid, float('inf'))
                    if seq < min_seq:
                        has_lower_active = True
                        break

                if has_lower_active:
                    # Can't flush completed task yet, an active task comes before it
                    break

                # Flush this completed task's buffer to sequenced sinks only
                # (bypass sinks already received messages immediately)
                if next_completed in self._task_buffers:
                    buffer = self._task_buffers[next_completed]
                    # Check if buffer contains TASK_OUT (last entry for completed tasks)
                    has_task_out = any(
                        (mt == "message" and hasattr(m, 'type') and
                         (m.type == MessageType.TASK_OUT or (hasattr(m.type, 'value') and m.type.value == "task_out")))
                        for mt, m in buffer
                    )
                    for msg_type, msg in buffer:
                        if msg_type == "message":
                            self._emit_to_sinks(msg, sequenced_only=True)
                        else:  # dict
                            self._emit_dict_to_sinks(msg, sequenced_only=True)
                    del self._task_buffers[next_completed]
                else:
                    # Zero-message task: buffer was empty but TASK_OUT was appended
                    # in emit_task_end. If somehow the buffer is missing entirely,
                    # the task still completes correctly since TASK_OUT was already
                    # sent to bypass sinks in emit_task_end.
                    has_task_out = False

                # Clean up completed task
                self._completed_tasks.discard(next_completed)
                self._task_sequence.pop(next_completed, None)

            # Now check active tasks
            if not self._active_tasks:
                self._current_task_id = None
                self._current_has_emitted = False
                return

            # Find active task with lowest sequence number
            next_task_id = None
            min_seq = float('inf')
            for tid in self._active_tasks:
                seq = self._task_sequence.get(tid, float('inf'))
                if seq < min_seq:
                    min_seq = seq
                    next_task_id = tid

            if next_task_id:
                self._current_task_id = next_task_id
                # Flush buffered messages for the newly promoted task to sequenced sinks
                # (bypass sinks already received messages immediately)
                # Keep _current_has_emitted = True during flush to prevent incoming
                # messages from bypassing the buffer (race condition fix)
                if next_task_id in self._task_buffers:
                    for msg_type, msg in self._task_buffers[next_task_id]:
                        if msg_type == "message":
                            self._emit_to_sinks(msg, sequenced_only=True)
                        else:  # dict
                            self._emit_dict_to_sinks(msg, sequenced_only=True)
                    del self._task_buffers[next_task_id]
                    # Buffer had messages, so this task has emitted
                    self._current_has_emitted = True
                else:
                    # No buffered messages, new task hasn't emitted yet
                    self._current_has_emitted = False
            else:
                self._current_task_id = None
                self._current_has_emitted = False

    # === Run lifecycle ===

    def emit_run_start(self, run_id: str, name: str, **metadata) -> None:
        """Signal start of a GenAI run."""
        self.emit(
            MessageType.RUN_START,
            f"Starting {name}",
            run_id=run_id,
            **metadata,
        )

    def emit_run_end(self, run_id: str, name: str, metrics: dict) -> None:
        """Signal end of a GenAI run and store metrics for aggregation."""
        self.emit(
            MessageType.RUN_END,
            f"Completed {name}",
            run_id=run_id,
            **metrics,
        )

        # Store for module summary aggregation
        self._pending_summaries.append(metrics)

    # === Summary aggregation ===

    def start_module(self, module: str) -> None:
        """Mark start of a module for timing."""
        self._module_start_times[module] = datetime.now()
        self.set_context(module=module)

    def emit_module_summary(self, module: str) -> None:
        """Aggregate all pending summaries for a module."""
        if not self._pending_summaries:
            return

        # Calculate wall clock time
        wall_time_ms = None
        if module in self._module_start_times:
            elapsed = (datetime.now() - self._module_start_times[module]).total_seconds()
            wall_time_ms = elapsed * 1000

        aggregated = self._aggregate(self._pending_summaries, wall_time_ms)
        aggregated["module"] = module  # Store module name for emit_all_module_summaries
        self.emit(
            MessageType.MODULE_SUMMARY,
            f"Module {module}",
            tool_name="MOD_SUM",
            agent_context=module.upper(),
            **aggregated,
        )

        self._module_summaries.append(aggregated)

        # If inside a module group, also store for parent aggregation
        if self._module_group_stack:
            parent_group = self._module_group_stack[-1]
            self._module_group_summaries[parent_group].append(aggregated)

        # Clear pending summaries after aggregation
        self._pending_summaries = []

    def start_module_group(self, name: str) -> None:
        """Start a module group for nested aggregation.

        Module groups can be nested (e.g., INVENTION_LOOP > iter_1 > modules).
        Call emit_module_group_summary(name) when the group completes.

        Args:
            name: Unique name for this module group (e.g., "iter_1", "INVENTION_LOOP")
        """
        self._module_group_start_times[name] = datetime.now()
        self._module_group_stack.append(name)
        self.set_context(module_group=name)

    def emit_module_group_summary(self, name: str) -> None:
        """Aggregate all module/module_group summaries within this group.

        Args:
            name: Name of the module group to summarize
        """
        summaries = self._module_group_summaries.get(name, [])
        if not summaries:
            return

        # Calculate wall clock time
        wall_time_ms = None
        if name in self._module_group_start_times:
            elapsed = (datetime.now() - self._module_group_start_times[name]).total_seconds()
            wall_time_ms = elapsed * 1000

        aggregated = self._aggregate(summaries, wall_time_ms)
        aggregated["module_group"] = name
        self.emit(
            MessageType.MODULE_GROUP_SUMMARY,
            f"ModuleGroup {name}",
            tool_name="MODGRP_SUM",
            agent_context=name.upper(),
            **aggregated,
        )

        # Pop from stack
        if self._module_group_stack and self._module_group_stack[-1] == name:
            self._module_group_stack.pop()

        # If there's a parent module group, propagate summary up
        if self._module_group_stack:
            parent_group = self._module_group_stack[-1]
            self._module_group_summaries[parent_group].append(aggregated)

        # Clear child summaries
        self._module_group_summaries[name] = []

    def get_last_module_summary(self) -> dict:
        """Get the most recent module summary."""
        return self._module_summaries[-1] if self._module_summaries else {}

    def emit_all_module_summaries(self) -> None:
        """Re-emit all stored module summaries (for end-of-run recap)."""
        for summary in self._module_summaries:
            module = summary.get("module", "MODULE")
            self.emit(
                MessageType.MODULE_SUMMARY,
                f"{module} summary",
                tool_name="MOD_SUM",
                agent_context=module.upper(),
                **summary,
            )

    def emit_pipeline_summary(self) -> None:
        """Aggregate all module summaries for the pipeline."""
        if not self._module_summaries:
            return

        # Calculate total wall clock time
        elapsed = (datetime.now() - self._start_time).total_seconds()
        wall_time_ms = elapsed * 1000

        aggregated = self._aggregate(self._module_summaries, wall_time_ms)
        self.emit(
            MessageType.PIPELINE_SUMMARY,
            "Pipeline complete",
            tool_name="RUN_SUM",
            agent_context="PIPELINE",
            **aggregated,
        )

    def _aggregate(self, summaries: list[dict], wall_time_ms: float | None = None) -> dict:
        """
        Aggregate multiple summaries into one.

        Supports both legacy nested format and new standardized SummaryMetrics format.
        Returns standardized SummaryMetrics format for consistent display.
        """
        if not summaries:
            return {}

        # Helper to get token value from nested or flat format
        def get_tokens(s: dict, key: str) -> int:
            # Try flat format first (standardized)
            if key in s and s[key]:
                return s[key]
            # Try nested format (legacy)
            tokens = s.get("tokens", {}) or {}
            key_map = {
                "input_tokens": "input",
                "output_tokens": "output",
                "reasoning_tokens": "reasoning",
            }
            nested_key = key_map.get(key, key.replace("_tokens", ""))
            return tokens.get(nested_key, 0) or 0

        # Run counts
        total_runs = sum(s.get("total_runs", s.get("num_calls", 1)) for s in summaries)
        completed = sum(s.get("completed", s.get("total_runs", s.get("num_calls", 1))) for s in summaries)
        failed = sum(s.get("failed", 0) for s in summaries)

        # Costs - support both formats
        total_cost = sum(s.get("total_cost", 0) or 0 for s in summaries)
        token_cost = sum(
            s.get("token_cost") if s.get("token_cost") is not None else (s.get("total_cost") or 0)
            for s in summaries
        )
        tool_cost = sum(s.get("tool_cost", 0) or 0 for s in summaries)

        # Token aggregation - support both nested and flat formats
        input_tokens = sum(get_tokens(s, "input_tokens") for s in summaries)
        output_tokens = sum(get_tokens(s, "output_tokens") for s in summaries)
        reasoning_tokens = sum(get_tokens(s, "reasoning_tokens") for s in summaries)
        cache_write_tokens = sum(s.get("cache_write_tokens", 0) or 0 for s in summaries)
        cache_read_tokens = sum(s.get("cache_read_tokens", 0) or 0 for s in summaries)

        # Tool call aggregation
        tool_calls = {}
        for s in summaries:
            for tool, count in (s.get("tool_calls") or {}).items():
                tool_calls[tool] = tool_calls.get(tool, 0) + count

        # Tool costs aggregation
        tool_costs = {}
        for s in summaries:
            for tool, cost_info in (s.get("tool_costs") or {}).items():
                if isinstance(cost_info, dict):
                    if tool not in tool_costs:
                        tool_costs[tool] = {"count": 0, "unit": cost_info.get("unit", 0), "total": 0}
                    tool_costs[tool]["count"] += cost_info.get("count", 0)
                    tool_costs[tool]["total"] += cost_info.get("total", 0)
                    # Always use the latest/most recent unit cost
                    unit_cost = cost_info.get("unit", 0)
                    if unit_cost:
                        tool_costs[tool]["unit"] = unit_cost

        # Time aggregation (convert ms to seconds for standardized format)
        runtime_seconds = (wall_time_ms / 1000) if wall_time_ms else 0.0

        def _get_llm_time_ms(s: dict) -> float:
            """Get LLM time in ms from summary, preferring llm_time_ms if available."""
            if s.get("llm_time_ms") is not None:
                return s["llm_time_ms"]
            if s.get("llm_time_seconds") is not None and s["llm_time_seconds"] != 0:
                return s["llm_time_seconds"] * 1000
            if s.get("duration_ms") is not None:
                return s["duration_ms"]
            if s.get("runtime_seconds") is not None and s["runtime_seconds"] != 0:
                return s["runtime_seconds"] * 1000
            return 0

        llm_time_ms = sum(_get_llm_time_ms(s) for s in summaries)
        llm_time_seconds = llm_time_ms / 1000

        # Get model from first summary (or "aggregated" if multiple different models)
        models = set(s.get("model", "") for s in summaries if s.get("model"))
        model = list(models)[0] if len(models) == 1 else "aggregated" if models else ""

        # Return standardized SummaryMetrics format
        return {
            "type": "summary",
            "total_cost": total_cost,
            "token_cost": token_cost,
            "tool_cost": tool_cost,
            "model": model,
            "status": "aggregated",
            "is_aggregated": True,
            "num_calls": total_runs,
            "runtime_seconds": runtime_seconds,
            "llm_time_seconds": llm_time_seconds,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cache_read_tokens": cache_read_tokens,
            "tool_calls": tool_calls if tool_calls else {},
            "tool_costs": tool_costs if tool_costs else {},
            "completed": completed,
            "failed": failed,
        }

    # === Callback creation ===

    def create_callback(self, run_id: str, name: str, group: str | None = None) -> Callable[[dict], None]:
        """Create callback for streaming output from LLM clients.

        Also stores summary messages for module aggregation.

        Args:
            run_id: Unique identifier for this run
            name: Display name for the run (shown in agent_context)
            group: Deprecated, ignored. Kept for backward compatibility.
        """
        def callback(message_dict: dict):
            # Enrich with context
            enriched = {
                **message_dict,
                "run_id": run_id,
                "agent_context": name,
            }

            self.emit_dict(enriched)

            # Store summary messages for module aggregation
            if message_dict.get("type") == "summary":
                self._pending_summaries.append(message_dict)

        return callback

    # === Context manager (auto exception handling) ===

    def __enter__(self) -> "AIITelemetry":
        """Enter context - returns self for use in with statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit context - catches exceptions, logs, flushes, and re-raises.

        Always flushes telemetry. If an exception occurred, logs it before flushing.
        Returns False to re-raise exceptions (never swallows them).
        """
        if exc_val is not None:
            module = self._context.get("module", "Module")
            self.emit(MessageType.ERROR, f"❌ {module} failed: {exc_val}")
        self.flush()
        return False  # Always re-raise exceptions

    # === Cleanup ===

    def flush(self) -> None:
        """Flush all sinks."""
        for sink in self.sinks:
            sink.flush()

    def close(self) -> None:
        """Close all sinks."""
        for sink in self.sinks:
            sink.close()

    def reset(self) -> None:
        """Reset all aggregation state and sequencer state."""
        # Aggregation state
        self._pending_summaries = []
        self._module_summaries.clear()
        self._module_start_times.clear()
        self._start_time = datetime.now()

        # Sequencer state
        with self._buffer_lock:
            self._task_buffers.clear()
            self._active_tasks.clear()
            self._completed_tasks.clear()
            self._task_sequence.clear()
            self._next_sequence = 0
            self._current_task_id = None
            self._current_has_emitted = False

        # Module buffer state
        with self._module_buffer_lock:
            self._module_buffers.clear()
            self._buffered_modules.clear()
            self._module_buffer_sequence.clear()
            self._next_module_sequence = 0
            self._current_streaming_module = None

    def get_sequencer_stats(self) -> dict:
        """Get current sequencer statistics for debugging."""
        with self._buffer_lock:
            return {
                "active_tasks": len(self._active_tasks),
                "completed_tasks": len(self._completed_tasks),
                "current_task": self._current_task_id,
                "buffered_messages": sum(len(msgs) for msgs in self._task_buffers.values()),
                "task_sequences": dict(self._task_sequence),
                "next_sequence": self._next_sequence,
            }

    # === Module-level buffering for parallel modules ===

    def start_buffered_module(self, module_name: str, sequence: int | None = None) -> None:
        """Start buffering output for a module.

        When multiple modules run in parallel, their output is buffered separately
        and flushed as complete blocks in sequence order when each module ends.

        The first module to start (by sequence number) streams immediately.
        Subsequent modules buffer until their turn.

        Uses contextvars to track which buffered module the current async task belongs to.

        Args:
            module_name: Unique identifier for this module
            sequence: Optional explicit sequence number for ordering.
                     If not provided, uses internal monotonic counter.
                     Lower sequence = higher priority (processed first).
        """
        # Set contextvar so this task's messages are associated with this module
        _current_buffered_module.set(module_name)

        with self._module_buffer_lock:
            self._buffered_modules.add(module_name)

            # Assign sequence number
            if sequence is not None:
                self._module_buffer_sequence[module_name] = sequence
            else:
                self._module_buffer_sequence[module_name] = self._next_module_sequence
                self._next_module_sequence += 1

            # If this is the first module or has lowest sequence, make it current
            if self._current_streaming_module is None:
                self._current_streaming_module = module_name
            else:
                # Check if new module has lower sequence than current
                current_seq = self._module_buffer_sequence.get(self._current_streaming_module, float('inf'))
                new_seq = self._module_buffer_sequence.get(module_name, float('inf'))
                if new_seq < current_seq:
                    self._current_streaming_module = module_name

    def end_buffered_module(self, module_name: str) -> None:
        """End buffering for a module and flush its output in sequence order.

        If this module is current, flush immediately and promote next module.
        If not current, mark as complete and buffer will flush when it's promoted.
        """
        with self._module_buffer_lock:
            # Clear contextvar for this task (under lock to avoid race)
            _current_buffered_module.set(None)

            is_current = self._current_streaming_module == module_name

            if is_current:
                # Flush this module's buffer to console
                self._flush_module_buffer(module_name)
                self._buffered_modules.discard(module_name)
                self._module_buffer_sequence.pop(module_name, None)

                # Promote next module
                self._promote_next_module()
            else:
                # Not current - flush buffer and clean up, then try to promote
                # in case this module was blocking the sequence order
                self._flush_module_buffer(module_name)
                self._buffered_modules.discard(module_name)
                self._module_buffer_sequence.pop(module_name, None)

                # Promote next module since this one is now removed from the set
                self._promote_next_module()

    def _flush_module_buffer(self, module_name: str) -> None:
        """Flush buffered messages for a module to sequenced sinks.

        Must be called while holding _module_buffer_lock.
        """
        if module_name not in self._module_buffers:
            return

        for msg_type, msg in self._module_buffers[module_name]:
            if msg_type == "message":
                self._emit_to_sinks(msg, sequenced_only=True)
            else:  # dict
                self._emit_dict_to_sinks(msg, sequenced_only=True)

        del self._module_buffers[module_name]

    def _promote_next_module(self) -> None:
        """Promote the next module in sequence order.

        Flushes any completed modules' buffers and sets the new current module.
        Must be called while holding _module_buffer_lock.
        """
        if not self._buffered_modules:
            self._current_streaming_module = None
            return

        # Find module with lowest sequence number
        next_module = None
        min_seq = float('inf')
        for mod in self._buffered_modules:
            seq = self._module_buffer_sequence.get(mod, float('inf'))
            if seq < min_seq:
                min_seq = seq
                next_module = mod

        if next_module:
            self._current_streaming_module = next_module
            # Flush any buffered messages for the newly promoted module
            self._flush_module_buffer(next_module)
        else:
            self._current_streaming_module = None

    def _module_buffer_message(self, message: TelemetryMessage) -> bool:
        """Buffer a message if module-level buffering is active for non-current module.

        Uses contextvar to determine which buffered module the current async task belongs to.
        Returns True if message was buffered, False if it should be emitted normally.
        """
        if not self._buffered_modules:
            return False

        # Get buffered module from contextvar (set by start_buffered_module)
        module_name = _current_buffered_module.get()
        if not module_name:
            return False

        with self._module_buffer_lock:
            # Check if this module is being buffered and is not current streaming module
            if module_name in self._buffered_modules:
                if module_name != self._current_streaming_module:
                    # Buffer this message
                    self._module_buffers[module_name].append(("message", message))
                    return True

        return False

    def _module_buffer_dict(self, message_dict: dict) -> bool:
        """Buffer a dict message if module-level buffering is active for non-current module.

        Uses contextvar to determine which buffered module the current async task belongs to.
        Returns True if message was buffered, False if it should be emitted normally.
        """
        if not self._buffered_modules:
            return False

        # Get buffered module from contextvar (set by start_buffered_module)
        module_name = _current_buffered_module.get()
        if not module_name:
            return False

        with self._module_buffer_lock:
            # Check if this module is being buffered and is not current streaming module
            if module_name in self._buffered_modules:
                if module_name != self._current_streaming_module:
                    # Buffer this message
                    self._module_buffers[module_name].append(("dict", message_dict))
                    return True

        return False

__all__ = ["AIITelemetry"]
