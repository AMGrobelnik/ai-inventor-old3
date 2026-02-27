"""
Public API for aii_lib agent backend.

This is the main entry point for users. Import Agent and AgentOptions to get started.
"""

import asyncio
import json
import time as _time
from pathlib import Path
from typing import Optional, Any
from dataclasses import replace
from tenacity import AsyncRetrying, stop_after_attempt, wait_random

from loguru import logger
from pydantic import ValidationError

from .models import AgentOptions, AgentResponse, PromptResult, TokenUsage
from .core import (
    initialize_agent,
    initialize_execution,
    execute_prompt_streaming,
    aggregate_prompt_results,
    MessageTimeoutError,
    SubscriptionAccessError,
)
from aii_lib.telemetry import MessageType
from aii_lib.agents.claude.utils import get_monitor


# Force output prompt (when max_turns is exceeded)
_FORCE_OUTPUT_GENERAL_TEMPLATE = '''STOP. You have reached the maximum number of turns.

Do NOT use any more tools. Finish what you are doing and provide your final output NOW.

Use whatever information you have gathered so far to produce the best response possible.
'''

_EXPECTED_FILES_FEEDBACK_TEMPLATE = '''The following required files are missing:

{missing_files}

Create these files now. The task is not complete until all required files exist.

IMPORTANT: When providing your structured output (title, summary, etc.), describe the ARTIFACT you built — NOT the file verification status. Your title and summary must describe what you created, not that you verified files.
'''

_EXPECTED_FILES_NO_PATHS_FEEDBACK_TEMPLATE = '''Your structured output did not include the expected file paths.

Issue: {detail}

Expected file fields in `{field}`:
{expected_fields}

You MUST include the `{field}` field in your structured output with ALL expected file paths filled in (as relative paths from your workspace). Also ensure those files actually exist in your workspace.

IMPORTANT: When providing your structured output (title, summary, etc.), describe the ARTIFACT you built — NOT the file verification status. Your title and summary must describe what you created, not that you verified files.
'''


_DIAG_PREFIX = "__DIAG__:"


def _make_retry_callback(
    max_retries: int,
    telemetry,
    run_id: str,
    task_name: str,
):
    """Create a retry callback for logging retries through telemetry.

    Args:
        max_retries: Max retry attempts for display
        telemetry: AIITelemetry instance for sequenced logging
        run_id: Task ID for sequencer routing
        task_name: Task name for display
    """
    def _log_retry_warning(retry_state) -> None:
        import traceback
        try:
            exc = retry_state.outcome.exception()
        except Exception:
            exc = None
        if exc:
            exc_type = type(exc).__name__
            exc_msg = str(exc) or repr(exc)
            # Include traceback for context
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            tb_short = "".join(tb_lines[-3:]).strip() if len(tb_lines) > 2 else "".join(tb_lines).strip()
            error_msg = f"{exc_type}: {exc_msg}\n{tb_short}"
        else:
            error_msg = "Unknown error"
        telemetry.emit_message(
            "WARNING",
            f"Retrying (attempt {retry_state.attempt_number}/{max_retries}): {error_msg}",
            task_name, run_id
        )
    return _log_retry_warning


class Agent:
    """
    Claude Code agent for executing prompts with full SDK capabilities.

    This is the main public interface. Create an agent with options, then call run()
    with one or more prompts.

    Example:
        >>> from aii_lib.agent_backend.claude import Agent, AgentOptions
        >>>
        >>> agent = Agent(AgentOptions(
        ...     model="sonnet",
        ...     max_turns=50
        ... ))
        >>>
        >>> result = await agent.run("Calculate 5 + 3")
        >>> print(result.final_response)

    Attributes:
        options: Configuration for this agent instance
    """

    def __init__(self, options: Optional[AgentOptions] = None):
        """
        Create a new agent.

        Args:
            options: Agent configuration. If None, uses default settings.

        Example:
            >>> agent = Agent()  # Default settings
            >>> agent = Agent(AgentOptions(model="opus"))  # Custom settings
        """
        self.options = options or AgentOptions()
        # Cache SDK options across run() calls to maintain conversation context
        # This allows retry/correction prompts to continue the same session
        self._sdk_options_cache = None
        self._prompt_count = 0  # Track total prompts across all run() calls
        # Track if expected files instructions have been added (only add on last prompt)
        self._expected_files_instructions_added = False
        # Track session_id for this specific agent instance
        # Used to resume the correct session (not "most recent" globally)
        self._session_id: str | None = None
        # Checkpoint storage: prompt_index -> session_id
        # Allows retrying from any point in the sequence
        self._checkpoints: dict[int, str] = {}
        # Context tracker state per session_id: {sid: {"_ctx_turn": int, "_ctx_cost": float}}
        # Persisted on every turn so forks inherit parent session's cumulative turn/cost
        self._ctx_states: dict[str, dict] = {}
        # Tracks what caused the last failure (for context-aware retry messages)
        self._last_failure_reason: str | None = None
        # Cumulative time spent waiting for capacity (excluded from agent_timeout)
        self._capacity_wait_total: float = 0.0

    def _persist_timeout_state(self, execution_state: dict, sdk_options) -> None:
        """Save partial session state after timeout for fork+resume recovery.

        Captures the session_id, context tracker state, and accumulated metrics
        so the next fork inherits the correct cumulative values.
        """
        partial_sid = execution_state.get("session_id")
        if not partial_sid:
            return
        self._session_id = partial_sid
        # Compute partial runtime for the timed-out session
        partial_runtime = 0.0
        start_iso = execution_state.get("module_start_time")
        if start_iso:
            from datetime import datetime
            try:
                partial_runtime = (datetime.now() - datetime.fromisoformat(start_iso)).total_seconds()
            except (ValueError, TypeError):
                pass
        # Get accumulated totals from parent (if this was already a fork)
        fork_parent_sid = sdk_options.resume if hasattr(sdk_options, 'resume') else None
        prev_acc = self._ctx_states.get(fork_parent_sid, {}) if fork_parent_sid else {}
        # Merge tool_calls from parent + this session
        prev_tool_calls = dict(prev_acc.get("_acc_tool_calls", {}))
        for name, count in execution_state.get("_tool_calls_count", {}).items():
            prev_tool_calls[name] = prev_tool_calls.get(name, 0) + count
        # Save ctx + all accumulated data so fork inherits
        self._ctx_states[partial_sid] = {
            "_ctx_turn": execution_state.get("_ctx_turn", 0),
            "_ctx_cost": execution_state.get("_ctx_cost", 0.0),
            "_ctx_window_used": execution_state.get("_ctx_window_used", 0),
            "_acc_turns": prev_acc.get("_acc_turns", 0) + execution_state.get("_ctx_turn", 0),
            "_acc_runtime": prev_acc.get("_acc_runtime", 0.0) + partial_runtime,
            "_acc_cost": prev_acc.get("_acc_cost", 0.0) + execution_state.get("_ctx_cost", 0.0),
            "_acc_input_tokens": prev_acc.get("_acc_input_tokens", 0) + execution_state.get("_ctx_input_tokens", 0),
            "_acc_output_tokens": prev_acc.get("_acc_output_tokens", 0) + execution_state.get("_ctx_output_tokens", 0),
            "_acc_cache_write_tokens": prev_acc.get("_acc_cache_write_tokens", 0) + execution_state.get("_ctx_cache_write_tokens", 0),
            "_acc_cache_read_tokens": prev_acc.get("_acc_cache_read_tokens", 0) + execution_state.get("_ctx_cache_read_tokens", 0),
            "_acc_tool_calls": prev_tool_calls,
            "_acc_message_count": prev_acc.get("_acc_message_count", 0) + execution_state.get("message_count", 0),
        }

    def _build_continue_prompt(self, original_prompt: str) -> str:
        """Build a context-aware continue prompt based on what caused the retry."""
        reason = self._last_failure_reason
        if reason == "message_timeout":
            timeout_val = self.options.message_timeout
            context = (
                f"YOUR PREVIOUS SESSION WAS INTERRUPTED: A single operation exceeded "
                f"the {timeout_val}s message timeout. Each individual operation must complete "
                f"within {timeout_val}s. Do NOT mock, skip, or compromise your execution — "
                f"still do the real work. Try to make operations run faster if possible. "
                f"If a command genuinely takes longer than {timeout_val}s, split it into "
                f"sequential parts that each complete within the time limit."
            )
        elif reason == "seq_prompt_timeout":
            timeout_val = self.options.seq_prompt_timeout
            context = (
                f"YOUR PREVIOUS SESSION WAS INTERRUPTED: The entire prompt execution exceeded "
                f"the {timeout_val}s prompt timeout. The total work for this prompt must complete "
                f"within {timeout_val}s. Do NOT mock, skip, or compromise your execution — "
                f"still do the real work. Reuse any partial results from the previous attempt. "
                f"Try to be more efficient — cut non-essential steps, but do not sacrifice "
                f"the quality of the core task."
            )
        elif reason == "agent_timeout":
            timeout_val = self.options.agent_timeout
            context = (
                f"YOUR PREVIOUS SESSION WAS INTERRUPTED: The entire agent run exceeded "
                f"the {timeout_val}s agent timeout. This is the final timeout level — "
                f"you have {timeout_val}s total. Do NOT mock, skip, or compromise your execution — "
                f"still do the real work. Use whatever partial work exists from previous attempts. "
                f"Do not start over or repeat completed steps. Focus only on what remains "
                f"and produce the required output."
            )
        elif reason == "connection_error":
            context = (
                "YOUR PREVIOUS SESSION WAS INTERRUPTED: A transient network/API error occurred "
                "(connection reset, rate limit, or service unavailability). This was not your fault. "
                "Continue exactly where you left off — the connection has been restored."
            )
        elif reason == "validation_error":
            context = (
                "YOUR PREVIOUS SESSION WAS INTERRUPTED: The output failed schema validation. "
                "Review the required output format carefully and ensure your response matches "
                "the expected JSON schema exactly."
            )
        elif reason == "subscription_error":
            context = (
                "YOUR PREVIOUS SESSION WAS INTERRUPTED: The Claude subscription/access was temporarily "
                "unavailable. This was not your fault — access has been restored. Continue exactly "
                "where you left off."
            )
        elif reason == "process_error":
            context = (
                "YOUR PREVIOUS SESSION WAS INTERRUPTED: The agent subprocess terminated unexpectedly. "
                "This was a transient infrastructure error, not your fault. Continue exactly where "
                "you left off."
            )
        else:
            context = (
                "YOUR PREVIOUS SESSION WAS INTERRUPTED due to an unexpected error. "
                "Continue where you left off."
            )
        return f"{context}\n\nCONTINUE FOLLOWING THESE INSTRUCTIONS:\n\n{original_prompt}"

    @property
    def checkpoints(self) -> dict[int, str]:
        """Get all checkpoints (prompt_index -> session_id).

        Returns:
            Dict mapping prompt index to session ID for each completed prompt.
        """
        return self._checkpoints.copy()

    def get_checkpoint(self, prompt_index: int) -> str | None:
        """Get checkpoint session_id for a specific prompt index.

        Args:
            prompt_index: The prompt index (0-based)

        Returns:
            Session ID for that checkpoint, or None if not found.
        """
        return self._checkpoints.get(prompt_index)

    def _build_expected_files_instructions(self) -> str:
        """Build expected files instructions to append to the last prompt.

        Returns empty string — the structured output schema description already
        tells the agent what file paths to report. No extra instructions needed.
        """
        return ""

    def _validate_expected_files(
        self,
        prompt_results: list | None = None,
    ) -> tuple[bool, list[str]]:
        """Validate that expected files exist (structured output mode).

        Extracts file paths from the agent's structured output and validates
        each one exists inside the workspace.

        Args:
            prompt_results: List of prompt results to extract file paths from.

        Returns:
            (all_exist, missing_files_list)
        """
        if not self.options.expected_files_struct_out_field:
            return True, []

        cwd = Path(self.options.cwd).resolve()

        # Extract file paths from structured output
        file_paths = self._extract_file_paths_from_structured_output(
            prompt_results or [], self.options.expected_files_struct_out_field
        )
        if not file_paths:
            # Build specific message about what's missing
            field = self.options.expected_files_struct_out_field
            # Check what the structured output actually contains
            so_keys = None
            field_value = None
            for result in reversed(prompt_results or []):
                if result.structured_output and isinstance(result.structured_output, dict):
                    so_keys = list(result.structured_output.keys())
                    field_value = result.structured_output.get(field)
                    break
            if so_keys is None:
                detail = f"no structured output returned (field `{field}` expected)"
            elif field not in (so_keys or []):
                detail = f"field `{field}` missing from structured output (got keys: {so_keys})"
            else:
                detail = f"field `{field}` is empty or contains no paths (value: {field_value!r})"
            return False, [f"{_DIAG_PREFIX}{detail}"]

        missing = []
        for rel_path in file_paths:
            file_path = (cwd / rel_path).resolve()

            # Security: ensure path is inside workspace
            if not str(file_path).startswith(str(cwd)):
                missing.append(f"`{rel_path}` (escapes workspace)")
                continue

            if not file_path.exists():
                missing.append(f"`{rel_path}`")

        return len(missing) == 0, missing

    def _extract_file_paths_from_structured_output(
        self,
        prompt_results: list,
        field_name: str,
    ) -> list[str]:
        """Extract file paths from the latest structured output in prompt_results.

        The field can contain any nested structure (dicts, lists, strings).
        All string values are recursively collected as file paths.
        """
        for result in reversed(prompt_results):
            if result.structured_output:
                data = result.structured_output if isinstance(result.structured_output, dict) else {}
                field_value = data.get(field_name)
                if field_value is not None:
                    return self._collect_paths_recursive(field_value)
        return []

    @staticmethod
    def _collect_paths_recursive(obj) -> list[str]:
        """Recursively collect all string values from a nested structure."""
        paths = []
        if isinstance(obj, str):
            paths.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                paths.extend(Agent._collect_paths_recursive(item))
        elif isinstance(obj, dict):
            for value in obj.values():
                paths.extend(Agent._collect_paths_recursive(value))
        return paths

    def _get_expected_file_fields(self) -> str:
        """Extract expected file field names from output_format schema.

        Resolves $ref to $defs and returns field names with descriptions.
        Returns formatted string for use in feedback prompts.
        """
        field_name = self.options.expected_files_struct_out_field
        schema = (self.options.output_format or {}).get("schema", {})
        if not schema or not field_name:
            return "(unknown — no output schema available)"

        # Find the expected files property in the schema
        ef_prop = schema.get("properties", {}).get(field_name, {})

        # Resolve $ref if present (e.g., "$ref": "#/$defs/ResearchExpectedFiles")
        ref = ef_prop.get("$ref", "")
        if ref.startswith("#/$defs/"):
            def_name = ref.split("/")[-1]
            ef_prop = schema.get("$defs", {}).get(def_name, {})

        # Extract properties with descriptions
        properties = ef_prop.get("properties", {})
        if not properties:
            return "(unknown — no properties found in schema)"

        lines = []
        for prop_name, prop_schema in properties.items():
            desc = prop_schema.get("description", "")
            lines.append(f"- `{prop_name}`: {desc}" if desc else f"- `{prop_name}`")
        return "\n".join(lines)

    def _build_expected_files_feedback(self, missing: list[str]) -> str:
        """Build feedback prompt for missing files retry.

        If missing contains a structured-output diagnostic (prefixed with
        _DIAG_PREFIX), uses the no-paths template instead.
        """
        if missing and missing[0].startswith(_DIAG_PREFIX):
            detail = missing[0][len(_DIAG_PREFIX):]
            field = self.options.expected_files_struct_out_field or "out_expected_files"
            expected_fields = self._get_expected_file_fields()
            return _EXPECTED_FILES_NO_PATHS_FEEDBACK_TEMPLATE.format(
                detail=detail, field=field, expected_fields=expected_fields,
            )
        return _EXPECTED_FILES_FEEDBACK_TEMPLATE.format(
            missing_files="\n".join(f"- {m}" for m in missing)
        )

    def _build_force_output_prompt(self) -> str:
        """Build force output prompt when max_turns is exceeded.

        Uses custom force_output_prompt from options if provided,
        otherwise falls back to the generic template.
        """
        return self.options.force_output_prompt or _FORCE_OUTPUT_GENERAL_TEMPLATE

    async def run(
        self,
        prompts: str | list[str],
    ) -> AgentResponse:
        """
        Execute one or more prompts with agent-level timeout and retry.

        Each prompt in the sequence can share conversation context with the previous
        prompt depending on your configuration (continue_seq_item setting).

        Agent-level retry/timeout wraps the entire run:
        - agent_timeout: Max time for entire agent run (all prompts)
        - agent_retries: Retries for entire agent on failure

        Per-prompt retry/timeout operates within each prompt:
        - seq_prompt_timeout: Max time per individual prompt
        - seq_prompt_retries: Retries per individual prompt

        NEVER CRASHES: On failure after all retries, logs error and returns empty response.

        Args:
            prompts: Single prompt string or list of prompts to execute sequentially

        Returns:
            AgentResponse containing:
                - final_response: Text response from the last prompt (empty string on failure)
                - total_cost: Total API cost in USD
                - prompt_results: Individual results for each prompt
                - final_session_id: Session ID for resuming/forking
                - all_messages: Complete message log
                - json_log_path: Path to saved JSON log (if enabled)
                - failed: True if agent failed after all retries

        Example:
            >>> # Single prompt
            >>> result = await agent.run("What is 2+2?")
            >>> if result.failed:
            ...     print("Agent failed")
            >>> else:
            ...     print(result.final_response)
        """
        agent_retries = self.options.agent_retries
        agent_timeout = self.options.agent_timeout
        last_error = None

        # Agent-level retry loop (never crashes, only warnings)
        for attempt_num in range(1, agent_retries + 1):
            try:
                # Reset state for retry (fresh start)
                if attempt_num > 1:
                    self._reset_for_retry()
                    self.options.telemetry.emit_message(
                        "WARNING",
                        f"Agent retry... (attempt {attempt_num}/{agent_retries}): {last_error}",
                        self.options.agent_context, self.options.run_id
                    )

                # Execute with optional agent-level timeout
                # Uses deadline-based approach so capacity wait time is excluded
                if agent_timeout is not None:
                    self._capacity_wait_total = 0.0
                    return await self._run_with_deadline(prompts, agent_timeout)
                else:
                    return await self._run_internal(prompts)

            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                self._last_failure_reason = "agent_timeout"
                last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                self.options.telemetry.emit_message(
                    "WARNING",
                    f"Agent timeout/cancel (attempt {attempt_num}/{agent_retries}): {last_error}",
                    self.options.agent_context, self.options.run_id
                )
                if attempt_num < agent_retries:
                    import random
                    await asyncio.sleep(random.uniform(1, 20))
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                self._last_failure_reason = "validation_error"
                last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                self.options.telemetry.emit_message(
                    "WARNING",
                    f"Agent validation error (attempt {attempt_num}/{agent_retries}): {last_error}",
                    self.options.agent_context, self.options.run_id
                )
                if attempt_num < agent_retries:
                    import random
                    await asyncio.sleep(random.uniform(1, 20))
            except (OSError, ConnectionError, TimeoutError) as e:
                self._last_failure_reason = "connection_error"
                last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                self.options.telemetry.emit_message(
                    "WARNING",
                    f"Agent connection error (attempt {attempt_num}/{agent_retries}): {last_error}",
                    self.options.agent_context, self.options.run_id
                )
                if attempt_num < agent_retries:
                    import random
                    await asyncio.sleep(random.uniform(1, 20))
            except Exception as e:
                self._last_failure_reason = "process_error"
                last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                self.options.telemetry.emit_message(
                    "WARNING",
                    f"Agent unexpected error (attempt {attempt_num}/{agent_retries}): {last_error}",
                    self.options.agent_context, self.options.run_id
                )
                if attempt_num < agent_retries:
                    import random
                    await asyncio.sleep(random.uniform(1, 20))

        # All retries exhausted - log error and return empty response (never crash)
        self.options.telemetry.emit(
            MessageType.ERROR,
            f"Agent failed after {agent_retries} retries: {last_error}",
        )

        # Return empty/failed response
        return AgentResponse(
            final_response="",
            total_cost=0.0,
            prompt_results=[],
            final_session_id=self._session_id,
            all_messages=[],
            json_log_path=None,
            failed=True,
            error_message=f"Agent failed after {agent_retries} retries: {last_error}",
        )

    def _reset_for_retry(self):
        """Reset agent state for a completely fresh retry attempt."""
        self._prompt_count = 0
        self._session_id = None
        self._sdk_options_cache = None
        self._checkpoints = {}
        self._expected_files_instructions_added = False

    async def _run_with_deadline(self, prompts, base_timeout: float):
        """Execute _run_internal with a deadline that excludes capacity wait time.

        The deadline is extended by any time spent in async_wait_for_capacity(),
        so rate-limit pauses don't eat into the agent's working time.

        Also directly checks the UsageMonitor: while the monitor reports rate-limited,
        the deadline is frozen (elapsed time doesn't count against the agent).
        This prevents the deadline from expiring DURING a capacity wait, which would
        cancel the task before _capacity_wait_total could be updated.

        Polls every 2s to check deadline and adjust for new capacity wait time.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + base_timeout
        task = asyncio.ensure_future(self._run_internal(prompts))
        prev_wait = 0.0
        last_check = loop.time()

        try:
            while not task.done():
                now = loop.time()
                elapsed = now - last_check
                last_check = now

                # Extend deadline by any new capacity wait time (retroactive)
                if self._capacity_wait_total > prev_wait:
                    deadline += self._capacity_wait_total - prev_wait
                    prev_wait = self._capacity_wait_total

                # Freeze deadline while rate limited: the agent is blocked in
                # async_wait_for_capacity() or SubscriptionAccessError sleep,
                # so elapsed time should not count against the timeout.
                monitor = get_monitor()
                if monitor.is_rate_limited():
                    deadline += elapsed

                remaining = deadline - now
                if remaining <= 0:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                    raise asyncio.TimeoutError(
                        f"Agent deadline exceeded (base={base_timeout}s, "
                        f"capacity_wait={self._capacity_wait_total:.0f}s)"
                    )

                await asyncio.wait({task}, timeout=min(remaining, 2.0))

            return task.result()
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            raise

    async def _run_internal(
        self,
        prompts: str | list[str],
    ) -> AgentResponse:
        """
        Internal run logic (called by run() with retry/timeout wrapper).

        This contains the actual prompt execution logic.
        """
        # Normalize to list
        if isinstance(prompts, str):
            prompts = [prompts]

        if not prompts:
            raise ValueError("At least one prompt must be provided")

        # Append instructions to LAST prompt (so agent writes output after completing all tasks)
        prompts = prompts.copy()  # Don't mutate original

        # If expected_files_struct_out_field is set, append instructions to LAST prompt
        if self.options.expected_files_struct_out_field and not self._expected_files_instructions_added:
            expected_files_instructions = self._build_expected_files_instructions()
            prompts[-1] = prompts[-1] + expected_files_instructions
            self._expected_files_instructions_added = True

        prompt_results = []
        total_prompts = len(prompts)

        # Execute each prompt sequentially
        # For multi-prompt: emit_summary=False for ALL, aggregate at end
        # For single prompt: emit_summary=True (normal behavior)
        has_post_validation = self.options.expected_files_struct_out_field is not None
        emit_individual_summaries = (total_prompts == 1) and not has_post_validation

        for i, prompt in enumerate(prompts):
            # Use global prompt count for continue_conversation logic
            global_prompt_idx = self._prompt_count + i

            # Execute this prompt (with per-prompt retry logic)
            # On failure after all per-prompt retries, logs warning and re-raises
            # to trigger agent-level retry (which handles the final error)
            try:
                is_last_prompt = (i == total_prompts - 1)
                result, self._sdk_options_cache = await self._execute_single_prompt(
                    prompt,
                    global_prompt_idx,
                    self._sdk_options_cache,
                    emit_summary=emit_individual_summaries,
                    with_output_format=is_last_prompt,
                )
            except asyncio.TimeoutError:
                # Timeout after all per-prompt retries - log warning (not error)
                # Re-raise to trigger agent-level retry
                self.options.telemetry.emit_message(
                    "WARNING",
                    f"Prompt {i+1}/{total_prompts} timed out after {self.options.seq_prompt_retries} retries",
                    self.options.agent_context, self.options.run_id
                )
                raise
            except (ConnectionError, OSError) as e:
                # Network/IO errors - log and re-raise for agent-level retry
                self.options.telemetry.emit_message(
                    "WARNING",
                    f"Prompt {i+1}/{total_prompts} failed (connection/IO error) after {self.options.seq_prompt_retries} retries: {e}",
                    self.options.agent_context, self.options.run_id
                )
                raise
            except (ValueError, ValidationError) as e:
                # Validation errors - log and re-raise for agent-level retry
                self.options.telemetry.emit_message(
                    "WARNING",
                    f"Prompt {i+1}/{total_prompts} failed (validation error) after {self.options.seq_prompt_retries} retries: {e}",
                    self.options.agent_context, self.options.run_id
                )
                raise
            except Exception as e:
                # Unexpected errors - log warning about unexpected type, re-raise for agent-level retry
                logger.warning(f"Unexpected exception type in prompt execution: {type(e).__name__}")
                self.options.telemetry.emit_message(
                    "WARNING",
                    f"Prompt {i+1}/{total_prompts} failed after {self.options.seq_prompt_retries} retries: {e}",
                    self.options.agent_context, self.options.run_id
                )
                raise

            prompt_results.append(result)

            # Check if max_turns was exceeded WITHOUT producing structured output.
            # If the agent's last action was the structured output, it finished normally
            # (just happened to use all its turns) — no intervention needed.
            # Only send force output when the agent ran out of turns mid-work.
            if result.num_turns >= self.options.max_turns and result.structured_output is None:
                self.options.telemetry.emit_message(
                    "WARNING",
                    f"Max turns ({self.options.max_turns}) reached after prompt {i+1}. Sending force output prompt.",
                    self.options.agent_context, self.options.run_id
                )

                # Build force output prompt (can be overridden by subclasses or via force_output_prompt option)
                force_prompt = self._build_force_output_prompt()
                force_prompt_idx = self._prompt_count + len(prompt_results)

                try:
                    force_result, self._sdk_options_cache = await self._execute_single_prompt(
                        force_prompt,
                        force_prompt_idx,
                        self._sdk_options_cache,
                        emit_summary=False,  # Will aggregate at end
                        with_output_format=True,
                    )
                    prompt_results.append(force_result)
                    if force_result.session_id:
                        self._session_id = force_result.session_id
                except (asyncio.TimeoutError, ConnectionError, OSError) as e:
                    # Force output failed due to timeout/connection — preserve collected prompt_results
                    logger.exception(f"Force output prompt failed ({type(e).__name__})")
                    self.options.telemetry.emit_message(
                        "ERROR",
                        f"Force output prompt failed ({type(e).__name__}): {e}",
                        self.options.agent_context, self.options.run_id
                    )
                except Exception as e:
                    # Force output failed unexpectedly — preserve collected prompt_results
                    logger.exception(f"Force output prompt failed ({type(e).__name__})")
                    self.options.telemetry.emit_message(
                        "ERROR",
                        f"Force output prompt failed: {e}",
                        self.options.agent_context, self.options.run_id
                    )

                # After force output, don't process remaining prompts
                break

        # Update total prompt count for future run() calls
        self._prompt_count += len(prompt_results)

        # Expected files validation and retry (if expected_files_struct_out_field is set)
        expected_files_valid = True
        if self.options.expected_files_struct_out_field:
            all_exist, missing = self._validate_expected_files(prompt_results)

            # Retry loop for missing files
            files_retry_count = 0
            max_retries = self.options.max_expected_files_retries
            while not all_exist and files_retry_count < max_retries:
                files_retry_count += 1
                display = [m.removeprefix(_DIAG_PREFIX) for m in missing]
                self.options.telemetry.emit_message(
                    "WARNING",
                    f"Expected files missing (retry {files_retry_count}/{max_retries}): {display}",
                    self.options.agent_context, self.options.run_id
                )

                # Send feedback prompt (continues conversation)
                feedback = self._build_expected_files_feedback(missing)
                global_prompt_idx = self._prompt_count
                self._prompt_count += 1

                try:
                    result, self._sdk_options_cache = await self._execute_single_prompt(
                        feedback,
                        global_prompt_idx,
                        self._sdk_options_cache,
                        emit_summary=False,
                        with_output_format=True,
                    )
                except (asyncio.TimeoutError, ConnectionError, OSError) as e:
                    logger.exception(f"Expected files retry failed ({type(e).__name__})")
                    self.options.telemetry.emit(MessageType.ERROR, f"Expected files retry failed ({type(e).__name__}): {e}")
                    # Preserve collected prompt_results and break retry loop
                    break
                except Exception as e:
                    logger.exception(f"Expected files retry failed ({type(e).__name__})")
                    self.options.telemetry.emit(MessageType.ERROR, f"Expected files retry failed: {e}")
                    break

                prompt_results.append(result)
                all_exist, missing = self._validate_expected_files(prompt_results)

            if not all_exist:
                # Log warning but don't fail - just mark as invalid
                display = [m.removeprefix(_DIAG_PREFIX) for m in missing]
                self.options.telemetry.emit_message(
                    "WARNING",
                    f"Expected files still missing after {files_retry_count} retries: {display}",
                    self.options.agent_context, self.options.run_id
                )
                expected_files_valid = False

        # SDK native structured output (when output_format is set)
        # Always from the last prompt result — either the main prompt captured it,
        # or the force output prompt produced it as its final response.
        structured_output = None
        if self.options.output_format and prompt_results:
            last_result = prompt_results[-1]
            if last_result.structured_output is not None:
                structured_output = last_result.structured_output

        # Aggregate all prompt results (includes emitting aggregated summary if multi-prompt)
        response = aggregate_prompt_results(self.options, prompt_results)
        response.structured_output = structured_output
        response.expected_files_valid = expected_files_valid
        return response

    async def _execute_single_prompt(
        self,
        prompt: str,
        prompt_index: int,
        sdk_options_cache,
        emit_summary: bool = True,
        with_output_format: bool = False,
    ) -> tuple[PromptResult, Any]:
        """
        Execute a single prompt with retry and timeout support.

        Steps:
            1. Prepare SDK options (build once, modify for subsequent prompts)
            2. Initialize execution (log coordinator)
            3. Execute via SDK streaming (with timeout if configured)
            4. Build result object

        Retries on failure/timeout using options.seq_prompt_retries.
        Timeout controlled by options.seq_prompt_timeout (None = no timeout).

        Args:
            prompt: The prompt text to execute
            prompt_index: Index in the sequence (0-based)
            sdk_options_cache: Cached SDK options from first prompt (or None)
            emit_summary: Whether to emit summary (False for multi-prompt aggregation)
            with_output_format: Whether to include output_format (structured output) for
                this prompt. Only True for the last prompt in a sequence and retries,
                so intermediate prompts don't waste turns producing structured output.

        Returns:
            Tuple of (PromptResult, sdk_options_cache)
        """
        max_retries = self.options.seq_prompt_retries
        timeout = self.options.seq_prompt_timeout

        # Check usage limits before executing (async wait, doesn't block event loop)
        monitor = get_monitor()
        if monitor._config["usage_tracking"]["enabled"]:
            # Start background monitor if not running
            monitor.start()
            # Wait for capacity (async — yields to event loop during wait)
            wait_start = _time.monotonic()
            await monitor.async_wait_for_capacity()
            wait_duration = _time.monotonic() - wait_start
            if wait_duration > 0.1:
                self._capacity_wait_total += wait_duration

        # Save original prompt — on retry with session resume, we send "continue" instead
        original_prompt = prompt

        # Retry loop with configurable attempts
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_random(min=1, max=20),
            before_sleep=_make_retry_callback(
                max_retries,
                telemetry=self.options.telemetry,
                run_id=self.options.run_id,
                task_name=self.options.agent_context,
            ),
            reraise=True,
        ):
            with attempt:
                # STEP 1: Prepare SDK options
                if prompt_index == 0:
                    if self._session_id:
                        # RETRY with session resume: fork from timed-out session
                        # and re-send the original instructions with a context-aware
                        # CONTINUE prefix so the agent knows what happened and adapts
                        if sdk_options_cache is None:
                            saved_output_format = self.options.output_format
                            self.options.output_format = None
                            sdk_options = initialize_agent(self.options)
                            self.options.output_format = saved_output_format
                            sdk_options_cache = sdk_options
                        sdk_options = replace(
                            sdk_options_cache,
                            resume=self._session_id,
                            fork_session=True,
                        )
                        effective_prompt = self._build_continue_prompt(original_prompt)
                        # Log fork config for visibility
                        self.options.telemetry.emit_message(
                            "DEBUG",
                            f"Forking from session {self._session_id[:12]}... | "
                            f"model={sdk_options.model} | "
                            f"mcp_servers={len(sdk_options.mcp_servers) if isinstance(sdk_options.mcp_servers, dict) else sdk_options.mcp_servers} | "
                            f"max_turns={sdk_options.max_turns} | "
                            f"permission_mode={sdk_options.permission_mode}",
                            self.options.agent_context, self.options.run_id,
                        )
                    else:
                        # First attempt (no previous session): build fresh SDK options
                        # Build WITHOUT output_format — it's only applied on the last
                        # prompt via with_output_format=True, so intermediate prompts
                        # don't force the agent to produce structured output.
                        saved_output_format = self.options.output_format
                        self.options.output_format = None
                        sdk_options = initialize_agent(self.options)
                        self.options.output_format = saved_output_format
                        sdk_options_cache = sdk_options
                        effective_prompt = original_prompt
                else:
                    # Subsequent prompts: FORK from previous session to create isolated checkpoint
                    # This is critical for retry-from-checkpoint functionality:
                    # - Each prompt creates a NEW session by forking from previous
                    # - If this prompt fails/timeouts, retry forks from same checkpoint
                    # - Previous prompt's checkpoint is never modified
                    resume_id = self._session_id if self.options.continue_seq_item else None
                    sdk_options = replace(
                        sdk_options_cache,
                        resume=resume_id,
                        fork_session=True if resume_id else False,  # Fork to create checkpoint
                    )
                    effective_prompt = original_prompt

                # Apply output_format only when requested (last prompt, retries)
                if with_output_format and self.options.output_format:
                    sdk_options = replace(sdk_options, output_format=self.options.output_format)

                # STEP 2: Initialize execution (setup logging for this prompt)
                telemetry, execution_state = initialize_execution(
                    self.options,
                    effective_prompt,
                    prompt_index,
                )

                # Carry over context tracker state from fork parent session
                fork_parent_sid = sdk_options.resume if hasattr(sdk_options, 'resume') else None
                if fork_parent_sid and fork_parent_sid in self._ctx_states:
                    parent_ctx = self._ctx_states[fork_parent_sid]
                    execution_state["_ctx_turn"] = parent_ctx.get("_ctx_turn", 0)
                    execution_state["_ctx_cost"] = parent_ctx.get("_ctx_cost", 0.0)
                    execution_state["_ctx_window_used"] = parent_ctx.get("_ctx_window_used", 0)

                # STEP 3: Execute via SDK streaming (with optional timeout)
                # Inner message retry loop: handles MessageTimeoutError with its own
                # retry budget (message_retries). When exhausted, escalates as
                # asyncio.TimeoutError to the outer seq_prompt retry loop.
                msg_retries_remaining = self.options.message_retries
                while True:
                    # Suppress summary emission when forking from a timed-out parent
                    has_fork_parent = (
                        fork_parent_sid
                        and fork_parent_sid in self._ctx_states
                        and self._ctx_states[fork_parent_sid].get("_acc_turns", 0) > 0
                    )
                    effective_emit_summary = emit_summary and not has_fork_parent
                    execution_coro = execute_prompt_streaming(
                        effective_prompt,
                        sdk_options,
                        telemetry,
                        execution_state,
                        emit_summary=effective_emit_summary,
                        message_timeout=self.options.message_timeout,
                    )

                    try:
                        if timeout is not None:
                            response_text, session_id, cost, usage, summary_data, num_turns, structured_output = await asyncio.wait_for(
                                execution_coro,
                                timeout=timeout,
                            )
                        else:
                            response_text, session_id, cost, usage, summary_data, num_turns, structured_output = await execution_coro
                        break  # Success — exit inner message retry loop

                    except MessageTimeoutError:
                        # Individual SDK message hung — save state and fork+resume
                        self._persist_timeout_state(execution_state, sdk_options)
                        msg_retries_remaining -= 1
                        if msg_retries_remaining < 0:
                            # Exhausted message retries — escalate to seq_prompt retry
                            raise asyncio.TimeoutError(
                                f"Message timeout exhausted after {self.options.message_retries} retries"
                            )
                        # Wait for usage monitor capacity before retrying
                        # (message timeout may have been caused by rate limiting)
                        _monitor = get_monitor()
                        if _monitor.is_rate_limited():
                            wait_start = _time.monotonic()
                            self.options.telemetry.emit_message(
                                "WARNING",
                                "Usage threshold exceeded — waiting for capacity before message retry...",
                                self.options.agent_context, self.options.run_id,
                            )
                            await _monitor.async_wait_for_capacity()
                            self._capacity_wait_total += _time.monotonic() - wait_start
                        self.options.telemetry.emit_message(
                            "WARNING",
                            f"Message timeout (attempt {self.options.message_retries - msg_retries_remaining}/{self.options.message_retries}) "
                            f"— forking from {self._session_id[:12] if self._session_id else '?'}...",
                            self.options.agent_context, self.options.run_id,
                        )
                        # Fork from partial session for next attempt
                        self._last_failure_reason = "message_timeout"
                        if self._session_id:
                            sdk_options = replace(
                                sdk_options_cache,
                                resume=self._session_id,
                                fork_session=True,
                            )
                            if with_output_format and self.options.output_format:
                                sdk_options = replace(sdk_options, output_format=self.options.output_format)
                            effective_prompt = self._build_continue_prompt(original_prompt)
                        # Re-initialize execution state for the fork
                        telemetry, execution_state = initialize_execution(
                            self.options, effective_prompt, prompt_index,
                        )
                        # Carry over ctx state from fork parent
                        fork_parent_sid = sdk_options.resume if hasattr(sdk_options, 'resume') else None
                        if fork_parent_sid and fork_parent_sid in self._ctx_states:
                            parent_ctx = self._ctx_states[fork_parent_sid]
                            execution_state["_ctx_turn"] = parent_ctx.get("_ctx_turn", 0)
                            execution_state["_ctx_cost"] = parent_ctx.get("_ctx_cost", 0.0)
                            execution_state["_ctx_window_used"] = parent_ctx.get("_ctx_window_used", 0)
                        continue  # Retry inner loop with forked session

                    except SubscriptionAccessError as e:
                        # Subscription/access lost — wait 60s then also wait for
                        # usage monitor capacity. Does NOT consume retries.
                        self._persist_timeout_state(execution_state, sdk_options)
                        self._last_failure_reason = "subscription_error"
                        wait_start = _time.monotonic()
                        self.options.telemetry.emit_message(
                            "WARNING",
                            f"Subscription/access error — waiting 60s before retry: {e}",
                            self.options.agent_context, self.options.run_id,
                        )
                        await asyncio.sleep(60)
                        # Also wait for usage monitor capacity (threshold may be exceeded)
                        _monitor = get_monitor()
                        if _monitor.is_rate_limited():
                            self.options.telemetry.emit_message(
                                "WARNING",
                                "Usage threshold also exceeded — waiting for capacity...",
                                self.options.agent_context, self.options.run_id,
                            )
                            await _monitor.async_wait_for_capacity()
                        self._capacity_wait_total += _time.monotonic() - wait_start
                        # Fork from partial session if available
                        if self._session_id:
                            sdk_options = replace(
                                sdk_options_cache,
                                resume=self._session_id,
                                fork_session=True,
                            )
                            if with_output_format and self.options.output_format:
                                sdk_options = replace(sdk_options, output_format=self.options.output_format)
                            effective_prompt = self._build_continue_prompt(original_prompt)
                        # Re-initialize execution state for the retry
                        telemetry, execution_state = initialize_execution(
                            self.options, effective_prompt, prompt_index,
                        )
                        # Carry over ctx state from fork parent
                        fork_parent_sid = sdk_options.resume if hasattr(sdk_options, 'resume') else None
                        if fork_parent_sid and fork_parent_sid in self._ctx_states:
                            parent_ctx = self._ctx_states[fork_parent_sid]
                            execution_state["_ctx_turn"] = parent_ctx.get("_ctx_turn", 0)
                            execution_state["_ctx_cost"] = parent_ctx.get("_ctx_cost", 0.0)
                            execution_state["_ctx_window_used"] = parent_ctx.get("_ctx_window_used", 0)
                        continue  # Retry inner loop

                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        # seq_prompt timeout or agent-level cancellation — save state and escalate
                        self._persist_timeout_state(execution_state, sdk_options)
                        self._last_failure_reason = "seq_prompt_timeout"
                        raise

                # Store session_id for this agent instance (used for subsequent prompts)
                if session_id:
                    self._session_id = session_id
                    # Store checkpoint for this prompt index (allows retry from any point)
                    self._checkpoints[prompt_index] = session_id
                    # Save ctx tracker state for potential future forks
                    self._ctx_states[session_id] = {
                        "_ctx_turn": execution_state.get("_ctx_turn", 0),
                        "_ctx_cost": execution_state.get("_ctx_cost", 0.0),
                        "_ctx_window_used": execution_state.get("_ctx_window_used", 0),
                    }

                # Adjust summary with accumulated data from timed-out parent sessions
                fork_parent_sid = sdk_options.resume if hasattr(sdk_options, 'resume') else None
                if fork_parent_sid and fork_parent_sid in self._ctx_states and summary_data:
                    prev_acc = self._ctx_states[fork_parent_sid]
                    # Turns and runtime
                    acc_turns = prev_acc.get("_acc_turns", 0)
                    acc_runtime = prev_acc.get("_acc_runtime", 0.0)
                    if acc_turns > 0:
                        summary_data["num_calls"] = summary_data.get("num_calls", 0) + acc_turns
                    if acc_runtime > 0:
                        summary_data["runtime_seconds"] = summary_data.get("runtime_seconds", 0) + acc_runtime
                    # Cost (from ContextTracker's per-turn pricing)
                    acc_cost = prev_acc.get("_acc_cost", 0.0)
                    if acc_cost > 0:
                        summary_data["total_cost"] = summary_data.get("total_cost", 0) + acc_cost
                        summary_data["token_cost"] = summary_data.get("token_cost", 0) + acc_cost
                    # Token counts
                    for field, acc_key in [
                        ("input_tokens", "_acc_input_tokens"),
                        ("output_tokens", "_acc_output_tokens"),
                        ("cache_write_tokens", "_acc_cache_write_tokens"),
                        ("cache_read_tokens", "_acc_cache_read_tokens"),
                    ]:
                        acc_val = prev_acc.get(acc_key, 0)
                        if acc_val > 0:
                            summary_data[field] = summary_data.get(field, 0) + acc_val
                    # Tool calls (merge dicts)
                    acc_tool_calls = prev_acc.get("_acc_tool_calls", {})
                    if acc_tool_calls:
                        merged = dict(acc_tool_calls)
                        for name, count in summary_data.get("tool_calls", {}).items():
                            merged[name] = merged.get(name, 0) + count
                        summary_data["tool_calls"] = merged
                    # Message count (in metadata)
                    acc_msg_count = prev_acc.get("_acc_message_count", 0)
                    if acc_msg_count > 0 and "message_metadata" in summary_data:
                        summary_data["message_metadata"]["message_count"] = (
                            summary_data["message_metadata"].get("message_count", 0) + acc_msg_count
                        )

                # Also adjust PromptResult.cost (used by aggregate_prompt_results)
                if fork_parent_sid and fork_parent_sid in self._ctx_states:
                    acc_cost = self._ctx_states[fork_parent_sid].get("_acc_cost", 0.0)
                    if acc_cost > 0:
                        cost += acc_cost

                # Re-emit patched summary (was suppressed during streaming)
                if has_fork_parent and emit_summary and summary_data:
                    # Update message_text to reflect patched cost
                    summary_data["message_text"] = f"Total cost: ${summary_data.get('total_cost', 0):.4f}"
                    # Emit via the same telemetry that was used for this execution
                    msg_dict = dict(summary_data)
                    if self.options.run_id:
                        msg_dict["run_id"] = self.options.run_id
                    if self.options.agent_context:
                        msg_dict["agent_context"] = self.options.agent_context
                    telemetry.emit_dict(msg_dict)

                # STEP 4: Build result object
                # Get messages from JSONSink if available
                collected_messages = []
                for sink in telemetry.sinks:
                    if hasattr(sink, 'messages'):
                        collected_messages = sink.messages
                        break

                result = PromptResult(
                    response=response_text,
                    session_id=session_id,
                    cost=cost,
                    prompt_index=prompt_index,
                    usage=usage,
                    messages=collected_messages,
                    summary_data=summary_data,  # Store for aggregation
                    num_turns=num_turns,  # For max_turns exceeded detection
                    structured_output=structured_output,  # SDK native structured output
                )

                return result, sdk_options_cache


__all__ = ['Agent']
