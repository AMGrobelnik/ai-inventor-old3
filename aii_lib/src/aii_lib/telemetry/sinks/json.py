"""
JSON sink - structured JSON file output.
"""

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Sink

if TYPE_CHECKING:
    from ..message import TelemetryMessage

_logger = logging.getLogger(__name__)


class JSONSink(Sink):
    """JSONL file output for visualization and analysis.

    Uses append-only JSONL format (one JSON object per line) for O(1) writes.
    Only logs LLM observability data (prompts, tool calls, summaries, errors).
    Filters out general logging messages (info, warning, success, debug).
    """

    # Message types to skip (general logging, not LLM observability)
    # NOTE: "error" is intentionally NOT skipped — critical errors must always be logged.
    SKIP_TYPES = {"info", "warning", "success", "debug"}

    # Number of consecutive failures before attempting to reopen the file
    _MAX_FAILURES_BEFORE_REOPEN = 3

    def __init__(self, path: Path | str, sequenced: bool = False):
        """
        Args:
            path: Path to JSONL log file (messages file)
            sequenced: If True, go through the sequencer (groups messages per task).
                      If False (default), bypass sequencer for immediate O(1) writes.
        """
        self.bypass_sequencer = not sequenced
        self.path = Path(path)
        self._lock = threading.Lock()
        self._file = None
        self._write_failure_count = 0
        self._write_failure_logged = False

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode (kept open for performance)
        self._file = open(self.path, "a", encoding="utf-8")

    def _should_log(self, msg_type: str) -> bool:
        """Check if message type should be logged to JSONL."""
        return msg_type.lower() not in self.SKIP_TYPES

    def emit(self, message: "TelemetryMessage") -> None:
        """Emit message to JSONL file (skips general logging types)."""
        if not self._should_log(str(message.type.value if hasattr(message.type, 'value') else message.type)):
            return
        self._write_line(message.to_dict())

    def emit_dict(self, message_dict: dict) -> None:
        """Emit from legacy dict format (for backward compatibility)."""
        if not self._should_log(message_dict.get("type", "")):
            return
        self._write_line(message_dict)

    def _write_line(self, data: dict) -> None:
        """Append single JSON line to file."""
        try:
            with self._lock:
                if self._file:
                    line = json.dumps(data, ensure_ascii=False, default=str)
                    self._file.write(line + "\n")
                    self._file.flush()  # Ensure durability
                    # Reset failure count on success
                    self._write_failure_count = 0
                    self._write_failure_logged = False
        except Exception as exc:
            self._write_failure_count += 1
            # Log warning on first failure to avoid log spam
            if not self._write_failure_logged:
                _logger.warning("JSONSink write failed for %s: %s", self.path, exc)
                _logger.debug("JSONSink write error details", exc_info=True)
                self._write_failure_logged = True
            # After multiple consecutive failures, attempt to reopen the file
            if self._write_failure_count >= self._MAX_FAILURES_BEFORE_REOPEN:
                self._try_reopen()

    def _try_reopen(self) -> None:
        """Attempt to reopen the file after multiple write failures.

        Must be called while NOT holding _lock (or use a reentrant lock).
        """
        try:
            with self._lock:
                if self._file:
                    try:
                        self._file.close()
                    except Exception:
                        pass
                self._file = open(self.path, "a", encoding="utf-8")
                self._write_failure_count = 0
                self._write_failure_logged = False
                _logger.debug("JSONSink successfully reopened %s", self.path)
        except Exception as exc:
            _logger.debug("JSONSink failed to reopen %s: %s", self.path, exc)
            self._file = None

    def flush(self) -> None:
        """Flush buffered writes to file."""
        with self._lock:
            if self._file:
                self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


# Standalone function for backward compatibility
def log_message(
    message_dict: dict,
    json_log_path: Path | str,
    session_id: str | None = None,
    prompt_index: int | None = None,
) -> None:
    """
    Log a single message to JSONL file (backward compatible function).

    Args:
        message_dict: Message dictionary
        json_log_path: Path to JSONL log file
        session_id: Optional session ID to add to message
        prompt_index: Optional prompt index to add to message
    """
    path = Path(json_log_path)

    # Enrich message
    enriched = message_dict.copy()
    if session_id:
        enriched["session_id"] = session_id
    if prompt_index is not None:
        enriched["prompt_index"] = prompt_index

    # Append single line (O(1) instead of O(n))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        line = json.dumps(enriched, ensure_ascii=False, default=str)
        f.write(line + "\n")


__all__ = [
    "JSONSink",
    "log_message",
]
