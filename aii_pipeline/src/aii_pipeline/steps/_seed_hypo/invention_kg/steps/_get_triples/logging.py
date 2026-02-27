#!/usr/bin/env python3
"""Logging configuration for building blocks extraction."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from aii_lib import AIITelemetry, MessageType


# Module-level telemetry (set by caller)
_telemetry: Optional[AIITelemetry] = None


def set_telemetry(telemetry: Optional[AIITelemetry]) -> None:
    """Set module-level telemetry instance."""
    global _telemetry
    _telemetry = telemetry


def _emit(msg_type: MessageType, msg: str) -> None:
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")


def setup_logging(base_dir: Path, resume_dir: Path | None = None):
    """
    Setup logging directory and return log file path.

    Note: Console logging is already configured via aii_lib.telemetry.
    This function just determines the log file path for reference.

    Args:
        base_dir: Base directory for the pipeline
        resume_dir: If resuming, the directory being resumed from (to reuse log file)

    Returns:
        Path to the log file (for reference, actual logging uses telemetry)
    """
    log_dir = base_dir / "log" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine log file name
    if resume_dir is not None:
        # Extract timestamp from resume directory name (format: {num}_{timestamp})
        dir_name = resume_dir.name
        # Try to extract timestamp (anything after first underscore)
        parts = dir_name.split('_', 1)
        if len(parts) > 1:
            timestamp_part = parts[1]
            # Look for existing log file with this timestamp
            existing_log = log_dir / f"bblocks_{timestamp_part}.log"
            if existing_log.exists():
                log_file = existing_log
                _emit(MessageType.INFO, f"Resuming - appending to existing log file: {log_file}")
            else:
                # Log file doesn't exist, create new one with same timestamp
                log_file = existing_log
        else:
            # Couldn't parse timestamp, create new log file
            log_file = log_dir / f"bblocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        # Fresh start - create new log file
        log_file = log_dir / f"bblocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    _emit(MessageType.INFO, f"Log directory: {log_dir}")
    return log_file
