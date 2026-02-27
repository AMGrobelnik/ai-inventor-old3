"""Centralized logging configuration for the pipeline."""

from pathlib import Path
from typing import Optional

from aii_lib.telemetry import AIITelemetry, ConsoleSink, JSONSink, load_telemetry_config

# ============================================================================
# Color constants for terminal output
# ============================================================================
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
WHITE = "\033[97m"
END = "\033[0m"

# ============================================================================
# Logging setup
# ============================================================================

_telemetry: Optional[AIITelemetry] = None
_logging_configured = False


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path | str] = None,
    log_file: Optional[str] = None,
    rotation: str = "30 MB"
) -> AIITelemetry:
    """
    Configure telemetry logging for the pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR) - currently ignored
        log_dir: Directory for log files
        log_file: Log file name (default: pipeline_messages.jsonl)
        rotation: Log rotation size - currently ignored (telemetry doesn't rotate)

    Returns:
        Configured AIITelemetry instance
    """
    global _telemetry, _logging_configured

    if _logging_configured and _telemetry:
        return _telemetry

    # Create telemetry with console sink (respect config)
    config = load_telemetry_config()
    truncation_val = config.get("console_msg_truncate", 150)
    if truncation_val is False or truncation_val is None:
        truncation = None
    else:
        truncation = int(truncation_val)
    log_messages = config.get("log_messages", True)

    _telemetry = AIITelemetry()
    _telemetry.add_sink(ConsoleSink(truncation=truncation, log_messages=log_messages))

    # Add JSON sink if log_dir specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_file or "pipeline_messages.jsonl"
        log_path = log_dir / log_file
        _telemetry.add_sink(JSONSink(log_path))

    _logging_configured = True
    return _telemetry


def get_logger(name: str = "") -> AIITelemetry:
    """
    Get the telemetry instance.

    Args:
        name: Module name (currently unused, kept for API compatibility)

    Returns:
        AIITelemetry instance
    """
    global _telemetry
    if _telemetry is None:
        _telemetry = setup_logging()
    return _telemetry
