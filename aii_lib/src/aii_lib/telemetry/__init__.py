"""
AIITelemetry - Unified logging system for aii_lib.

Architecture:
- AIITelemetry: Main class with emit() and emit_message() for logging
- Sinks: ConsoleSink, JSONSink (destinations for messages)
- MessageType: Enum of standardized message types
- TelemetryMessage: Standard message dataclass

Usage:
    from aii_lib.telemetry import AIITelemetry, ConsoleSink, JSONSink, MessageType

    telemetry = AIITelemetry()
    telemetry.add_sink(ConsoleSink())
    telemetry.add_sink(JSONSink("logs/messages.json"))

    # For task-associated messages (sequenced):
    telemetry.emit_message(MessageType.INFO, "Processing...", task_name, task_id)
    telemetry.emit_message(MessageType.SUCCESS, "Done!", task_name, task_id)
    telemetry.emit_message("VERIFY", "Verifying...", task_name, task_id)  # custom label

    # For general logging (not task-associated):
    telemetry.emit(MessageType.INFO, "Starting...")
    telemetry.emit(MessageType.ERROR, "Error occurred")

Simple logging (drop-in replacement for loguru):
    from aii_lib.telemetry import log

    log.info("Starting...")
    log.debug("Debug message")
    log.warning("Warning!")
    log.error("Error occurred")
    log.success("Done!")
"""

from .telemetry import AIITelemetry, create_telemetry, load_telemetry_config
from .message import MessageType, TelemetryMessage, SummaryMetrics, AggregatedMetrics
from .sinks import Sink, ConsoleSink, JSONSink
from .sinks.console import Colors, get_color, colorize


class _DefaultLogger:
    """
    Simple logger with console output - drop-in replacement for loguru.

    Uses ConsoleSink for formatted output. Respects console_msg_truncate from config.yaml.

    All logging methods accept an optional `exc` parameter to include traceback:
        logger.warning("Request failed", exc=exc)  # includes full traceback
    """

    def __init__(self):
        self._telemetry = AIITelemetry()
        # Respect config for truncation
        config = load_telemetry_config()
        truncation_val = config.get("console_msg_truncate", 150)
        if truncation_val is False or truncation_val is None:
            truncation = None
        else:
            truncation = int(truncation_val)
        log_messages = config.get("log_messages", True)
        self._telemetry.add_sink(ConsoleSink(truncation=truncation, log_messages=log_messages))

    def _format_with_exc(self, msg: str, exc: BaseException | None) -> str:
        """Format message with traceback if exception provided."""
        if exc is None:
            return str(msg)
        import traceback
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return f"{msg}\n{tb_str}"

    def info(self, msg: str, exc: BaseException | None = None, **kwargs) -> None:
        """Log info message. Pass exc= to include traceback."""
        self._telemetry.emit(MessageType.INFO, self._format_with_exc(msg, exc), **kwargs)

    def debug(self, msg: str, exc: BaseException | None = None, **kwargs) -> None:
        """Log debug message. Pass exc= to include traceback."""
        self._telemetry.emit(MessageType.DEBUG, self._format_with_exc(msg, exc), **kwargs)

    def warning(self, msg: str, exc: BaseException | None = None, **kwargs) -> None:
        """Log warning message. Pass exc= to include traceback."""
        self._telemetry.emit(MessageType.WARNING, self._format_with_exc(msg, exc), **kwargs)

    def error(self, msg: str, exc: BaseException | None = None, **kwargs) -> None:
        """Log error message. Pass exc= to include traceback."""
        self._telemetry.emit(MessageType.ERROR, self._format_with_exc(msg, exc), **kwargs)

    def success(self, msg: str, exc: BaseException | None = None, **kwargs) -> None:
        """Log success message. Pass exc= to include traceback."""
        self._telemetry.emit(MessageType.SUCCESS, self._format_with_exc(msg, exc), **kwargs)

    def exception(self, msg: str, **kwargs) -> None:
        """Log error with exception info from current context (use inside except block)."""
        import traceback
        full_msg = f"{msg}\n{traceback.format_exc()}"
        self._telemetry.emit(MessageType.ERROR, full_msg, **kwargs)

    def catch(self, *args, **kwargs):
        """Decorator for catching exceptions (mimics loguru.catch).

        Works with both sync and async functions.
        """
        reraise = kwargs.get("reraise", True)

        def decorator(func):
            import functools
            import asyncio

            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*func_args, **func_kwargs):
                    try:
                        return await func(*func_args, **func_kwargs)
                    except Exception as e:
                        self.exception(f"Exception in {func.__name__}: {e}")
                        if reraise:
                            raise
                return async_wrapper
            else:
                @functools.wraps(func)
                def wrapper(*func_args, **func_kwargs):
                    try:
                        return func(*func_args, **func_kwargs)
                    except Exception as e:
                        self.exception(f"Exception in {func.__name__}: {e}")
                        if reraise:
                            raise
                return wrapper

        # Handle @log.catch vs @log.catch()
        if args and callable(args[0]):
            return decorator(args[0])
        return decorator


# Default logger instance - use like loguru's logger
log = _DefaultLogger()

# Backward compatibility alias
logger = log


__all__ = [
    "AIITelemetry",
    "create_telemetry",
    "load_telemetry_config",
    "MessageType",
    "TelemetryMessage",
    "SummaryMetrics",
    "AggregatedMetrics",
    "Sink",
    "ConsoleSink",
    "JSONSink",
    "Colors",
    "get_color",
    "colorize",
    "log",
    "logger",
]
