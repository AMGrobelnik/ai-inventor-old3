"""
Base Sink protocol for telemetry destinations.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..message import TelemetryMessage


class Sink(ABC):
    """Base class for telemetry destinations."""

    # If True, this sink bypasses the sequencer and receives messages immediately.
    # If False, messages go through the sequencer (buffered for parallel tasks).
    # Default True - most sinks want immediate writes (e.g., JSON for crash recovery).
    # ConsoleSink sets this to False for clean sequential display of parallel tasks.
    bypass_sequencer: bool = True

    @abstractmethod
    def emit(self, message: "TelemetryMessage") -> None:
        """Emit a message to this sink."""
        pass

    def flush(self) -> None:
        """Flush any buffered data. Override if needed."""
        pass

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass


__all__ = ["Sink"]
