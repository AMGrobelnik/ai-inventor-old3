"""
AIITelemetry sinks - destinations for telemetry messages.
"""

from .base import Sink
from .console import ConsoleSink
from .json import JSONSink

__all__ = [
    "Sink",
    "ConsoleSink",
    "JSONSink",
]
