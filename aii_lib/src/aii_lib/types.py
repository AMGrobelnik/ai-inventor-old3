"""Core types for aii_lib."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TokenUsage:
    """Token usage tracking."""

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.reasoning


@dataclass
class GenAIRun:
    """Result record for a single GenAI invocation."""

    # Identity
    id: str
    name: str
    group: str | None = None

    # Input
    prompt: str = ""
    system: str | None = None

    # Output
    result: dict | str | None = None
    status: str = "completed"  # completed/failed
    error: str | None = None

    # Metrics
    cost: float = 0.0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    duration_ms: float = 0.0
    tool_calls: dict[str, int] = field(default_factory=dict)

    # Metadata
    model: str | None = None
    backend_type: str | None = None  # "llm" or "agent"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "group": self.group,
            "prompt": self.prompt[:500] if self.prompt else None,
            "system": self.system[:200] if self.system else None,
            "result": self.result,
            "status": self.status,
            "error": self.error,
            "cost": self.cost,
            "tokens": {
                "input": self.tokens.input,
                "output": self.tokens.output,
                "reasoning": self.tokens.reasoning,
                "total": self.tokens.total,
            },
            "duration_ms": self.duration_ms,
            "tool_calls": self.tool_calls,
            "model": self.model,
            "backend_type": self.backend_type,
            "timestamp": self.timestamp.isoformat(),
        }
