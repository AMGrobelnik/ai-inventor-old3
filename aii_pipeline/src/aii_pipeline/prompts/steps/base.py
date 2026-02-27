"""Base schema for pipeline step outputs."""

from pydantic import BaseModel, Field


class BaseStepOut(BaseModel):
    """Base class for all pipeline step outputs.

    Every step must produce an output_dir and metadata dict.
    """
    output_dir: str = Field(default="", description="Output directory path")
    metadata: dict = Field(default_factory=dict, description="Module metadata")
