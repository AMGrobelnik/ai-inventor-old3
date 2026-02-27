"""Schema for invention loop step output."""

from __future__ import annotations

from pydantic import Field

from aii_pipeline.prompts.steps.base import BaseStepOut
from aii_pipeline.prompts.steps._3_invention_loop._4_gen_narr.schema import Narrative
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.schema import BaseArtifact


class InventionLoopOut(BaseStepOut):
    """Output of the invention_loop module."""
    pools_dir: str = Field(default="", description="Path to pools directory")
    narrative: Narrative | None = Field(default=None, description="Selected narrative")
    artifacts: list[BaseArtifact] = Field(default_factory=list, description="All artifacts produced")
    hypothesis: dict = Field(default_factory=dict, description="Input hypothesis")
