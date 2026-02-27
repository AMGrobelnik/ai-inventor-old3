"""Schema for seed hypothesis step output."""

from pydantic import Field

from aii_pipeline.prompts.steps.base import BaseStepOut


class SeedHypoOut(BaseStepOut):
    """Output of the seed_hypo module."""
    agent_prompts: list[list[dict]] = Field(default_factory=list, description="Per-agent seed prompts")
    agent_topics: list[list[str]] = Field(default_factory=list, description="Per-agent topic assignments")
    selected_topics: list[str] = Field(default_factory=list, description="Topics selected for sampling")
    pools: dict[str, list[str]] = Field(default_factory=dict, description="Sampling pools: topic -> seed IDs")
    all_hypo_prompts: list[dict] = Field(default_factory=list, description="All available hypothesis prompts")
