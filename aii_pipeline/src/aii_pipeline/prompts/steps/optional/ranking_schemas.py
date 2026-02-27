"""Shared schemas for all LLM-as-judge ranking modules.

LLM Output Schemas (structured output from judge LLM):
- PairwisePreference: A/B choice with justification
- PairwisePreferenceSimple: A/B choice only

Pool / Result Schemas (used by ranking steps and invention loop):
- Gap: Research gap identified during narrative ranking
- PairwiseJudgment: Result of a pairwise comparison
- RankingResult: Full ranking result with scores and judgments
- IterationStats: Statistics for a single iteration
"""

from pydantic import BaseModel, Field
from typing import Annotated, Literal

from aii_lib.prompts import LLMPromptModel, LLMStructOutModel, LLMPrompt, LLMStructOut
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.schema import PlanType


# =============================================================================
# LLM OUTPUT SCHEMAS — used by SwissBTRanker as response_schema
# =============================================================================


class PairwisePreference(LLMPromptModel, LLMStructOutModel):
    """Pairwise comparison preference (with justification)."""

    preferred: Annotated[Literal["A", "B"], LLMPrompt, LLMStructOut] = Field(
        description="Which item is preferred: 'A' or 'B'"
    )
    justification: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Brief justification for the preference (~75 words)."
    )


class PairwisePreferenceSimple(LLMPromptModel, LLMStructOutModel):
    """Pairwise comparison preference — just the choice, no justification."""

    preferred: Annotated[Literal["A", "B"], LLMPrompt, LLMStructOut] = Field(
        description="Which item is preferred: 'A' or 'B'"
    )


# =============================================================================
# POOL / RESULT SCHEMAS — used by ranking steps and invention loop
# =============================================================================


class Gap(BaseModel):
    """A gap identified during narrative ranking."""
    description: str = Field(description="Description of what's missing")
    frequency: int = Field(default=1, description="How often this gap was mentioned")
    source_narratives: list[str] = Field(default_factory=list, description="Narratives with this gap")
    suggested_type: PlanType | None = Field(default=None, description="Suggested artifact type to fill gap")


class PairwiseJudgment(BaseModel):
    """Result of a pairwise comparison."""
    item_a_id: str = Field(description="ID of item A")
    item_b_id: str = Field(description="ID of item B")
    preferred: str = Field(description="Which item won: 'A' or 'B'")
    reasoning: str = Field(description="Brief reasoning for the decision")
    judge_model: str = Field(description="Which LLM judged this")
    gap_a: str | None = Field(default=None, description="Gap in item A")
    gap_b: str | None = Field(default=None, description="Gap in item B")


class RankingResult(BaseModel):
    """Result of sparse pairwise ranking."""
    ranked_items: list[str] = Field(description="Item IDs in rank order (best first)")
    bt_updates: dict[str, float] = Field(description="New BT scores for each item")
    win_rates: dict[str, float] = Field(description="Win rates for each item")
    all_judgments: list[PairwiseJudgment] = Field(description="All pairwise comparisons made")
    gaps_extracted: list[Gap] = Field(default_factory=list, description="Gaps extracted (narrative ranking)")


class IterationStats(BaseModel):
    """Statistics for a single iteration."""
    iteration: int
    narratives_produced: list[str] = Field(default_factory=list)
    best_narrative_id: str = Field(default="")
