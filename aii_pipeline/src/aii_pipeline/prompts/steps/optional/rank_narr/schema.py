"""Narrative-specific ranking schemas.

Extends the shared PairwisePreference with gap extraction fields
specific to narrative ranking.

Pool schemas (Gap, RankingResult, etc.) live in ranking_schemas.py.
"""

from pydantic import Field
from typing import Annotated

from aii_lib.prompts import LLMPrompt, LLMStructOut

from aii_pipeline.prompts.steps.optional.ranking_schemas import (
    PairwisePreference as _BasePairwise,
    PairwisePreferenceSimple as _BaseSimple,
)


class PairwisePreference(_BasePairwise):
    """Pairwise narrative comparison — adds gap extraction to shared base."""

    gap_a: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="What's missing from Narrative A that would strengthen it (empty string if none)"
    )
    gap_b: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="What's missing from Narrative B that would strengthen it (empty string if none)"
    )


class PairwisePreferenceSimple(_BaseSimple):
    """Pairwise narrative comparison (no justification) — adds gap extraction."""

    gap_a: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="What's missing from Narrative A that would strengthen it (empty string if none)"
    )
    gap_b: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="What's missing from Narrative B that would strengthen it (empty string if none)"
    )
