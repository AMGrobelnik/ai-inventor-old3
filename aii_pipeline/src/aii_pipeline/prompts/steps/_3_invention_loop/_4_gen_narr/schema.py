"""Schemas for narrative generation — pool objects and LLM output."""

from typing import Annotated

from pydantic import Field

from aii_lib.prompts import LLMPromptModel, LLMStructOutModel, LLMPrompt, LLMStructOut


# ============================================================================
# Schemas
# ============================================================================

class Narrative(LLMPromptModel, LLMStructOutModel):
    """A research narrative.

    Content fields have LLMPrompt + LLMStructOut markers.
    ``id`` is code-assigned (LLMPrompt only — visible in prompts, not LLM-generated).

    ID format: narr_v{N}_it{iteration}__{model}
    """
    id: Annotated[str, LLMPrompt] = Field(default="", description="Unique narrative ID (e.g., narr_v1_it1__gpt5)")
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(default="", description="Short title for this narrative")
    narrative: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="The full narrative text")
    artifacts_used: Annotated[list[str], LLMPrompt, LLMStructOut] = Field(description="Artifact IDs referenced")
    gaps: Annotated[list[str], LLMPrompt, LLMStructOut] = Field(description="Weaknesses or missing pieces in the current research story")
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Brief summary of the narrative")
