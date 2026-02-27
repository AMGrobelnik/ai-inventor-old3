"""Schemas for plan generation — single inheritance hierarchy.

Plan (base) holds code-assigned metadata + common content fields.
Type-specific subclasses add their own typed fields.
AnyPlan is a discriminated union for deserialization.

Structured output:
- Content fields are marked with LLMStructOut (metadata excluded automatically)
- cls.plan_output_format() returns the Claude/OpenRouter output_format
  for a single plan object (one plan per artifact direction)
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import ConfigDict, Field
from enum import Enum

from aii_lib.prompts import LLMPromptModel, LLMStructOutModel, LLMPrompt, LLMStructOut


# =============================================================================
# ENUMS
# =============================================================================

class PlanType(str, Enum):
    """Types of plans that can be generated."""
    EXPERIMENT = "experiment"
    RESEARCH = "research"
    PROOF = "proof"
    EVALUATION = "evaluation"
    DATASET = "dataset"


# =============================================================================
# BASE PLAN
# =============================================================================

class BasePlan(LLMPromptModel, LLMStructOutModel):
    """Base plan — common fields + code-assigned metadata.

    ID format: plan_{direction_id}_v{N}_it{iteration}__{model}_idx{N}

    Content fields (LLMPrompt + LLMStructOut) are included in prompts and schemas.
    ``id`` and ``type`` are LLMPrompt only (visible in prompts, not LLM-generated).
    Other metadata fields (no markers) are excluded from both.
    """
    model_config = ConfigDict(extra="ignore")

    # Code-assigned metadata (LLMPrompt = visible in prompts, not LLM-generated)
    id: Annotated[str, LLMPrompt] = Field(default="", description="Unique plan ID")
    type: Annotated[PlanType, LLMPrompt] = Field(default=PlanType.EXPERIMENT)
    artifact_dependencies: list[str] = Field(default_factory=list)
    in_art_direction_id: str | None = Field(default=None)
    in_strat_id: str | None = Field(default=None)

    # Common content (LLM-filled)
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Short title for the plan")
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(default="", description="Brief summary")

    @classmethod
    def plan_output_format(cls) -> dict[str, Any]:
        """Build output_format for a single plan object.

        Uses LLMStructOut markers to auto-filter to content fields only.

        Returns:
            {"type": "json_schema", "schema": ...} ready for output_format= or
            access ["schema"] for response_format=.
        """
        return cls.to_struct_output()


# =============================================================================
# TYPE-SPECIFIC PLANS
# =============================================================================

class ProofPlan(BasePlan):
    """Plan for a PROOF artifact."""
    type: Annotated[Literal[PlanType.PROOF], LLMPrompt] = PlanType.PROOF

    informal_proof_draft: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Initial proof sketch in plain language - this is a first draft that may be refined or corrected during execution")
    explanation: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Why this proof matters and how it advances the research")


class ResearchPlan(BasePlan):
    """Plan for a RESEARCH artifact."""
    type: Annotated[Literal[PlanType.RESEARCH], LLMPrompt] = PlanType.RESEARCH

    question: Annotated[str, LLMPrompt, LLMStructOut] = Field(default="", description="The specific research question to investigate")
    research_plan: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Step-by-step plan for web research to gather this research")
    explanation: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Why this research matters and what question it answers")


class DatasetPlan(BasePlan):
    """Plan for a DATASET artifact."""
    type: Annotated[Literal[PlanType.DATASET], LLMPrompt] = PlanType.DATASET

    ideal_dataset_criteria: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="What makes an ideal dataset for this purpose - size, format, content requirements")
    dataset_search_plan: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Step-by-step plan for finding/creating this dataset - sources to check, fallback options")
    target_num_datasets: Annotated[int, LLMPrompt, LLMStructOut] = Field(description="How many individual datasets should be delivered. Count each dataset separately, not collections — a benchmark suite of N datasets counts as N. This controls how broadly the executor searches, so setting it too low will under-collect.")


class ExperimentPlan(BasePlan):
    """Plan for an EXPERIMENT artifact."""
    type: Annotated[Literal[PlanType.EXPERIMENT], LLMPrompt] = PlanType.EXPERIMENT

    implementation_pseudocode: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="High-level pseudocode for the experiment implementation")
    fallback_plan: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="What to do if the primary approach fails - alternative methods, simplified versions")
    testing_plan: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="How to validate the experiment works: start with small/fast tests, look for confirmation signals before running full-scale experiments")


class EvaluationPlan(BasePlan):
    """Plan for an EVALUATION artifact."""
    type: Annotated[Literal[PlanType.EVALUATION], LLMPrompt] = PlanType.EVALUATION

    metrics_descriptions: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="What metrics will be computed and how they're defined")
    metrics_justification: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Why these metrics are the right ones - what do they tell us about the hypothesis")


# =============================================================================
# DISCRIMINATED UNION
# =============================================================================

AnyPlan = Annotated[
    Union[ProofPlan, ResearchPlan, DatasetPlan,
          ExperimentPlan, EvaluationPlan],
    Field(discriminator="type"),
]


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

PLAN_SCHEMAS: dict[str, type[BasePlan]] = {
    "proof": ProofPlan,
    "research": ResearchPlan,
    "dataset": DatasetPlan,
    "experiment": ExperimentPlan,
    "evaluation": EvaluationPlan,
}


def get_plan_schema(artifact_type: str) -> type[BasePlan]:
    """Get the plan schema class for a given artifact type."""
    return PLAN_SCHEMAS.get(artifact_type, ResearchPlan)
