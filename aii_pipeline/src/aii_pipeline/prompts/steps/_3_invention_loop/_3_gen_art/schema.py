"""Schemas for artifact generation — base classes and pool objects.

Base Classes:
- BaseArtifact: Base for all artifact types (pool + per-type inheritance)
- BaseExpectedFiles: Base for per-type expected file specifications

Enums:
- ArtifactType: Enum for artifact types

Per-type subclasses live in their own subdirectories:
- research/schema.py, experiment/schema.py, dataset/schema.py, etc.
"""

from typing import Annotated

from pydantic import Field
from enum import Enum

from aii_lib.prompts import LLMPromptModel, LLMStructOutModel, BaseExpectedFiles, LLMPrompt, LLMStructOut
from aii_lib.agent_backend import ExpectedFile


# =============================================================================
# POOL SCHEMAS
# =============================================================================

class ArtifactType(str, Enum):
    """Types of artifacts that can be produced."""
    EXPERIMENT = "experiment"
    RESEARCH = "research"
    PROOF = "proof"
    EVALUATION = "evaluation"
    DATASET = "dataset"


class BaseArtifact(LLMPromptModel, LLMStructOutModel):
    """A completed artifact.

    Content fields (title, summary) have LLMPrompt + LLMStructOut markers.
    ``id`` and ``type`` are LLMPrompt only (visible in prompts, not LLM-generated).
    Other metadata fields are code-assigned (no markers, excluded from both).

    Only successful artifacts are stored in the pool.

    ID format: {type}_id{N}_it{iteration}__{model}
    """
    id: Annotated[str, LLMPrompt] = Field(default="", description="Unique artifact ID (e.g., exp_id1_it1__sonnet)")
    type: Annotated[ArtifactType, LLMPrompt] = Field(default=ArtifactType.RESEARCH, description="Type of artifact")
    in_plan_id: str = Field(default="", description="ID of the plan this artifact was created from")
    in_dependency_artifact_ids: list[str] = Field(default_factory=list, description="IDs of artifacts this artifact depended on at execution time")
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(default="", description="Short descriptive title (max 15 characters). Must describe content, NOT a status message.")
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(default="", description="Summary for downstream artifacts: what this artifact provides (max 200 words)")
    workspace_path: Annotated[str | None, LLMPrompt] = Field(default=None, description="Absolute path to artifact workspace")
    out_expected_files: list[str] = Field(default_factory=list, description="Files executor should create (for verification)")
    out_demo_files: Annotated[list[ExpectedFile], LLMPrompt] = Field(default_factory=list, description="Primary file(s) to convert to demo formats")
    out_dependency_files: Annotated[dict[str, str | list[str] | None], LLMPrompt] = Field(
        default_factory=dict,
        description="Output files that dependent artifacts can consume.",
    )
