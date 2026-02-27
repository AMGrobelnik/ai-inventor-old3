"""Building Block extraction schema for LLM structured output.

This schema is used by Gemini 2.5 Flash to extract building blocks from papers.
Wikipedia URL verification and Wikidata enrichment happen in postprocessing.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# Type definitions for structured output
ENTITY_TYPES = Literal["task", "method", "data", "artifact", "tool", "concept", "other"]
RELATION_TYPES = Literal["uses", "proposes"]
PAPER_TYPES = Literal["contribution", "survey"]


class BuildingBlock(BaseModel):
    """A building block extracted from a research paper.

    Entity Types:
        task     - Problem being solved (image classification, theorem proving)
        method   - Technique, algorithm (gradient descent, CRISPR)
        data     - Datasets, benchmarks (ImageNet, MNIST)
        artifact - Pre-built: models, libraries (GPT-4, Mathlib)
        tool     - Software, instruments (PyTorch, Lean prover)
        concept  - Abstract ideas, theories (attention, category theory)
        other    - Catch-all

    Relations:
        uses     - Anything EXISTING that the paper uses
        proposes - Anything NEW/NOVEL that the paper creates
    """

    name: str = Field(
        description="Wikipedia article title for this entity (e.g., 'Gradient descent', 'ImageNet')"
    )
    entity_type: ENTITY_TYPES = Field(
        description="Type: task, method, data, artifact, tool, concept, or other"
    )
    relation: RELATION_TYPES = Field(
        description="Relation: uses (existing), proposes (new/novel)"
    )
    relevance: str = Field(
        description="Why this entity matters to the paper (1 sentence)"
    )

    # Fields added in postprocessing (not from LLM)
    wikipedia_url: Optional[str] = Field(
        default=None,
        description="Wikipedia URL (added in postprocessing)"
    )
    wikidata_id: Optional[str] = Field(
        default=None,
        description="Wikidata QID (added in postprocessing)"
    )
    wikidata_desc: Optional[str] = Field(
        default=None,
        description="Wikidata description (added in postprocessing)"
    )
    external_ids: Optional[dict] = Field(
        default=None,
        description="External IDs from Wikidata (added in postprocessing)"
    )


class PaperExtraction(BaseModel):
    """Extraction result for a single paper.

    Paper Types:
        contribution - proposes something new (method, dataset, framework, etc.)
        survey       - literature review, meta-analysis (only references existing work)
    """

    paper_type: PAPER_TYPES = Field(
        description="Paper type: contribution or survey"
    )
    building_blocks: list[BuildingBlock] = Field(
        description="List of building blocks extracted from the paper"
    )


# Schema for LLM extraction (excludes postprocessing fields)
class BuildingBlockLLM(BaseModel):
    """Building block schema for LLM extraction (no postprocessing fields)."""

    name: str = Field(
        description="Wikipedia article title for this entity (e.g., 'Gradient descent', 'ImageNet')"
    )
    entity_type: ENTITY_TYPES = Field(
        description="Type: task, method, data, artifact, tool, concept, or other"
    )
    relation: RELATION_TYPES = Field(
        description="Relation: uses (existing), proposes (new/novel)"
    )
    relevance: str = Field(
        description="Why this entity matters to the paper (1 sentence)"
    )


class PaperExtractionLLM(BaseModel):
    """Extraction schema for LLM (used with Gemini structured output)."""

    paper_type: PAPER_TYPES = Field(
        description="Paper type: contribution or survey"
    )
    building_blocks: list[BuildingBlockLLM] = Field(
        description="List of building blocks extracted from the paper"
    )
