"""Invention Loop - Iterative scientific invention with four-step loop.

The Loop: GEN_STRAT → GEN_PLAN → GEN_ART → GEN_NARR → loop

Four Pools:
- StrategyPool: Research strategies per iteration
- PlanPool: Pending work elaborated from strategy directions
- ArtifactPool: Completed work (only successes stored)
- NarrativePool: Research stories synthesized from artifacts
"""

from .pools import StrategyPool, PlanPool, ArtifactPool, NarrativePool, parse_iteration, get_type_abbrev

from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.schema import Strategy, ArtifactDirection
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.schema import (
    BasePlan, PlanType, AnyPlan,
    ProofPlan, ResearchPlan, DatasetPlan,
    ExperimentPlan, EvaluationPlan,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.schema import BaseArtifact, ArtifactType
from aii_pipeline.prompts.steps._3_invention_loop._4_gen_narr.schema import Narrative

from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.research.schema import ResearchArtifact

__all__ = [
    # Pools
    "StrategyPool",
    "PlanPool",
    "ArtifactPool",
    "NarrativePool",
    # Utilities
    "parse_iteration",
    "get_type_abbrev",
    # Enums
    "PlanType",
    "ArtifactType",
    # Core schemas
    "Strategy",
    "ArtifactDirection",
    "BasePlan",
    "AnyPlan",
    "ProofPlan",
    "ResearchPlan",
    "DatasetPlan",
    "ExperimentPlan",
    "EvaluationPlan",
    "BaseArtifact",
    "Narrative",
    # Artifact results
    "ResearchArtifact",
]
