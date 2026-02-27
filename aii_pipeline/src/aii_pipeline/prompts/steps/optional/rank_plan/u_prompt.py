"""User prompt for plan ranking (Optional: RANK_PROP).

Read top-to-bottom to understand the full prompt structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....components.artifact_summaries import get_artifact_context


if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.schema import Strategy, ArtifactDirection
    from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.schema import BasePlan
    from aii_pipeline.steps._invention_loop.pools import ArtifactPool


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(
    artifact_pool_summary: str,
    strategy_text: str,
    artifact_direction_text: str,
    artifact_type_description: str,
    plan_a_formatted: str,
    plan_b_formatted: str,
) -> str:
    return f"""<artifact_pool>
Artifacts produced so far. Use this to understand what exists and what can be built upon.

{artifact_pool_summary if artifact_pool_summary else "(No artifacts produced yet)"}
</artifact_pool>

<current_strategy>
The strategy guiding this iteration's work. Both plans below were
generated to implement an artifact direction from this strategy.

{strategy_text}
</current_strategy>

<source_artifact_direction>
Both plans below are elaborations of this specific artifact direction.
The plans are PLANS that describe how to implement this direction.

{artifact_direction_text}
</source_artifact_direction>

<artifact_type_context>
What this artifact type is capable of - use this to evaluate if plans
are making good use of the available capabilities.

{artifact_type_description}
</artifact_type_context>

{plan_a_formatted}

{plan_b_formatted}

<evaluation_criteria>
These plans are PLANS that will be passed to an executor agent for implementation.
Evaluate them as plans, not as executed work.

Compare plans on:
1. FAITHFULNESS - Does it follow the artifact direction's objective and approach?
2. ACTIONABILITY - Can an executor implement this plan without ambiguity?
3. FEASIBILITY - Is the plan realistic given available resources and dependencies?
4. QUALITY - Will the planned output be useful for downstream artifacts and the research?

FOR DATASET PROPOSALS SPECIFICALLY:
- STRONGLY PREFER plans that plan to use REAL third-party datasets (HuggingFace, Kaggle, academic)
- PENALIZE plans that plan to generate synthetic data as PRIMARY approach
- Synthetic data is acceptable ONLY as fallback when no real datasets exist
</evaluation_criteria>

<task>
Which plan is a better PLAN that would be more likely to succeed when an
executor agent implements it?

You MUST choose A or B.
</task>"""


# =============================================================================
# HELPERS
# =============================================================================

def _format_plan(plan: BasePlan, label: str) -> str:
    """Format a plan as YAML for LLM readability."""
    label_lower = label.lower()
    return f"""<plan_{label_lower}>
{plan.to_prompt_yaml()}
</plan_{label_lower}>"""


def _format_strategy(strategy: Strategy) -> str:
    """Format strategy as YAML for LLM readability."""
    return strategy.to_prompt_yaml()


def _format_artifact_direction(direction: ArtifactDirection) -> str:
    """Format artifact direction as YAML for LLM readability."""
    return direction.to_prompt_yaml()


def _get_artifact_direction_by_id(strategy: Strategy, direction_id: str | None) -> ArtifactDirection | None:
    """Find artifact direction in strategy by ID."""
    if not direction_id:
        return None
    for direction in strategy.artifact_directions:
        if direction.id == direction_id:
            return direction
    return None


# =============================================================================
# EXPORTS
# =============================================================================

def get(
    plan_a: BasePlan,
    plan_b: BasePlan,
    artifact_pool: ArtifactPool,
    strategy: Strategy,
) -> str:
    """Build user prompt for pairwise plan comparison."""
    # Get the artifact direction that these plans elaborate
    # (both should share the same in_art_direction_id)
    artifact_direction = _get_artifact_direction_by_id(strategy, plan_a.in_art_direction_id)

    # Get artifact type description for this plan type
    artifact_type = plan_a.type.value
    artifact_type_description = get_artifact_context([artifact_type])

    return PROMPT(
        artifact_pool_summary=artifact_pool.get_prompt(
            include={"id", "type", "title", "summary", "out_dependency_files"},
        ),
        strategy_text=_format_strategy(strategy),
        artifact_direction_text=_format_artifact_direction(artifact_direction) if artifact_direction else "(Not available)",
        artifact_type_description=artifact_type_description,
        plan_a_formatted=_format_plan(plan_a, "A"),
        plan_b_formatted=_format_plan(plan_b, "B"),
    )
