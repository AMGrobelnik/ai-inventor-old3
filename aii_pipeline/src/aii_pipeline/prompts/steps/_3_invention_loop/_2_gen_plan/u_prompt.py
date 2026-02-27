"""User prompt for plan generation (GEN_PLAN).

Expands a single artifact_direction into a detailed plan.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from aii_pipeline.utils import to_prompt_yaml
from aii_pipeline.prompts.components.artifact_planning import get_artifact_planning
from aii_pipeline.prompts.components.artifact_scope import get_artifact_scope

if TYPE_CHECKING:
    from aii_pipeline.steps._invention_loop.pools import ArtifactPool
    from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.schema import ArtifactDirection


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

_PLANNING_ONLY = """YOUR ROLE: Write a detailed PLAN for the artifact. A separate executor agent runs the actual artifact later.

You are a PLANNER, not an executor. Your output is a plan that tells the executor what to do and how.
Do NOT execute the artifact itself — a separate agent handles that. Your job is to plan it so well that the executor can follow your plan step by step.

You CAN and SHOULD: search the web, read papers, and explore library docs to make your plan concrete.
You CANNOT run Bash commands or scripts — code execution is disabled. Research via web tools only.

Do NOT do the executor's job: don't download datasets, don't implement code, don't run experiments, don't write proofs, don't compute evaluations."""


def PROMPT(
    hypothesis_text: str,
    artifact_direction_text: str,
    dependencies_text: str,
    artifact_planning_guidance: str,
    artifact_scope_text: str,
) -> str:
    return f"""<hypothesis>
{hypothesis_text}
</hypothesis>

<artifact_direction>
Make this direction concrete and actionable. Keep the same type and respect dependencies.

{artifact_direction_text}
</artifact_direction>

{f'''<dependencies>
Completed artifacts this artifact can use during execution.

{dependencies_text}
</dependencies>''' if dependencies_text else ''}

<instructions>
{_PLANNING_ONLY}

{artifact_scope_text}

{artifact_planning_guidance}

GOOD PLANS: specific, actionable, consider failure scenarios, build on the suggested approach.
BAD PLANS: vague hand-waving, ignoring the suggested approach, missing critical executor details.
</instructions>"""


# =============================================================================
# HELPERS
# =============================================================================


def _format_hypothesis(hypo: dict) -> str:
    """Format hypothesis dict as YAML for LLM readability."""
    return to_prompt_yaml(hypo)


def _format_artifact_direction(direction: ArtifactDirection) -> str:
    """Format artifact direction as YAML for LLM readability."""
    return direction.to_prompt_yaml()


_PLAN_DEPENDENCY_FIELDS: set[str] = {"id", "type", "title", "summary", "out_dependency_files"}


def _format_dependencies_context(
    direction: ArtifactDirection,
    artifact_pool: ArtifactPool,
) -> str:
    """Format dependency artifacts as YAML for LLM readability."""
    if not direction.depends_on:
        return ""
    return artifact_pool.get_prompt(
        ids=list(direction.depends_on),
        include=_PLAN_DEPENDENCY_FIELDS,
        label="Dependency",
    )


# =============================================================================
# EXPORTS
# =============================================================================


def get(
    hypothesis: dict,
    artifact_pool: ArtifactPool,
    artifact_direction: ArtifactDirection,
) -> str:
    """Build user prompt for expanding an artifact direction into a plan."""
    hypo_filtered = {
        k: v for k, v in hypothesis.items()
        if k not in ["hypothesis_id", "is_seeded", "model"]
        and not (k == "seeds" and not hypothesis.get("is_seeded"))
    }

    artifact_type = artifact_direction.type

    return PROMPT(
        hypothesis_text=_format_hypothesis(hypo_filtered),
        artifact_direction_text=_format_artifact_direction(artifact_direction),
        dependencies_text=_format_dependencies_context(artifact_direction, artifact_pool),
        artifact_planning_guidance=get_artifact_planning([artifact_type]),
        artifact_scope_text=get_artifact_scope([artifact_type]),
    )
