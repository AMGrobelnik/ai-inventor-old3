"""User prompt for narrative generation (GEN_NARR)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_pipeline.utils import to_prompt_yaml

if TYPE_CHECKING:
    from aii_pipeline.steps._invention_loop.pools import ArtifactPool


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(
    hypothesis_formatted: str,
    artifacts_json: str,
    artifact_count: int,
) -> str:
    return f"""<hypothesis>
{hypothesis_formatted}
</hypothesis>

<artifacts>
All {artifact_count} completed artifacts. Synthesize ALL into a coherent narrative.
Reference each by ID (e.g., "As shown in [exp_exec_iter1_idx1], ...").

{artifacts_json}
</artifacts>

<task>
Write a research narrative synthesizing all artifacts into a coherent story. This serves two purposes: (1) basis for the final research paper, and (2) input for planning the next research iteration.

1. Reference every artifact by ID (e.g., [exp_exec_iter1_idx1]) and explain its role
2. Show how artifacts connect — what built on what, what enabled what
3. Highlight key findings, quantitative results, and their significance
4. Be honest about what worked, what failed, and why
5. Identify specific gaps: what evidence is missing, what claims are unsupported, what alternative explanations haven't been tested
</task>"""


# =============================================================================
# HELPERS
# =============================================================================

def _format_hypothesis(hypo: dict) -> str:
    """Format hypothesis dict as YAML for LLM readability."""
    return to_prompt_yaml(hypo)


# =============================================================================
# EXPORTS
# =============================================================================

def get(
    hypothesis: dict,
    artifact_pool: ArtifactPool,
) -> str:
    """Build user prompt for narrative generation."""
    hypo_display = {
        k: v for k, v in hypothesis.items()
        if k not in ["hypothesis_id", "is_seeded", "model"]
        and not (k == "seeds" and not hypothesis.get("is_seeded"))
    }

    return PROMPT(
        hypothesis_formatted=_format_hypothesis(hypo_display),
        artifacts_json=artifact_pool.get_prompt(
            include={"id", "type", "title", "summary", "out_dependency_files"},
        ) or "No artifacts yet.",
        artifact_count=len(artifact_pool.get_all()),
    )
