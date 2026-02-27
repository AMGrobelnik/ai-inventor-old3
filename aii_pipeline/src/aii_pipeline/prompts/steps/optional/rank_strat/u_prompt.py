"""User prompt for strategy ranking (Optional: RANK_STRAT).

Read top-to-bottom to understand the full prompt structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_pipeline.utils import to_prompt_yaml

if TYPE_CHECKING:
    from aii_pipeline.steps._invention_loop.pools import ArtifactPool


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(
    current_iteration: int,
    max_iterations: int,
    remaining: int,
    artifact_pool_text: str,
    strategy_a_text: str,
    strategy_b_text: str,
) -> str:
    return f"""<iteration_context>
Current iteration: {current_iteration} of {max_iterations}
Remaining (including this one): {remaining}
</iteration_context>

<artifact_pool>
Artifacts produced so far. Strategies can only build on these existing artifacts.

{artifact_pool_text if artifact_pool_text else "(No artifacts produced yet)"}
</artifact_pool>

<strategy_a>
{strategy_a_text}
</strategy_a>

<strategy_b>
{strategy_b_text}
</strategy_b>

<evaluation_criteria>
Compare the two strategies on:

1. QUALITY - Will this meaningfully advance the research toward a novel contribution?
2. COHERENCE - Do the artifact directions work together toward the objective?
3. FEASIBILITY - Can the planned artifacts be completed this iteration given dependencies?
4. GUIDANCE - Does it set up the next iteration well with clear next steps?
6. DATA QUALITY - For strategies with dataset artifacts:
   - PREFER strategies that use REAL third-party datasets (HuggingFace, Kaggle, academic)
   - PENALIZE strategies that plan synthetic/generated data as primary approach
   - Real datasets have established benchmarks and real-world distributions
</evaluation_criteria>

<task>
Select the better strategy: A or B.

You MUST choose one - no ties allowed.
Consider the WHOLE strategy holistically, not just individual artifacts.
</task>"""


# =============================================================================
# HELPERS
# =============================================================================

def _format_strategy(strategy: dict) -> str:
    """Format strategy dict as YAML for LLM readability."""
    return to_prompt_yaml(strategy)


# =============================================================================
# EXPORTS
# =============================================================================

def get(
    strategy_a: dict,
    strategy_b: dict,
    artifact_pool: ArtifactPool,
    current_iteration: int = 1,
    max_iterations: int = 8,
) -> str:
    """Build user prompt for strategy pairwise comparison."""
    remaining = max_iterations - current_iteration + 1

    return PROMPT(
        current_iteration=current_iteration,
        max_iterations=max_iterations,
        remaining=remaining,
        artifact_pool_text=artifact_pool.get_prompt(
            include={"id", "type", "title", "summary", "out_dependency_files"},
        ),
        strategy_a_text=_format_strategy(strategy_a),
        strategy_b_text=_format_strategy(strategy_b),
    )
