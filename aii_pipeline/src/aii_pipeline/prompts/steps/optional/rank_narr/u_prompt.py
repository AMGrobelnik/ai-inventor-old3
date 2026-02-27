"""User prompt for narrative ranking (Optional: NARR_RANK).

Read top-to-bottom to understand the full prompt structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._4_gen_narr.schema import Narrative


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(
    narrative_a_content: str,
    narrative_b_content: str,
) -> str:
    return f"""<task_preview>
You will compare two research narratives and decide which tells a better research story.
</task_preview>

<narrative_a>
{narrative_a_content}
</narrative_a>

<narrative_b>
{narrative_b_content}
</narrative_b>

<comparison_criteria>
Compare these research narratives:
1. Which tells a more compelling, complete research story?
2. What's MISSING from each narrative that would strengthen it?

You MUST choose A or B.
</comparison_criteria>"""


# =============================================================================
# EXPORTS
# =============================================================================

def get(narrative_a: Narrative, narrative_b: Narrative) -> str:
    """Build user prompt for pairwise narrative comparison."""
    return PROMPT(
        narrative_a_content=narrative_a.narrative,
        narrative_b_content=narrative_b.narrative,
    )
