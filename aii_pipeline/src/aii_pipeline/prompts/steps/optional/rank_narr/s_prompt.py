"""System prompt for narrative ranking (Optional: NARR_RANK).

Read top-to-bottom to understand the full prompt structure.
"""

from ....components.aii_context import get_aii_context
from ....components.subagents import get_no_subagents_guidance
from ....components.work_solo_reminder import get_work_solo_reminder


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(context: str) -> str:
    return f"""{context}

<ranking_criteria>
Consider when comparing narratives:
1. Clarity - Is the narrative clear and well-structured?
2. Coherence - Do the findings build on each other logically?
3. Completeness - Does it address the research question fully?
4. Evidence - Is the story well-supported by the artifacts?
5. Significance - Does it make meaningful contributions?

You must also identify what's MISSING from each narrative.
These gaps will seed the next round of research plans.

You MUST choose A or B - no ties allowed.
</ranking_criteria>

{get_work_solo_reminder()}

{get_no_subagents_guidance()}"""


# =============================================================================
# EXPORTS
# =============================================================================

def get() -> str:
    """Get system prompt for narrative ranking."""
    return PROMPT(context=get_aii_context(focus="rank_narr"))
