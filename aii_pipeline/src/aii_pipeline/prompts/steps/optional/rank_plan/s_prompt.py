"""System prompt for plan ranking (Optional: RANK_PROP).

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
Consider when comparing:
1. Utility - How much would completing this plan advance the research?
2. Timing - Is this the right thing to do NOW given current artifacts?
3. Dependencies - Are the dependencies satisfied? Is it buildable?
4. Risk/Reward - What's the upside if it succeeds? Cost if it fails?
5. Coherence - Does it fit the research narrative?

You MUST choose A or B - no ties allowed.
</ranking_criteria>

{get_work_solo_reminder()}

{get_no_subagents_guidance()}"""


# =============================================================================
# EXPORTS
# =============================================================================

def get() -> str:
    """Get system prompt for plan ranking."""
    return PROMPT(context=get_aii_context(focus="prop_rank"))
