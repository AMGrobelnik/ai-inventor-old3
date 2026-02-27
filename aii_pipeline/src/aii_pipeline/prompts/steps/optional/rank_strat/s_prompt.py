"""System prompt for strategy ranking (Optional: RANK_STRAT).

Read top-to-bottom to understand the full prompt structure.
"""

from ....components.aii_context import get_aii_context
from ....components.subagents import get_no_subagents_guidance
from ....components.artifact_summaries import get_artifact_context
from ....components.work_solo_reminder import get_work_solo_reminder


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(context: str, artifact_summaries: str) -> str:
    return f"""{context}

{artifact_summaries}

<ranking_principles>
1. BE DECISIVE - you MUST pick A or B, never tie or refuse
2. HOLISTIC VIEW - evaluate the whole strategy, not just individual artifacts
3. COHERENCE MATTERS - fewer well-integrated artifacts beats many disconnected ones
4. DEPENDENCY VALIDITY - artifacts can only depend on EXISTING artifacts, not other artifacts in the same strategy
</ranking_principles>

{get_work_solo_reminder()}

{get_no_subagents_guidance()}"""


# =============================================================================
# EXPORTS
# =============================================================================

def get() -> str:
    """Build system prompt for strategy ranking."""
    return PROMPT(
        context=get_aii_context(focus="rank_strat"),
        artifact_summaries=get_artifact_context(),
    )
