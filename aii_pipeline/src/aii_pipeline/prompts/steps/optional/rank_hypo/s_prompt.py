"""System prompt for hypothesis ranking via pairwise comparison."""

from ....components.aii_context import get_aii_context
from ....components.subagents import get_no_subagents_guidance
from ....components.work_solo_reminder import get_work_solo_reminder


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(context: str) -> str:
    return f"""{context}

<how_to_use_audit_evidence>
Each hypothesis has been audited with VERIFIED citations:
- NOVELTY arguments: Evidence about whether the idea is genuinely new
  - Positive: Citations showing gaps in existing work, unexplored territory
  - Negative: Citations showing similar prior work exists
- FEASIBILITY arguments: Evidence about whether it can be implemented
  - Positive: Quotes from available resources showing required capabilities exist
  - Negative: Quotes showing missing capabilities or barriers

<CRITICAL>
The audit arguments contain VERIFIED quotes from real sources.
Treat them as ground truth evidence - these citations have been checked.
</CRITICAL>

When comparing:
1. Read the cited arguments carefully - they are verified evidence
2. Weigh the strength of positive vs negative arguments for each dimension
3. A hypothesis with strong positive arguments and weak negative arguments is better
4. Consider the quality and relevance of citations, not just quantity
</how_to_use_audit_evidence>

<ranking_principles>
<CRITICAL>
- Be DECISIVE - you MUST pick A or B, never tie or refuse
- Reference SPECIFIC audit arguments that influenced your choice
</CRITICAL>

1. Compare holistically - consider both the hypothesis itself AND its audit arguments
2. Justify concretely - cite specific quotes from the audit that swayed you
3. Prioritize novelty over feasibility when both are acceptable
4. Penalize hypotheses with strong negative arguments that aren't countered
5. If both seem equal, pick the one with clearer methodology
</ranking_principles>

{get_work_solo_reminder()}

{get_no_subagents_guidance()}"""


# =============================================================================
# EXPORTS
# =============================================================================

def get() -> str:
    """Get system prompt for hypothesis ranker."""
    return PROMPT(context=get_aii_context(focus="rank_hypo"))
