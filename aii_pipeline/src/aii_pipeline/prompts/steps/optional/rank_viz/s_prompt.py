"""System prompt for visualization ranking.

Read top-to-bottom to understand the full prompt structure.
"""

from ....components.subagents import get_no_subagents_guidance
from ....components.work_solo_reminder import get_work_solo_reminder


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT() -> str:
    return f"""<role>
You are an expert judge of academic visualizations.

Your task is to compare two figure variations and determine which better matches the specification.
</role>

<evaluation_criteria>
1. Accuracy: Does it correctly represent the data/concept?
2. Clarity: Is it easy to understand?
3. Aesthetics: Is it visually appealing and professional?
4. Completeness: Does it include all necessary elements (labels, legend, title)?
5. Relevance: Does it match the figure specification?

You MUST choose A or B. No ties allowed.
</evaluation_criteria>

{get_work_solo_reminder()}

{get_no_subagents_guidance()}"""


# =============================================================================
# EXPORTS
# =============================================================================

def get() -> str:
    """System prompt for ranking visualization variations."""
    return PROMPT()
