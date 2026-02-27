"""System prompt for narrative generation (Step 3.4: NARRATE).

Read top-to-bottom to understand the full prompt structure.
"""

from ....components.aii_context import get_aii_context
from ....components.research_practices import get_research_practices
from ....components.subagents import get_no_subagents_guidance
from ....components.tool_calling import get_web_tool_guidance, get_tool_calling_guidance
from ....components.work_solo_reminder import get_work_solo_reminder


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(context: str) -> str:
    return f"""{context}

{get_research_practices("gen_narr")}

{get_web_tool_guidance()}

{get_tool_calling_guidance()}

{get_work_solo_reminder()}

{get_no_subagents_guidance()}"""


# =============================================================================
# EXPORTS
# =============================================================================

def get() -> str:
    """Get system prompt for narrative generation."""
    return PROMPT(context=get_aii_context(focus="gen_narr"))
