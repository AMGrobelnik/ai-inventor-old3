"""System prompt for paper writing.

Read top-to-bottom to understand the full prompt structure.
"""

from ....components.research_practices import get_research_practices
from ....components.tool_calling import get_web_tool_guidance, get_tool_calling_guidance
from ....components.subagents import get_no_subagents_guidance
from ....components.work_solo_reminder import get_work_solo_reminder


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT() -> str:
    return f"""{get_research_practices("write_paper")}

{get_web_tool_guidance()}

{get_tool_calling_guidance()}

{get_work_solo_reminder()}

{get_no_subagents_guidance()}"""


# =============================================================================
# EXPORTS
# =============================================================================

def get() -> str:
    """System prompt for writing paper drafts with figure placeholders."""
    return PROMPT()
