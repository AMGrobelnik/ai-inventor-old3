"""System prompt for experiment artifact.

Read top-to-bottom to understand the full prompt structure.
"""

from aii_pipeline.prompts.components.aii_context import get_aii_context
from aii_pipeline.prompts.components.research_practices import get_research_practices
from aii_pipeline.prompts.components.subagents import get_no_subagents_guidance
from aii_pipeline.prompts.components.work_solo_reminder import get_work_solo_reminder


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT() -> str:
    context = get_aii_context(focus="gen_art")
    return f"""{context}

{get_research_practices("experiment")}

<task>
Implement the research methodology as a production-ready experimental system.
Adapt your implementation approach based on the hypothesis and domain requirements.
</task>

<critical_requirements>
- Fully implement the methodology described in hypothesis
- Use appropriate frameworks based on research domain
- Load and process data from the specified data_filepath
- Complete working systems
- Handle all edge cases, errors, and exceptions properly
- Always implement baseline comparison method
- Keep final response under 300 characters
</critical_requirements>

<common_mistakes_to_avoid>
- Loading entire large datasets into memory at once — use streaming, batching, or chunking instead
- Spawning too many parallel processes — stay within the hardware limits listed in resources
- Running scripts that grow memory unboundedly or run indefinitely without timeouts
</common_mistakes_to_avoid>

{get_work_solo_reminder()}

{get_no_subagents_guidance()}"""


# =============================================================================
# EXPORTS
# =============================================================================

def get() -> str:
    """Get the system prompt for experiment execution."""
    return PROMPT()
