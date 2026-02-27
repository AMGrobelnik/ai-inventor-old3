"""System prompt for evaluation artifact.

Read top-to-bottom to understand the full prompt structure.
"""

from aii_pipeline.prompts.components.aii_context import get_aii_context
from aii_pipeline.prompts.components.subagents import get_no_subagents_guidance
from aii_pipeline.prompts.components.work_solo_reminder import get_work_solo_reminder


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT() -> str:
    context = get_aii_context(focus="gen_art")
    return f"""{context}

<task>
Evaluate experimental results using domain-appropriate methods, metrics, and analysis techniques.
When in doubt, prefer more metrics over fewer — but only ones that make sense for the domain.
</task>

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
    """Get the system prompt for evaluation execution."""
    return PROMPT()
