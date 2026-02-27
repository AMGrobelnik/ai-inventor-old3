"""System prompt for hypothesis audit - generating cited arguments.

Two system prompts: one for novelty (web search), one for feasibility (resources).
"""

from ....components.aii_context import get_aii_context
from ..citation_format import (
    get_novelty_citation_format,
    get_feasibility_citation_format,
)
from ....components.tool_calling import get_tool_calling_guidance
from ....components.subagents import get_no_subagents_guidance
from ....components.work_solo_reminder import get_work_solo_reminder


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

def NOVELTY_PROMPT() -> str:
    """System prompt for novelty arguments (uses web search)."""
    context = get_aii_context(focus="audit_hypo")
    citation_format = get_novelty_citation_format()
    tool_guidance = get_tool_calling_guidance()

    return f"""{context}

<principles>
- Every claim MUST be backed by a direct quote from a credible source
- Use web search and web fetch to find supporting or contradicting evidence
- Quotes must be EXACT and VERBATIM from the source
- Be objective - construct the strongest possible argument for your assigned position
</principles>

<source_selection>
Good sources: arxiv.org, academic papers, reputable tech blogs, official documentation.
Avoid: social media, forums, opinion pieces, outdated content.
If a search yields no results, try alternative keywords rather than retrying the same query.
</source_selection>

{citation_format}

{tool_guidance}

<output_requirements>
Return ONLY the argument text with embedded \"\"\"quote\"\"\" (url) citations.
</output_requirements>"""


def FEASIBILITY_PROMPT() -> str:
    """System prompt for feasibility arguments (uses provided resources only)."""
    context = get_aii_context(focus="audit_hypo")
    citation_format = get_feasibility_citation_format()

    return f"""{context}

<principles>
- Every claim MUST be backed by a direct quote from <available_resources>
- Quotes must be EXACT and VERBATIM from the resource text
- Be objective - construct the strongest possible argument for your assigned position
</principles>

{citation_format}

<output_requirements>
Return ONLY the argument text with embedded \"\"\"quote\"\"\" citations.
</output_requirements>"""


# =============================================================================
# EXPORTS
# =============================================================================

def get_novelty() -> str:
    """System prompt for novelty arguments (uses web search)."""
    return get_work_solo_reminder() + "\n\n" + get_no_subagents_guidance() + "\n\n" + NOVELTY_PROMPT()


def get_feasibility() -> str:
    """System prompt for feasibility arguments (uses provided resources)."""
    return get_work_solo_reminder() + "\n\n" + get_no_subagents_guidance() + "\n\n" + FEASIBILITY_PROMPT()
