"""System prompt for strategy generation (Step 3.1: GEN_STRAT).

Read top-to-bottom to understand the full prompt structure.
"""

from ....components.aii_context import get_aii_context
from ....components.research_practices import get_research_practices
from ....components.subagents import get_no_subagents_guidance
from ....components.tool_calling import get_web_tool_guidance, get_tool_calling_guidance
from ....components.resources import get_resources_prompt, ARTIFACT_RESOURCES
from ....components.work_solo_reminder import get_work_solo_reminder


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(context: str, resources_text: str) -> str:
    return f"""{context}

{resources_text}

{get_web_tool_guidance()}

{get_tool_calling_guidance()}

{get_research_practices("gen_strat")}

<principles>
1. FOCUS ON NOVELTY - every strategy must lead to a genuinely novel contribution
2. MAXIMIZE PARALLELIZATION - all artifacts in your strategy run in parallel
3. BUILD ON EXISTING WORK - use completed artifacts from previous iterations, learn from failures
4. ITERATE ON THE METHOD - don't settle on one approach. When results are weak or negative, try different variations (different algorithms, parameters, techniques) in parallel. A good research strategy explores multiple method variants, not just one.
5. SET DEPENDENCIES WISELY - depends_on uses IDs from existing artifacts (not your strategy's); the right dependencies help the executor understand what was already done and build on it
6. PLAN FOR DEPENDENCIES - if an artifact depends on another (e.g. experiments need datasets), ensure prerequisites exist first or plan them this iteration for the next
</principles>

{get_work_solo_reminder()}

{get_no_subagents_guidance()}"""


# =============================================================================
# HELPERS
# =============================================================================

def _get_combined_resource_keys(allowed_artifacts: list[str] | None = None) -> list[str]:
    """Get union of all resource keys needed for the allowed artifact types."""
    if allowed_artifacts is None:
        allowed_artifacts = list(ARTIFACT_RESOURCES.keys())
    combined: set[str] = set()
    for art_type in allowed_artifacts:
        combined |= ARTIFACT_RESOURCES.get(art_type, set())
    return list(combined)


# =============================================================================
# EXPORTS
# =============================================================================

def get(allowed_artifacts: list[str] | None = None) -> str:
    """Build system prompt for strategy generation."""
    resource_keys = _get_combined_resource_keys(allowed_artifacts)
    return PROMPT(
        context=get_aii_context(focus="gen_strat"),
        resources_text=get_resources_prompt(include=resource_keys) if resource_keys else get_resources_prompt(),
    )
