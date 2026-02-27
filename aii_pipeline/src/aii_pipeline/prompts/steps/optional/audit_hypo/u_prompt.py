"""User prompt for hypothesis audit - generating cited arguments for/against feasibility and novelty."""

from aii_pipeline.utils import to_prompt_yaml
from ....components.resources import get_resources_prompt


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

def NOVELTY_PROMPT(hypothesis: dict, stance: str) -> str:
    """User prompt for novelty argument (uses web search)."""
    return f"""<task_preview>
Generate a {"positive" if stance == "positive" else "negative"} argument about this hypothesis's NOVELTY using web search.
</task_preview>

<available_tools>
Use aii_web_search_fast instead of web_search.
Use aii_web_fetch_direct instead of web_fetch.
</available_tools>

<hypothesis>
{_format_hypothesis(hypothesis)}
</hypothesis>

<novelty_factors>
Consider: existing research, prior work, similar approaches already published,
unexplored aspects, genuine contribution beyond current state-of-art.
</novelty_factors>

<YOUR_TASK>
{"Construct a POSITIVE argument for the novelty of this hypothesis. Find evidence that this approach is genuinely new and unexplored." if stance == "positive" else "Construct a NEGATIVE argument against the novelty of this hypothesis. Find existing work that already addresses similar ideas."}

<CRITICAL>
- Build your argument FROM direct quotes, not the other way around
- First search, then find quotes, then structure argument AROUND those quotes
- Each claim must be grounded in a real quote
- Format: \"\"\"exact verbatim quote\"\"\" (https://source-url)

EXACT QUOTES ONLY:
- Quotes must be EXACT word-for-word copies from the source page
- NO paraphrasing, summarizing, or rewording
- NO combining multiple quotes into one
- Copy-paste the exact text as it appears on the page
- Citations will be verified against the source - modified quotes will FAIL

Do NOT make claims then try to find quotes to support them - that leads to hallucination.
Instead, let the quotes you find DRIVE what claims you make.
</CRITICAL>

Return ONLY the argument text with embedded citations.
</YOUR_TASK>"""


def FEASIBILITY_PROMPT(hypothesis: dict, stance: str) -> str:
    """User prompt for feasibility argument (uses provided resources only)."""
    resources = get_resources_prompt()
    return f"""<task_preview>
Generate a {"positive" if stance == "positive" else "negative"} argument about this hypothesis's FEASIBILITY using the provided resources.
</task_preview>

<hypothesis>
{_format_hypothesis(hypothesis)}
</hypothesis>

<feasibility_instructions>
Evaluate feasibility using ONLY the resources listed in <available_resources> below.
Do NOT assume access to any tools, APIs, compute, or data beyond what is explicitly listed.
Consider: Can this be built with these specific resources? What's missing?
</feasibility_instructions>

{resources}

<YOUR_TASK>
{"Construct a POSITIVE argument for the feasibility of this hypothesis. Find evidence that it CAN be implemented using ONLY the resources listed in <available_resources>." if stance == "positive" else "Construct a NEGATIVE argument against the feasibility of this hypothesis. Find evidence of barriers or limitations given ONLY the resources in <available_resources>."}

<CRITICAL>
- Build your argument FROM direct quotes from <available_resources> above
- Review resources, find quotes, then structure argument AROUND those quotes
- Each claim must be grounded in a real quote from <available_resources>
- Format: \"\"\"exact verbatim quote\"\"\"

EXACT QUOTES ONLY:
- Quotes must be EXACT word-for-word copies from <available_resources>
- NO paraphrasing, summarizing, or rewording
- NO combining multiple quotes into one
- Copy-paste the exact text as it appears in the resources
- Citations will be verified against the resources - modified quotes will FAIL

Do NOT make claims then try to find quotes to support them.
Instead, let the quotes you find DRIVE what claims you make.
</CRITICAL>

Return ONLY the argument text with embedded citations.
</YOUR_TASK>"""


NOVELTY_FORCE_OUTPUT_PROMPT = """STOP SEARCHING. You have gathered enough research.

Now write your final novelty argument using ONLY the sources you have already found.
Include citations with exact quotes: \"\"\"verbatim quote\"\"\"(url)

Do not search for more sources. Write your argument now."""


# =============================================================================
# HELPERS
# =============================================================================

def _format_hypothesis(hypothesis: dict) -> str:
    """Format hypothesis dict as YAML for LLM readability."""
    display = {k: v for k, v in hypothesis.items()
               if k not in ["hypothesis_id", "is_seeded", "model"]
               and v is not None}
    return to_prompt_yaml(display)


# =============================================================================
# EXPORTS
# =============================================================================

def get_novelty(hypothesis: dict, stance: str) -> str:
    """Generate prompt for novelty argument (uses web search)."""
    return NOVELTY_PROMPT(hypothesis, stance)


def get_feasibility(hypothesis: dict, stance: str) -> str:
    """Generate prompt for feasibility argument (uses provided resources only)."""
    return FEASIBILITY_PROMPT(hypothesis, stance)


def get_force_output_prompt() -> str:
    """Prompt to force output when tool iterations are exhausted (novelty only)."""
    return NOVELTY_FORCE_OUTPUT_PROMPT


# Convenience functions
def get_novelty_positive(hypothesis: dict) -> str:
    return get_novelty(hypothesis, "positive")


def get_novelty_negative(hypothesis: dict) -> str:
    return get_novelty(hypothesis, "negative")


def get_feasibility_positive(hypothesis: dict) -> str:
    return get_feasibility(hypothesis, "positive")


def get_feasibility_negative(hypothesis: dict) -> str:
    return get_feasibility(hypothesis, "negative")


# =============================================================================
# RETRY PROMPT BUILDERS
# =============================================================================

NOVELTY_NO_CITATIONS_PROMPT = """Your previous response did not contain any valid citations.

You MUST provide an argument with properly formatted citations.

<task>
1. Use aii_web_search_fast to find relevant sources that support your argument
2. Use aii_web_fetch_direct to read the full content of promising URLs
3. Copy EXACT quotes from the fetched content (word-for-word, no paraphrasing)
4. Format each citation as: \"\"\"exact quote from source\"\"\"(full_url)

The argument must contain at least one verified citation.
</task>"""


FEASIBILITY_NO_CITATIONS_PROMPT = """Your previous response did not contain any valid citations.

You MUST provide an argument with properly formatted citations from the RESOURCES provided.

<task>
1. Read the <available_resources> section carefully
2. Find EXACT quotes from the resources that support your argument
3. Copy quotes word-for-word (no paraphrasing or summarizing)
4. Format each citation as: \"\"\"exact quote from resources\"\"\"

The argument must contain at least one verified citation.
</task>"""


NOVELTY_SOME_VALID_TASK = """<task>
1. Keep verified citations EXACTLY as-is
2. Remove failed citations and claims depending solely on them
3. Restructure argument using ONLY verified citations
4. Do NOT search for new citations
</task>"""


NOVELTY_NOT_ENOUGH_VALID_TASK = """<task>
NOT ENOUGH VALID CITATIONS. Search for more:
1. Keep verified citations - you may reuse those sources
2. Use aii_web_search_fast to find ADDITIONAL sources
3. Use aii_web_fetch_direct to read full page content
4. Copy EXACT quotes (word-for-word, no paraphrasing)
5. Format: \"\"\"exact quote\"\"\"(full_url)
</task>"""


NOVELTY_ALL_FAILED_TASK = """<task>
ALL CITATIONS FAILED. Search again:
1. Use aii_web_search_fast to find NEW sources
2. Use aii_web_fetch_direct to read full page content
3. Copy EXACT quotes (word-for-word, no paraphrasing)
4. Format: \"\"\"exact quote\"\"\"(full_url)
</task>"""


FEASIBILITY_SOME_VALID_TASK = """<task>
1. Keep verified citations EXACTLY as-is
2. Remove failed citations and claims depending solely on them
3. Restructure argument using ONLY verified citations
4. Do NOT invent new citations
</task>"""


FEASIBILITY_NOT_ENOUGH_VALID_TASK = """<task>
NOT ENOUGH VALID CITATIONS. Find more from resources:
1. Keep verified citations
2. Re-read <available_resources> for ADDITIONAL quotes
3. Copy quotes word-for-word (no paraphrasing)
4. Format: \"\"\"exact quote\"\"\"
</task>"""


FEASIBILITY_ALL_FAILED_TASK = """<task>
ALL CITATIONS FAILED. Re-read <available_resources>:
1. Find EXACT quotes from the resources
2. Copy quotes word-for-word (no paraphrasing)
3. Format: \"\"\"exact quote\"\"\"
</task>"""


def build_novelty_retry_prompt(verification: dict, no_citations: bool = False, min_valid: int = 5) -> str:
    """Build retry prompt for novelty citations."""
    if no_citations:
        return NOVELTY_NO_CITATIONS_PROMPT

    # Dedupe by quote text to count unique citations
    seen_valid = set()
    valid = []
    for r in verification.get('results', []):
        if r['status'] == 'valid' and r['quote'] not in seen_valid:
            seen_valid.add(r['quote'])
            valid.append(r)

    seen_failed = set()
    failed = []
    for r in verification.get('results', []):
        if r['status'] != 'valid' and r['quote'] not in seen_failed:
            seen_failed.add(r['quote'])
            failed.append(r)

    results: dict = {}
    if valid:
        results["verified"] = [{"quote": r["quote"], "url": r["url"]} for r in valid]
    if failed:
        results["failed"] = [
            {"quote": r["quote"], "reason": "not found on page" if r["status"] == "invalid" else "fetch failed"}
            for r in failed
        ]

    verification_block = f"<verification_results>\n{to_prompt_yaml(results)}\n</verification_results>"

    # Choose task based on how many valid citations
    if len(valid) >= min_valid:
        task = NOVELTY_SOME_VALID_TASK
    elif len(valid) > 0:
        task = NOVELTY_NOT_ENOUGH_VALID_TASK
    else:
        task = NOVELTY_ALL_FAILED_TASK
    return verification_block + "\n\n" + task


def build_feasibility_retry_prompt(verification: dict, no_citations: bool = False, min_valid: int = 5) -> str:
    """Build retry prompt for feasibility citations (resources-based)."""
    if no_citations:
        return FEASIBILITY_NO_CITATIONS_PROMPT

    # Dedupe by quote text to count unique citations
    seen_valid = set()
    valid = []
    for r in verification.get('results', []):
        if r['status'] == 'valid' and r['quote'] not in seen_valid:
            seen_valid.add(r['quote'])
            valid.append(r)

    seen_failed = set()
    failed = []
    for r in verification.get('results', []):
        if r['status'] != 'valid' and r['quote'] not in seen_failed:
            seen_failed.add(r['quote'])
            failed.append(r)

    results: dict = {}
    if valid:
        results["verified"] = [r["quote"] for r in valid]
    if failed:
        results["failed"] = [r["quote"] for r in failed]

    verification_block = f"<verification_results>\n{to_prompt_yaml(results)}\n</verification_results>"

    # Choose task based on how many valid citations
    if len(valid) >= min_valid:
        task = FEASIBILITY_SOME_VALID_TASK
    elif len(valid) > 0:
        task = FEASIBILITY_NOT_ENOUGH_VALID_TASK
    else:
        task = FEASIBILITY_ALL_FAILED_TASK
    return verification_block + "\n\n" + task
