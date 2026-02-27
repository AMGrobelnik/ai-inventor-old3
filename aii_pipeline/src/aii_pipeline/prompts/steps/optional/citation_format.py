"""Citation format guidance for audit_hypo prompts.

Provides standardized citation format instructions for LLMs generating cited arguments.
Used by both novelty (web search) and feasibility (resources) audit prompts.
"""


def get_quoting_techniques() -> str:
    """Shared quoting techniques guidance."""
    return """<quoting_techniques>
MULTIPLE QUOTES:
\"\"\"first quote\"\"\" and \"\"\"second quote\"\"\"
Group related quotes together.

EDITORIAL BRACKETS [text]:
\"\"\"the system [uses] advanced techniques\"\"\" - brackets indicate YOUR addition, not original text
CAUTION: Use SPARINGLY and ONLY when absolutely necessary to make quote grammatically fit.
Do NOT use brackets to add meaning or misrepresent the original quote.

NO SHORT QUOTES:
Avoid quoting single words like \"\"\"contracts\"\"\" or short phrases - these provide no real evidence.
Quotes must be meaningful phrases (4+ words) that actually support your claim.
</quoting_techniques>"""


def get_novelty_citation_format() -> str:
    """Citation format for novelty arguments (with URLs).

    Returns text wrapped in <citation_format> tags — callers should embed directly.
    """
    return f"""<citation_format>
Use TRIPLE quote marks: \"\"\"exact verbatim quote\"\"\" (https://source-url.com)

EXAMPLE:
\"\"\"LLMs can coordinate through shared representations\"\"\" (https://arxiv.org/abs/2301.12345)

{get_quoting_techniques()}

RULES:
- Quote must be EXACT text from the webpage (will be verified)
- Text in [brackets] is ignored during verification - use responsibly
- URL must be the FULL URL where the quote appears
- Multiple quotes can share one URL (put URL after last quote)
- At least ONE citation with URL is REQUIRED
</citation_format>"""


def get_feasibility_citation_format() -> str:
    """Citation format for feasibility arguments (no URLs, quotes from resources).

    Returns text wrapped in <citation_format> tags — callers should embed directly.
    """
    return f"""<citation_format>
Use TRIPLE quote marks: \"\"\"exact verbatim quote\"\"\"

EXAMPLES:
\"\"\"32GB RAM\"\"\"
\"\"\"Python only implementation\"\"\"
\"\"\"OpenRouter API: 300+ models\"\"\"

{get_quoting_techniques()}

RULES:
- Quote ONLY from text between <available_resources> and </available_resources> tags
- Do NOT quote from <hypothesis> or any other section - ONLY <available_resources>
- Quote must be EXACT text (will be verified against resources)
- Text in [brackets] is ignored during verification - use responsibly
- At least ONE citation is REQUIRED
</citation_format>"""
