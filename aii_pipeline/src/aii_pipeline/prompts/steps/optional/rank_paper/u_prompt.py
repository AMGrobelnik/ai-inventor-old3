"""User prompt for paper ranking.

Read top-to-bottom to understand the full prompt structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._4_gen_paper_repo._1a_write_papers.schema import PaperText


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(
    summary_a: str,
    summary_b: str,
) -> str:
    return f"""Compare these two research paper drafts.

<paper_a>
{summary_a}
</paper_a>

<paper_b>
{summary_b}
</paper_b>

<task>
Which paper is better overall?

Consider:
1. Clarity: Is the writing clear and well-organized?
2. Coherence: Does it tell a compelling research story?
3. Completeness: Are all sections adequately developed?
4. Technical quality: Are methods and results well-presented?
5. Academic tone: Is the language appropriate?

You MUST choose A or B. Respond with:
- preferred: "A" or "B"
- justification: Brief explanation (2-3 sentences)
</task>"""


# =============================================================================
# HELPERS
# =============================================================================

def _summarize_paper(paper: PaperText) -> str:
    """Format full paper for comparison."""
    parts = []
    if paper.title:
        parts.append(f"TITLE: {paper.title}")
    if paper.abstract:
        parts.append(f"ABSTRACT:\n{paper.abstract}")
    if paper.paper_text:
        parts.append(paper.paper_text)
    return "\n\n".join(parts) if parts else ""


# =============================================================================
# EXPORTS
# =============================================================================

def get(paper_a: PaperText, paper_b: PaperText) -> str:
    """Build prompt for comparing two paper drafts."""
    return PROMPT(
        summary_a=_summarize_paper(paper_a),
        summary_b=_summarize_paper(paper_b),
    )
