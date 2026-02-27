"""User prompt for paper writing.

Read top-to-bottom to understand the full prompt structure.

Two modes:
- Agent mode: Artifacts copied to ./artifacts/ folder (for Claude agent)
- LLM mode: Only title + summary provided (for OpenRouter LLM)

Figures are embedded inline as XML in the text sections.
They are parsed out by schema.py - NO separate array output needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_pipeline.utils import to_prompt_yaml

if TYPE_CHECKING:
    from aii_pipeline.steps._invention_loop.pools import ArtifactPool, NarrativePool

from ....components.data_files import get_reading_mini_preview_full
from ....components.read_skills import get_read_skills
from ....components.todo import get_todo_header
from ....components.tool_calling import get_web_tool_guidance


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

_NARR_SUMMARY_FIELDS: set[str] = {"id", "title", "summary", "artifacts_used"}
_NARR_FULL_FIELDS: set[str] = {"id", "title", "narrative", "summary", "artifacts_used", "gaps"}


def PROMPT(
    hypothesis_text: str,
    narrative_history: str,
    final_narrative: str,
    artifacts_str: str,
) -> str:
    return f"""<hypothesis>
{hypothesis_text}
</hypothesis>

<narrative_history>
All research narratives across iterations (summaries only).

{narrative_history}
</narrative_history>

<research_narrative>
The final narrative — basis for the paper.

{final_narrative}
</research_narrative>

<data_files>
{get_reading_mini_preview_full()}
</data_files>

<artifact_descriptions>
Research artifacts (experiments, datasets, research, proofs).

{artifacts_str}
</artifact_descriptions>

<task>
Write a publication-ready top-conference research paper in LaTeX with BibTeX citations and figure placeholders.
</task>

<figure_instructions>
Follow aii_paper_writing skill for the figure placeholder XML format, data precision requirements, and figure vs table decision rules.

CRITICAL: Before writing figure placeholders, look through artifact workspace output files (*_out.json) and code to find ALL the exact values that will be included in the figure. The figure generator cannot read files — every exact number and value MUST be in the description. Be as detailed as possible: aspect ratio, preferred colors, all data values, axis labels, ranges, legend entries, and any other visual details.
</figure_instructions>

{get_web_tool_guidance()}

{get_todo_header()}
{_format_todos(TODOS)}"""


TODOS = [
    get_read_skills("aii_paper_writing"),

    """EXTENSIVE LITERATURE REVIEW: Be exhaustive, meticulous, and thorough. Use web search tools (see <available_tools>) to research the landscape — search key terms from <hypothesis> and <research_narrative> to find foundational works, recent advances, competing approaches, and papers this work builds upon. Then use dblp_bib skill (dblp_bib_search + dblp_bib_fetch) to look up each paper found and collect real BibTeX entries. Build a comprehensive Related Work section. Do NOT fabricate entries.""",

    """READ ARTIFACTS: Before writing each section, READ the relevant artifact source code, output files, and data in the workspace. Extract concrete implementation details, technical innovations, algorithmic specifics, and quantitative results. Do NOT write surface-level descriptions.""",

    """WRITE PAPER: Write the full paper text with figure placeholders per <figure_instructions>. Cite with numeric references [1], [2], etc. Include all relevant citations from the literature review. At the end of the paper text, include a full bibliography section listing every cited paper with all BibTeX information (authors, title, venue, year, DOI/URL).""",
]


# =============================================================================
# HELPERS
# =============================================================================

def _format_todos(todos: list[str]) -> str:
    """Format TODO items into a single <todos> block."""
    lines = ["<todos>"]
    for i, item in enumerate(todos, start=1):
        lines.append(f"TODO {i}. {item}")
    lines.append("</todos>")
    return "\n".join(lines)


_PAPER_FIELDS: set[str] = {"id", "type", "title", "summary", "workspace_path", "out_expected_files"}


def _format_artifacts(artifact_pool: ArtifactPool) -> str:
    """Format artifacts for paper writing context.

    Shows out_expected_files (what files the artifact produced).
    Excludes out_demo_files / out_dependency_files (irrelevant for paper writing).
    """
    return artifact_pool.get_prompt(include=_PAPER_FIELDS) or "No artifacts available."


def _format_narrative_history(narrative_pool: NarrativePool) -> str:
    """All narratives as summaries (no full text)."""
    return narrative_pool.get_all_prompt(include=_NARR_SUMMARY_FIELDS, label="Narrative") or "No narratives available."


def _format_final_narrative(narrative_pool: NarrativePool) -> str:
    """Last narrative with full text included."""
    items = narrative_pool.get_all()
    if not items:
        return "No narrative available."
    return items[-1].to_prompt_yaml(include=_NARR_FULL_FIELDS, strip_nulls=True)


def _format_hypothesis(hypo: dict) -> str:
    """Format hypothesis dict as YAML for LLM readability (same as gen_strat)."""
    filtered = {
        k: v for k, v in hypo.items()
        if k not in ["hypothesis_id", "is_seeded", "model"]
        and not (k == "seeds" and not hypo.get("is_seeded"))
    }
    return to_prompt_yaml(filtered)


# =============================================================================
# EXPORTS
# =============================================================================

def get(
    narrative_pool: NarrativePool,
    artifact_pool: ArtifactPool,
    hypothesis: dict | None = None,
) -> str:
    """Build prompt for writing a paper draft with figure placeholders.

    Args:
        narrative_pool: NarrativePool — all narratives (summaries) + last one's full text
        artifact_pool: ArtifactPool with completed artifacts
        hypothesis: Hypothesis dict from invention loop
    """
    return PROMPT(
        hypothesis_text=_format_hypothesis(hypothesis) if hypothesis else "No hypothesis provided.",
        narrative_history=_format_narrative_history(narrative_pool),
        final_narrative=_format_final_narrative(narrative_pool),
        artifacts_str=_format_artifacts(artifact_pool),
    )


# =============================================================================
# RETRY PROMPT BUILDERS (for figure verification failures)
# =============================================================================

def build_figure_retry_prompt(
    verification: dict,
) -> str:
    """Build retry prompt for figure placeholder verification failures.

    Args:
        verification: Dict from verify_figures() with keys:
            - id_errors: list of duplicate/invalid figure ID errors
            - field_errors: list of missing title/description errors
            - figures_valid: count of valid figures
            - figures_total: total figures parsed

    Returns:
        Retry prompt string explaining issues and requesting fixes
    """
    lines = ["<verification_results>", "Your figure placeholders have issues that need fixing:", ""]

    id_errors = verification.get("id_errors", [])
    field_errors = verification.get("field_errors", [])
    figures_valid = verification.get("figures_valid", 0)
    figures_total = verification.get("figures_total", 0)

    if id_errors:
        lines.append("FIGURE ID ERRORS:")
        for err in id_errors:
            lines.append(f"  - {err}")
        lines.append("")

    if field_errors:
        lines.append("CONTENT ERRORS:")
        for err in field_errors:
            lines.append(f"  - {err}")
        lines.append("")

    lines.append(f"Summary: {figures_valid}/{figures_total} figures are valid")
    lines.append("</verification_results>")
    lines.append("")
    lines.append("<task>")
    lines.append("Fix ALL issues above in your figure placeholders:")
    lines.append("")

    step_num = 1
    if id_errors:
        lines.append(f"{step_num}. Fix figure IDs:")
        lines.append("   - Each figure must have a unique ID (e.g., fig_1, fig_2, ...)")
        lines.append("   - Do not duplicate IDs across figures")
        step_num += 1

    if field_errors:
        lines.append(f"{step_num}. Fix missing content:")
        lines.append("   - Every figure must have a title and description")
        lines.append("   - Description must include all data values and visualization details")

    lines.append("")
    lines.append("Output the corrected paper text with fixed figure placeholders.")
    lines.append("</task>")

    return "\n".join(lines)
