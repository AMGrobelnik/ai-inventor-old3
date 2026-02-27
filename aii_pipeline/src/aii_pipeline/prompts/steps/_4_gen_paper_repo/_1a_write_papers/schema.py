"""Schema for Write Paper Text step.

PaperText is the structured output schema for paper writing.
Figures are parsed from inline XML in the text sections via parse_figures_from_xml().

Includes verification logic for figure placeholder validation.
"""

import re

from typing import Annotated

from pydantic import Field

from aii_lib.prompts import LLMPromptModel, LLMStructOutModel, LLMPrompt, LLMStructOut

from .._2a_viz_gen.schema import Figure


# =============================================================================
# SCHEMAS
# =============================================================================

class PaperText(LLMPromptModel, LLMStructOutModel):
    """Paper text — structured output from paper writing agent.

    Structured output fields (LLMPrompt + LLMStructOut):
    - title, abstract, paper_text, summary

    Metadata fields (plain, set by pipeline code):
    - id
    """

    # Structured output fields (agent fills these)
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Paper title - concise, descriptive, captures the main contribution")
    abstract: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Paper abstract")
    paper_text: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Full paper body text with markdown section headers (# Introduction, # Methods, # Results, # Discussion, # Conclusion). Include inline <figure> XML placeholders where figures should appear.")
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Brief summary of the paper's main contribution and findings")

    # Metadata fields (set by pipeline code, not by agent)
    id: str = Field(default="", description="Draft ID")


# =============================================================================
# XML PARSING
# =============================================================================

def parse_figures_from_xml(data: dict) -> list[Figure]:
    """Parse figures from inline XML in paper text.

    Looks for <figure> tags in paper_text and abstract fields.

    Args:
        data: Dict with paper_text and abstract keys.

    Returns:
        List of Figure objects parsed from inline XML
    """
    combined_text = "\n".join(filter(None, [data.get("abstract", ""), data.get("paper_text", "")]))

    return _parse_figures_from_text(combined_text)


def _parse_figures_from_text(text: str) -> list[Figure]:
    """Parse <figure> XML tags from text and return Figure objects.

    Expected XML format:
    <figure id="fig_1">
      <title>Figure Title</title>
      <description>what this figure should communicate</description>
      <summary>brief summary of what this figure shows</summary>
    </figure>
    """
    figures = []

    # Find all <figure>...</figure> blocks
    figure_pattern = re.compile(r'<figure[^>]*>(.*?)</figure>', re.DOTALL | re.IGNORECASE)

    for match in figure_pattern.finditer(text):
        figure_content = match.group(0)
        figure = _parse_single_figure(figure_content)
        if figure:
            figures.append(figure)

    return figures


def _parse_single_figure(figure_xml: str) -> Figure | None:
    """Parse a single <figure> XML block into a Figure."""
    # Extract id from <figure id="..."> or <id>...</id>
    fig_id = _extract_xml_attr(figure_xml, "figure", "id") or _extract_xml_value(figure_xml, "id")
    if not fig_id:
        fig_id = f"fig_{hash(figure_xml) % 10000}"

    # Extract fields
    title = _extract_xml_value(figure_xml, "title") or ""
    description = _extract_xml_value(figure_xml, "description") or ""
    summary = _extract_xml_value(figure_xml, "summary") or ""

    return Figure(
        id=fig_id,
        title=title,
        description=description,
        summary=summary,
    )


def _extract_xml_value(xml: str, tag: str) -> str | None:
    """Extract value from <tag>value</tag>."""
    pattern = re.compile(rf'<{tag}[^>]*>(.*?)</{tag}>', re.DOTALL | re.IGNORECASE)
    match = pattern.search(xml)
    return match.group(1).strip() if match else None


def _extract_xml_attr(xml: str, tag: str, attr: str) -> str | None:
    """Extract attribute value from <tag attr="value">."""
    pattern = re.compile(rf'<{tag}[^>]*{attr}=["\']([^"\']*)["\']', re.IGNORECASE)
    match = pattern.search(xml)
    return match.group(1).strip() if match else None


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_figures(
    figures: list[Figure],
) -> dict:
    """Verify figures for duplicate IDs and missing fields.

    Args:
        figures: List of parsed Figure objects

    Returns dict with:
    - valid: bool - True if all checks pass
    - id_errors: list - Duplicate or invalid figure IDs
    - field_errors: list - Missing title/description
    - figures_valid: int - Count of fully valid figures
    - figures_total: int - Total figures parsed
    """
    id_errors: list[str] = []
    field_errors: list[str] = []

    seen_ids: set[str] = set()
    figures_valid = 0

    for figure in figures:
        is_valid = True

        # Check for duplicate IDs
        if figure.id in seen_ids:
            id_errors.append(f"Duplicate figure ID: {figure.id}")
            is_valid = False
        seen_ids.add(figure.id)

        # Check required fields
        if not figure.title:
            field_errors.append(f"Figure {figure.id}: Missing title")
            is_valid = False
        if not figure.description:
            field_errors.append(f"Figure {figure.id}: Missing description")
            is_valid = False

        if is_valid:
            figures_valid += 1

    valid = not id_errors and not field_errors

    return {
        "valid": valid,
        "id_errors": id_errors,
        "field_errors": field_errors,
        "figures_valid": figures_valid,
        "figures_total": len(figures),
    }
