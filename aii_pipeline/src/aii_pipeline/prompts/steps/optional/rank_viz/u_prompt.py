"""User prompt for visualization ranking.

Read top-to-bottom to understand the full prompt structure.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._4_gen_paper_repo._2a_viz_gen.schema import Figure


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(
    fig_id: str,
    title: str,
    description: str,
    placement: str,
    context_section: str,
) -> str:
    return f"""Compare these two figure variations for the paper.

<figure_specification>
ID: {fig_id}
Title: {title}
Description: {description}
Section: {placement}
</figure_specification>

{context_section}

<task>
Look at FIGURE A and FIGURE B. Which figure better matches the specification and fits the paper?

Consider:
1. Does it correctly visualize what the specification asks for?
2. Is it clear and easy to understand?
3. Is it visually professional and publication-ready?
4. Does it have proper labels, title, and legend?
5. Does it fit the paper context and section?

You MUST choose A or B. Respond with:
- preferred: "A" or "B"
- justification: Brief explanation (2-3 sentences)
</task>"""


# =============================================================================
# HELPERS
# =============================================================================

def _load_image_as_base64(figure_path: str) -> str | None:
    """Load image file and return as base64 data URL."""
    try:
        path = Path(figure_path)
        if not path.exists():
            return None

        suffix = path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/png")

        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{data}"
    except Exception:
        raise


def _format_context_section(paper_context: str | None, placement: str) -> str:
    """Format context section if provided."""
    if paper_context:
        return f"""<paper_context>
This figure will appear in the {placement} section of the paper.
Surrounding text excerpt:
{paper_context[:500]}...
</paper_context>"""
    return ""


# =============================================================================
# EXPORTS
# =============================================================================

def get(
    placeholder: Figure,
    fig_a: Figure,
    fig_b: Figure,
    paper_context: str | None = None,
    use_claude_agent: bool = False,
    local_path_a: str | None = None,
    local_path_b: str | None = None,
) -> list | str:
    """Build multimodal prompt for comparing two figure variations.

    Args:
        placeholder: Figure placeholder with specification
        fig_a: First figure result (will be labeled as Figure A)
        fig_b: Second figure result (will be labeled as Figure B)
        paper_context: Optional surrounding text from paper
        use_claude_agent: If True, return text prompt telling agent to read local files
        local_path_a: Local path to Figure A (for Claude agent mode)
        local_path_b: Local path to Figure B (for Claude agent mode)

    Returns:
        For OpenRouter: list of content blocks (text + images) for multimodal comparison
        For Claude agent: string prompt instructing agent to read files
    """
    context_section = _format_context_section(paper_context, "results")

    text_prompt = PROMPT(
        fig_id=placeholder.id,
        title=placeholder.title,
        description=placeholder.description,
        placement="results",
        context_section=context_section,
    )

    # Claude agent mode: tell agent which files to read
    if use_claude_agent:
        path_a = local_path_a or fig_a.figure_path
        path_b = local_path_b or fig_b.figure_path
        return f"""{text_prompt}

<figure_files>
The two figures have been copied to your workspace. Use the Read tool to view them:

**FIGURE A:** `{path_a}`
**FIGURE B:** `{path_b}`

Read BOTH images using the Read tool, then compare them and decide which is better.
</figure_files>"""

    # OpenRouter mode: embed images as base64
    img_a_url = _load_image_as_base64(fig_a.figure_path)
    img_b_url = _load_image_as_base64(fig_b.figure_path)

    # If both images loaded, return multimodal content
    if img_a_url and img_b_url:
        return [
            {"type": "text", "text": text_prompt},
            {"type": "text", "text": "\n\n<figure_a>"},
            {"type": "image_url", "image_url": {"url": img_a_url}},
            {"type": "text", "text": "</figure_a>\n\n<figure_b>"},
            {"type": "image_url", "image_url": {"url": img_b_url}},
            {"type": "text", "text": "</figure_b>"},
        ]

    # Fallback to text-only (no images)
    return f"""{text_prompt}

<figure_a>
File: {fig_a.figure_path}
(Image could not be loaded)
</figure_a>

<figure_b>
File: {fig_b.figure_path}
(Image could not be loaded)
</figure_b>"""
