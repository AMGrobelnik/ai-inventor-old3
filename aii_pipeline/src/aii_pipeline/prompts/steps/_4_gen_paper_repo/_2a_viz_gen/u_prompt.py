"""User prompt for visualization image generation.

Read top-to-bottom to understand the full prompt structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Figure


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

def PROMPT(
    workspace: str,
    figure_id: str,
    output_filename: str,
    iterations_folder: str,
    title: str,
    description: str,
    summary: str,
) -> str:
    return f"""{workspace}

<task>
Generate a NeurIPS publication-quality figure for a publication-ready research paper for a top conference that exactly follows the provided specification.

Use the aii_image_gen_nano_banana skill to generate the figure with Nano Banana Pro in the right aspect ratio. Be as detailed as possible in your image generation prompt: include all data values, axis labels, ranges, legend entries, preferred colors, and describe where each element should be positioned.

IMPORTANT — File saving rules:
- Create a subfolder `{iterations_folder}/` in your workspace for ALL iteration attempts
- Save each attempt as: `{iterations_folder}/{figure_id}_v0_it1.png`, `{iterations_folder}/{figure_id}_v0_it2.png`, etc.
- After you are satisfied with the final version, copy ONLY the best iteration to your workspace root as: {output_filename}
- The final file `{output_filename}` is the deliverable — iterations in `{iterations_folder}/` are for reference

After generating, you MUST read the image back and carefully verify it matches the specification. Check for:
- Layout issues (e.g. text too close together, figure looks cluttered, elements crammed into corners)
- Overlapping or touching labels, legends, or annotations
- Cut-off or truncated text, axis labels, or titles
- Wrong or missing data values, bars, lines, or data points
- Incorrect axis ranges, tick marks, or scales
- Missing or misplaced legend entries
- Blurry text, unreadable font sizes, or poor contrast
- Wrong font family (MUST be sans-serif like Helvetica/Arial — reject any serif fonts like Times New Roman)

If ANY issue is found — even minor ones — you MUST regenerate with a corrected prompt. Do NOT accept a figure that has problems. Iterate until the figure is clean, accurate, and publication-ready. Expect to regenerate 2-3 times minimum.
</task>

<figure_specification>
Figure ID: {figure_id}
Title: {title}
Description: {description}
Summary: {summary}
</figure_specification>

<critical_requirements>
1. Accurately represent ALL data values described above — include every number mentioned
2. Do NOT invent additional data points beyond what is described
3. Include clear axis labels only if the figure has axes (not for diagrams/flowcharts)
4. FONT: ALL text MUST use sans-serif font (Helvetica/Arial). NO serif fonts (Times New Roman). Always include "Sans-serif font throughout (Helvetica/Arial style, NOT Times New Roman)" in your image generation prompt. This is the #1 most common issue — check it first during verification
5. NeurIPS camera-ready style: white backgrounds, properly formatted axes, no 3D effects/shadows/gradients. Follow aii_image_gen_nano_banana skill for image generation, prompting best practices, and figure type templates
6. TEXT SPACING: Ensure generous spacing between ALL text labels. Labels MUST NOT overlap or touch. Use large readable font sizes (minimum 12pt equivalent). If labels would overlap, stagger them vertically, use leader lines, or abbreviate. For multi-panel figures, add clear padding between panels
7. RESOLUTION: For multi-panel or detail-heavy figures, request higher resolution (2K or 4K) and wider aspect ratios to give elements room
8. MANDATORY VERIFICATION: After EVERY generation attempt, read the image and check font first (sans-serif?), then layout, data accuracy, and readability. If anything is wrong, regenerate. Do NOT stop at the first attempt
</critical_requirements>
"""


# =============================================================================
# EXPORTS
# =============================================================================

def get(figure_spec: Figure, workspace_path: str = "") -> str:
    """Build prompt for generating a visualization image."""
    from ....components.workspace import get_workspace_prompt
    from .schema import get_output_filename, get_iterations_folder
    return PROMPT(
        workspace=get_workspace_prompt(workspace_path) if workspace_path else "",
        figure_id=figure_spec.id,
        output_filename=get_output_filename(figure_spec.id, 0),
        iterations_folder=get_iterations_folder(figure_spec.id),
        title=figure_spec.title,
        description=figure_spec.description,
        summary=figure_spec.summary,
    )
