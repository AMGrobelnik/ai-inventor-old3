"""Schema for visualization generation.

Defines the unified Figure class and output file helpers for viz generation.

Figure lifecycle:
1. XML parser creates Figure from <figure> tags in paper text (figure_path="")
2. viz gen fills in figure_path after image generation
3. gen_full_paper uses Figure with figure_path for LaTeX insertion
"""

from typing import Annotated

from pydantic import Field

from aii_lib.prompts import LLMPromptModel, LLMStructOutModel, BaseExpectedFiles, LLMPrompt, LLMStructOut
from aii_lib.agent_backend import ExpectedFile


# =============================================================================
# FIGURE
# =============================================================================

class Figure(LLMPromptModel):
    """A figure — parsed from paper XML, with optional generated image path.

    All fields are LLMPrompt-annotated for YAML serialization in prompts.
    """

    id: Annotated[str, LLMPrompt] = Field(
        description="Figure ID (e.g., 'fig_1'). Links back to the <figure> placeholder in paper text."
    )
    title: Annotated[str, LLMPrompt] = Field(
        description="Figure title/caption"
    )
    description: Annotated[str, LLMPrompt] = Field(
        default="",
        description="What this figure should communicate — axes, labels, key message"
    )
    summary: Annotated[str, LLMPrompt] = Field(
        default="",
        description="Brief summary of what this figure shows"
    )
    figure_path: Annotated[str, LLMPrompt] = Field(
        default="",
        description="Path to the generated image file (e.g., 'figures/fig_1_v0.png'). Empty before viz gen."
    )


# =============================================================================
# STRUCTURED OUTPUT (agent output schema for expected files validation)
# =============================================================================

class VizExpectedFiles(BaseExpectedFiles):
    """Expected output files from viz generation."""
    image_path: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Path to the generated figure image file. Example: 'fig_1_v0.png'"
    )


class VizFigureOutput(LLMPromptModel, LLMStructOutModel):
    """Structured output from viz figure generation agent."""
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Short descriptive title for the generated figure (max 50 characters)"
    )
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Brief summary of the generated figure: what it shows, style, any issues fixed (max 200 words)"
    )
    out_expected_files: Annotated[VizExpectedFiles, LLMPrompt, LLMStructOut] = Field(
        description="Output file you created. Must include the generated figure image path."
    )


# =============================================================================
# CONSTANTS
# =============================================================================

VIZ_OUTPUT_FORMAT = "png"


# =============================================================================
# OUTPUT FILE HELPERS
# =============================================================================

def get_expected_out_file(figure_id: str, variation_idx: int) -> ExpectedFile:
    """Get expected output file for a single figure variation."""
    filename = f"{figure_id}_v{variation_idx}.{VIZ_OUTPUT_FORMAT}"
    return ExpectedFile(filename, f"Figure image for {figure_id} variation {variation_idx}")


def get_expected_out_files(figure_id: str, num_variations: int) -> list[ExpectedFile]:
    """Get all expected output files for a figure with multiple variations."""
    return [
        get_expected_out_file(figure_id, i)
        for i in range(num_variations)
    ]


def get_output_filename(figure_id: str, variation_idx: int) -> str:
    """Get the output filename for a figure."""
    return f"{figure_id}_v{variation_idx}.{VIZ_OUTPUT_FORMAT}"


def get_iterations_folder(figure_id: str) -> str:
    """Get the subfolder name for iteration attempts (e.g. 'fig_1_all')."""
    return f"{figure_id}_all"
