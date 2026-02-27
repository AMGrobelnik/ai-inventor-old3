"""Schema for full paper generation step.

Defines:
- FullPaper, FullPaperExpectedFiles: Structured output for LaTeX paper generation
- GenPaperRepoOut: Final output of gen_paper module
"""

from typing import Annotated

from pydantic import BaseModel, Field

from aii_lib.prompts import LLMPromptModel, LLMStructOutModel, BaseExpectedFiles, LLMPrompt, LLMStructOut
from aii_pipeline.prompts.steps.base import BaseStepOut

from .._2a_viz_gen.schema import Figure
from .._1a_write_papers.schema import PaperText
from ..schema import GistDeployment


# =============================================================================
# STRUCTURED OUTPUT (agent output schema)
# =============================================================================

class FullPaperExpectedFiles(BaseExpectedFiles):
    """All expected output files from full paper generation."""
    paper_tex_path: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Path to LaTeX source file. Example: 'paper.tex'")
    paper_pdf_path: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Path to compiled PDF. Example: 'paper.pdf'")
    references_bib_path: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Path to BibTeX bibliography file. Example: 'references.bib'")
    figure_paths: Annotated[list[str], LLMPrompt, LLMStructOut] = Field(description="Paths to all figure image files. Example: ['figures/fig_1_v0.png', 'figures/fig_2_v0.png']")


class FullPaper(LLMPromptModel, LLMStructOutModel):
    """Full paper — structured output from paper generation."""
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Short descriptive title for this paper generation task (max 50 characters)"
    )
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Brief summary of the generated paper: sections written, figures included, compilation status (max 200 words)"
    )
    out_expected_files: Annotated[FullPaperExpectedFiles, LLMPrompt, LLMStructOut] = Field(
        description="All output files you created. Must include paper.tex, paper.pdf, references.bib, and paths to all figure files."
    )


# =============================================================================
# RESULT
# =============================================================================

class GenPaperRepoOut(BaseStepOut):
    """Final result of gen_paper module."""
    repo_url: str | None = Field(default=None, description="GitHub repo URL if created")

    # Artifacts
    gist_deployments: list[GistDeployment] = Field(default_factory=list)

    # Visualizations
    figures: list[Figure] = Field(default_factory=list)

    # Paper
    paper: PaperText | None = Field(default=None)
