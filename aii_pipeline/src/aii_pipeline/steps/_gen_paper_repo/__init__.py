"""Gen Paper module - step functions and schemas.

Pipeline (2 parallel tracks):

  Track A (PAPER → VIZ):
    _1a_write_paper_text (with inline figure XML)
    _2a_gen_viz (image_gen from figure specs)

  Track B (ARTIFACT DEMOS):
    _0b_create_repo
    _1b_gen_artifact_demos
    _2b_deploy_to_repo

  Final: _5_gen_full_paper (combines paper + figures into LaTeX)

Paper writer embeds viz specs as inline XML — parsed into Figure objects.
"""

from aii_pipeline.prompts.steps._4_gen_paper_repo._2a_viz_gen.schema import Figure
from aii_pipeline.prompts.steps._4_gen_paper_repo._1a_write_papers.schema import PaperText
from aii_pipeline.prompts.steps._4_gen_paper_repo._1b_artifact_demos.schema_code import (
    BaseDemo,
    CodeDemo,
    LeanDemo,
    MarkdownDemo,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo.schema import GistDeployment
from aii_pipeline.prompts.steps._4_gen_paper_repo._5_gen_full_paper.schema import GenPaperRepoOut

__all__ = [
    "Figure",
    "PaperText",
    "BaseDemo",
    "CodeDemo",
    "LeanDemo",
    "MarkdownDemo",
    "GistDeployment",
    "GenPaperRepoOut",
]
