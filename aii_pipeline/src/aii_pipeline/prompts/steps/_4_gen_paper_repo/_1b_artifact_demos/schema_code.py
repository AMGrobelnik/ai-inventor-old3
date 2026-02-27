"""Schema for artifact demo generation step.

Defines:
- DemoType enum and BaseDemo hierarchy (CodeDemo, LeanDemo, MarkdownDemo)
- Demo: Structured output from demo notebook generation
- DemoExpectedFiles: Expected output files
"""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from aii_lib.prompts import LLMPromptModel, LLMStructOutModel, BaseExpectedFiles, LLMPrompt, LLMStructOut


# =============================================================================
# DEMO TYPE HIERARCHY
# =============================================================================

class DemoType(str, Enum):
    CODE = "code"
    LEAN = "lean"
    MARKDOWN = "markdown"


class BaseDemo(BaseModel):
    """Base demo — common fields for all demo types."""
    id: str = Field(description="Artifact ID this demo belongs to")
    type: DemoType = Field(description="Demo type discriminator")
    title: str = Field(default="", description="Short descriptive title for this demo")
    summary: str = Field(default="", description="Brief summary of what this demo shows")
    original_path: str = Field(default="", description="Path to source workspace")
    demo_path: str = Field(default="", description="Path to demo output")


class DemoExpectedFiles(BaseExpectedFiles):
    """Expected output files from code demo notebook generation."""
    notebook: Annotated[str, LLMPrompt, LLMStructOut] = Field(default="", description="Path to the generated demo notebook. Example: 'code_demo.ipynb'")
    mini_data_file: Annotated[str, LLMPrompt, LLMStructOut] = Field(default="", description="Path to the mini demo data JSON (curated subset). Example: 'mini_demo_data.json'")


class LeanDemoExpectedFiles(BaseExpectedFiles):
    """Expected output files from lean demo generation."""
    lean_file: Annotated[str, LLMPrompt, LLMStructOut] = Field(default="", description="Path to the Lean 4 proof file. Example: 'proof.lean'")


class CodeDemo(BaseDemo, LLMPromptModel, LLMStructOutModel):
    """Dataset/experiment/evaluation → Jupyter notebook demo.

    Serves as both structured output schema (agent fills title, summary,
    out_expected_files) and pipeline data model (notebook_path set by code).
    """
    type: Literal[DemoType.CODE] = DemoType.CODE
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(default="", description="Short descriptive title for this demo (max 50 characters)")
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(default="", description="Brief summary of the demo notebook: what it demonstrates, data used (max 200 words)")
    out_expected_files: Annotated[DemoExpectedFiles, LLMPrompt, LLMStructOut] = Field(
        default_factory=DemoExpectedFiles,
        description="All output files you created. Must include the demo notebook."
    )
    notebook_path: str = Field(default="", description="Path to generated notebook")


class LeanDemo(BaseDemo):
    """Proof → markdown + Lean playground link."""
    type: Literal[DemoType.LEAN] = DemoType.LEAN
    out_expected_files: LeanDemoExpectedFiles = Field(
        default_factory=LeanDemoExpectedFiles,
        description="Expected output files from lean demo."
    )
    playground_url: str = Field(default="", description="Lean playground URL")


class MarkdownDemo(BaseDemo):
    """Research → markdown summary."""
    type: Literal[DemoType.MARKDOWN] = DemoType.MARKDOWN


AnyDemo = Union[CodeDemo, LeanDemo, MarkdownDemo]


