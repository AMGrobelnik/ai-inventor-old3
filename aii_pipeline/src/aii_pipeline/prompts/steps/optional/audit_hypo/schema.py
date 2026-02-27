"""Pydantic schemas for hypothesis audit output structure."""

from typing import Annotated

from aii_lib.prompts import LLMPromptModel, LLMStructOutModel, LLMPrompt, LLMStructOut


class AuditArgs(LLMPromptModel, LLMStructOutModel):
    """Arguments for and against a dimension (feasibility, novelty, etc.)."""
    positive_args: Annotated[list[str], LLMPrompt, LLMStructOut]
    negative_args: Annotated[list[str], LLMPrompt, LLMStructOut]


class HypoAudit(LLMPromptModel, LLMStructOutModel):
    """Complete audit of a hypothesis with verified cited arguments."""
    feasibility: Annotated[AuditArgs, LLMPrompt, LLMStructOut]
    novelty: Annotated[AuditArgs, LLMPrompt, LLMStructOut]


class AuditedHypothesis(LLMPromptModel, LLMStructOutModel):
    """A hypothesis with its audit results (capped args only)."""
    hypothesis: Annotated[dict, LLMPrompt, LLMStructOut]
    audit: Annotated[HypoAudit, LLMPrompt, LLMStructOut]


class AuditOutput(LLMPromptModel, LLMStructOutModel):
    """Final output of the audit stage - all hypotheses with audits."""
    hypotheses: Annotated[list[AuditedHypothesis], LLMPrompt, LLMStructOut]


class SingleArgument(LLMPromptModel, LLMStructOutModel):
    """Single argument output from Claude agent (one agent call = one argument)."""
    argument: Annotated[str, LLMPrompt, LLMStructOut]
