"""AI Inventor prompt components.

This module provides re-exports for convenient imports of prompt utilities.
Individual prompt files use simple static strings with .format() calls.

Usage:
    from aii_pipeline.prompts.components import (
        get_aii_context,
        get_resources_prompt,
        get_tool_calling_guidance,
    )
"""

# Context utilities
from .aii_context import (
    get_aii_context,
    FocusArea,
)

# Artifact summaries
from .artifact_summaries import get_artifact_context

# Artifact planning
from .artifact_planning import get_artifact_planning

# Resources utilities
from .resources import get_resources_prompt

# Tool calling utilities
from .tool_calling import get_tool_calling_guidance

# Subagent restriction
from .subagents import get_no_subagents_guidance

# Todo utilities
from .todo import get_todo_header

# Data files utilities
from .data_files import get_reading_mini_preview_full

# Read skills instruction
from .read_skills import get_read_skills

__all__ = [
    # Context
    "get_aii_context",
    "FocusArea",
    # Artifact summaries
    "get_artifact_context",
    # Artifact planning
    "get_artifact_planning",
    # Resources
    "get_resources_prompt",
    # Tool calling
    "get_tool_calling_guidance",
    # Subagent restriction
    "get_no_subagents_guidance",
    # Todo
    "get_todo_header",
    # Data files
    "get_reading_mini_preview_full",
    # Read skills
    "get_read_skills",
]
