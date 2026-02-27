"""Shared dependency prompt builder for all gen_art artifact types.

Builds the <dependencies> prompt section using ArtifactPool.get_prompt()
with explicit field selection so downstream executor agents see:
id, type, title, summary, workspace_path, and out_dependency_files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_pipeline.prompts.components.data_files import get_reading_mini_preview_full

if TYPE_CHECKING:
    from aii_pipeline.steps._invention_loop.pools import ArtifactPool

# Fields to include in the dependency prompt for each artifact.
# These are the fields the downstream executor agent needs to see.
DEPENDENCY_FIELDS: set[str] = {
    "id",
    "type",
    "title",
    "summary",
    "workspace_path",
    "out_dependency_files",
}


def build_dependencies_prompt(
    artifact_pool: ArtifactPool,
    dependency_ids: list[str],
) -> str:
    """Build the <dependencies> prompt section from the artifact pool.

    Args:
        artifact_pool: The artifact pool to resolve IDs from.
        dependency_ids: Artifact IDs to include as dependencies.

    Returns:
        Formatted <dependencies>...</dependencies> block, or empty string if none.
    """
    if not dependency_ids:
        return ""

    content = artifact_pool.get_prompt(
        ids=dependency_ids,
        include=DEPENDENCY_FIELDS,
        label="Dependency",
    )
    if not content:
        return ""

    return f"""<dependencies>
Read the files in these dependency workspaces to understand what's available, then copy any you need into your working directory.

{content}

{get_reading_mini_preview_full()}
</dependencies>"""
