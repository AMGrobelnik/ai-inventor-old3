"""Shared schemas for Gen Paper module.

Classes used across multiple gen_paper_repo steps live here.
"""

from pydantic import BaseModel, Field


class GistDeployment(BaseModel):
    """Information about a deployed artifact in GitHub repo."""
    artifact_id: str = Field(description="ID of the artifact")
    gist_url: str = Field(description="URL to the artifact in GitHub repo")
    gist_id: str = Field(description="GitHub artifact ID")
    files: list[str] = Field(default_factory=list, description="Files deployed")
    colab_url: str | None = Field(default=None, description="Google Colab URL for notebooks")
