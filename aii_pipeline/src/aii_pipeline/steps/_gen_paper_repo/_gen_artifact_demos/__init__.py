"""Artifact demo converters — one per file type."""

from .gen_py_demo import github_to_colab_url, convert_to_notebook
from .gen_lean_demo import lean_playground_url, create_proof_markdown
from .gen_md_demo import create_research_markdown
