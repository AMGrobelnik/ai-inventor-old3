#!/usr/bin/env python3
"""
Configuration loading for building blocks extraction.

Handles loading pipeline config, agent config, and paper data.
"""

import json
from pathlib import Path
from typing import Optional

from aii_lib import AIITelemetry, MessageType
import yaml


# Module-level telemetry (set by caller)
_telemetry: Optional[AIITelemetry] = None


def set_telemetry(telemetry: Optional[AIITelemetry]) -> None:
    """Set module-level telemetry instance."""
    global _telemetry
    _telemetry = telemetry


def _emit(msg_type: MessageType, msg: str) -> None:
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")


def load_pipeline_config(config_path: Path) -> dict:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Pipeline configuration dict
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_agent_config(config_path: Path) -> dict:
    """
    Load agent configuration from YAML file.

    Args:
        config_path: Path to agent config (e.g., bblocks_config.yaml)

    Returns:
        Agent configuration dict
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Agent config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _emit(MessageType.SUCCESS, f"Loaded config from: {config_path}")
    return config


def load_papers_from_directory(
    papers_dir: Path, max_papers: int = -1
) -> tuple[list[dict], list[dict], int]:
    """
    Load all papers from paper_XXXXX/ directories.

    Args:
        papers_dir: Directory containing paper_XXXXX/ folders with paper.json
        max_papers: Maximum number of papers to load (-1 for all)

    Returns:
        Tuple of (all_papers, papers_to_process, num_paper_dirs) where:
        - all_papers: All papers loaded from all paper directories
        - papers_to_process: Papers after applying max_papers limit
        - num_paper_dirs: Number of paper directories found
    """
    # Find all paper_* directories and sort by index
    paper_dirs = sorted(papers_dir.glob("paper_*"))

    if not paper_dirs:
        raise FileNotFoundError(f"No paper directories found in {papers_dir}")

    _emit(MessageType.INFO, f"Found {len(paper_dirs)} paper directories")

    # Load all papers
    _emit(MessageType.INFO, "Loading all papers...")
    all_papers = []
    for paper_dir in paper_dirs:
        paper_file = paper_dir / "paper.json"
        if not paper_file.exists():
            _emit(MessageType.WARNING, f"Skipping {paper_dir.name}: no paper.json found")
            continue

        try:
            with open(paper_file, "r", encoding="utf-8") as f:
                paper = json.load(f)
                all_papers.append(paper)
        except Exception as e:
            _emit(MessageType.ERROR, f"Error loading {paper_file}: {e}")
            continue

    _emit(MessageType.INFO, f"Loaded {len(all_papers)} papers from {len(paper_dirs)} directories")

    # Apply max_papers limit
    if max_papers == -1:
        papers_to_process = all_papers
    else:
        papers_to_process = all_papers[:max_papers]

    return all_papers, papers_to_process, len(paper_dirs)


def validate_paths(agent_cwd_template: Path) -> None:
    """
    Validate that required paths exist.

    Args:
        agent_cwd_template: Path to agent_cwd template directory

    Raises:
        FileNotFoundError: If required paths don't exist
    """
    if not agent_cwd_template.exists():
        raise FileNotFoundError(
            f"Agent CWD template not found: {agent_cwd_template}\n"
            "Please ensure agent_cwd/ directory exists with validate_bblock_json.py"
        )
