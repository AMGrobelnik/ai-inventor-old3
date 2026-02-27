#!/usr/bin/env python3
"""
Utility functions for Building Blocks Knowledge Graph Pipeline.

This module provides common utilities used across all pipeline steps,
including run ID management, path resolution, and directory helpers.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

# Get base directory (invention_kg/)
BASE_DIR = Path(__file__).parent.resolve()

# Import constants
from aii_pipeline.steps._seed_hypo.invention_kg.constants import *  # noqa: F401, F403


# ============================================================================
# Run ID Management
# ============================================================================

def create_run_id(num_papers_requested: int) -> str:
    """
    Create a new run ID with format: {num_papers}_{timestamp}.

    Args:
        num_papers_requested: Total number of papers requested

    Returns:
        Run ID string (e.g., "460_20251111_204728")

    Example:
        >>> create_run_id(460)
        '460_20251111_204728'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{num_papers_requested}_{timestamp}"


def find_most_recent_run_id(
    step_name: str,
    base_dir: Optional[Path] = None
) -> Optional[str]:
    """
    Find the most recent run_id by looking at existing directories.

    Looks in: runs/<run_id>/1_seed_hypo/<step_name>/

    Args:
        step_name: Step directory name (e.g., '_1_sel_topics', '_7_hypo_seeds')
        base_dir: Base directory for runs (defaults to aii_pipeline/runs/)

    Returns:
        Most recent run_id string, or None if no runs found

    Example:
        >>> find_most_recent_run_id('_7_hypo_seeds')
        'novak_hypo_seed'
    """
    from .constants import RUNS_DIR, SEED_HYPO_SUBDIR

    if base_dir is None:
        base_dir = RUNS_DIR

    if not base_dir.exists():
        return None

    # Find all run directories that have the step completed
    run_dirs = []
    for run_dir in base_dir.iterdir():
        if run_dir.is_dir():
            step_dir = run_dir / SEED_HYPO_SUBDIR / step_name
            if step_dir.exists():
                run_dirs.append(run_dir)

    if not run_dirs:
        return None

    # Sort by modification time and return most recent
    run_dirs.sort(key=lambda x: x.stat().st_mtime)
    return run_dirs[-1].name


def get_run_dir(
    step_name: str,
    run_id: str,
    base_dir: Optional[Path] = None
) -> Path:
    """
    Get the output directory path for a specific step and run.

    Structure: runs/<run_id>/1_seed_hypo/<step_name>/

    Args:
        step_name: Step directory name (e.g., '_1_sel_topics', '_7_hypo_seeds')
        run_id: Run ID string (e.g., 'novak_hypo_seed')
        base_dir: Base directory for runs (defaults to aii_pipeline/runs/)

    Returns:
        Path to step's run directory

    Example:
        >>> get_run_dir('_7_hypo_seeds', 'novak_hypo_seed')
        PosixPath('.../aii_pipeline/runs/novak_hypo_seed/1_seed_hypo/_7_hypo_seeds')
    """
    from .constants import RUNS_DIR, SEED_HYPO_SUBDIR

    if base_dir is None:
        base_dir = RUNS_DIR

    return base_dir / run_id / SEED_HYPO_SUBDIR / step_name


# ============================================================================
# Path Resolution Helpers
# ============================================================================

def resolve_graph_path(
    graph_file_config: str,
    base_dir: Optional[Path] = None
) -> Path:
    """
    Resolve graph file path from config value.

    Args:
        graph_file_config: Graph file path from config (relative to invention_kg/)
        base_dir: Base directory (defaults to invention_kg/)

    Returns:
        Absolute path to graph file

    Example:
        >>> resolve_graph_path('data/_5_bblocks_graph/bblocks_graph_all.json')
        PosixPath('.../invention_kg/data/_5_bblocks_graph/bblocks_graph_all.json')
    """
    if base_dir is None:
        base_dir = BASE_DIR

    return base_dir / graph_file_config


def get_graph_symlink_path(
    filename: str,
    base_dir: Optional[Path] = None
) -> Path:
    """
    Get path to graph symlink in base directory.

    Args:
        filename: Graph filename (e.g., 'bblocks_graph_all.json')
        base_dir: Base directory (defaults to invention_kg/)

    Returns:
        Path to symlink file

    Example:
        >>> get_graph_symlink_path('bblocks_graph_all.json')
        PosixPath('.../invention_kg/data/_5_bblocks_graph/bblocks_graph_all.json')
    """
    if base_dir is None:
        base_dir = BASE_DIR

    from .constants import GRAPH_BASE_DIR
    return base_dir / GRAPH_BASE_DIR / filename


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_run_id_format(run_id: str) -> bool:
    """
    Validate run_id format: {number}_{YYYYMMDD_HHMMSS}.

    Args:
        run_id: Run ID string to validate

    Returns:
        True if valid format, False otherwise

    Example:
        >>> validate_run_id_format('460_20251111_204728')
        True
        >>> validate_run_id_format('invalid')
        False
    """
    try:
        parts = run_id.split('_')
        if len(parts) != 3:
            return False

        # Check number part
        int(parts[0])

        # Check date part (YYYYMMDD)
        if len(parts[1]) != 8:
            return False
        int(parts[1])

        # Check time part (HHMMSS)
        if len(parts[2]) != 6:
            return False
        int(parts[2])

        return True
    except (ValueError, IndexError):
        return False
