"""
Cache cleanup utilities for run directories.

Cleans up temporary caches created during pipeline runs:
- .venv directories (virtual environments created by agents)
- .hf_cache directories (HuggingFace dataset downloads)
"""

import shutil
from pathlib import Path


def cleanup_run_caches(
    run_dir: Path | str,
    clear_venv: bool = True,
    clear_hf: bool = True,
) -> dict:
    """
    Clean up cache directories in a run folder.

    Args:
        run_dir: Path to the run directory (e.g., runs/20250112_123456/)
        clear_venv: Remove .venv directories (virtual environments)
        clear_hf: Remove .hf_cache directories (HuggingFace downloads)

    Returns:
        Dict with cleanup stats: {removed: [...], total_size_mb: float}
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return {"removed": [], "total_size_mb": 0.0}

    removed = []
    total_size_mb = 0.0

    # Clean up .venv directories (can be nested anywhere in the run)
    if clear_venv:
        for venv_dir in run_dir.rglob(".venv"):
            if venv_dir.is_dir():
                try:
                    size_mb = _get_dir_size_mb(venv_dir)
                    shutil.rmtree(venv_dir)
                    total_size_mb += size_mb
                    removed.append(f".venv ({size_mb:.1f} MB) at {venv_dir.relative_to(run_dir)}")
                except Exception as e:
                    from loguru import logger
                    logger.exception(f"Failed to remove .venv at {venv_dir}: {e}")

    # Clean up .hf_cache directories
    if clear_hf:
        for hf_dir in run_dir.rglob(".hf_cache"):
            if hf_dir.is_dir():
                try:
                    size_mb = _get_dir_size_mb(hf_dir)
                    shutil.rmtree(hf_dir)
                    total_size_mb += size_mb
                    removed.append(f".hf_cache ({size_mb:.1f} MB) at {hf_dir.relative_to(run_dir)}")
                except Exception as e:
                    from loguru import logger
                    logger.exception(f"Failed to remove .hf_cache at {hf_dir}: {e}")

    return {"removed": removed, "total_size_mb": total_size_mb}


def _get_dir_size_mb(path: Path) -> float:
    """Get directory size in MB."""
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
    except Exception:
        return 0.0
