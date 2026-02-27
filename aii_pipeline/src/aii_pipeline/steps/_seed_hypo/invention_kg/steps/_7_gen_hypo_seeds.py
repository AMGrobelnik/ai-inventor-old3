#!/usr/bin/env python3
"""
Step 7: Generate Hypothesis Seeds

Extracts seeds for hypothesis generation from the knowledge graph:
1. Topic Blind Spots - concepts a topic is missing from dissimilar topics

Input: data/_6_paper_triples/{run_id}/paper_triples_pr.json
Output: data/_7_hypo_seeds/{run_id}/
    - topic_blind_spots.json
"""

import sys
import json
import shutil
from pathlib import Path
from typing import Optional
from aii_lib import AIITelemetry, MessageType

from ._gen_hypo_seeds import generate_topic_blind_spots
from ._gen_hypo_seeds.topic_blind_spots import set_telemetry as set_blind_spots_telemetry


# Module-level telemetry (set by main)
_telemetry: Optional[AIITelemetry] = None


def _emit(msg_type: MessageType, msg: str):
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")

__all__ = ["main"]


def load_papers(input_dir: Path) -> list:
    """Load papers from combined JSON file."""
    json_file = input_dir / "paper_triples_pr.json"

    if not json_file.exists():
        _emit(MessageType.ERROR, f"Papers file not found: {json_file}")
        return []

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        _emit(MessageType.SUCCESS, f"Loaded {len(papers)} papers")
        return papers
    except Exception as e:
        _emit(MessageType.ERROR, f"Failed to load papers: {e}")
        raise


def main(run_id: str, config: dict = None, telemetry: Optional[AIITelemetry] = None):
    """
    Main entry point for hypothesis seed extraction.

    Args:
        run_id: Run ID for pipeline orchestration.
        config: Optional config dict. If None, loads from config.yaml.
        telemetry: Optional AIITelemetry instance.
    """
    global _telemetry
    _telemetry = telemetry

    # Set telemetry for helper modules
    set_blind_spots_telemetry(telemetry)

    from ..utils import get_run_dir
    from ..constants import (
        RUNS_DIR,
        STEP_6_PAPER_TRIPLES,
        STEP_7_HYPO_SEEDS,
    )

    # Config must be provided (passed from pipeline)
    if config is None:
        _emit(MessageType.ERROR, "Config not provided. Run via aii_pipeline, not standalone.")
        return 1

    # Get hypo seeds config with defaults
    hypo_config = config.get("gen_hypo_seeds", {})
    blind_spots_cfg = hypo_config.get("blind_spots", {}) if hypo_config else {}

    _emit(MessageType.INFO, f"Run ID: {run_id}")

    # Get input/output directories (uses RUNS_DIR for pipeline runs)
    input_dir = get_run_dir(STEP_6_PAPER_TRIPLES, run_id)
    output_dir = get_run_dir(STEP_7_HYPO_SEEDS, run_id)

    _emit(MessageType.INFO, f"Input: {input_dir.relative_to(RUNS_DIR)}")
    _emit(MessageType.INFO, f"Output: {output_dir.relative_to(RUNS_DIR)}")

    # Clean up output directory
    if output_dir.exists():
        _emit(MessageType.INFO, "Removing existing output directory")
        shutil.rmtree(output_dir, ignore_errors=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check input exists
    if not input_dir.exists():
        _emit(MessageType.ERROR, f"Input directory not found: {input_dir}")
        _emit(MessageType.ERROR, "Run step 6 first")
        return 1

    # Load papers
    _emit(MessageType.INFO, "Loading papers...")
    papers = load_papers(input_dir)

    if not papers:
        _emit(MessageType.ERROR, "No papers loaded")
        return 1

    _emit(MessageType.INFO, f"Loaded {len(papers)} papers")

    results = {}

    # 1. Topic Blind Spots (sorted by score descending)
    _emit(MessageType.INFO, "1. Finding Topic Blind Spots")
    results["blind_spots"] = generate_topic_blind_spots(
        papers,
        output_dir / "topic_blind_spots.json",
        min_shared_concepts=blind_spots_cfg.get("min_shared_concepts", 1),
        max_similarity=blind_spots_cfg.get("max_similarity", 1.0),
        entity_types=blind_spots_cfg.get("entity_types", []),  # Empty = all types
    )

    # Summary
    _emit(MessageType.INFO, "Summary")
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    _emit(MessageType.INFO, f"Generated {success_count}/{total_count} seed types")

    # List output files
    output_files = list(output_dir.glob("*.json"))
    for f in output_files:
        size = f.stat().st_size
        _emit(MessageType.INFO, f"  {f.name} ({size:,} bytes)")

    _emit(MessageType.SUCCESS, "Done!")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1]))
    else:
        print("Usage: python _7_gen_hypo_seeds.py <run_id>")
        sys.exit(1)
