#!/usr/bin/env python3
"""
Step 9: Generate Graphs

Creates multiple graph representations:
1. Concepts graph: Co-occurrence of concepts
2. Concept Ontology: Wikidata hierarchy (subclass_of, part_of)
3. Paper to Concepts: Semantic KG with UMAP embeddings
4. Blind Spots: Topic gaps from hypo_seeds

Input: data/_6_paper_triples/{run_id}/, data/_7_hypo_seeds/{run_id}/
Output: data/_9_graphs/{run_id}/
"""

import sys
import shutil
import yaml
from pathlib import Path
from typing import Optional
from aii_lib import AIITelemetry, MessageType

from ._gen_graphs import (
    load_all_papers,
    load_papers_by_year,
    generate_cooccurrence_graph,
    generate_ontology_graph,
    generate_semantic_graph,
    generate_blind_spots_graph,
    set_telemetry as set_submodule_telemetry,
)


# Module-level telemetry (set by main)
_telemetry: Optional[AIITelemetry] = None


def _emit(msg_type: MessageType, msg: str):
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")

__all__ = ["main"]


def main(run_id: str, config: dict = None, telemetry: Optional[AIITelemetry] = None):
    """
    Main entry point for graph generation.

    Args:
        run_id: Run ID for pipeline orchestration.
        config: Optional config dict. If None, loads from local config.yaml.
        telemetry: Optional AIITelemetry instance.
    """
    global _telemetry
    _telemetry = telemetry
    # Propagate telemetry to all submodules
    set_submodule_telemetry(telemetry)

    from ..utils import get_run_dir
    from ..constants import (
        BASE_DIR,
        RUNS_DIR,
        STEP_6_PAPER_TRIPLES,
        STEP_7_HYPO_SEEDS,
        STEP_9_GRAPHS,
    )

    _emit(MessageType.INFO, f"Run ID: {run_id}")

    # Config must be provided (passed from pipeline)
    if config is None:
        _emit(MessageType.ERROR, "Config not provided. Run via aii_pipeline, not standalone.")
        return 1

    # Get temporal windows from config
    gen_graph_config = config.get("gen_graph", {})
    temporal_windows = gen_graph_config.get("temporal_windows", [[2018, 2020], [2021, 2023], [2024, 2025]])

    _emit(MessageType.INFO, f"Temporal windows: {temporal_windows}")

    # Get input/output directories (uses RUNS_DIR for pipeline runs)
    input_dir = get_run_dir(STEP_6_PAPER_TRIPLES, run_id)
    hypo_seeds_dir = get_run_dir(STEP_7_HYPO_SEEDS, run_id)
    output_dir = get_run_dir(STEP_9_GRAPHS, run_id)

    _emit(MessageType.INFO, f"Input (triples): {input_dir.relative_to(RUNS_DIR)}")
    _emit(MessageType.INFO, f"Input (hypo_seeds): {hypo_seeds_dir.relative_to(RUNS_DIR)}")
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
    papers = load_all_papers(input_dir)

    if not papers:
        _emit(MessageType.ERROR, "No papers loaded")
        return 1

    _emit(MessageType.INFO, f"Loaded {len(papers)} papers")

    # Generate all graphs
    results = {}

    # 1. Concepts graph (co-occurrence, all years + per-year)
    _emit(MessageType.INFO, "1. Concepts graph (co-occurrence)")
    cooccur_dir = output_dir / "cooccurrence"
    cooccur_dir.mkdir(parents=True, exist_ok=True)

    results["concepts"] = generate_cooccurrence_graph(
        papers,
        cooccur_dir / "all.json",
        temporal_windows
    )

    # Per-year co-occurrence
    papers_by_year = load_papers_by_year(input_dir)
    by_year_dir = cooccur_dir / "by_year"
    by_year_dir.mkdir(parents=True, exist_ok=True)

    for year in sorted(papers_by_year.keys()):
        year_papers = papers_by_year[year]
        generate_cooccurrence_graph(
            year_papers,
            by_year_dir / f"{year}.json",
            temporal_windows
        )

    # 2. Concept Ontology graph
    _emit(MessageType.INFO, "2. Concept Ontology graph")
    ontology_dir = output_dir / "ontology"
    results["ontology"] = generate_ontology_graph(
        papers,
        ontology_dir / "full.json"
    )

    # 3. Paper to Concepts graph (semantic/UMAP)
    _emit(MessageType.INFO, "3. Paper to Concepts graph (semantic)")
    semantic_dir = output_dir / "semantic"
    results["paper_concepts"] = generate_semantic_graph(
        papers,
        semantic_dir
    )

    # 4. Blind Spots graph (from hypo_seeds)
    _emit(MessageType.INFO, "4. Blind Spots graph")
    derived_dir = output_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    if hypo_seeds_dir.exists():
        results["blind_spots"] = generate_blind_spots_graph(
            hypo_seeds_dir,
            derived_dir / "blind_spots.json"
        )
    else:
        _emit(MessageType.WARNING, "Hypo seeds directory not found, skipping blind spots")
        results["blind_spots"] = False

    # Summary
    _emit(MessageType.INFO, "Summary")
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    _emit(MessageType.INFO, f"Generated {success_count}/{total_count} graph types")

    # List output files
    _emit(MessageType.INFO, "Output structure:")
    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            files = list(subdir.rglob("*.json"))
            _emit(MessageType.INFO, f"  {subdir.name}/ ({len(files)} files)")

    _emit(MessageType.SUCCESS, "Done!")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1]))
    else:
        print("Usage: python _9_gen_graphs.py <run_id>")
        sys.exit(1)
