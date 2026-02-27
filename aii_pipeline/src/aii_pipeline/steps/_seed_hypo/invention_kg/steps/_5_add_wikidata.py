#!/usr/bin/env python3
"""
Step 5: Add Wikidata to Triples

This step enriches all extracted triples with full Wikidata information:
- QID, label, description, aliases
- All claims/properties (external IDs, ontology, relationships)
- Resolved Q-ID labels for linked entities

Input: data/_4_triples/{run_id}/ (triple JSON files per paper)
Output: data/_5_wikidata/{run_id}/ (enriched triple JSON files)

The enriched data is stored under triple["wikidata"] = {...}
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from aii_lib import AIITelemetry, MessageType


# Module-level telemetry (set by main)
_telemetry: Optional[AIITelemetry] = None


def _emit(msg_type: MessageType, msg: str):
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")


def load_triples_from_run(triples_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all triples from a triples run directory.

    Args:
        triples_dir: Path to _4_triples/{run_id}/

    Returns:
        List of paper data with triples
    """
    papers_with_triples = []

    # Find all paper directories
    paper_dirs = sorted([d for d in triples_dir.iterdir() if d.is_dir() and d.name.startswith("paper_")])

    for paper_dir in paper_dirs:
        # Step 4 outputs to agent_cwd/triples_output.json
        triples_file = paper_dir / "agent_cwd" / "triples_output.json"
        if not triples_file.exists():
            continue

        try:
            with open(triples_file, 'r', encoding='utf-8') as f:
                triples_data = json.load(f)

            # Skip if no triples
            if not triples_data.get("triples"):
                continue

            papers_with_triples.append({
                "paper_id": paper_dir.name,
                "paper_dir": paper_dir,
                "triples": triples_data
            })
        except Exception as e:
            _emit(MessageType.WARNING, f"Failed to load {triples_file}: {e}")

    return papers_with_triples


async def enrich_with_wikidata(papers_with_triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich all triples with Wikidata information.

    Args:
        papers_with_triples: List of paper data with triples

    Returns:
        Same list with triples enriched with wikidata
    """
    from aii_pipeline.steps._seed_hypo.invention_kg.grounding.wikidata import (
        enrich_triples_async,
        set_telemetry as set_wikidata_telemetry,
    )
    # Pass telemetry to wikidata module
    set_wikidata_telemetry(_telemetry)

    # Collect all triples from all papers
    all_triples = []
    triple_locations = []  # (paper_idx, triple_idx)

    for paper_idx, paper_data in enumerate(papers_with_triples):
        triples = paper_data.get("triples", {}).get("triples", [])
        for triple_idx, triple in enumerate(triples):
            all_triples.append(triple)
            triple_locations.append((paper_idx, triple_idx))

    if not all_triples:
        _emit(MessageType.WARNING, "No triples found to enrich")
        return papers_with_triples

    _emit(MessageType.INFO, f"Enriching {len(all_triples)} triples with Wikidata...")

    # Enrich all triples at once (async for speed) - await directly
    enriched_triples = await enrich_triples_async(all_triples, resolve_labels=True)

    # Update triples back in papers
    for i, (paper_idx, triple_idx) in enumerate(triple_locations):
        papers_with_triples[paper_idx]["triples"]["triples"][triple_idx] = enriched_triples[i]

    return papers_with_triples


def save_enriched_triples(
    papers_with_triples: List[Dict[str, Any]],
    output_dir: Path
) -> int:
    """
    Save enriched triples to output directory.

    Args:
        papers_with_triples: List of paper data with enriched triples
        output_dir: Output directory path

    Returns:
        Number of papers saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for paper_data in papers_with_triples:
        paper_id = paper_data["paper_id"]
        triples_data = paper_data["triples"]

        # Create paper output directory
        paper_output_dir = output_dir / paper_id
        paper_output_dir.mkdir(parents=True, exist_ok=True)

        # Save enriched triples
        output_file = paper_output_dir / "triples.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(triples_data, f, indent=2, ensure_ascii=False)
            saved += 1
        except Exception as e:
            _emit(MessageType.ERROR, f"Failed to save {output_file}: {e}")

    return saved


async def main(run_id: str, telemetry: Optional[AIITelemetry] = None):
    """
    Main entry point.

    Args:
        run_id: Run ID for pipeline orchestration mode.
        telemetry: Optional AIITelemetry instance.
    """
    global _telemetry
    _telemetry = telemetry

    from aii_pipeline.steps._seed_hypo.invention_kg.utils import get_run_dir
    from aii_pipeline.steps._seed_hypo.invention_kg.constants import (
        RUNS_DIR,
        STEP_4_TRIPLES,
        STEP_5_WIKIDATA
    )

    _emit(MessageType.INFO, f"Run ID: {run_id}")

    # Get input/output directories (uses RUNS_DIR for pipeline runs)
    triples_dir = get_run_dir(STEP_4_TRIPLES, run_id)
    output_dir = get_run_dir(STEP_5_WIKIDATA, run_id)

    _emit(MessageType.INFO, f"Input: {triples_dir.relative_to(RUNS_DIR)}")
    _emit(MessageType.INFO, f"Output: {output_dir.relative_to(RUNS_DIR)}")

    # Load triples
    papers_with_triples = load_triples_from_run(triples_dir)

    if not papers_with_triples:
        _emit(MessageType.WARNING, f"No triples found in {triples_dir}")
        return 1

    _emit(MessageType.INFO, f"Found {len(papers_with_triples)} papers with triples")

    # Count total triples
    total_triples = sum(
        len(p.get("triples", {}).get("triples", []))
        for p in papers_with_triples
    )
    _emit(MessageType.INFO, f"Total triples: {total_triples}")

    # Enrich with Wikidata (async)
    papers_with_triples = await enrich_with_wikidata(papers_with_triples)

    # Save enriched triples
    saved = save_enriched_triples(papers_with_triples, output_dir)

    _emit(MessageType.SUCCESS, f"Saved {saved} papers with enriched triples to {output_dir.relative_to(RUNS_DIR)}")

    # Stats
    enriched_count = 0
    for paper_data in papers_with_triples:
        for triple in paper_data.get("triples", {}).get("triples", []):
            if triple.get("wikidata"):
                enriched_count += 1

    _emit(MessageType.INFO, f"Enrichment rate: {enriched_count}/{total_triples} ({enriched_count/total_triples*100:.1f}%)")

    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python _5_add_wikidata.py <run_id>")
        sys.exit(1)
