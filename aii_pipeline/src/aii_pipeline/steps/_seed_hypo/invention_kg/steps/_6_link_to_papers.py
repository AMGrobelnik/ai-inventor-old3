#!/usr/bin/env python3
"""
Step 6: Add Papers to Triples

Combines clean paper data with Wikidata-enriched triples.

Input:
  - data/_3_papers_clean/{run_id}/ (clean paper.json files)
  - data/_5_wikidata/{run_id}/ (enriched triples.json files)

Output:
  - data/_6_paper_triples/{run_id}/paper_triples_pr.json

The combined output has structure:
{
  "index": 0,
  "paper": {...},  # Clean paper data
  "triples": {...}  # Enriched triples with wikidata
}
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from aii_lib import AIITelemetry, MessageType


# Module-level telemetry (set by main)
_telemetry: Optional[AIITelemetry] = None


def _emit(msg_type: MessageType, msg: str):
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")


def load_paper_data(paper_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load paper.json from paper directory.

    Args:
        paper_dir: Path to the paper directory

    Returns:
        Paper data dictionary, or None if not found/invalid
    """
    paper_file = paper_dir / "paper.json"

    if not paper_file.exists():
        _emit(MessageType.WARNING, f"No paper.json found in {paper_dir.name}")
        return None

    try:
        with open(paper_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        _emit(MessageType.ERROR, f"Failed to parse JSON in {paper_file}: {e}")
        return None
    except Exception as e:
        _emit(MessageType.ERROR, f"Error loading {paper_file}: {e}")
        return None


def load_enriched_triples(triples_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load triples.json from enriched triples directory (step 5 output).

    Args:
        triples_dir: Path to the paper's triples directory

    Returns:
        Triples data, or None if not found/invalid
    """
    triples_file = triples_dir / "triples.json"

    if not triples_file.exists():
        return None

    try:
        with open(triples_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        _emit(MessageType.ERROR, f"Failed to parse JSON in {triples_file}: {e}")
        return None
    except Exception as e:
        _emit(MessageType.ERROR, f"Error loading {triples_file}: {e}")
        return None


def process_papers(clean_papers_dir: Path, enriched_triples_dir: Path) -> Dict[str, Any]:
    """
    Process all paper directories and combine clean papers with enriched triples.

    Args:
        clean_papers_dir: Directory containing paper_XXXXX/ folders with paper.json
        enriched_triples_dir: Directory containing paper_XXXXX/ folders with triples.json

    Returns:
        Dictionary with combined papers and statistics
    """
    _emit(MessageType.INFO, "Combining papers with enriched triples")

    # Get all paper directories from clean_papers_dir
    paper_dirs = sorted(clean_papers_dir.glob("paper_*"))
    if not paper_dirs:
        _emit(MessageType.WARNING, f"No paper directories found in {clean_papers_dir}")
        return {"combined_papers": [], "stats": {"total": 0, "not_enriched": 0, "no_triples": 0, "with_triples": 0}}

    _emit(MessageType.INFO, f"Found {len(paper_dirs)} paper directories")

    combined_papers = []
    stats = {
        "total": len(paper_dirs),
        "not_enriched": 0,  # Papers not processed in step 5
        "no_triples": 0,     # Papers with empty triples
        "with_triples": 0,   # Papers with triples (saved)
    }

    for paper_dir in paper_dirs:
        # Extract index - handle both paper_XXXXX and paper_idxXXXXX formats
        paper_name = paper_dir.name
        paper_index = int(paper_name.replace("paper_idx", "").replace("paper_", ""))

        # Load clean paper
        clean_paper = load_paper_data(paper_dir)
        if clean_paper is None:
            _emit(MessageType.WARNING, f"Skipping {paper_dir.name}: no paper.json")
            continue

        # Load enriched triples from step 5
        # Use same folder name as clean papers (paper_XXXXX format)
        triples_dir = enriched_triples_dir / paper_name
        triples_data = load_enriched_triples(triples_dir)

        # Paper not enriched in step 5
        if triples_data is None:
            stats["not_enriched"] += 1
            continue

        # Check if triples were extracted
        triples = triples_data.get("triples", [])
        if not triples:
            stats["no_triples"] += 1
            continue

        # Has triples = success
        stats["with_triples"] += 1
        combined_papers.append({
            "index": paper_index,
            "paper": clean_paper,
            "triples": triples_data
        })

    return {"combined_papers": combined_papers, "stats": stats}


def main(run_id: str, telemetry: Optional[AIITelemetry] = None):
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
        STEP_3_PAPERS_CLEAN,
        STEP_5_WIKIDATA,
        STEP_6_PAPER_TRIPLES
    )

    _emit(MessageType.INFO, f"Run ID: {run_id}")

    # Get input/output directories (uses RUNS_DIR for pipeline runs)
    clean_papers_dir = get_run_dir(STEP_3_PAPERS_CLEAN, run_id)
    enriched_triples_dir = get_run_dir(STEP_5_WIKIDATA, run_id)
    output_dir = get_run_dir(STEP_6_PAPER_TRIPLES, run_id)

    _emit(MessageType.INFO, f"Clean papers: {clean_papers_dir.relative_to(RUNS_DIR)}")
    _emit(MessageType.INFO, f"Enriched triples: {enriched_triples_dir.relative_to(RUNS_DIR)}")
    _emit(MessageType.INFO, f"Output: {output_dir.relative_to(RUNS_DIR)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all papers
    result = process_papers(clean_papers_dir, enriched_triples_dir)
    combined_papers = result["combined_papers"]
    stats = result["stats"]

    # Save combined data
    output_file = output_dir / "paper_triples_pr.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_papers, f, indent=2, ensure_ascii=False)

        _emit(MessageType.SUCCESS, f"Saved {len(combined_papers)} papers to {output_file.name}")
    except Exception as e:
        _emit(MessageType.ERROR, f"Failed to save {output_file}: {e}")
        return 1

    # Print summary
    _emit(MessageType.INFO, "Summary")
    _emit(MessageType.INFO, f"Total papers: {stats['total']}")
    _emit(MessageType.INFO, f"  Not enriched (step 5): {stats['not_enriched']}")
    enriched = stats['total'] - stats['not_enriched']
    _emit(MessageType.INFO, f"  Enriched: {enriched}")
    _emit(MessageType.INFO, f"    With triples (saved): {stats['with_triples']}")
    _emit(MessageType.INFO, f"    No triples (excluded): {stats['no_triples']}")

    if stats["total"] > 0 and enriched > 0:
        enrich_rate = (enriched / stats['total']) * 100
        extraction_rate = (stats['with_triples'] / enriched) * 100
        _emit(MessageType.INFO, f"Enrichment rate: {enrich_rate:.1f}%")
        _emit(MessageType.INFO, f"Triple extraction rate: {extraction_rate:.1f}%")

    _emit(MessageType.SUCCESS, "Done!")

    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python _6_link_to_papers.py <run_id>")
        sys.exit(1)
