#!/usr/bin/env python3
"""
Paper loading functions for graph generation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from aii_lib import AIITelemetry, MessageType

# ============================================================================
# Module-level telemetry (set by caller via set_telemetry)
# ============================================================================
_telemetry: Optional[AIITelemetry] = None


def set_telemetry(telemetry: Optional[AIITelemetry]):
    """Set module telemetry from calling code."""
    global _telemetry
    _telemetry = telemetry


def _emit(msg_type: MessageType, msg: str):
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")


def load_all_papers(combined_dir: Path) -> List[Dict]:
    """Load all papers from combined JSON file."""
    json_file = combined_dir / "paper_triples_pr.json"

    if not json_file.exists():
        _emit(MessageType.ERROR, f"Combined papers file not found: {json_file}")
        return []

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        _emit(MessageType.INFO, f"Loaded {len(papers)} papers")
        return papers
    except Exception as e:
        _emit(MessageType.ERROR, f"Failed to load {json_file}: {e}")
        return []


def load_papers_by_year(combined_dir: Path) -> Dict[int, List[Dict]]:
    """Load papers grouped by year."""
    all_papers = load_all_papers(combined_dir)
    if not all_papers:
        return {}

    papers_by_year = {}
    for paper_entry in all_papers:
        paper_data = paper_entry.get("paper", {})
        year = paper_data.get("publication_year")

        if year is not None:
            if year not in papers_by_year:
                papers_by_year[year] = []
            papers_by_year[year].append(paper_entry)

    _emit(MessageType.INFO, f"Loaded papers from {len(papers_by_year)} years")
    return papers_by_year
