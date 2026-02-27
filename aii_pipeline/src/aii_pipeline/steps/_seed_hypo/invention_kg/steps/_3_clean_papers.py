#!/usr/bin/env python3
"""
Step 3: Clean Papers

Extract minimal data needed for agent prompts from raw OpenAlex papers.

Input: data/_2_papers/{run_id}/topic_{id}/papers_{year}.json (raw OpenAlex data)
Output: data/_3_papers_clean/{run_id}/paper_XXXXX/paper.json (individual paper files)

Minimal format includes:
- index: Global paper index
- id: OpenAlex ID
- doi: Paper DOI
- title: Paper title
- abstract: Reconstructed from inverted index
- publication_year: Year published
- topic_id: OpenAlex topic ID the paper belongs to
- topic_name: Human readable topic name
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

from aii_lib import AIITelemetry, MessageType


# Module-level telemetry (set by run_clean_papers)
_telemetry: Optional[AIITelemetry] = None


def _emit(msg_type: MessageType, msg: str):
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")


def reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    """
    Reconstruct abstract text from OpenAlex inverted index.

    Args:
        inverted_index: Dict mapping words to position indices

    Returns:
        Reconstructed abstract text
    """
    if not inverted_index:
        return ""

    # Create list of (position, word) tuples
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    # Sort by position and join
    word_positions.sort(key=lambda x: x[0])
    abstract = " ".join(word for _, word in word_positions)

    return abstract


def extract_minimal_paper(
    paper: Dict[str, Any],
    global_index: int,
    topic_id: str,
    topic_name: str
) -> Dict[str, Any]:
    """
    Extract fields needed for agent analysis and graph construction.

    Args:
        paper: Full OpenAlex paper object
        global_index: Global paper index across all years/topics
        topic_id: OpenAlex topic ID (e.g., "T10456")
        topic_name: Human readable topic name

    Returns:
        Paper dict with metadata, title, abstract, and author information
    """
    # Reconstruct abstract
    abstract = ""
    if 'abstract_inverted_index' in paper and paper['abstract_inverted_index']:
        abstract = reconstruct_abstract(paper['abstract_inverted_index'])

    return {
        "index": global_index,
        "id": paper.get('id', ''),
        "doi": paper.get('doi', ''),
        "title": paper.get('title', ''),
        "abstract": abstract,
        "publication_year": paper.get('publication_year'),
        "publication_date": paper.get('publication_date', ''),
        "cited_by_count": paper.get('cited_by_count', 0),
        "type": paper.get('type', ''),
        "language": paper.get('language', ''),
        "topic_id": topic_id,
        "topic_name": topic_name,
        "authorships": paper.get('authorships', [])
    }


def run_clean_papers(
    papers_dir: Path,
    output_dir: Path,
    telemetry: Optional[AIITelemetry] = None,
) -> Dict:
    """
    Clean all papers from the topic-based directory structure.

    Args:
        papers_dir: Path to papers directory (e.g., data/_2_papers/{run_id}/)
        output_dir: Directory to save individual paper folders
        telemetry: Optional AIITelemetry instance

    Returns:
        Summary dict with counts
    """
    global _telemetry
    _telemetry = telemetry

    _emit(MessageType.INFO, "=== Step 3: Clean Papers ===")
    _emit(MessageType.INFO, f"Input: {papers_dir}")
    _emit(MessageType.INFO, f"Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load topic summaries to get topic names
    topic_names = {}
    for topic_dir in papers_dir.glob("topic_*"):
        summary_file = topic_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                topic_id = summary.get('topic_id', topic_dir.name.replace('topic_', ''))
                topic_names[topic_id] = summary.get('topic_name', '')

    # Process papers from all topic directories
    global_index = 0
    topic_stats = {}
    all_papers = []

    for topic_dir in sorted(papers_dir.glob("topic_*")):
        topic_id = topic_dir.name.replace('topic_', '')
        topic_name = topic_names.get(topic_id, topic_id)
        topic_papers = 0

        _emit(MessageType.INFO, f"Processing topic: {topic_name} ({topic_id})")

        # Process each year file in the topic
        for year_file in sorted(topic_dir.glob("papers_*.json")):
            year = year_file.stem.replace('papers_', '')

            with open(year_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)

            for paper in papers:
                clean_paper = extract_minimal_paper(
                    paper, global_index, topic_id, topic_name
                )

                # Create paper directory
                paper_dir = output_dir / f"paper_{global_index:05d}"
                paper_dir.mkdir(exist_ok=True)

                # Save paper.json
                paper_file = paper_dir / "paper.json"
                with open(paper_file, 'w', encoding='utf-8') as f:
                    json.dump(clean_paper, f, indent=2, ensure_ascii=False)

                all_papers.append({
                    "index": global_index,
                    "topic_id": topic_id,
                    "year": year,
                    "title": clean_paper["title"][:60],
                })

                global_index += 1
                topic_papers += 1

            _emit(MessageType.INFO, f"  {year}: {len(papers)} papers")

        topic_stats[topic_id] = {
            "name": topic_name,
            "papers": topic_papers,
        }
        _emit(MessageType.SUCCESS, f"  Total: {topic_papers} papers for {topic_name}")

    # Save summary
    result = {
        "total_papers": global_index,
        "topic_stats": topic_stats,
        "papers": all_papers,
    }

    summary_file = output_dir / "clean_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Final summary
    _emit(MessageType.SUCCESS, "=== Clean Papers Complete ===")
    _emit(MessageType.INFO, f"Total papers cleaned: {global_index}")
    for topic_id, stats in topic_stats.items():
        _emit(MessageType.INFO, f"  {stats['name']}: {stats['papers']} papers")
    _emit(MessageType.INFO, f"Output: {output_dir}")

    return result


if __name__ == "__main__":
    """Standalone execution for testing."""
    from aii_lib import create_telemetry

    # Load config
    from aii_pipeline.steps._seed_hypo.invention_kg.utils import find_most_recent_run_id, get_run_dir
    from aii_pipeline.steps._seed_hypo.invention_kg.constants import BASE_DIR, STEP_2_PAPERS, STEP_3_PAPERS_CLEAN

    # Find most recent step 2 run
    step2_run_id = find_most_recent_run_id(STEP_2_PAPERS)
    if not step2_run_id:
        print("No step 2 output found! Run _2_get_papers.py first.")
        sys.exit(1)

    papers_dir = get_run_dir(STEP_2_PAPERS, step2_run_id)

    # Use same run_id for output
    output_dir = get_run_dir(STEP_3_PAPERS_CLEAN, step2_run_id)

    # Create telemetry
    telemetry = create_telemetry(output_dir, "clean_papers")

    telemetry.emit(MessageType.INFO, f"Using papers from: {papers_dir}")

    # Run
    result = run_clean_papers(papers_dir, output_dir, telemetry=telemetry)

    telemetry.emit(MessageType.SUCCESS, f"Step 3 complete! Cleaned {result['total_papers']} papers.")
    telemetry.flush()
