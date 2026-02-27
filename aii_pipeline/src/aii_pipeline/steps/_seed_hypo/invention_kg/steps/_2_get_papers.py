#!/usr/bin/env python3
"""
Step 2: Get Papers

Fetch papers from OpenAlex for each selected topic.

Input: data/_1_sel_topics/{run_id}/topics.json (resolved topics from step 1)
Output: data/_2_papers/{run_id}/topic_{id}/papers_{year}.json

Loop structure: for each topic -> for each year -> fetch papers
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional
import sys

from pyalex import Works, config as pyalex_config
from aii_lib import AIITelemetry, MessageType
from tenacity import retry, stop_after_attempt, wait_exponential


# Module-level telemetry (set by run_get_papers)
_telemetry: Optional[AIITelemetry] = None


def _emit(msg_type: MessageType, msg: str):
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True
)
def fetch_papers_for_topic_year(
    topic_id: str,
    year: int,
    limit: int = 100,
    sort_by: str = "cited_by_count"
) -> List[Dict]:
    """
    Fetch papers for a specific topic and year from OpenAlex.

    Args:
        topic_id: OpenAlex topic ID (e.g., "T10456")
        year: Publication year
        limit: Maximum number of papers to fetch
        sort_by: Sort criterion (default: "cited_by_count")

    Returns:
        List of paper dictionaries with metadata
    """
    try:
        # Query OpenAlex for papers with this topic
        query = (
            Works()
            .filter(publication_year=year)
            .filter(topics={"id": topic_id})
            .filter(has_abstract=True)
            .sort(**{sort_by: "desc"})
            .select([
                # Core identification
                "id", "doi", "title", "publication_year", "publication_date",
                "cited_by_count", "type", "language",

                # Abstract (inverted index format)
                "abstract_inverted_index",

                # Topics & concepts
                "topics",
                "primary_topic",
                "keywords",
                "concepts",

                # Graph building
                "referenced_works",
                "related_works",

                # Open access
                "best_oa_location",
                "open_access",

                # Authorship
                "authorships",

                # Additional metadata
                "biblio",
            ])
        )

        # Paginate to get papers up to limit
        all_papers = []
        per_page = min(200, limit)

        pager = query.paginate(per_page=per_page, n_max=limit)

        for page in pager:
            all_papers.extend(page)
            if len(all_papers) >= limit:
                all_papers = all_papers[:limit]
                break

        # Filter out papers with null abstracts
        original_count = len(all_papers)
        all_papers = [p for p in all_papers if p.get('abstract_inverted_index')]
        filtered_count = original_count - len(all_papers)

        return all_papers, filtered_count

    except Exception as e:
        _emit(MessageType.ERROR, f"Error fetching papers for topic {topic_id}, year {year}: {e}")
        raise


async def fetch_papers_for_topic(
    topic: Dict,
    start_year: int,
    end_year: int,
    papers_per_year: int,
    output_dir: Path,
    sort_by: str = "cited_by_count",
    max_concurrent: int = 6,
) -> Dict:
    """
    Fetch papers for a single topic across all years (parallel).

    Args:
        topic: Topic dict with id, display_name, openalex_id
        start_year: First year to fetch
        end_year: Last year to fetch (inclusive)
        papers_per_year: Papers to fetch per year
        output_dir: Directory to save results (topic-specific)
        sort_by: Sort criterion
        max_concurrent: Max concurrent year fetches

    Returns:
        Summary dict with counts
    """
    topic_name = topic['display_name']
    topic_id = topic['openalex_id']

    _emit(MessageType.INFO, f"=== Topic: {topic_name} ({topic_id}) ===")

    # Create topic output directory
    topic_dir = output_dir / f"topic_{topic_id}"
    topic_dir.mkdir(parents=True, exist_ok=True)

    total_papers = 0
    years_data = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    executor = ThreadPoolExecutor(max_workers=max_concurrent)
    loop = asyncio.get_event_loop()

    async def _fetch_year(year: int) -> tuple:
        """Fetch papers for a single year, return (year, count)."""
        output_file = topic_dir / f"papers_{year}.json"

        # Skip if already exists
        if output_file.exists():
            _emit(MessageType.INFO, f"  {year}: already exists, skipping")
            with open(output_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            return year, len(papers)

        # Fetch papers (run sync call in thread pool)
        try:
            async with semaphore:
                papers, filtered = await loop.run_in_executor(
                    executor,
                    lambda: fetch_papers_for_topic_year(
                        topic_id=topic_id,
                        year=year,
                        limit=papers_per_year,
                        sort_by=sort_by,
                    )
                )

            if papers:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(papers, f, indent=2, ensure_ascii=False)
                filter_note = f" (filtered {filtered})" if filtered else ""
                _emit(MessageType.INFO, f"  {year}: {len(papers)} papers{filter_note}")
                return year, len(papers)
            else:
                _emit(MessageType.WARNING, f"  {year}: no papers found")
                return year, 0

        except Exception as e:
            _emit(MessageType.ERROR, f"  {year}: failed - {e}")
            return year, 0

    # Fetch all years in parallel
    years = list(range(start_year, end_year + 1))
    results = await asyncio.gather(*[_fetch_year(y) for y in years])
    executor.shutdown(wait=False)

    for year, count in results:
        years_data[year] = count
        total_papers += count

    # Save topic summary
    summary = {
        'topic_id': topic_id,
        'topic_name': topic_name,
        'years': years_data,
        'total_papers': total_papers,
    }
    summary_file = topic_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    _emit(MessageType.SUCCESS, f"  Total: {total_papers} papers for {topic_name}")
    return summary


async def run_get_papers(
    topics_file: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
    papers_per_topic_per_year: int,
    sort_by: str = "cited_by_count",
    email: str = "adrian.m.grobelnik@ijs.si",
    telemetry: Optional[AIITelemetry] = None,
) -> Dict:
    """
    Run the get papers step for all topics (one topic at a time, years in parallel).

    Args:
        topics_file: Path to resolved topics JSON from step 1
        output_dir: Base output directory
        start_year: First year
        end_year: Last year (inclusive)
        papers_per_topic_per_year: Papers to fetch per topic per year
        sort_by: Sort criterion
        email: Email for OpenAlex API
        telemetry: Optional AIITelemetry instance

    Returns:
        Summary dict
    """
    global _telemetry
    _telemetry = telemetry

    # Configure pyalex
    pyalex_config.email = email

    _emit(MessageType.INFO, "=== Step 2: Get Papers ===")

    # Load topics from step 1
    with open(topics_file, 'r', encoding='utf-8') as f:
        topics_data = json.load(f)

    topics = topics_data['topics']
    _emit(MessageType.INFO, f"Topics: {len(topics)}")
    _emit(MessageType.INFO, f"Years: {start_year}-{end_year} ({end_year - start_year + 1} years)")
    _emit(MessageType.INFO, f"Papers per topic per year: {papers_per_topic_per_year}")

    expected_total = len(topics) * (end_year - start_year + 1) * papers_per_topic_per_year
    _emit(MessageType.INFO, f"Expected max papers: ~{expected_total:,}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch papers for each topic (one topic at a time, years in parallel)
    all_summaries = []
    grand_total = 0

    for topic in topics:
        summary = await fetch_papers_for_topic(
            topic=topic,
            start_year=start_year,
            end_year=end_year,
            papers_per_year=papers_per_topic_per_year,
            output_dir=output_dir,
            sort_by=sort_by,
        )
        all_summaries.append(summary)
        grand_total += summary['total_papers']

    # Save overall summary
    result = {
        'topics_file': str(topics_file),
        'start_year': start_year,
        'end_year': end_year,
        'papers_per_topic_per_year': papers_per_topic_per_year,
        'topic_summaries': all_summaries,
        'grand_total_papers': grand_total,
    }

    result_file = output_dir / "fetch_summary.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    # Final summary
    _emit(MessageType.SUCCESS, "=== Fetch Complete ===")
    _emit(MessageType.INFO, f"Total papers: {grand_total:,}")
    for s in all_summaries:
        _emit(MessageType.INFO, f"  {s['topic_name']}: {s['total_papers']:,}")

    return result


if __name__ == "__main__":
    """Standalone execution for testing."""
    import yaml
    from aii_lib import create_telemetry

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get settings from config
    get_papers_config = config.get('get_papers', {})
    year_range = get_papers_config.get('year_range', {})
    start_year = year_range.get('start', 2020)
    end_year = year_range.get('end', 2025)
    papers_per_topic_per_year = get_papers_config.get('papers_per_topic_per_year', 10)
    sort_by = get_papers_config.get('sort_by', 'cited_by_count')
    email = get_papers_config.get('email', 'adrian.m.grobelnik@ijs.si')

    # Find latest topics file from step 1
    from aii_pipeline.steps._seed_hypo.invention_kg.utils import find_most_recent_run_id, get_run_dir, create_run_id
    from aii_pipeline.steps._seed_hypo.invention_kg.constants import BASE_DIR, STEP_1_SEL_TOPICS, STEP_2_PAPERS

    step1_run_id = find_most_recent_run_id(STEP_1_SEL_TOPICS)
    if not step1_run_id:
        print("No step 1 output found! Run _1_sel_topics.py first.")
        sys.exit(1)

    topics_file = get_run_dir(STEP_1_SEL_TOPICS, step1_run_id) / "topics.json"

    # Create output directory with new run_id
    with open(topics_file) as f:
        topics_data = json.load(f)
    num_topics = len(topics_data['topics'])
    num_years = end_year - start_year + 1
    expected_papers = num_topics * num_years * papers_per_topic_per_year

    run_id = create_run_id(expected_papers)
    output_dir = BASE_DIR / "data" / STEP_2_PAPERS / run_id

    # Create telemetry
    telemetry = create_telemetry(output_dir, "get_papers")

    telemetry.emit(MessageType.INFO, f"Using topics from: {topics_file}")

    # Run
    result = asyncio.run(run_get_papers(
        topics_file=topics_file,
        output_dir=output_dir,
        start_year=start_year,
        end_year=end_year,
        papers_per_topic_per_year=papers_per_topic_per_year,
        sort_by=sort_by,
        email=email,
        telemetry=telemetry,
    ))

    telemetry.emit(MessageType.SUCCESS, f"Step 2 complete! Fetched {result['grand_total_papers']:,} papers.")
    telemetry.flush()
