#!/usr/bin/env python3
"""
Step 1: Select Topics

Resolves topic names from config to OpenAlex topic IDs.
This step validates that the configured topics exist in OpenAlex
and retrieves their full metadata for use in subsequent steps.

Input: config.yaml sel_topics.topics (list of topic names)
Output: data/_1_sel_topics/{run_id}/topics.json (resolved topic metadata)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pyalex import Topics, config as pyalex_config
from aii_lib import AIITelemetry, MessageType
from tenacity import retry, stop_after_attempt, wait_exponential


# Module-level telemetry (set by run_sel_topics)
_telemetry: Optional[AIITelemetry] = None


def _emit(msg_type: MessageType, msg: str):
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True
)
def resolve_topic(topic_name: str) -> Optional[Dict]:
    """
    Resolve a topic name to its OpenAlex metadata.

    Args:
        topic_name: Display name of the topic (e.g., "Multi-Agent Systems and Negotiation")

    Returns:
        Topic metadata dict if found, None otherwise
    """
    # Search for the topic
    results = list(Topics().search(topic_name).get())

    if not results:
        _emit(MessageType.WARNING, f"No results for topic: {topic_name}")
        return None

    # Find exact match or best match
    for topic in results:
        if topic['display_name'].lower() == topic_name.lower():
            _emit(MessageType.INFO, f"Found exact match: {topic['display_name']}")
            return topic

    # Return first result as best match
    best_match = results[0]
    _emit(
        MessageType.WARNING,
        f"No exact match for '{topic_name}', using best match: '{best_match['display_name']}'"
    )
    return best_match


def resolve_topics(
    topic_names: List[str],
    email: str = "adrian.m.grobelnik@ijs.si"
) -> List[Dict]:
    """
    Resolve a list of topic names to OpenAlex topic metadata.

    Args:
        topic_names: List of topic display names
        email: Email for OpenAlex polite pool access

    Returns:
        List of resolved topic metadata dicts
    """
    # Configure pyalex
    pyalex_config.email = email

    _emit(MessageType.INFO, f"Resolving {len(topic_names)} topics from OpenAlex")

    resolved = []
    failed = []

    for name in topic_names:
        topic = resolve_topic(name)
        if topic:
            resolved.append({
                'display_name': topic['display_name'],
                'id': topic['id'],
                'openalex_id': topic['id'].split('/')[-1],  # e.g., "T12345"
                'description': topic.get('description', ''),
                'keywords': topic.get('keywords', []),
                'works_count': topic.get('works_count', 0),
                'domain': topic.get('domain', {}),
                'field': topic.get('field', {}),
                'subfield': topic.get('subfield', {}),
            })
        else:
            failed.append(name)

    if failed:
        _emit(MessageType.ERROR, f"Failed to resolve {len(failed)} topics: {failed}")

    _emit(MessageType.SUCCESS, f"Resolved {len(resolved)}/{len(topic_names)} topics")
    return resolved


def run_sel_topics(
    topic_names: List[str],
    output_dir: Path,
    email: str = "adrian.m.grobelnik@ijs.si",
    telemetry: Optional[AIITelemetry] = None,
) -> Dict:
    """
    Run the topic selection step.

    Args:
        topic_names: List of topic names from config
        output_dir: Directory to save results
        email: Email for OpenAlex API
        telemetry: Optional AIITelemetry instance

    Returns:
        Dict with resolved topics and metadata
    """
    global _telemetry
    _telemetry = telemetry

    _emit(MessageType.INFO, "=== Step 1: Select Topics ===")
    _emit(MessageType.INFO, f"Topics to resolve: {len(topic_names)}")
    for name in topic_names:
        _emit(MessageType.INFO, f"  - {name}")

    # Resolve topics
    resolved_topics = resolve_topics(topic_names, email=email)

    if not resolved_topics:
        raise ValueError("No topics could be resolved!")

    # Create output
    result = {
        'resolved_at': datetime.now().isoformat(),
        'requested_topics': topic_names,
        'resolved_count': len(resolved_topics),
        'topics': resolved_topics,
    }

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'topics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    _emit(MessageType.SUCCESS, f"Saved resolved topics to {output_file}")

    # Log summary
    _emit(MessageType.INFO, "Resolved Topics:")
    for topic in resolved_topics:
        _emit(MessageType.INFO, f"  Topic:    {topic['display_name']}")
        _emit(MessageType.INFO, f"  Domain:   {topic['domain'].get('display_name', '?')}")
        _emit(MessageType.INFO, f"  Field:    {topic['field'].get('display_name', '?')}")
        _emit(MessageType.INFO, f"  Subfield: {topic['subfield'].get('display_name', '?')}")
        _emit(MessageType.INFO, f"  Works:    {topic['works_count']:,}")

    return result


if __name__ == "__main__":
    """Standalone execution for testing."""
    import sys
    import yaml
    from aii_lib import create_telemetry

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get topics from config
    topic_names = config.get('sel_topics', {}).get('topics', [])
    if not topic_names:
        print("No topics configured in sel_topics.topics!")
        sys.exit(1)

    # Create output directory
    from aii_pipeline.steps._seed_hypo.invention_kg.utils import create_run_id
    from aii_pipeline.steps._seed_hypo.invention_kg.constants import BASE_DIR, STEP_1_SEL_TOPICS

    run_id = create_run_id(len(topic_names))
    output_dir = BASE_DIR / "data" / STEP_1_SEL_TOPICS / run_id

    # Get email from config
    email = config.get('get_papers', {}).get('email', 'adrian.m.grobelnik@ijs.si')

    # Create telemetry
    telemetry = create_telemetry(output_dir, "sel_topics")

    # Run
    result = run_sel_topics(topic_names, output_dir, email=email, telemetry=telemetry)

    telemetry.emit(MessageType.SUCCESS, f"Step 1 complete! Resolved {result['resolved_count']} topics.")
    telemetry.flush()
