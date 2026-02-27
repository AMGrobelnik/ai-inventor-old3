#!/usr/bin/env python3
"""
Step 8: Generate Seed Prompts

Takes hypothesis seeds from step 7 and formats them into prompt snippets.
These snippets describe opportunities that can be inserted into a larger
LLM prompt elsewhere.

Input: data/_7_hypo_seeds/{run_id}/
    - topic_blind_spots.json

Output: data/_8_seed_prompt/{run_id}/
    - blind_spot_prompts.json
"""

import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from aii_lib import AIITelemetry, MessageType

from ._gen_seed_prompt import format_opportunity_prompt


# Module-level telemetry (set by main)
_telemetry: Optional[AIITelemetry] = None


def _emit(msg_type: MessageType, msg: str):
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")

__all__ = ["main"]


def extract_score_info(opp: Dict[str, Any]) -> Dict[str, Any]:
    """Extract score and breakdown from concept-centric blind spot opportunity."""
    # Get hierarchical scores from sub-objects
    topic_pair = opp.get("topic_pair", {})
    importance = opp.get("importance", {})
    transferability = opp.get("transferability", {})
    novelty = opp.get("novelty", {})

    return {
        "score": opp.get("seed_score", 0),
        "score_breakdown": {
            "topic_pair": topic_pair.get("score", 0),
            "importance": importance.get("score", 0),
            "transferability": transferability.get("score", 0),
            "novelty": novelty.get("score", 0),
        },
        "percentile_breakdown": {
            "topic_pair": topic_pair.get("percentile", 0),
            "importance": importance.get("percentile", 0),
            "transferability": transferability.get("percentile", 0),
            "novelty": novelty.get("percentile", 0),
        }
    }


def extract_topics(opp: Dict[str, Any]) -> List[str]:
    """Extract topic names from concept-centric blind spot opportunity."""
    topics = []

    # In concept-centric format, blind_topic and ref_topic are strings
    blind_topic = opp.get("blind_topic")
    if blind_topic:
        if isinstance(blind_topic, dict):
            topics.append(blind_topic.get("name", ""))
        else:
            topics.append(str(blind_topic))

    ref_topic = opp.get("ref_topic")
    if ref_topic:
        if isinstance(ref_topic, dict):
            topics.append(ref_topic.get("name", ""))
        else:
            topics.append(str(ref_topic))

    return [t for t in topics if t]


def format_opportunities(opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format blind spot opportunities into prompt snippets with scores."""
    _emit(MessageType.INFO, f"Formatting {len(opportunities)} blind spot opportunities")

    results = []
    for i, opp in enumerate(opportunities):
        try:
            prompt_text = format_opportunity_prompt(opp)
            score_info = extract_score_info(opp)
            topics = extract_topics(opp)

            # Use existing id from concept-centric format, or generate one
            opp_id = opp.get("id", f"blind_spot_idx{i}")

            # Extract blind_topic and ref_topic as separate fields
            blind_topic = opp.get("blind_topic", "")
            if isinstance(blind_topic, dict):
                blind_topic = blind_topic.get("name", "")
            ref_topic = opp.get("ref_topic", "")
            if isinstance(ref_topic, dict):
                ref_topic = ref_topic.get("name", "")

            results.append({
                "id": opp_id,
                "type": "blind_spot",
                "concept": opp.get("concept", ""),
                "entity_type": opp.get("entity_type", ""),
                "blind_topic": blind_topic,
                "ref_topic": ref_topic,
                "topics": topics,
                "score": score_info["score"],
                "score_breakdown": score_info["score_breakdown"],
                "score_percentile": opp.get("score_percentile", 0),
                "percentile_breakdown": score_info["percentile_breakdown"],
                "prompt": prompt_text,
            })
        except Exception as e:
            _emit(MessageType.WARNING, f"Failed to format opportunity {i}: {e}")

    _emit(MessageType.INFO, f"Formatted {len(results)}/{len(opportunities)} prompts")
    return results




def main(run_id: str, telemetry: Optional[AIITelemetry] = None):
    """
    Main entry point for seed prompt formatting.

    Args:
        run_id: Run ID for pipeline orchestration.
        telemetry: Optional AIITelemetry instance.
    """
    global _telemetry
    _telemetry = telemetry

    from ..utils import get_run_dir
    from ..constants import (
        RUNS_DIR,
        STEP_7_HYPO_SEEDS,
        STEP_8_SEED_PROMPT,
    )

    _emit(MessageType.INFO, f"Run ID: {run_id}")

    # Get input/output directories (uses RUNS_DIR for pipeline runs)
    input_dir = get_run_dir(STEP_7_HYPO_SEEDS, run_id)
    output_dir = get_run_dir(STEP_8_SEED_PROMPT, run_id)

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
        _emit(MessageType.ERROR, "Run step 7 first")
        return 1

    # Load opportunities
    blind_spots_file = input_dir / "topic_blind_spots.json"

    if not blind_spots_file.exists():
        _emit(MessageType.ERROR, "No topic_blind_spots.json found")
        return 1

    with open(blind_spots_file, 'r') as f:
        blind_spots = json.load(f)
    _emit(MessageType.INFO, f"Loaded {len(blind_spots)} blind spot opportunities")

    if not blind_spots:
        _emit(MessageType.ERROR, "No opportunities found")
        return 1

    # Format blind spot prompts
    _emit(MessageType.INFO, "1. Formatting Blind Spot Opportunities")
    all_prompts = format_opportunities(blind_spots)

    if all_prompts:
        # Sort by score_percentile descending (already computed in step 7)
        all_prompts.sort(key=lambda x: x.get('score_percentile', 0), reverse=True)

        with open(output_dir / "blind_spot_prompts.json", 'w') as f:
            json.dump(all_prompts, f, indent=2, ensure_ascii=False)

    # Summary
    _emit(MessageType.INFO, "Summary")
    _emit(MessageType.INFO, f"Total prompts formatted: {len(all_prompts)}")

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
        print("Usage: python _8_gen_seed_prompt.py <run_id>")
        sys.exit(1)
