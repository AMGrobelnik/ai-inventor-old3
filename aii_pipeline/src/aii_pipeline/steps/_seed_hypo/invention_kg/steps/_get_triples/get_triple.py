#!/usr/bin/env python3
"""
Core logic for extracting triples from a single research paper.

Uses the gen_kg workflow from aii_lib for:
1. Initial prompt to extract triples
2. Wikipedia URL verification
3. Retry loop with conversation continuity for failed URLs
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from aii_lib import AIITelemetry, GenKGConfig, generate_kg_triples

# Import prompt, schema, and system prompt from standard location
from aii_pipeline.prompts.steps._1_seed_hypo._invention_kg import (
    triples_prompt,
    build_retry_prompt,
    Triples,
    get_system_prompt,
)


async def get_triples_for_paper(
    paper_id: int,
    paper_index: int,
    title: str,
    abstract: str,
    parent_run_dir: Path,
    agent_cwd_template: Path,
    config: Dict[str, Any],
    telemetry: Optional[AIITelemetry] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract knowledge graph triples from a single paper.

    Uses gen_kg workflow which:
    1. Runs agent to extract triples
    2. Verifies Wikipedia URLs exist
    3. Retries with conversation continuity if URLs are invalid

    Args:
        paper_id: Paper ID within the year
        paper_index: Global paper index across all years
        title: Paper title
        abstract: Paper abstract
        parent_run_dir: Parent run directory where this paper's folder will be created
        agent_cwd_template: Path to agent_cwd/ template directory
        config: Agent configuration from config.yaml (get_triples section with claude_agent nested)
        telemetry: Optional AIITelemetry instance for logging

    Returns:
        Dict with analysis results or None if failed:
        {
            "paper_id": int,
            "paper_index": int,
            "title": str,
            "cost": float,
            "run_dir": str,
            "analysis": dict
        }
    """
    # Create run directory: parent_run_dir/paper_{index:05d}/
    run_name = f"paper_{paper_index:05d}"
    run_dir = parent_run_dir / run_name
    agent_cwd_dir = run_dir / "agent_cwd"

    # Setup workspace from template
    run_dir.mkdir(parents=True, exist_ok=True)
    if agent_cwd_template.exists():
        if agent_cwd_dir.exists():
            shutil.rmtree(agent_cwd_dir)
        shutil.copytree(agent_cwd_template, agent_cwd_dir)
    else:
        agent_cwd_dir.mkdir(parents=True, exist_ok=True)

    # Create prompt
    prompt = triples_prompt(title, abstract)

    # Extract claude_agent config (nested under get_triples in config.yaml)
    claude_cfg = config.get("claude_agent", {})

    # Build workflow config
    # Note: gen_kg workflow uses aii_web_search_fast and aii_web_fetch_direct MCP tools automatically
    # via get_tooluniverse_mcp_config() and disallows WebSearch/WebFetch built-in tools
    kg_config = GenKGConfig(
        paper_id=paper_id,
        paper_index=paper_index,
        title=title,
        abstract=abstract,
        prompt=prompt,
        system_prompt=get_system_prompt(),  # From prompts module
        model=claude_cfg.get("model", "claude-haiku-4-5"),
        max_turns=claude_cfg.get("max_turns", 100),
        agent_timeout=claude_cfg.get("agent_timeout"),
        agent_retries=claude_cfg.get("agent_retries", 2),
        seq_prompt_timeout=claude_cfg.get("seq_prompt_timeout", 600),
        seq_prompt_retries=claude_cfg.get("seq_prompt_retries", 5),
        cwd=str(agent_cwd_dir),
        response_schema=Triples,
        verify_retries=config.get("url_verification_retries", 2),
        min_valid_urls=config.get("min_valid_urls", 0),
        build_retry_prompt_fn=build_retry_prompt,
    )

    # Run workflow
    result = await generate_kg_triples(kg_config, telemetry=telemetry)

    # Check result
    if result.error:
        return None

    if result.triples is None:
        return None

    # Write structured output to disk for downstream validation
    # (Previously written by StructJsonOutConfig, now written by caller)
    output_data = {"paper_type": result.paper_type, "triples": result.triples}
    (agent_cwd_dir / "triples_output.json").write_text(
        json.dumps(output_data, indent=2), encoding="utf-8"
    )

    # Build analysis dict from result
    analysis = {
        "paper_type": result.paper_type,
        "triples": result.triples,
    }

    return {
        "paper_id": paper_id,
        "paper_index": paper_index,
        "title": title,
        "cost": result.cost,
        "run_dir": str(run_dir),
        "analysis": analysis,
        "verified": result.verified,
    }
