#!/usr/bin/env python3
"""
Invention Knowledge Graph Pipeline

Main orchestrator for the entire pipeline:
1. Select Topics from OpenAlex
2. Get Papers from OpenAlex (for each topic)
3. Clean Papers (extract minimal data)
4. Get Triples (using agents)
5. Add Wikidata (enrich triples with Wikidata)
6. Link to Papers (combine papers with enriched triples)
7. Generate Hypothesis Seeds (blind spots, breakthroughs)
8. Generate Seed Prompts (research ideas from blind spots)
9. Generate Graphs (co-occurrence, ontology, etc.)
10. Visualize Graphs (interactive web visualization - optional/manual)

All settings controlled via config.yaml
"""

import sys
from pathlib import Path
from typing import Optional

from aii_lib import AIITelemetry, MessageType, create_telemetry


# Import utilities and constants
from .utils import create_run_id, find_most_recent_run_id, get_run_dir
from .constants import (
    STEP_1_SEL_TOPICS,
    STEP_2_PAPERS,
    STEP_3_PAPERS_CLEAN,
    STEP_4_TRIPLES,
    STEP_5_WIKIDATA,
    STEP_6_PAPER_TRIPLES,
    STEP_7_HYPO_SEEDS,
    STEP_8_SEED_PROMPT,
    STEP_9_GRAPHS,
    Colors
)
from .validation import validate_config


def run_step_1_sel_topics(config, run_id, telemetry: AIITelemetry):
    """Step 1: Select Topics from OpenAlex."""
    telemetry.emit(MessageType.INFO, f"Step 1: Select Topics (run_id: {run_id})")

    # Import step module
    from .steps._1_sel_topics import run_sel_topics

    # Get settings from config
    topic_names = config['sel_topics']['topics']
    email = config['get_papers']['email']

    # Create output directory with run_id
    output_dir = get_run_dir(STEP_1_SEL_TOPICS, run_id, config.base_dir)

    # Run step
    try:
        run_sel_topics(
            topic_names=topic_names,
            output_dir=output_dir,
            email=email,
            telemetry=telemetry,
        )
        telemetry.emit(MessageType.SUCCESS, f"Step 1 completed: {len(topic_names)} topics resolved")
        return True
    except Exception as e:
        telemetry.emit(MessageType.ERROR, f"Step 1 failed: {e}")
        raise


async def run_step_2_get_papers(config, run_id, telemetry: AIITelemetry):
    """Step 2: Get Papers from OpenAlex."""
    telemetry.emit(MessageType.INFO, f"Step 2: Get Papers (run_id: {run_id})")

    # Import step module
    from .steps._2_get_papers import run_get_papers

    # Get settings from config
    step_config = config['get_papers']
    start_year = step_config['year_range']['start']
    end_year = step_config['year_range']['end']
    papers_per_year = step_config['papers_per_year']
    sort_by = step_config['sort_by']
    email = step_config['email']

    # Get topics file from step 1
    topics_file = get_run_dir(STEP_1_SEL_TOPICS, run_id, config.base_dir) / 'topics.json'
    if not topics_file.exists():
        telemetry.emit(MessageType.ERROR, f"Topics file not found: {topics_file}")
        return False

    # Create output directory with run_id
    output_dir = get_run_dir(STEP_2_PAPERS, run_id, config.base_dir)

    # Fetch papers
    try:
        await run_get_papers(
            topics_file=topics_file,
            output_dir=output_dir,
            start_year=start_year,
            end_year=end_year,
            papers_per_topic_per_year=papers_per_year,
            sort_by=sort_by,
            email=email,
            telemetry=telemetry,
        )
        telemetry.emit(MessageType.SUCCESS, f"Step 2 completed")
        return True
    except Exception as e:
        telemetry.emit(MessageType.ERROR, f"Step 2 failed: {e}")
        raise


def run_step_3_clean_papers(config, run_id, telemetry: AIITelemetry):
    """Step 3: Clean Papers (extract minimal data)."""
    telemetry.emit(MessageType.INFO, f"Step 3: Clean Papers (run_id: {run_id})")

    # Import step module
    from .steps._3_clean_papers import run_clean_papers

    # Get input/output directories
    input_dir = get_run_dir(STEP_2_PAPERS, run_id, config.base_dir)
    output_dir = get_run_dir(STEP_3_PAPERS_CLEAN, run_id, config.base_dir)

    # Run step
    try:
        run_clean_papers(papers_dir=input_dir, output_dir=output_dir, telemetry=telemetry)
        telemetry.emit(MessageType.SUCCESS, f"Step 3 completed")
        return True
    except Exception as e:
        telemetry.emit(MessageType.ERROR, f"Step 3 failed: {e}")
        raise


async def run_step_4_get_triples(config, run_id, telemetry: AIITelemetry):
    """Step 4: Get Triples using agents."""
    telemetry.emit(MessageType.INFO, f"Step 4: Get Triples (run_id: {run_id})")

    # Import step module
    from .steps._4_get_triples import main as get_triples_main

    # Run step with config, run_id and telemetry (main is now async)
    exit_code = await get_triples_main(run_id, config=config.raw_config, telemetry=telemetry)

    if exit_code != 0:
        telemetry.emit(MessageType.ERROR, f"Step 4 failed with exit code {exit_code}")
        return False

    telemetry.emit(MessageType.SUCCESS, "Step 4 completed")
    return True


async def run_step_5_add_wikidata(config, run_id, telemetry: AIITelemetry):
    """Step 5: Add Wikidata (enrich triples with Wikidata)."""
    telemetry.emit(MessageType.INFO, f"Step 5: Add Wikidata (run_id: {run_id})")

    # Import step module
    from .steps._5_add_wikidata import main as add_wikidata_main

    # Run step with run_id and telemetry (main is now async)
    exit_code = await add_wikidata_main(run_id, telemetry=telemetry)

    if exit_code != 0:
        telemetry.emit(MessageType.ERROR, f"Step 5 failed with exit code {exit_code}")
        return False

    telemetry.emit(MessageType.SUCCESS, "Step 5 completed")
    return True


def run_step_6_link_to_papers(config, run_id, telemetry: AIITelemetry):
    """Step 6: Link to Papers (combine papers with enriched triples)."""
    telemetry.emit(MessageType.INFO, f"Step 6: Link to Papers (run_id: {run_id})")

    # Import step module
    from .steps._6_link_to_papers import main as link_main

    # Run step with run_id and telemetry
    exit_code = link_main(run_id, telemetry=telemetry)

    if exit_code != 0:
        telemetry.emit(MessageType.ERROR, f"Step 6 failed with exit code {exit_code}")
        return False

    telemetry.emit(MessageType.SUCCESS, "Step 6 completed")
    return True


def run_step_7_gen_hypo_seeds(config, run_id, telemetry: AIITelemetry):
    """Step 7: Generate Hypothesis Seeds (blind spots, breakthroughs)."""
    telemetry.emit(MessageType.INFO, f"Step 7: Generate Hypothesis Seeds (run_id: {run_id})")

    # Import step module
    from .steps._7_gen_hypo_seeds import main as hypo_seeds_main

    # Run step with config and telemetry
    exit_code = hypo_seeds_main(run_id, config=config.raw_config, telemetry=telemetry)

    if exit_code != 0:
        telemetry.emit(MessageType.ERROR, f"Step 7 failed with exit code {exit_code}")
        return False

    telemetry.emit(MessageType.SUCCESS, "Step 7 completed")
    return True


def run_step_8_gen_seed_prompt(config, run_id, telemetry: AIITelemetry):
    """Step 8: Generate Seed Prompts (research ideas from blind spots)."""
    telemetry.emit(MessageType.INFO, f"Step 8: Generate Seed Prompts (run_id: {run_id})")

    # Import step module
    from .steps._8_gen_seed_prompt import main as seed_prompt_main

    # Run step with telemetry
    exit_code = seed_prompt_main(run_id, telemetry=telemetry)

    if exit_code != 0:
        telemetry.emit(MessageType.ERROR, f"Step 8 failed with exit code {exit_code}")
        return False

    telemetry.emit(MessageType.SUCCESS, "Step 8 completed")
    return True


def run_step_9_gen_graphs(config, run_id, telemetry: AIITelemetry):
    """Step 9: Generate Graphs (co-occurrence, ontology, etc.)."""
    telemetry.emit(MessageType.INFO, f"Step 9: Generate Graphs (run_id: {run_id})")

    # Import step module
    from .steps._9_gen_graphs import main as gen_graphs_main

    # Run step with config and telemetry
    exit_code = gen_graphs_main(run_id, config=config.raw_config, telemetry=telemetry)

    if exit_code != 0:
        telemetry.emit(MessageType.ERROR, f"Step 9 failed with exit code {exit_code}")
        return False

    telemetry.emit(MessageType.SUCCESS, "Step 9 completed")
    return True


import asyncio
import inspect


async def run_pipeline(
    config,
    start_step: int = 1,
    end_step: int = 9,
    telemetry: Optional[AIITelemetry] = None,
) -> dict:
    """
    Run the full pipeline or subset of steps.

    Args:
        config: Configuration object
        start_step: First step to run (1-9)
        end_step: Last step to run (1-9)
        telemetry: Optional AIITelemetry instance (creates local one if not provided)

    Returns:
        Dict with 'success', 'prompts', 'output_dir', 'run_id'
    """
    import json

    # Create local telemetry if not provided
    local_telemetry = telemetry is None
    if local_telemetry:
        telemetry = create_telemetry(config.base_dir, "invention_kg")

    # Start module group for the entire invention_kg pipeline
    telemetry.start_module_group("INVENTION_KG")

    steps = {
        1: ("Select Topics", run_step_1_sel_topics),
        2: ("Get Papers", run_step_2_get_papers),
        3: ("Clean Papers", run_step_3_clean_papers),
        4: ("Get Triples", run_step_4_get_triples),
        5: ("Add Wikidata", run_step_5_add_wikidata),
        6: ("Link to Papers", run_step_6_link_to_papers),
        7: ("Gen Hypo Seeds", run_step_7_gen_hypo_seeds),
        8: ("Gen Seed Prompt", run_step_8_gen_seed_prompt),
        9: ("Generate Graphs", run_step_9_gen_graphs),
    }

    # Validate configuration
    validation_errors = validate_config(config.raw_config)
    if validation_errors:
        telemetry.emit(MessageType.ERROR, "Configuration validation failed:")
        for error in validation_errors:
            telemetry.emit(MessageType.ERROR, f"  ❌ {error}")
        telemetry.emit(MessageType.ERROR, "Please fix the errors in config.yaml and try again.")
        telemetry.emit_module_group_summary("INVENTION_KG")
        return {'success': False, 'prompts': [], 'output_dir': '', 'run_id': ''}
    telemetry.emit(MessageType.SUCCESS, "✅ Configuration validated successfully")

    # Map step -> required out_dir from previous step
    # To start from step N, you need to set the out_dir for step N-1
    step_to_prev_out_dir = {
        2: 'sel_topics_out_dir',
        3: 'get_papers_out_dir',
        4: 'clean_papers_out_dir',
        5: 'get_triples_out_dir',
        6: 'add_wikidata_out_dir',
        7: 'link_to_papers_out_dir',
        8: 'gen_hypo_seeds_out_dir',
        9: 'gen_seed_prompt_out_dir',
    }

    run_id = None

    # Check if resuming from checkpoint
    if start_step > 1:
        out_dir_key = step_to_prev_out_dir.get(start_step)
        out_dir_path = (config.get(out_dir_key) or '').strip() if out_dir_key else ''

        if not out_dir_path:
            telemetry.emit(MessageType.ERROR, f"❌ Starting from step {start_step} requires {out_dir_key} to be set")
            telemetry.emit_module_group_summary("INVENTION_KG")
            return {'success': False, 'prompts': [], 'output_dir': '', 'run_id': ''}

        # Path must be absolute
        out_dir = Path(out_dir_path)
        if not out_dir.is_absolute():
            telemetry.emit(MessageType.ERROR, f"❌ {out_dir_key} must be an absolute path: {out_dir_path}")
            telemetry.emit_module_group_summary("INVENTION_KG")
            return {'success': False, 'prompts': [], 'output_dir': '', 'run_id': ''}

        if not out_dir.exists():
            telemetry.emit(MessageType.ERROR, f"❌ {out_dir_key} does not exist: {out_dir}")
            telemetry.emit_module_group_summary("INVENTION_KG")
            return {'success': False, 'prompts': [], 'output_dir': '', 'run_id': ''}

        # Extract run_id from path: .../runs/{run_id}/1_seed_hypo/_N_step/
        # Path structure: runs/novak_hypo_seed/1_seed_hypo/_6_paper_triples
        parts = out_dir.parts
        try:
            seed_hypo_idx = parts.index('1_seed_hypo')
            run_id = parts[seed_hypo_idx - 1]
        except (ValueError, IndexError) as e:
            telemetry.emit(MessageType.ERROR, f"❌ Could not extract run_id from path: {out_dir}")
            telemetry.emit_module_group_summary("INVENTION_KG")
            raise ValueError(f"Could not extract run_id from path: {out_dir}") from e

        telemetry.emit(MessageType.INFO, f"Resuming from {out_dir_key}: {out_dir}")
        telemetry.emit(MessageType.INFO, f"Extracted run_id: {run_id}")

    # Create new run_id if starting from step 1
    if run_id is None:
        step_config = config['get_papers']
        start_year = step_config['year_range']['start']
        end_year = step_config['year_range']['end']
        papers_per_year = step_config['papers_per_year']
        num_topics = len(config['sel_topics']['topics'])

        num_years = end_year - start_year + 1
        total_requested = num_years * papers_per_year * num_topics

        run_id = create_run_id(total_requested)
        telemetry.emit(MessageType.INFO, f"Created run_id: {run_id} ({total_requested} papers requested)")

    # Determine steps to run
    steps_to_run = range(start_step, end_step + 1)
    telemetry.emit(MessageType.INFO, f"Running steps {start_step} to {end_step}")

    # Run steps - each step is a module for cost aggregation
    for step_num in steps_to_run:
        if step_num not in steps:
            telemetry.emit(MessageType.ERROR, f"Invalid step number: {step_num}")
            continue

        step_name, step_func = steps[step_num]
        module_name = f"KG_{step_num}_{step_name.upper().replace(' ', '_')}"

        # Start module for this step
        telemetry.start_module(module_name)

        try:
            # All steps need run_id and telemetry
            # Some steps are async (e.g., step 4), so await them if needed
            if inspect.iscoroutinefunction(step_func):
                result = await step_func(config, run_id, telemetry)
            else:
                result = step_func(config, run_id, telemetry)

            if result is False:
                telemetry.emit(MessageType.ERROR, f"Pipeline stopped at step {step_num} due to error")
                telemetry.emit_module_summary(module_name)
                telemetry.emit_module_group_summary("INVENTION_KG")
                return {'success': False, 'prompts': [], 'output_dir': '', 'run_id': run_id}

            telemetry.emit_module_summary(module_name)
        except Exception as e:
            telemetry.emit(MessageType.ERROR, f"Error in step {step_num} ({step_name}): {e}")
            telemetry.emit_module_summary(module_name)
            telemetry.emit_module_group_summary("INVENTION_KG")
            raise

    # Load prompts from step 8 output
    prompts = []
    output_dir = get_run_dir(STEP_8_SEED_PROMPT, run_id, config.base_dir)
    prompts_file = output_dir / "blind_spot_prompts.json"
    if prompts_file.exists():
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        telemetry.emit(MessageType.INFO, f"Loaded {len(prompts)} prompts from {prompts_file}")

    # Pipeline footer
    telemetry.emit(MessageType.SUCCESS, "INVENTION_KG pipeline completed successfully!")

    # Emit module group summary for the entire invention_kg pipeline
    telemetry.emit_module_group_summary("INVENTION_KG")

    # Flush local telemetry if we created it
    if local_telemetry:
        telemetry.flush()

    return {
        'success': True,
        'prompts': prompts,
        'output_dir': str(output_dir),
        'run_id': run_id,
    }


def main():
    """Main entry point - standalone execution not supported."""
    print("Error: Standalone execution not supported.")
    print("Run via aii_pipeline command instead.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
