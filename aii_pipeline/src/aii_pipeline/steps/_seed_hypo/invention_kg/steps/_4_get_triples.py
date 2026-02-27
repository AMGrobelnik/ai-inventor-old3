#!/usr/bin/env python3
"""
Triple extraction orchestrator with parallel processing.

This script handles:
1. Loading configuration and paper data
2. Parallel processing of papers with concurrency control
3. Progress tracking and cost/timing statistics
4. Summary reporting

The core logic for processing a single paper is in _get_triples/get_triple.py
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from aii_lib import AIITelemetry, MessageType, JSONSink


# Module-level telemetry (set by main)
_telemetry: Optional[AIITelemetry] = None


def _emit(msg_type: MessageType, msg: str):
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")

from ._get_triples import (  # type: ignore[import-not-found]
    get_triples_for_paper,
    setup_logging,
    get_completed_paper_indices,
    create_progress_tracker,
    print_summary,
    print_header,
    print_completion,
    console,
    load_pipeline_config,
    load_agent_config,
    load_papers_from_directory,
    validate_paths,
    set_telemetry as set_submodule_telemetry,
)

# ============================================================================
# Main Processing Orchestration
# ============================================================================


async def process_year(
    year: int | str,
    papers: list,
    parent_run_dir: Path,
    agent_cwd_template: Path,
    config: Dict[str, Any],
    max_concurrent: int = 10,
):
    """
    Process all papers for a single year in parallel.

    Args:
        year: Year to process (int or "all_years" for combined processing)
        papers: List of paper dicts with title and abstract (must have 'index' field)
        parent_run_dir: Parent run directory where individual paper folders will be created
        agent_cwd_template: Path to agent_cwd/ template directory
        config: Agent configuration from triples_config.yaml
        max_concurrent: Maximum number of concurrent paper processing tasks
    """
    start_time = time.time()

    print_header(len(papers), max_concurrent)

    # Semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)

    # Progress tracking file
    progress_file = parent_run_dir / "completed_papers.txt"
    progress_lock = asyncio.Lock()  # Lock only for file writes

    # Shared state with single lock to prevent race conditions
    total_cost: float = 0.0
    successful: int = 0
    failed: int = 0
    in_progress: int = 0
    state_lock = asyncio.Lock()  # Single lock for all shared state

    # Helper function to write completed paper to progress file
    async def write_progress(paper_index: int):
        """Write successfully completed paper index to progress file with lock."""
        async with progress_lock:
            with open(progress_file, 'a') as f:
                f.write(f"{paper_index}\n")

    # Create progress tracker (simple logging)
    update_task_display = create_progress_tracker()

    # Modified wrapper to use new display system
    async def process_paper_with_display(idx: int, paper: dict):
        """Process paper and update display."""
        nonlocal total_cost, successful, failed, in_progress

        paper_index = paper["index"]
        title = paper["title"]
        abstract = paper["abstract"]

        async with semaphore:
            try:
                # Update status to running and calculate stats atomically
                async with state_lock:
                    in_progress += 1
                    done = successful + failed
                    pending = len(papers) - done - in_progress

                # Debug logging handled by update_task_display

                await update_task_display(
                    f"[bold blue]Started Paper {paper_index}[/bold blue] | Pending: {pending}, In Progress: {in_progress}, Done: {done} | {title}"
                )

                paper_start = time.time()
                result = await get_triples_for_paper(
                    paper_id=idx,
                    paper_index=paper_index,
                    title=title,
                    abstract=abstract,
                    parent_run_dir=parent_run_dir,
                    agent_cwd_template=agent_cwd_template,
                    config=config,
                    telemetry=_telemetry,
                )
                paper_time = time.time() - paper_start

                # Write to progress file immediately (only successful completions, outside state_lock for minimal blocking)
                if result:
                    await write_progress(paper_index)

                # Remove from running and update stats atomically
                async with state_lock:
                    in_progress -= 1

                    if result:
                        total_cost += result["cost"]
                        successful += 1
                    else:
                        failed += 1

                    done = successful + failed
                    pending = len(papers) - done - in_progress

                # Debug logging handled by update_task_display

                if result:
                    await update_task_display(
                        f"[bold green]✓ Done Paper {paper_index}[/bold green] | Pending: {pending}, In Progress: {in_progress}, Done: {done} | {title} [yellow]${result['cost']:.3f}[/yellow] [cyan]{paper_time:.0f}s[/cyan]"
                    )
                else:
                    await update_task_display(
                        f"[bold red]✗ Failed Paper {paper_index}[/bold red] | Pending: {pending}, In Progress: {in_progress}, Done: {done} | {title}"
                    )

                return result

            except Exception as e:
                # Log full exception
                _emit(MessageType.ERROR,
                    f"Error processing paper {paper_index} ('{title}'): {type(e).__name__}: {str(e)}"
                )

                # Update state atomically
                async with state_lock:
                    in_progress -= 1
                    failed += 1
                    done = successful + failed
                    pending = len(papers) - done - in_progress

                await update_task_display(
                    f"[bold red]✗ Error Paper {paper_index}[/bold red] | Pending: {pending}, In Progress: {in_progress}, Done: {done} | {title} ({type(e).__name__})"
                )
                return None

    # Launch all tasks concurrently
    tasks = [process_paper_with_display(idx, paper) for idx, paper in enumerate(papers)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Allow time for subprocess cleanup before event loop closes
    # Multiple subprocesses need time to terminate gracefully
    await asyncio.sleep(3)

    # Calculate total time
    total_time = time.time() - start_time

    # Summary
    print_summary(
        successful, failed, len(papers), total_cost, total_time, "Processing Summary"
    )

    return total_time, successful, failed, total_cost


async def main(run_id: str, config: Dict[str, Any] = None, telemetry: Optional[AIITelemetry] = None):
    """
    Process papers from all years.

    Args:
        run_id: Run ID for pipeline orchestration mode.
        config: Config dict from pipeline (required). Must include 'get_triples' section
                with 'claude_agent' nested config.
        telemetry: Optional AIITelemetry instance.
    """
    global _telemetry
    _telemetry = telemetry
    # Set telemetry for all submodules
    set_submodule_telemetry(telemetry)

    from aii_pipeline.steps._seed_hypo.invention_kg.utils import get_run_dir
    from aii_pipeline.steps._seed_hypo.invention_kg.constants import STEP_3_PAPERS_CLEAN, STEP_4_TRIPLES, RUNS_DIR

    # Paths - module_dir for config files, RUNS_DIR for run output
    module_dir = Path(__file__).parent.parent  # invention_kg/

    # Config must be provided (passed from pipeline)
    if config is None:
        _emit(MessageType.ERROR, "Config not provided. Run via aii_pipeline, not standalone.")
        return 1

    # Get settings from config (passed from main pipeline config.yaml)
    step_config = config.get("get_triples", {})
    max_papers = step_config.get("max_papers", -1)
    max_concurrent = step_config.get("max_concurrent_agents", 10)

    # Agent config is now embedded in step_config (from config.yaml get_triples.claude_agent)
    agent_config = step_config  # Pass the full step_config which includes claude_agent

    # Get agent_cwd template path
    agent_cwd_template = module_dir / "agent_cwd"

    # Setup comprehensive logging to logs/ directory
    log_file = setup_logging(module_dir, resume_dir=None)

    _emit(MessageType.INFO, f"Starting triple extraction")
    _emit(MessageType.INFO, f"max_papers: {max_papers}, max_concurrent: {max_concurrent}")

    # Determine input/output directories using run_id (uses RUNS_DIR for pipeline runs)
    _emit(MessageType.INFO, f"Run ID: {run_id}")

    papers_clean_dir = get_run_dir(STEP_3_PAPERS_CLEAN, run_id)
    parent_run_dir = get_run_dir(STEP_4_TRIPLES, run_id)
    is_resume = parent_run_dir.exists()

    if is_resume:
        _emit(MessageType.INFO, f"Resuming step 4 from: {parent_run_dir}")
    else:
        _emit(MessageType.INFO, f"Creating step 4 output: {parent_run_dir}")

    parent_run_dir.mkdir(parents=True, exist_ok=True)

    # Start module for telemetry aggregation (like other pipeline steps)
    # Add JSONSink AFTER parent_run_dir is created
    module_sinks = []
    if telemetry:
        s1 = JSONSink(parent_run_dir / "get_triples_pipeline_messages.jsonl")
        s2 = JSONSink(parent_run_dir / "get_triples_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])
        telemetry.start_module("GET_TRIPLES")

    # Check agent_cwd template exists (optional - agent will create if needed)
    if not agent_cwd_template.exists():
        _emit(MessageType.WARNING, f"Agent CWD template not found: {agent_cwd_template}")
        _emit(MessageType.INFO, "Agent will create workspace directory as needed")

    _emit(MessageType.INFO, f"Input directory: {papers_clean_dir}")
    _emit(MessageType.INFO, f"Agent CWD template: {agent_cwd_template}")
    _emit(MessageType.INFO, f"Output directory: {parent_run_dir}")

    # Load papers from directory (uses config.py module)
    try:
        all_papers, papers_to_process, num_year_files = load_papers_from_directory(
            papers_clean_dir, max_papers
        )
    except FileNotFoundError as e:
        _emit(MessageType.ERROR, str(e))
        return 1

    # Create limit message for display
    if max_papers == -1:
        limit_msg = f"Processing all {len(papers_to_process)} papers"
    else:
        limit_msg = f"Processing {len(papers_to_process)} papers (max_papers={max_papers}, total available: {len(all_papers)})"

    # Check for already completed papers (dynamic resume) - only if resume mode is enabled
    if is_resume:
        # Validation check includes its own detailed logging and folder cleanup
        completed_indices = get_completed_paper_indices(parent_run_dir, agent_cwd_template)

        if completed_indices:
            _emit(MessageType.INFO, f"Completed paper indices: {sorted(list(completed_indices))[:10]}..." if len(completed_indices) > 10 else f"Completed paper indices: {sorted(list(completed_indices))}")

            # Filter out completed papers
            papers_before = len(papers_to_process)
            papers_to_process = [p for p in papers_to_process if p["index"] not in completed_indices]
            papers_skipped = papers_before - len(papers_to_process)

            _emit(MessageType.INFO, f"Skipping {papers_skipped} already completed papers")
            _emit(MessageType.INFO, f"Will process {len(papers_to_process)} remaining papers")
        else:
            _emit(MessageType.INFO, f"No completed papers found, will process all {len(papers_to_process)} papers")
    else:
        _emit(MessageType.INFO, f"Starting fresh - will process all {len(papers_to_process)} papers")

    _emit(MessageType.INFO, f"{limit_msg}")
    _emit(MessageType.INFO, f"Max concurrent: {max_concurrent}, Year files: {num_year_files}")

    # Process all papers at once
    overall_start_time = time.time()
    total_time, successful, failed, total_cost = await process_year(
        "all_years",
        papers_to_process,
        parent_run_dir,
        agent_cwd_template,
        agent_config,
        max_concurrent=max_concurrent,
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    overall_time = time.time() - overall_start_time

    # Count all folders in directory for reporting
    total_folders = len(list(parent_run_dir.glob("paper_*")))

    print_summary(
        successful,
        failed,
        total_folders,
        total_cost,
        overall_time,
        "Final Summary",
    )

    print_completion()
    _emit(MessageType.INFO,
        f"Final results: {successful} successful, {failed} failed out of {total_folders} total papers"
    )

    # Print helpful message for next steps
    console.print(f"\n[bold cyan]Output saved to:[/bold cyan]")
    console.print(f"  {parent_run_dir}")

    # Emit module summary for telemetry aggregation
    if telemetry:
        telemetry.emit_module_summary("GET_TRIPLES")

    # Remove module sinks to prevent leak into subsequent modules
    for sink in module_sinks:
        sink.flush()
        telemetry.remove_sink(sink)

    return 0
