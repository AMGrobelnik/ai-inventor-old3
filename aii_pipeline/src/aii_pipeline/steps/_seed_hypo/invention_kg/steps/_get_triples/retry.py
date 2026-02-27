#!/usr/bin/env python3
"""
Retry logic for failed building blocks extractions.

Handles retrying failed paper processing runs.
"""

import asyncio
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional

from aii_lib import AIITelemetry, MessageType

from .get_triple import get_triples_for_paper
from .display import create_progress_tracker, print_header, print_summary, console


# Module-level telemetry (set by caller)
_telemetry: Optional[AIITelemetry] = None


def set_telemetry(telemetry: Optional[AIITelemetry]) -> None:
    """Set module-level telemetry instance."""
    global _telemetry
    _telemetry = telemetry


def _emit(msg_type: MessageType, msg: str) -> None:
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")


async def retry_failed_runs(
    failed_runs: list[dict],
    parent_run_dir: Path,
    agent_cwd_template: Path,
    config: Dict[str, Any],
    max_concurrent: int = 10,
) -> tuple[int, int, float]:
    """
    Retry processing failed runs after deleting their folders.

    Args:
        failed_runs: List of failed run dicts from validate_all_runs
        parent_run_dir: Parent run directory
        agent_cwd_template: Path to agent_cwd/ template directory
        config: Agent configuration
        max_concurrent: Maximum number of concurrent paper processing tasks

    Returns:
        (successful, failed, total_cost)
    """
    if not failed_runs:
        return 0, 0, 0.0

    console.print(f"\n[bold magenta]{'='*80}[/bold magenta]")
    console.print(f"[bold]Retrying {len(failed_runs)} Failed Runs[/bold]")
    console.print(f"[bold magenta]{'='*80}[/bold magenta]\n")

    # Delete failed run folders
    _emit(MessageType.INFO, f"Deleting {len(failed_runs)} failed run folders...")
    for failed_run in failed_runs:
        run_dir = failed_run["run_dir"]
        try:
            shutil.rmtree(run_dir)
            _emit(MessageType.DEBUG, f"Deleted folder: {run_dir}")
        except Exception as e:
            _emit(MessageType.ERROR, f"Failed to delete {run_dir}: {e}")

    # Extract papers to retry
    papers_to_retry = [failed_run["paper"] for failed_run in failed_runs]

    # Process using the same mechanism as process_year
    start_time = time.time()

    print_header(len(papers_to_retry), max_concurrent)

    # Semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)

    # Progress tracking file (same as main processing)
    progress_file = parent_run_dir / "completed_papers.txt"
    progress_lock = asyncio.Lock()

    # Helper function to write completed paper to progress file
    async def write_progress(paper_index: int):
        """Write successfully completed paper index to progress file with lock."""
        async with progress_lock:
            with open(progress_file, 'a') as f:
                f.write(f"{paper_index}\n")

    # Shared state with single lock
    total_cost: float = 0.0
    successful: int = 0
    failed: int = 0
    in_progress: int = 0
    state_lock = asyncio.Lock()

    # Create progress tracker
    update_task_display = create_progress_tracker()

    # Modified wrapper for retry
    async def process_paper_with_display(idx: int, paper: dict):
        """Process paper and update display."""
        nonlocal total_cost, successful, failed, in_progress

        paper_index = paper["index"]
        title = paper["title"]
        abstract = paper["abstract"]

        async with semaphore:
            try:
                # Update status to running
                async with state_lock:
                    in_progress += 1
                    done = successful + failed
                    pending = len(papers_to_retry) - done - in_progress

                _emit(MessageType.DEBUG, f"Retrying paper {paper_index} | {in_progress} concurrent")

                await update_task_display(
                    f"[bold blue]Retry Paper {paper_index}[/bold blue] | Pending: {pending}, In Progress: {in_progress}, Done: {done} | {title}"
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

                # Write to progress file immediately (only successful completions)
                if result:
                    await write_progress(paper_index)

                # Update stats
                async with state_lock:
                    in_progress -= 1

                    if result:
                        total_cost += result["cost"]
                        successful += 1
                    else:
                        failed += 1

                    done = successful + failed
                    pending = len(papers_to_retry) - done - in_progress

                _emit(MessageType.DEBUG, f"Finished retry {paper_index} | {in_progress} concurrent")

                if result:
                    await update_task_display(
                        f"[bold green]✓ Retry Success {paper_index}[/bold green] | Pending: {pending}, In Progress: {in_progress}, Done: {done} | {title} [yellow]${result['cost']:.3f}[/yellow] [cyan]{paper_time:.0f}s[/cyan]"
                    )
                else:
                    await update_task_display(
                        f"[bold red]✗ Retry Failed {paper_index}[/bold red] | Pending: {pending}, In Progress: {in_progress}, Done: {done} | {title}"
                    )

                return result

            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                _emit(MessageType.ERROR,
                    f"Error retrying paper {paper_index} ('{title}'): {type(e).__name__}: {str(e)}\n{tb_str}"
                )

                async with state_lock:
                    in_progress -= 1
                    failed += 1
                    done = successful + failed
                    pending = len(papers_to_retry) - done - in_progress

                await update_task_display(
                    f"[bold red]✗ Retry Error {paper_index}[/bold red] | Pending: {pending}, In Progress: {in_progress}, Done: {done} | {title} ({type(e).__name__})"
                )
                return None

    # Launch all retry tasks
    tasks = [
        process_paper_with_display(idx, paper)
        for idx, paper in enumerate(papers_to_retry)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Allow time for subprocess cleanup
    await asyncio.sleep(3)

    # Calculate total time
    total_time = time.time() - start_time

    # Summary
    print_summary(
        successful,
        failed,
        len(papers_to_retry),
        total_cost,
        total_time,
        "Retry Summary",
    )

    return successful, failed, total_cost
