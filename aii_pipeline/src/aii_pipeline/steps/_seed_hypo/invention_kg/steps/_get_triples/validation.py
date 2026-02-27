#!/usr/bin/env python3
"""
Validation logic for triples extraction.

Validates paper processing results and checks output JSON files.
"""

import json
from pathlib import Path
from typing import Optional

from aii_lib import AIITelemetry, MessageType

from .resume import load_validation_module
from .display import console


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


def validate_all_runs(
    parent_run_dir: Path,
    papers: list,
    agent_cwd_template: Path,
) -> tuple[list[dict], list[dict]]:
    """
    Validate all paper run outputs using the validation script.

    Args:
        parent_run_dir: Parent run directory containing all paper folders
        papers: List of papers that were processed (with 'index' field)
        agent_cwd_template: Path to agent_cwd/ template directory

    Returns:
        (valid_runs, failed_runs) where each is a list of dicts with:
        {
            "paper_index": int,
            "paper": dict,
            "run_dir": Path,
            "error": str (only for failed_runs)
        }
    """
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold]Final Validation Check[/bold]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")

    # Load validation module
    validator_module = load_validation_module(agent_cwd_template)

    valid_runs = []
    failed_runs = []

    # Find all run directories (paper_XXXXX format)
    run_dirs = sorted(parent_run_dir.glob("paper_*"))

    _emit(MessageType.INFO, f"Found {len(run_dirs)} run directories to validate")

    for run_dir in run_dirs:
        # Extract paper index from folder name (format: paper_{index})
        try:
            # Handle both paper_XXXXX and paper_idxXXXXX formats
            name = run_dir.name
            paper_index = int(name.replace("paper_idx", "").replace("paper_", ""))
        except ValueError:
            _emit(MessageType.ERROR, f"Cannot parse paper index from folder name: {run_dir.name}")
            continue

        # Find corresponding paper
        paper = next((p for p in papers if p["index"] == paper_index), None)
        if not paper:
            _emit(MessageType.ERROR, f"Paper index {paper_index} not found in original papers list")
            continue

        # Check if triples_output.json exists
        output_file = run_dir / "agent_cwd" / "triples_output.json"

        if not output_file.exists():
            failed_runs.append(
                {
                    "paper_index": paper_index,
                    "paper": paper,
                    "run_dir": run_dir,
                    "error": "triples_output.json not found",
                }
            )
            _emit(MessageType.WARNING, f"Paper {paper_index}: Missing output file")
            continue

        # Validate using the validation module
        # Load JSON to get triples count and paper_type for progress message
        try:
            with open(output_file) as f:
                data = json.load(f)
            num_triples = len(data.get("triples", []))
            paper_type = data.get("paper_type", "UNKNOWN")
            console.print(
                f"  [dim]Validating Paper {paper_index}: {paper_type}, {num_triples} triples...[/dim]"
            )
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            pass  # Will be caught by validation

        is_valid, errors = validator_module.validate_analysis(
            output_file, verify_urls=False
        )

        if is_valid:
            valid_runs.append(
                {
                    "paper_index": paper_index,
                    "paper": paper,
                    "run_dir": run_dir,
                }
            )
            _emit(MessageType.DEBUG, f"Paper {paper_index}: Valid ✓")
        else:
            error_summary = f"{len(errors)} validation errors: " + "; ".join(errors[:3])
            if len(errors) > 3:
                error_summary += f"... and {len(errors) - 3} more"

            failed_runs.append(
                {
                    "paper_index": paper_index,
                    "paper": paper,
                    "run_dir": run_dir,
                    "error": error_summary,
                    "full_errors": errors,
                }
            )
            _emit(MessageType.WARNING, f"Paper {paper_index}: Invalid - {error_summary}")

    # Print summary
    console.print(
        f"\n[bold green]Valid runs:[/bold green] {len(valid_runs)}/{len(run_dirs)}"
    )
    console.print(
        f"[bold red]Failed runs:[/bold red] {len(failed_runs)}/{len(run_dirs)}"
    )

    if failed_runs:
        console.print(f"\n[bold yellow]Failed Runs Details:[/bold yellow]")
        for failed in failed_runs:
            console.print(
                f"  [red]✗[/red] Paper {failed['paper_index']}: {failed['paper']['title'][:60]}"
            )
            console.print(f"    [yellow]Error:[/yellow] {failed['error']}")
            console.print(f"    [dim]Folder:[/dim] {failed['run_dir'].name}\n")

    return valid_runs, failed_runs
