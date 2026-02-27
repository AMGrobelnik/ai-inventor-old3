"""Standardized module output builder for ALL pipeline modules.

Every module emits a MODULE_OUTPUT with the same 3-key structure:

    {
        "outputs": [...],       # This module's new outputs (full model dumps)
        "metadata": {...},      # This module's run metadata
        "cumulative": {         # Cumulative snapshot of everything so far
            "seed_hypo": [...],       # Pre-loop modules at top level
            "gen_hypo": [...],
            "hypothesis": {...},
            "invention_loop": {       # All invention loop data nested here
                "iterations": {
                    "1": {
                        "gen_strat": [...],
                        "gen_plan": [...],
                        "gen_art": [...],
                        "gen_narr": [...]
                    },
                    "2": { ... }
                },
                "result": [...]       # Final invention_loop summary
            },
            "gen_paper_repo": [...]   # Post-loop modules at top level
        }
    }

Usage:
    # Pipeline creates cumulative dict once:
    cumulative = init_cumulative()

    # Pre-loop modules (no iteration):
    module_output = build_module_output(
        module="seed_hypo",
        outputs=[seed_data],
        cumulative=cumulative,
    )

    # Invention loop modules (with iteration):
    module_output = build_module_output(
        module="gen_strat",
        outputs=strategies,
        cumulative=cumulative,
        iteration=1,
        total_cost_usd=5.07,
    )
    emit_module_output(module_output, telemetry, output_dir=iter_dir)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aii_lib import AIITelemetry
    from pydantic import BaseModel


def init_cumulative(hypothesis: dict | None = None) -> dict:
    """Initialize the cumulative state dict passed through all modules.

    Args:
        hypothesis: Optional research hypothesis dict. Can be set later
            via cumulative["hypothesis"] = hypothesis.

    Returns:
        Mutable dict that modules will add their outputs to.
    """
    d: dict = {}
    if hypothesis:
        d["hypothesis"] = hypothesis
    return d


def build_module_output(
    module: str,
    outputs: list,
    cumulative: dict,
    iteration: int | None = None,
    total_cost_usd: float = 0.0,
    output_dir: Path | None = None,
    llm_provider: str = "",
    **extra_metadata,
) -> dict:
    """Build standardized module output with 3 keys: outputs, metadata, cumulative.

    Dumps each output item via model_dump(mode="json") if it's a Pydantic model,
    otherwise includes it as-is. Mutates `cumulative` to add this module's outputs.

    For iteration modules (gen_strat, gen_plan, gen_art, gen_narr):
        Stored under cumulative["invention_loop"]["iterations"][iteration][module]

    For the invention_loop module (final summary):
        Stored under cumulative["invention_loop"]["result"] (merges with iterations)

    For non-iteration modules (seed_hypo, gen_hypo, gen_paper_repo):
        Stored at cumulative[module] (top level)

    Args:
        module: Module name (e.g., "gen_strat", "seed_hypo", "gen_paper_repo").
        outputs: List of Pydantic models or dicts produced by this module.
        cumulative: Mutable cumulative state (from init_cumulative).
        iteration: Current iteration number, or None for non-iteration modules.
        total_cost_usd: Total cost for this module run.
        output_dir: Output directory path (for metadata).
        llm_provider: LLM provider name (for metadata).
        **extra_metadata: Any additional metadata fields.

    Returns:
        Dict with keys: outputs, metadata, cumulative.
    """
    # Dump outputs
    output_dicts = [
        o.model_dump(mode="json") if hasattr(o, "model_dump") else o
        for o in outputs
    ]

    # Add to cumulative state
    if iteration is not None:
        # Iteration module: store under invention_loop/iterations/{N}/{module}
        if "invention_loop" not in cumulative:
            cumulative["invention_loop"] = {"iterations": {}}
        if "iterations" not in cumulative["invention_loop"]:
            cumulative["invention_loop"]["iterations"] = {}
        iter_key = str(iteration)
        if iter_key not in cumulative["invention_loop"]["iterations"]:
            cumulative["invention_loop"]["iterations"][iter_key] = {}
        cumulative["invention_loop"]["iterations"][iter_key][module] = output_dicts
    elif module == "invention_loop":
        # Invention loop final result: merge into existing dict (preserves iterations)
        if "invention_loop" not in cumulative:
            cumulative["invention_loop"] = {}
        cumulative["invention_loop"]["result"] = output_dicts
    else:
        # Non-iteration module: store at top level
        cumulative[module] = output_dicts

    # Build metadata
    metadata = {
        "module": module,
        "iteration": iteration,
        "total_cost_usd": total_cost_usd,
        "count": len(output_dicts),
        "generated_at": datetime.now().isoformat(),
        "output_dir": str(output_dir) if output_dir else None,
        "llm_provider": llm_provider,
        **extra_metadata,
    }

    return {
        "outputs": output_dicts,
        "metadata": metadata,
        "cumulative": cumulative,
    }


def emit_module_output(
    module_output: dict,
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
) -> None:
    """Emit MODULE_OUTPUT via telemetry and save to file.

    Args:
        module_output: Dict from build_module_output().
        telemetry: Telemetry instance.
        output_dir: Directory to save {module}_output.json. If None, skips file save.
    """
    from aii_lib import MessageType
    from aii_pipeline.utils import rel_path

    telemetry.emit(
        MessageType.MODULE_OUTPUT,
        json.dumps(module_output, indent=2, ensure_ascii=False, default=str),
    )

    if output_dir:
        module = module_output["metadata"]["module"]
        output_file = output_dir / f"{module}_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(module_output, f, indent=2, ensure_ascii=False, default=str)
        telemetry.emit(MessageType.INFO, f"   Results saved to: {rel_path(output_file)}")
