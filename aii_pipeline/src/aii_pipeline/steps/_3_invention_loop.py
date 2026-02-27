#!/usr/bin/env python3
"""
Invention Loop Module - Iterative Scientific Invention

Implements the four-step invention loop:
1. GEN_STRAT  (module) - Generate strategic plans from multiple LLMs
2. GEN_PLAN   (module) - Generate plans from strategy's artifact directions
3. GEN_ART    (MODULE GROUP) - Execute selected plans to generate artifacts
   └── RESEARCH, DATASET, EXPERIMENT, EVALUATION, PROOF (sub-modules)
4. GEN_NARR   (module) - Generate narratives from artifact pool

Telemetry hierarchy:
- INVENTION_LOOP (module group)
  └── iter_1 (module group per iteration)
      └── GEN_STRAT, GEN_PLAN, GEN_ART, GEN_NARR

The loop continues until:
- Max iterations reached
- Human signals exit

Three pools track state:
- PlanPool: Pending work
- ArtifactPool: Completed work (successes and failures)
- NarrativePool: Research stories

Uses aii_lib for:
- AIITelemetry: Centralized task tracking and logging
"""

import asyncio
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from aii_lib import create_telemetry, AIITelemetry, MessageType

from aii_pipeline.utils import PipelineConfig, rel_path, init_cumulative
from aii_pipeline.prompts.steps._3_invention_loop.schema import InventionLoopOut
from ._invention_loop.pools import StrategyPool, PlanPool, ArtifactPool, NarrativePool, save_all_pools, load_all_pools

# Import loop steps from flat structure
from ._3_1_gen_strat import run_gen_strat_module
from ._3_2_gen_plan import run_gen_plan_module
from ._3_3_gen_art import run_gen_art_module
from ._3_4_gen_narr import run_gen_narr_module


# Valid step names for invention_loop_last_step config (order matches execution)
# GEN_ART is a MODULE GROUP containing sub-modules for each artifact type (RESEARCH,
# DATASET, EXPERIMENT, EVALUATION, PROOF). When used as end_step, execution stops
# after all artifacts complete (you cannot stop mid-execution of parallel artifacts).
LOOP_STEPS = ["gen_strat", "gen_plan", "gen_art", "gen_narr"]


def detect_last_iteration(resume_dir: Path) -> tuple[int, bool, str | None]:
    """
    Detect the last iteration from a resume directory and whether it's complete.

    Looks for iter_N directories and returns (highest N, is_complete, last_step).
    An iteration is complete if it has gen_narr_output.json (the final step).
    Returns (0, False, None) if no iterations found.
    """
    last_iter = 0
    for item in resume_dir.iterdir():
        if item.is_dir() and item.name.startswith("iter_"):
            match = re.match(r"iter_(\d+)", item.name)
            if match:
                iter_num = int(match.group(1))
                last_iter = max(last_iter, iter_num)

    if last_iter == 0:
        return 0, False, None

    # Check which steps completed in this iteration
    last_iter_dir = resume_dir / f"iter_{last_iter}"
    last_step = detect_last_step(last_iter_dir)
    is_complete = last_step == "gen_narr"

    return last_iter, is_complete, last_step


def detect_last_step(iter_dir: Path) -> str | None:
    """
    Detect the last completed step within an iteration directory.

    Returns the step name (gen_strat, gen_plan, etc.) or None if no steps completed.
    """
    # Map output files to step names (in execution order)
    step_outputs = [
        ("gen_strat_output.json", "gen_strat"),
        ("gen_plan_output.json", "gen_plan"),
        ("gen_art_output.json", "gen_art"),
        ("gen_narr_output.json", "gen_narr"),
    ]

    last_step = None
    for output_file, step_name in step_outputs:
        if (iter_dir / output_file).exists():
            last_step = step_name

    return last_step


def copy_iter_dir_from_resume(resume_dir: Path, output_dir: Path, iteration: int) -> bool:
    """
    Copy an iteration directory from resume_dir to output_dir.

    This preserves output files from already-completed steps when resuming.
    Returns True if copied, False if source didn't exist.
    """
    src = resume_dir / f"iter_{iteration}"
    dst = output_dir / f"iter_{iteration}"

    if src.exists() and src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return True
    return False


async def run_invention_loop_module(
    config: PipelineConfig,
    hypothesis: dict,
    run_dir: Path | None = None,
    workspace_dir: Path | None = None,
    telemetry: AIITelemetry | None = None,
    cumulative: dict | None = None,
):
    """
    Run the full invention loop.

    The loop: GEN_STRAT → GEN_PLAN → GEN_ART → GEN_NARR → loop

    Args:
        config: Pipeline configuration
        hypothesis: The selected hypothesis to investigate
        run_dir: Run output directory
        workspace_dir: Workspace directory for Claude agents

    Returns:
        Result dictionary with all pools, stats, and metadata
    """
    # Create output directory
    if run_dir:
        output_dir = run_dir / "3_invention_loop"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = config.init.outputs_directory
        output_dir = Path(f"{output_base}/{timestamp}_invention_loop")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up workspace
    if not workspace_dir:
        workspace_dir = output_dir / "model_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Get invention loop config
    invention_cfg = config.invention_loop
    max_iterations = invention_cfg.max_iterations
    # patience = invention_cfg.early_stopping.patience  # Convergence check disabled
    start_narrative_at = invention_cfg.narrative.start_at_iteration

    # Iteration control: resume from specific iteration and step (from init.pipeline)
    pipeline_cfg = config.init.pipeline
    resume_dir = Path(pipeline_cfg.invention_loop_resume_dir) if pipeline_cfg.invention_loop_resume_dir else None
    first_step_cfg = pipeline_cfg.invention_loop_first_step  # "auto", step name, or None
    end_step = pipeline_cfg.invention_loop_last_step

    # Handle resume from previous run
    resumed_from = None
    resume_start_step = None  # Step to start from in first iteration (None = gen_strat)
    resume_pools_dir = None  # Path to pools/ dir from which to load state
    if resume_dir and resume_dir.exists():
        # Auto-detect: find last iteration, completion status, and last step
        last_iter, is_complete, last_step = detect_last_iteration(resume_dir)

        # Load pools from top-level pools/ directory
        candidate = resume_dir / "pools"
        if candidate.exists():
            resume_pools_dir = candidate
        if last_iter == 0:
            start_iteration = 1
        elif is_complete:
            # Last iteration finished all steps, start fresh at next
            start_iteration = last_iter + 1
        else:
            # Last iteration is incomplete, resume at that iteration
            start_iteration = last_iter
            # Copy the incomplete iter directory to preserve existing outputs
            copy_iter_dir_from_resume(resume_dir, output_dir, last_iter)

        # Determine which step to start from
        if first_step_cfg is None or first_step_cfg == "auto":
            # Auto-detect: find step AFTER last completed step
            if not is_complete and last_step and last_step in LOOP_STEPS:
                step_idx = LOOP_STEPS.index(last_step)
                if step_idx + 1 < len(LOOP_STEPS):
                    resume_start_step = LOOP_STEPS[step_idx + 1]
        elif first_step_cfg in LOOP_STEPS:
            # Explicit step specified - use it
            resume_start_step = first_step_cfg
        else:
            raise ValueError(f"Invalid first_step: {first_step_cfg}. Must be 'auto' or one of: {LOOP_STEPS}")

        resumed_from = {
            "dir": str(resume_dir),
            "pools_dir": str(resume_pools_dir) if resume_pools_dir else None,
            "last_iter": last_iter,
            "last_iter_complete": is_complete,
            "last_step": last_step,
            "resuming_at": start_iteration,
            "resume_start_step": resume_start_step,
            "first_step_mode": first_step_cfg or "auto",
        }
    else:
        # No resume - start at 1
        start_iteration = 1

    # Validate end_step
    if end_step and end_step not in LOOP_STEPS:
        raise ValueError(f"Invalid end_step: {end_step}. Must be one of: {LOOP_STEPS}")

    # Helper to check if we should stop after a step
    def should_stop_at_step(current_step: str) -> bool:
        """Check if we should stop after completing this step."""
        return end_step and current_step == end_step

    # Helper to check if we should skip a step (already completed in resumed iteration)
    def should_skip_step(current_iteration: int, current_step: str) -> bool:
        """Check if we should skip this step (already completed in resume)."""
        # Only skip in the first iteration when resuming
        if current_iteration != start_iteration:
            return False
        if not resume_start_step:
            return False
        # Skip all steps before resume_start_step
        try:
            step_idx = LOOP_STEPS.index(current_step)
            resume_idx = LOOP_STEPS.index(resume_start_step)
        except ValueError:
            raise ValueError(f"Invalid loop step name, valid steps: {LOOP_STEPS}")
        return step_idx < resume_idx

    # Use provided telemetry or create local one
    local_telemetry = telemetry is None
    if local_telemetry:
        telemetry = create_telemetry(output_dir, "invention_loop")

    try:
        # Extract model info for logging
        gen_plan_cfg = invention_cfg.gen_plan
        gen_plan_models = gen_plan_cfg.llm_client.models
        gen_plan_model_names = [m.model for m in gen_plan_models]

        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, "INVENTION LOOP - Iterative Scientific Invention")
        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, f"   Hypothesis: {hypothesis.get('title', 'N/A')}")
        telemetry.emit(MessageType.INFO, f"   Max iterations: {max_iterations}" + (f" (stop at {end_step})" if end_step else ""))
        if resumed_from:
            telemetry.emit(MessageType.INFO, f"   Resuming from: {rel_path(resumed_from['dir'])}")
            if resume_pools_dir:
                telemetry.emit(MessageType.INFO, f"   Pools dir: {rel_path(resume_pools_dir)}")
            telemetry.emit(MessageType.INFO, f"   First step mode: {resumed_from.get('first_step_mode', 'auto')}")
            if resume_start_step:
                telemetry.emit(MessageType.INFO, f"   Starting at step: {resume_start_step} (skipping prior steps)")
        telemetry.emit(MessageType.INFO, f"   Start narratives at: {start_narrative_at}")
        # telemetry.emit(MessageType.INFO, f"   Convergence patience: {patience}")  # Convergence check disabled
        telemetry.emit(MessageType.INFO, f"   Gen prop models: {gen_plan_model_names}")
        telemetry.emit(MessageType.INFO, f"   Output: {rel_path(output_dir)}")
        telemetry.emit(MessageType.INFO, "=" * 60)

        # Start module group (INVENTION_LOOP is only a group, not a module - modules are inside iterations)
        telemetry.start_module_group("INVENTION_LOOP")

        # Initialize the four pools (in-memory, no output_dir)
        if resume_pools_dir:
            strategy_pool, plan_pool, artifact_pool, narrative_pool = load_all_pools(resume_pools_dir)
            telemetry.emit(MessageType.INFO, f"Loaded pools: {len(strategy_pool.get_all())} strategies, {len(plan_pool.get_all())} plans, {len(artifact_pool.get_all())} artifacts, {len(narrative_pool.get_all())} narratives")
        else:
            strategy_pool = StrategyPool()
            plan_pool = PlanPool()
            artifact_pool = ArtifactPool()
            narrative_pool = NarrativePool()

        # Initialize cumulative state for module outputs (use provided or create new)
        if cumulative is None:
            cumulative = init_cumulative()
        cumulative["hypothesis"] = hypothesis

        # Track completed iterations
        iterations_completed = 0

        # Main loop
        for iteration in range(start_iteration, max_iterations + 1):
            telemetry.emit(MessageType.INFO, "")
            telemetry.emit(MessageType.INFO, f"{'=' * 60}")
            telemetry.emit(MessageType.INFO, f"ITERATION {iteration}/{max_iterations}")
            if resume_start_step and iteration == start_iteration:
                telemetry.emit(MessageType.INFO, f"   (resuming from {resume_start_step})")
            telemetry.emit(MessageType.INFO, f"{'=' * 60}")

            # Start iteration module group for nested aggregation
            iter_group_name = f"iter_{iteration}"
            telemetry.start_module_group(iter_group_name)

            # Create iteration output directory
            iter_dir = output_dir / f"iter_{iteration}"
            iter_dir.mkdir(exist_ok=True)

            # =================================================================
            # STEP 1: GEN_STRAT - Generate strategic plans
            # =================================================================
            strategies = []
            if should_skip_step(iteration, "gen_strat"):
                telemetry.emit(MessageType.INFO, "\nSTEP 1: GEN_STRAT [SKIPPED - already completed]")
            else:
                telemetry.emit(MessageType.INFO, "\nSTEP 1: GEN_STRAT")
                # Get previous iteration's strategies for continuity
                prev_strats = [s.model_dump() for s in strategy_pool.get_by_iteration(iteration - 1)]

                strategies = await run_gen_strat_module(
                    config=config,
                    hypothesis=hypothesis,
                    artifact_pool=artifact_pool,
                    narrative_pool=narrative_pool,
                    iteration=iteration,
                    telemetry=telemetry,
                    output_dir=iter_dir,
                    previous_strategies=prev_strats or None,
                    cumulative=cumulative,
                )
                telemetry.emit(MessageType.INFO, f"   Generated {len(strategies)} strategies")

                if not strategies:
                    telemetry.emit(MessageType.WARNING, "No strategies generated, ending loop")
                    telemetry.emit_module_group_summary(iter_group_name)
                    break

                # Add strategies to the pool
                strategy_pool.add_many(strategies)

                # Check if we should stop at this module
                if should_stop_at_step("gen_strat"):
                    telemetry.emit(MessageType.INFO, f"\nSTOPPING - Reached end_step=gen_strat at iteration {iteration}")
                    telemetry.emit_module_group_summary(iter_group_name)
                    break

            # =================================================================
            # STEP 2: GEN_PLAN - Generate plans from ALL strategies
            # =================================================================
            plans = []
            if should_skip_step(iteration, "gen_plan"):
                telemetry.emit(MessageType.INFO, "\nSTEP 2: GEN_PLAN [SKIPPED - already completed]")
            else:
                telemetry.emit(MessageType.INFO, "\nSTEP 2: GEN_PLAN")
                plans = await run_gen_plan_module(
                    config=config,
                    hypothesis=hypothesis,
                    plan_pool=plan_pool,
                    artifact_pool=artifact_pool,
                    iteration=iteration,
                    telemetry=telemetry,
                    output_dir=iter_dir,
                    strategies=strategies,
                    cumulative=cumulative,
                )
                telemetry.emit(MessageType.INFO, f"   Generated {len(plans)} plans")

                if not plans:
                    telemetry.emit(MessageType.WARNING, "No plans generated — retrying gen_plan once...")
                    plans = await run_gen_plan_module(
                        config=config,
                        hypothesis=hypothesis,
                        plan_pool=plan_pool,
                        artifact_pool=artifact_pool,
                        iteration=iteration,
                        telemetry=telemetry,
                        output_dir=iter_dir,
                        strategies=strategies,
                        cumulative=cumulative,
                    )
                    telemetry.emit(MessageType.INFO, f"   Retry generated {len(plans)} plans")

                if not plans:
                    telemetry.emit(MessageType.WARNING, "No plans after retry, ending loop")
                    telemetry.emit_module_group_summary(iter_group_name)
                    break

                # Check if we should stop at this module
                if should_stop_at_step("gen_plan"):
                    telemetry.emit(MessageType.INFO, f"\nSTOPPING - Reached end_step=gen_plan at iteration {iteration}")
                    telemetry.emit_module_group_summary(iter_group_name)
                    break

            # =================================================================
            # STEP 3: GEN_ART - Execute plans to generate artifacts
            # =================================================================
            artifacts = []
            if should_skip_step(iteration, "gen_art"):
                telemetry.emit(MessageType.INFO, "\nSTEP 3: GEN_ART [SKIPPED - already completed]")
            else:
                telemetry.emit(MessageType.INFO, "\nSTEP 3: GEN_ART")
                artifacts = await run_gen_art_module(
                    config=config,
                    plan_pool=plan_pool,
                    artifact_pool=artifact_pool,
                    iteration=iteration,
                    run_dir=run_dir or output_dir,
                    telemetry=telemetry,
                    output_dir=iter_dir,
                    cumulative=cumulative,
                    plans=plans,
                )
                telemetry.emit(MessageType.INFO, f"   Produced {len(artifacts)} artifacts")

                # Check if we should stop at this step
                if should_stop_at_step("gen_art"):
                    telemetry.emit(MessageType.INFO, f"\nSTOPPING - Reached end_step=gen_art at iteration {iteration}")
                    telemetry.emit_module_group_summary(iter_group_name)
                    break

            # =================================================================
            # STEP 4: GEN_NARR - Generate narratives (starts at configured iteration)
            # =================================================================
            narratives = []  # Initialize for iteration summary
            if iteration >= start_narrative_at:
                if should_skip_step(iteration, "gen_narr"):
                    telemetry.emit(MessageType.INFO, "\nSTEP 4: GEN_NARR [SKIPPED - already completed]")
                else:
                    telemetry.emit(MessageType.INFO, "\nSTEP 4: GEN_NARR")
                    narratives = await run_gen_narr_module(
                        config=config,
                        hypothesis=hypothesis,
                        artifact_pool=artifact_pool,
                        narrative_pool=narrative_pool,
                        iteration=iteration,
                        telemetry=telemetry,
                        output_dir=iter_dir,
                        cumulative=cumulative,
                    )
                    telemetry.emit(MessageType.INFO, f"   Generated {len(narratives)} narratives")

                    # Check if we should stop at this module
                    if should_stop_at_step("gen_narr"):
                        telemetry.emit(MessageType.INFO, f"\nSTOPPING - Reached end_step=gen_narr at iteration {iteration}")
                        telemetry.emit_module_group_summary(iter_group_name)
                        break

            # Log iteration summary
            telemetry.emit(MessageType.INFO, "")
            telemetry.emit(MessageType.INFO, f"Iteration {iteration} Summary:")
            # Per-iteration counts
            telemetry.emit(MessageType.INFO, f"   It {iteration} Strategies: {len(strategies)}")
            telemetry.emit(MessageType.INFO, f"   It {iteration} Plans: {len(plans)}")
            telemetry.emit(MessageType.INFO, f"   It {iteration} Artifacts: {len(artifacts)}")
            telemetry.emit(MessageType.INFO, f"   It {iteration} Narratives: {len(narratives)}")
            # Cumulative totals
            telemetry.emit(MessageType.INFO, f"   Total Strategies: {len(strategy_pool.get_all())}")
            telemetry.emit(MessageType.INFO, f"   Total Plans: {len(plan_pool.get_all())}")
            telemetry.emit(MessageType.INFO, f"   Total Artifacts: {len(artifact_pool.get_all())}")
            telemetry.emit(MessageType.INFO, f"   Total Narratives: {len(narrative_pool.get_all())}")
            all_narrs = narrative_pool.get_all()
            if all_narrs:
                telemetry.emit(MessageType.INFO, f"   Latest narrative: {all_narrs[-1].id}")

            # Emit iteration module group summary (aggregates all modules in this iteration)
            telemetry.emit_module_group_summary(iter_group_name)

            iterations_completed += 1

            stats = {
                "iteration": iteration,
                "strategies": len(strategy_pool.get_all()),
                "plans": len(plan_pool.get_all()),
                "artifacts": len(artifact_pool.get_all()),
                "narratives": len(narrative_pool.get_all()),
                "timestamp": datetime.now().isoformat(),
            }
            with open(iter_dir / "stats.json", 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)

            # Save pool snapshot after each iteration (enables resume on crash)
            save_all_pools(output_dir / "pools", strategy_pool, plan_pool, artifact_pool, narrative_pool)
            telemetry.emit(MessageType.INFO, f"   Pools saved to {rel_path(output_dir / 'pools')}")

        # =====================================================================
        # POST-LOOP: Finalize
        # =====================================================================
        telemetry.emit(MessageType.INFO, "")
        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, "invention loop Complete")
        telemetry.emit(MessageType.INFO, "=" * 60)

        # Save final pool state
        save_all_pools(output_dir / "pools", strategy_pool, plan_pool, artifact_pool, narrative_pool)

        # Get narrative (last one produced)
        all_narrs = narrative_pool.get_all()
        narrative = all_narrs[-1] if all_narrs else None

        # Build final result (returned for pipeline flow + saved as legacy file)
        result = InventionLoopOut(
            output_dir=str(output_dir),
            pools_dir=str(output_dir / "pools"),
            narrative=narrative,
            artifacts=artifact_pool.get_all(),
            hypothesis=hypothesis,
            metadata={
                'generated_at': datetime.now().isoformat(),
                'module': 'invention_loop',
                'iterations_completed': iterations_completed,
                'max_iterations': max_iterations,
                'start_iteration': start_iteration,
                'end_step': end_step,
                'total_strategies': len(strategy_pool.get_all()),
                'total_plans': len(plan_pool.get_all()),
                'total_artifacts': len(artifact_pool.get_all()),
                'total_narratives': len(narrative_pool.get_all()),
            },
        )

        # Save legacy result file (for pipeline resume from gen_paper_repo)
        result_file = output_dir / "invention_loop_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, ensure_ascii=False, default=str)

        # Emit standardized module output (cumulative already has all iteration data)
        from aii_pipeline.utils import build_module_output, emit_module_output
        total_cost_usd = sum(s.get("total_cost", 0.0) or 0.0 for s in telemetry._module_summaries)
        std_output = build_module_output(
            module="invention_loop",
            outputs=[result],
            cumulative=cumulative,
            output_dir=output_dir,
            total_cost_usd=total_cost_usd,
            iterations_completed=iterations_completed,
            max_iterations=max_iterations,
        )
        emit_module_output(std_output, telemetry, output_dir=output_dir)

        telemetry.emit(MessageType.SUCCESS, f"invention loop completed:")
        telemetry.emit(MessageType.INFO, f"   Iterations: {iterations_completed}")
        telemetry.emit(MessageType.INFO, f"   Artifacts: {len(artifact_pool.get_all())}")
        telemetry.emit(MessageType.INFO, f"   Narratives: {len(narrative_pool.get_all())}")
        if narrative:
            telemetry.emit(MessageType.INFO, f"   Narrative: {narrative.id}")
        telemetry.emit(MessageType.INFO, f"   Results: {rel_path(result_file)}")

        # Emit module group summary (aggregates all iteration module groups)
        telemetry.emit_module_group_summary("INVENTION_LOOP")

        return result

    finally:
        # Only flush/close if we created local telemetry
        if local_telemetry:
            telemetry.flush()


async def main():
    """Main function for standalone execution."""
    # Load config
    config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
    config = PipelineConfig.from_yaml(config_path)

    # Sample hypothesis for testing
    sample_hypothesis = {
        "title": "Multi-Agent Specialization Improves Task Performance",
        "hypothesis": "A multi-agent system with specialized roles outperforms monolithic agents on complex tasks",
        "description": "We hypothesize that decomposing complex tasks across specialized agents leads to better outcomes than single-agent approaches.",
    }

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/{timestamp}_invention_loop")
    run_dir.mkdir(parents=True, exist_ok=True)

    workspace_dir = run_dir / "model_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    result = await run_invention_loop_module(
        config=config,
        hypothesis=sample_hypothesis,
        run_dir=run_dir,
        workspace_dir=workspace_dir,
    )

    if result:
        print("invention loop completed successfully")
        return 0
    return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
