"""GEN_ART Module - Execute selected plans via type-specific artifact executors.

Contains executors for each artifact type:
- RESEARCH: Research questions answered via web search (OpenRouter or Claude agent)
- EXPERIMENT: Methodology implementations (Claude Code agent)
- DATASET: HuggingFace/OWID dataset search and download (Claude Code agent)
- EVALUATION: Experiment result assessment (Claude Code agent)
- PROOF: Lean 4 formal proofs (Claude Code agent)

All artifact types execute in parallel, controlled by max_concurrent_artifacts semaphore.
This allows mixed execution (e.g., 2 research + 1 experiment + 2 datasets at once).

Telemetry structure:
- GEN_ART (module)
  └── Tasks for each artifact execution (FND_exec_*, EXP_exec_*, etc.)

Adds artifacts to pool (both successes AND failures).

Uses aii_lib for:
- AIITelemetry: Task tracking and logging (module/task hierarchy)
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from aii_lib import AIITelemetry, MessageType, JSONSink, get_model_short

from aii_pipeline.utils import PipelineConfig, rel_path

from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.schema import BasePlan, PlanType
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.schema import BaseArtifact, ArtifactType
from ._invention_loop.pools import get_type_abbrev
from ._invention_loop.pools import PlanPool, ArtifactPool

# Import artifact executors
from ._invention_loop.executors import (
    execute_research,
    execute_experiment,
    execute_dataset,
    execute_evaluation,
    execute_proof,
)

# Import schema modules for file metadata functions
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.research.schema import ResearchArtifact
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.experiment.schema import ExperimentArtifact
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset.schema import DatasetArtifact
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.evaluation.schema import EvaluationArtifact
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof.schema import ProofArtifact

# Map plan types to their artifact schema classes (for file metadata)
PLAN_TO_SCHEMA = {
    PlanType.RESEARCH: ResearchArtifact,
    PlanType.EXPERIMENT: ExperimentArtifact,
    PlanType.DATASET: DatasetArtifact,
    PlanType.EVALUATION: EvaluationArtifact,
    PlanType.PROOF: ProofArtifact,
}


# Map plan types to artifact types
PLAN_TO_ARTIFACT = {
    PlanType.RESEARCH: ArtifactType.RESEARCH,
    PlanType.EXPERIMENT: ArtifactType.EXPERIMENT,
    PlanType.DATASET: ArtifactType.DATASET,
    PlanType.EVALUATION: ArtifactType.EVALUATION,
    PlanType.PROOF: ArtifactType.PROOF,
}


async def exec_plan(
    plan: BasePlan,
    config: PipelineConfig,
    artifact_pool: ArtifactPool,
    iteration: int,
    run_dir: Path,
    research_idx: int = 0,
    dataset_idx: int = 0,
    experiment_idx: int = 0,
    evaluation_idx: int = 0,
    proof_idx: int = 0,
    telemetry: AIITelemetry | None = None,
    task_id: str | None = None,
    task_name: str | None = None,
    task_sequence: int | None = None,
) -> BaseArtifact | None:
    """
    Execute a single plan.

    Dispatches to the appropriate unit based on plan type.
    Handles inner refinement loops and failure cases.

    Returns:
        Created artifact (success or failure), or None on critical error
    """
    # Note: emit_task_start is called inside each executor (research.py, etc.)
    # to avoid duplicate registration

    if telemetry and task_id and task_name:
        telemetry.emit_message("INFO", f"Executing plan: {plan.id} ({plan.type.value})", task_name, task_id)
        telemetry.emit_message("INFO", f"  Title: {plan.title}", task_name, task_id)

    # Log missing dependencies (pool resolves them on demand)
    for dep_id in plan.artifact_dependencies:
        if not artifact_pool.get_by_id(dep_id):
            if telemetry and task_id and task_name:
                telemetry.emit_message("WARN", f"Missing dependency: {dep_id}", task_name, task_id)

    # Dispatch to appropriate unit
    try:
        if plan.type == PlanType.RESEARCH:
            result, is_success, cost = await execute_research(
                plan=plan,
                artifact_pool=artifact_pool,
                config=config,
                run_dir=run_dir,
                iteration=iteration,
                research_idx=research_idx,
                telemetry=telemetry,
                task_id=task_id,
                task_name=task_name,
                task_sequence=task_sequence,
            )

        elif plan.type == PlanType.EXPERIMENT:
            result, is_success, cost = await execute_experiment(
                plan=plan,
                artifact_pool=artifact_pool,
                config=config,
                run_dir=run_dir,
                iteration=iteration,
                experiment_idx=experiment_idx,
                telemetry=telemetry,
                task_id=task_id,
                task_name=task_name,
                task_sequence=task_sequence,
            )

        elif plan.type == PlanType.DATASET:
            result, is_success, cost = await execute_dataset(
                plan=plan,
                artifact_pool=artifact_pool,
                config=config,
                run_dir=run_dir,
                iteration=iteration,
                dataset_idx=dataset_idx,
                telemetry=telemetry,
                task_id=task_id,
                task_name=task_name,
                task_sequence=task_sequence,
            )

        elif plan.type == PlanType.EVALUATION:
            result, is_success, cost = await execute_evaluation(
                plan=plan,
                artifact_pool=artifact_pool,
                config=config,
                run_dir=run_dir,
                iteration=iteration,
                evaluation_idx=evaluation_idx,
                telemetry=telemetry,
                task_id=task_id,
                task_name=task_name,
                task_sequence=task_sequence,
            )

        elif plan.type == PlanType.PROOF:
            result, is_success, cost = await execute_proof(
                plan=plan,
                artifact_pool=artifact_pool,
                config=config,
                run_dir=run_dir,
                iteration=iteration,
                proof_idx=proof_idx,
                telemetry=telemetry,
                task_id=task_id,
                task_name=task_name,
                task_sequence=task_sequence,
            )

        else:
            if telemetry and task_id and task_name:
                telemetry.emit_message("ERROR", f"Unknown plan type: {plan.type}", task_name, task_id)
            return None

    except Exception as e:
        if telemetry and task_id and task_name:
            telemetry.emit_message("ERROR", f"Execution failed for {plan.id}: {e}", task_name, task_id)
        raise

    # Only add successful artifacts to pool (pool only stores successes)
    if not is_success:
        error_msg = result.get('error', 'Unknown error')
        # Research executor handles its own task_end; other types need it here
        if telemetry and task_id and task_name and plan.type != PlanType.RESEARCH:
            telemetry.emit_message("WARN", f"Failed: {error_msg}", task_name, task_id)
            telemetry.emit_task_end(task_id, task_name, f"Failed: {error_msg}")
        return None

    # Get file metadata from schema class
    schema_class = PLAN_TO_SCHEMA.get(plan.type)
    if schema_class:
        # Convert ExpectedFile objects to strings (just paths)
        expected_files_raw = schema_class.get_expected_out_files()
        out_expected_files = [f.path if hasattr(f, 'path') else str(f) for f in expected_files_raw]
        out_demo_files = schema_class.model_fields["out_demo_files"].default or []
    else:
        out_expected_files = []
        out_demo_files = []

    # Extract fields from result dict (populated by executors)
    artifact_title = result.get("title", "") or plan.title
    artifact_summary = result.get("summary", "")

    out_dependency_files: dict = {}
    if "file_list" in result:
        out_dependency_files["file_list"] = result["file_list"]
    elif out_expected_files:
        # Fallback: use expected output files as dependency file list
        out_dependency_files["file_list"] = out_expected_files
    if "data_file_paths" in result:
        out_dependency_files["data_file_paths"] = result["data_file_paths"]

    artifact = artifact_pool.add(
        id=task_id,
        plan=plan,
        title=artifact_title,
        summary=artifact_summary,
        workspace_path=result.get("workspace_path"),
        out_expected_files=out_expected_files,
        out_demo_files=out_demo_files,
        out_dependency_files=out_dependency_files,
    )

    result_summary = f"Success: {artifact.id}"

    # Emit task messages and end if telemetry provided
    # Note: research.py handles its own task lifecycle (emit_message + emit_task_end)
    if telemetry and task_id and task_name and plan.type != PlanType.RESEARCH:
        telemetry.emit_message("SUCCESS", f"{artifact.id} created", task_name, task_id)
        telemetry.emit_task_end(task_id, task_name, result_summary)

    return artifact


async def run_gen_art_module(
    config: PipelineConfig,
    plan_pool: PlanPool,
    artifact_pool: ArtifactPool,
    iteration: int,
    run_dir: Path,
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
    cumulative: dict | None = None,
    plans: list[BasePlan] | None = None,
) -> list[BaseArtifact]:
    """
    Run the EXECUTE step.

    Executes all selected plans via their type-specific units.
    Can run independent plans in parallel.

    Args:
        config: Pipeline configuration
        plan_pool: Pool with selected plans
        artifact_pool: Pool to add artifacts to
        iteration: Current iteration
        run_dir: Run output directory
        telemetry: AIITelemetry instance for logging
        output_dir: Directory to save outputs

    Returns:
        List of created artifacts (successes and failures)
    """
    # Step subdir within iteration dir (consistent with gen_strat, gen_plan, gen_narr)
    module_sinks = []
    if output_dir:
        step_dir = (output_dir / "gen_art").resolve()
        step_dir.mkdir(parents=True, exist_ok=True)
        output_dir = step_dir
        s1 = JSONSink(output_dir / "gen_art_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "gen_art_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    # GEN_ART is a single module - all artifact executions are tasks within it
    telemetry.start_module("GEN_ART")

    telemetry.emit(MessageType.INFO, "=" * 60)
    telemetry.emit(MessageType.INFO, f"GEN_ART - Running selected plans for iteration {iteration}")
    telemetry.emit(MessageType.INFO, "=" * 60)

    allowed_artifacts = config.invention_loop.allowed_artifacts


    # Use explicitly passed plans, or all plans from pool for this iteration
    if plans:
        selected = plans
    else:
        selected = plan_pool.get_by_iteration(iteration)

    # Filter by allowed artifacts if specified
    if allowed_artifacts:
        before_count = len(selected)
        selected = [p for p in selected if p.type.value in allowed_artifacts]
        if len(selected) < before_count:
            telemetry.emit(MessageType.INFO, f"Filtered to allowed artifacts {allowed_artifacts}: {before_count} -> {len(selected)}")

    if not selected:
        telemetry.emit(MessageType.WARNING, "No plans selected for execution")
        telemetry.emit_module_summary("GEN_ART")
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        return []

    telemetry.emit(MessageType.INFO, f"Executing {len(selected)} plans:")
    for p in selected:
        deps_str = f" (deps: {', '.join(p.artifact_dependencies)})" if p.artifact_dependencies else ""
        telemetry.emit(MessageType.INFO, f"  - [{p.id}] {p.type.value}: {p.title[:50]}{deps_str}")

    # Get concurrency limit from config
    max_concurrent = config.invention_loop.execute.max_concurrent_artifacts
    semaphore = asyncio.Semaphore(max_concurrent)
    telemetry.emit(MessageType.INFO, f"Max concurrent artifacts: {max_concurrent}")

    # Count plans by type for logging and idx tracking
    type_counts: dict[PlanType, int] = {}
    for p in selected:
        type_counts[p.type] = type_counts.get(p.type, 0) + 1

    # Log breakdown by type
    for ptype, count in sorted(type_counts.items(), key=lambda x: x[0].value):
        telemetry.emit(MessageType.INFO, f"  {ptype.value.upper()}: {count}")

    # Track idx per type (for workspace naming)
    type_idx_counters: dict[PlanType, int] = {t: 0 for t in PlanType}

    async def execute_with_semaphore(plan: BasePlan, task_counter: int) -> BaseArtifact | None:
        """Execute a plan with semaphore-controlled concurrency."""
        async with semaphore:
            type_short = get_type_abbrev(plan.type.value)
            # Get model from executor config for task_id suffix
            exec_cfg = getattr(config.invention_loop.execute, plan.type.value, None)
            exec_model = get_model_short(exec_cfg.claude_agent.model) if exec_cfg and exec_cfg.claude_agent else "agent"
            task_id = f"{type_short}_id{task_counter}_it{iteration}__{exec_model}"
            task_name = task_id

            # Get and increment idx for this type
            type_idx = type_idx_counters[plan.type]
            type_idx_counters[plan.type] += 1

            # Build kwargs with the correct idx parameter for this type
            kwargs = {
                "plan": plan,
                "config": config,
                "artifact_pool": artifact_pool,
                "iteration": iteration,
                "run_dir": output_dir or run_dir,  # iter_N/gen_art/ — executors create workspaces here
                "telemetry": telemetry,
                "task_id": task_id,
                "task_name": task_name,
                "task_sequence": task_counter,
                # Set all idx to 0 by default, then override the specific one
                "research_idx": 0,
                "dataset_idx": 0,
                "experiment_idx": 0,
                "evaluation_idx": 0,
                "proof_idx": 0,
            }

            # Override the specific idx for this type
            idx_key = f"{plan.type.value}_idx"
            kwargs[idx_key] = type_idx

            return await exec_plan(**kwargs)

    # Execute ALL plans in parallel (semaphore controls concurrency)
    tasks = []
    for task_counter, p in enumerate(selected, start=1):
        tasks.append(execute_with_semaphore(p, task_counter))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            telemetry.emit(MessageType.WARNING, f"GenArt task failed: {r}")
    artifacts: list[BaseArtifact] = [r for r in results if r is not None and not isinstance(r, Exception)]

    # Summary - artifacts only contains successes (pool only stores successes)
    successes = len(artifacts)
    failures = len(selected) - successes

    telemetry.emit(MessageType.SUCCESS, f"EXECUTE complete: {successes} successes, {failures} failures")

    # Compute total cost from pending summaries before emit_module_summary clears them
    total_cost_usd = sum(s.get("total_cost", 0.0) for s in telemetry._pending_summaries)

    # Build and emit module output
    from aii_pipeline.utils import build_module_output, emit_module_output
    module_output = build_module_output(
        module="gen_art",
        iteration=iteration,
        outputs=[a.model_dump(mode="json") for a in artifacts],
        cumulative=cumulative or {},
        total_cost_usd=total_cost_usd,
        output_dir=output_dir,
        llm_provider="claude_agent",
        num_executed=len(selected),
        successes=successes,
        failures=failures,
    )
    emit_module_output(module_output, telemetry, output_dir=output_dir)

    # Emit module summary (aggregates all artifact executions)
    telemetry.emit_module_summary("GEN_ART")

    # Remove module sinks to prevent leak into subsequent modules
    for sink in module_sinks:
        sink.flush()
        telemetry.remove_sink(sink)

    return artifacts
