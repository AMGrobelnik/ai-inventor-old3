"""Experiment executor - Claude Code agent for methodology implementation.

Experiments implement and run research methodologies based on hypotheses.
Uses 10 sequential prompts for iterative development and validation.

Uses aii_lib for:
- Agent: Claude Code SDK wrapper
- AgentOptions: Agent configuration with expected_files_struct_out_field for file validation
- AgentInitializer/AgentFinalizer: Pre/post-agent utilities
- AIITelemetry: Task tracking

File validation uses structured output (expected_files_struct_out_field on AgentOptions):
- Agent reports all created file paths in typed expected_files structure
- SDK recursively extracts paths and validates they exist inside workspace
- Automatically retries with feedback on missing files
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from aii_lib import Agent, AgentOptions, AgentInitializer, AgentFinalizer, AIITelemetry
from aii_lib.agent_backend import aggregate_summaries
from aii_lib.abilities.mcp_server.config import get_tooluniverse_mcp_config

from aii_pipeline.utils import PipelineConfig
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.experiment import u_prompt
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.experiment.s_prompt import get as get_system
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.experiment.schema import (
    ExperimentArtifact,
    verify_experiment_output,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.experiment.u_prompt import build_experiment_retry_prompt
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.schema import BasePlan

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aii_pipeline.steps._invention_loop.pools import ArtifactPool

# Path to workspace template (schemas, etc.)
WORKSPACE_TEMPLATE = Path(__file__).parent.parent.parent.parent / "prompts" / "_3_invention_loop" / "_3_gen_art" / "experiment_workspace"


async def execute_experiment(
    plan: BasePlan,
    artifact_pool: ArtifactPool,
    config: PipelineConfig,
    run_dir: Path,
    iteration: int = 1,
    experiment_idx: int = 0,
    telemetry: AIITelemetry | None = None,
    task_id: str | None = None,
    task_name: str | None = None,
    task_sequence: int | None = None,
) -> tuple[dict, bool, float]:
    """
    Execute an experiment plan using Claude Code agent.

    The agent runs 10 sequential prompts to:
    1. Read and understand the data structure
    2. Test framework functionality
    3. Implement method.py with baseline comparison
    4-7. Iteratively run and fix the method
    8. Switch to full data and run
    9. Generate output file versions (full, mini, preview)
    10. Return final output paths

    Args:
        plan: The experiment plan to execute
        artifact_pool: Artifact pool for resolving dependencies
        config: Pipeline configuration
        run_dir: Run output directory
        iteration: Current invention loop iteration number
        experiment_idx: Index of this experiment within the iteration
        telemetry: Optional AIITelemetry for message logging
        task_id: Task ID for telemetry
        task_name: Task name for telemetry

    Returns:
        (result_dict, status, cost_usd)
    """
    # Create initializer and finalizer
    initializer = AgentInitializer(telemetry=telemetry, task_id=task_id, task_name=task_name)
    finalizer = AgentFinalizer(telemetry=telemetry, task_id=task_id, task_name=task_name)

    # Setup workspace from template
    workspace_dir = run_dir / (task_id or f"experiment_workspace_idx{experiment_idx}")
    initializer.setup_workspace(workspace_dir, template_dir=WORKSPACE_TEMPLATE)

    # Start task (must be before any emit_message calls)
    initializer.start_task(sequence=task_sequence)

    # Create callback for summary aggregation (agent suppresses summaries due to expected_files_struct_out_field)
    callback = telemetry.create_callback(task_id, task_name, group="experiment") if telemetry and task_id and task_name else None

    # Log execution info
    if telemetry and task_id and task_name:
        telemetry.emit_message("INFO", f"Executing EXPERIMENT: {plan.title}", task_name, task_id)
        telemetry.emit_message("INFO", f"Workspace: {workspace_dir}", task_name, task_id)

    # Get experiment executor config
    experiment_cfg = config.invention_loop.execute.experiment
    agent_cfg = experiment_cfg.claude_agent
    model = agent_cfg.model
    max_turns = agent_cfg.max_turns
    timeout_seconds = agent_cfg.seq_prompt_timeout

    # Build plan text from typed fields
    plan_text = plan.to_prompt_yaml()

    workspace_str = str(workspace_dir)

    # Build sequential prompts with dependencies using new API
    prompts = u_prompt.get_all_prompts(
        plan_text=plan_text,
        artifact_pool=artifact_pool,
        dependency_ids=plan.artifact_dependencies,
        workspace_path=workspace_str,
    )

    # Get system prompt
    system_prompt = get_system()

    # Get verification config
    verify_retries = experiment_cfg.verify_retries
    min_examples = experiment_cfg.min_examples

    # Create agent options
    options = AgentOptions(
        model=model,
        max_turns=max_turns,
        agent_timeout=agent_cfg.agent_timeout,
        agent_retries=agent_cfg.agent_retries,
        seq_prompt_timeout=timeout_seconds,
        seq_prompt_retries=agent_cfg.seq_prompt_retries,
        message_timeout=agent_cfg.message_timeout,
        message_retries=agent_cfg.message_retries,
        cwd=workspace_str,
        system_prompt=system_prompt,
        continue_seq_item=True,  # Continue conversation between prompts
        # Always connect MCP for aii_web_fetch_grep.
        # When use_aii_web_tools=True: all 3 MCP tools replace built-in WebSearch/WebFetch.
        # When use_aii_web_tools=False: only grep via MCP, built-in WebSearch/WebFetch for the rest.
        mcp_servers=get_tooluniverse_mcp_config(use_aii_server=True),
        disallowed_tools=(
            ["WebSearch", "WebFetch"] if agent_cfg.use_aii_web_tools
            else ["mcp__aii_tooluniverse__aii_web_search_fast", "mcp__aii_tooluniverse__aii_web_fetch_direct"]
        ) + ["mcp__aii_tooluniverse__dblp_bib_search", "mcp__aii_tooluniverse__dblp_bib_fetch"],
        # AIITelemetry integration for sequenced parallel execution
        telemetry=telemetry,
        run_id=task_id,
        agent_context=task_name,  # Display name for logs
        # SDK native structured output for title/summary
        output_format=ExperimentArtifact.to_struct_output(),
        # Expected files validation via structured output (auto-retry on missing)
        expected_files_struct_out_field="out_expected_files",
        max_expected_files_retries=verify_retries,
    )

    try:
        # Create and run agent
        agent = Agent(options)
        all_responses: list = []  # Track for cost aggregation

        if telemetry and task_id and task_name:
            telemetry.emit_message("INFO", f"Running {len(prompts)} sequential prompts", task_name, task_id)

        # Run all prompts sequentially
        # SDK handles file existence validation + retry automatically
        result = await agent.run(prompts)
        all_responses.append(result)
        cost = result.total_cost

        # Emit aggregated summary (agent suppresses individual summaries due to expected_files_struct_out_field)
        if callback and result.prompt_results:
            aggregated = aggregate_summaries(result.prompt_results)
            if aggregated:
                callback(aggregated)

        if result.failed:
            err = result.error_message or "unknown error"
            finalizer.end_task_failure(f"Agent failed: {err}", cost=cost)
            return {}, False, cost

        # =================================================================
        # POST-VALIDATION: Schema and content quality check (with retry)
        # =================================================================
        expected_files = ExperimentArtifact.get_expected_out_files()
        schema_max_retries = experiment_cfg.schema_retries

        for schema_attempt in range(1, schema_max_retries + 2):
            verification = verify_experiment_output(
                workspace_dir=workspace_dir,
                expected_files=expected_files,
                min_examples=min_examples,
            )

            if verification.get("schema_errors"):
                for err in verification["schema_errors"][:3]:
                    if telemetry and task_id and task_name:
                        telemetry.emit_message("WARNING", f"Schema: {err}", task_name, task_id)
            if verification.get("content_warnings"):
                for warn in verification["content_warnings"][:3]:
                    if telemetry and task_id and task_name:
                        telemetry.emit_message("WARNING", f"Content: {warn}", task_name, task_id)

            if verification.get("valid", False) or schema_attempt > schema_max_retries:
                break

            retry_prompt = build_experiment_retry_prompt(
                verification=verification,
                attempt=schema_attempt,
                max_attempts=schema_max_retries,
            )
            if telemetry and task_id and task_name:
                telemetry.emit_message(
                    "RETRY",
                    f"Schema validation failed, retrying ({schema_attempt}/{schema_max_retries})...",
                    task_name, task_id,
                )
            retry_result = await agent.run(retry_prompt)
            all_responses.append(retry_result)
            cost += retry_result.total_cost

        # =================================================================
        # COLLECT RESULTS
        # =================================================================
        experiment_result = _find_experiment_files(workspace_dir)

        if experiment_result.get("error"):
            finalizer.end_task_failure(experiment_result['error'], cost=cost)
            return experiment_result, False, cost

        # Add metadata
        experiment_result["hypothesis"] = plan.title
        experiment_result["workspace_path"] = str(workspace_dir)
        experiment_result["example_count"] = verification.get("example_count", 0)

        # Extract title/summary from structured output (use last response with data)
        title = ""
        summary = ""
        for resp in reversed(all_responses):
            if resp.structured_output:
                data = resp.structured_output if isinstance(resp.structured_output, dict) else {}
                title = data.get("title", "")
                summary = data.get("summary", "")
                break

        if not title and not summary:
            raise RuntimeError(f"Experiment executor produced no structured output for {plan.id} — cannot extract title/summary")

        title = title or plan.title.strip()

        experiment_result["summary"] = summary

        if telemetry and task_id and task_name:
            schema_ok = verification.get("valid", False)
            telemetry.emit_message(
                "SUCCESS" if schema_ok else "WARNING",
                f"Experiment {'complete' if schema_ok else 'complete (schema issues)'}: {experiment_result['summary']}, ${cost:.4f}",
                task_name, task_id
            )
        finalizer.end_task_success(cost=cost)

        return experiment_result, True, cost

    except asyncio.TimeoutError:
        finalizer.end_task_timeout(timeout_seconds)
        raise

    except Exception as e:
        finalizer.end_task_error(str(e))
        raise


def _find_experiment_files(workspace_dir: Path) -> dict:
    """Find experiment output files in workspace.

    Claude agent creates these files directly. We just need to locate them.

    Returns:
        Dict with experiment paths or error
    """
    result = {
        "full_path": None,
        "mini_path": None,
        "preview_path": None,
        "code_files": [],
        "valid": False,
    }

    # Look for standard output files
    full_file = workspace_dir / "full_method_out.json"
    mini_file = workspace_dir / "mini_method_out.json"
    preview_file = workspace_dir / "preview_method_out.json"

    if full_file.exists():
        result["full_path"] = str(full_file)
        result["full_path_exists"] = True

    if mini_file.exists():
        result["mini_path"] = str(mini_file)
        result["mini_path_exists"] = True

    if preview_file.exists():
        result["preview_path"] = str(preview_file)
        result["preview_path_exists"] = True

    # Find code files
    for f in workspace_dir.glob("*.py"):
        result["code_files"].append(f.name)

    # Valid if at least full path exists
    result["valid"] = result.get("full_path_exists", False)

    if not result["valid"]:
        result["error"] = "No experiment output files found in workspace"

    return result
