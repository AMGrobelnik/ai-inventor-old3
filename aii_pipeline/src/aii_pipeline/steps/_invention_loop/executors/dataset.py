"""Dataset executor - Claude Code agent with HuggingFace/OWID dataset tools.

Uses aii_lib for:
- Agent: Claude Code SDK wrapper
- AgentOptions: Agent configuration
- AgentInitializer/AgentFinalizer: Pre/post-agent utilities
- AIITelemetry: Task tracking

File validation uses structured output (DatasetArtifact.expected_files):
- Agent reports all created file paths in typed expected_files structure
- SDK recursively extracts paths and validates they exist inside workspace
- Post-execution: verify JSON schema and content quality
- Retry with feedback if validation fails
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from aii_lib import Agent, AgentOptions, AgentInitializer, AgentFinalizer, AIITelemetry
from aii_lib.agent_backend import aggregate_summaries
from aii_lib.abilities.mcp_server.config import get_tooluniverse_mcp_config

from aii_pipeline.utils import PipelineConfig
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset import u_prompt
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset.s_prompt import get as get_system
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset.schema import (
    DatasetArtifact,
    verify_dataset_output,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset.u_prompt import build_dataset_retry_prompt
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.schema import BasePlan

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aii_pipeline.steps._invention_loop.pools import ArtifactPool

# Path to workspace template (schemas, etc.)
WORKSPACE_TEMPLATE = Path(__file__).parent.parent.parent.parent / "prompts" / "_3_invention_loop" / "_3_gen_art" / "dataset_workspace"


async def execute_dataset(
    plan: BasePlan,
    artifact_pool: ArtifactPool,
    config: PipelineConfig,
    run_dir: Path,
    iteration: int = 1,
    dataset_idx: int = 0,
    telemetry: AIITelemetry | None = None,
    task_id: str | None = None,
    task_name: str | None = None,
    task_sequence: int | None = None,
) -> tuple[dict, bool, float]:
    """
    Execute a dataset plan using Claude Code agent.

    The agent runs 8 sequential prompts to search, download, evaluate,
    and select a single best dataset.

    Args:
        plan: The dataset plan to execute
        context: Context from dependent artifacts
        config: Pipeline configuration
        run_dir: Run output directory
        iteration: Current invention loop iteration number
        dataset_idx: Index of this dataset within the iteration
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
    workspace_dir = run_dir / (task_id or f"dataset_workspace_idx{dataset_idx}")
    initializer.setup_workspace(workspace_dir, template_dir=WORKSPACE_TEMPLATE)

    # Start task (must be before any emit_message calls)
    initializer.start_task(sequence=task_sequence)

    # Create callback for summary aggregation (agent suppresses summaries due to expected_files_struct_out_field)
    callback = telemetry.create_callback(task_id, task_name, group="dataset") if telemetry and task_id and task_name else None

    # Log execution info
    if telemetry and task_id and task_name:
        telemetry.emit_message("INFO", f"Executing DATASET: {plan.title}", task_name, task_id)
        telemetry.emit_message("INFO", f"Workspace: {workspace_dir}", task_name, task_id)

    # Get dataset executor config
    dataset_cfg = config.invention_loop.execute.dataset
    agent_cfg = dataset_cfg.claude_agent
    model = agent_cfg.model
    max_turns = agent_cfg.max_turns
    timeout_seconds = agent_cfg.seq_prompt_timeout

    # Extract plan fields (typed on DatasetPlan)
    target_num_datasets = min(plan.target_num_datasets, dataset_cfg.dataset_chosen_final_cap)
    plan_text = plan.to_prompt_yaml()

    workspace_str = str(workspace_dir)

    # Build sequential prompts with dependencies — funnel narrows to target_num_datasets
    prompts = u_prompt.get_all_prompts(
        plan_text=plan_text,
        artifact_pool=artifact_pool,
        dependency_ids=plan.artifact_dependencies,
        target_num_datasets=target_num_datasets,
        max_dataset_size=dataset_cfg.dataset_max_size,
        search_tool_cap=dataset_cfg.dataset_search_tool_cap,
        chosen_for_preview_cap=dataset_cfg.dataset_chosen_for_preview_cap,
        chosen_for_download_cap=dataset_cfg.dataset_chosen_for_download_cap,
        workspace_path=workspace_str,
    )

    # Get system prompt
    system_prompt = get_system()

    # Get verification config
    min_examples = dataset_cfg.min_examples

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
        agent_context=task_name,  # Display name for logs (e.g., "data-0")
        # Structured output: agent reports title, summary, and all file paths it created
        output_format=DatasetArtifact.to_struct_output(),
        # Expected files validation via structured output (auto-retry on missing)
        expected_files_struct_out_field="out_expected_files",
        max_expected_files_retries=dataset_cfg.verify_retries,
    )

    try:
        # Create and run agent
        agent = Agent(options)
        all_responses: list = []  # Track for cost aggregation

        if telemetry and task_id and task_name:
            telemetry.emit_message("INFO", f"Running {len(prompts)} sequential prompts", task_name, task_id)

        # Run all prompts sequentially
        # SDK handles expected_files validation + retry via structured output
        result = await agent.run(prompts)
        all_responses.append(result)
        cost = result.total_cost

        # Emit aggregated summary
        if callback and result.prompt_results:
            aggregated = aggregate_summaries(result.prompt_results)
            if aggregated:
                callback(aggregated)


        if result.failed:
            err = result.error_message or "unknown error"
            finalizer.end_task_failure(f"Agent failed: {err}", cost=cost)
            return {}, False, cost

        # =================================================================
        # EXTRACT STRUCTURED OUTPUT
        # =================================================================
        title = plan.title.strip()
        summary = ""
        all_file_paths: list[str] = []
        data_file_paths: list[str] = []

        # Helper to extract from latest structured output (may be called after retries)
        def _extract_structured_output() -> None:
            nonlocal title, summary, all_file_paths, data_file_paths
            for resp in reversed(all_responses):
                if resp.structured_output:
                    data = resp.structured_output if isinstance(resp.structured_output, dict) else {}
                    title = data.get("title", "") or title
                    summary = data.get("summary", "")
                    # Extract all file paths from nested expected_files structure
                    out_expected_files = data.get("out_expected_files", {})
                    all_file_paths = Agent._collect_paths_recursive(out_expected_files)
                    # Extract data file paths (JSON only) from datasets
                    data_file_paths = []
                    for ds in out_expected_files.get("datasets", []):
                        if isinstance(ds, dict):
                            data_file_paths.extend(ds.get("full", []))
                            if ds.get("mini"):
                                data_file_paths.append(ds["mini"])
                            if ds.get("preview"):
                                data_file_paths.append(ds["preview"])
                    return

        _extract_structured_output()

        # =================================================================
        # POST-VALIDATION: Schema and content quality check (with retry)
        # Uses file paths reported by the agent — no filesystem scanning.
        # =================================================================
        schema_max_retries = dataset_cfg.schema_retries

        for schema_attempt in range(1, schema_max_retries + 2):  # +2: 1 initial + N retries
            verification = verify_dataset_output(
                workspace_dir=workspace_dir,
                file_paths=all_file_paths,
                min_examples=min_examples,
            )

            if verification.get("valid", False) or schema_attempt > schema_max_retries:
                break

            # Log issues
            for err in verification.get("schema_errors", [])[:3]:
                if telemetry and task_id and task_name:
                    telemetry.emit_message("WARNING", f"Schema: {err}", task_name, task_id)

            retry_prompt = build_dataset_retry_prompt(
                verification=verification,
                attempt=schema_attempt,
                max_attempts=schema_max_retries,
            )
            if telemetry and task_id and task_name:
                telemetry.emit_message("RETRY", f"Schema validation retry ({schema_attempt}/{schema_max_retries})...", task_name, task_id)
            retry_result = await agent.run(retry_prompt)
            all_responses.append(retry_result)
            cost += retry_result.total_cost
            _extract_structured_output()

        # =================================================================
        # POST-VERIFICATION: Check file sizes
        # =================================================================
        oversized, size_retry_cost = await finalizer.verify_file_sizes_and_retry(
            workspace_dir=workspace_dir,
            agent=agent,
            max_retries=1,
        )
        cost += size_retry_cost
        if size_retry_cost > 0:
            _extract_structured_output()  # Re-extract if size retry changed things

        # =================================================================
        # COLLECT RESULTS
        # =================================================================
        dataset_result: dict = {}
        dataset_result["workspace_path"] = str(workspace_dir)
        dataset_result["example_count"] = verification.get("example_count", 0)
        dataset_result["summary"] = summary
        # File lists come directly from structured output (already validated by SDK)
        dataset_result["file_list"] = all_file_paths
        dataset_result["data_file_paths"] = data_file_paths or [
            p for p in all_file_paths if p.endswith(".json")
        ]

        if not dataset_result["file_list"]:
            finalizer.end_task_failure("No dataset files found", cost=cost)
            raise RuntimeError("No dataset files found in workspace")

        if telemetry and task_id and task_name:
            n_files = len(dataset_result["file_list"])
            schema_ok = verification.get("valid", False)
            telemetry.emit_message(
                "SUCCESS" if schema_ok else "WARNING",
                f"Dataset {'complete' if schema_ok else 'complete (schema issues)'}: {n_files} files, ${cost:.4f}",
                task_name, task_id
            )
        finalizer.end_task_success(cost=cost)

        return dataset_result, True, cost

    except asyncio.TimeoutError:
        finalizer.end_task_timeout(timeout_seconds)
        raise

    except Exception as e:
        finalizer.end_task_error(str(e))
        raise


