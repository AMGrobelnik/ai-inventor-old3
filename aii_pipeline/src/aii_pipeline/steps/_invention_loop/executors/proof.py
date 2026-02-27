"""Proof executor - Claude Code agent with Lean 4 verification tool.

Proofs are formal mathematical proofs in Lean 4.
Uses Claude agent with access to lean_run_code MCP tool for verification.

The agent:
1. Analyzes the theorem and plans approach
2. Writes initial Lean 4 code
3. Tests with lean_run_code tool
4. Iterates on errors until verified
5. Returns structured JSON with proof results

Uses aii_lib for:
- Agent: Claude Code SDK wrapper
- AgentOptions: Agent configuration with expected_files_struct_out_field for file validation
- AgentInitializer/AgentFinalizer: Pre/post-agent utilities
- AIITelemetry: Task tracking

File validation uses structured output (expected_files_struct_out_field on AgentOptions):
- Agent reports all created file paths in typed expected_files structure
- SDK recursively extracts paths and validates they exist inside workspace
- Automatically retries with feedback on missing files

TODO: Upgrade to use LeanExplore API for improved proof verification and search.
      See: https://leanexplore.com/docs/api
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aii_lib import Agent, AgentOptions, AgentInitializer, AgentFinalizer, AIITelemetry
from aii_lib.agent_backend import aggregate_summaries
from aii_lib.abilities.mcp_server.config import get_tooluniverse_mcp_config

from aii_pipeline.utils import PipelineConfig

# Path to workspace template
WORKSPACE_TEMPLATE = Path(__file__).parent.parent.parent.parent / "prompts" / "_3_invention_loop" / "_3_gen_art" / "proof_workspace"
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof import u_prompt
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof.s_prompt import get as get_system
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof.schema import (
    ProofArtifact,
    verify_proof_output,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof.u_prompt import build_proof_retry_prompt
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.schema import BasePlan

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aii_pipeline.steps._invention_loop.pools import ArtifactPool


async def execute_proof(
    plan: BasePlan,
    artifact_pool: ArtifactPool,
    config: PipelineConfig,
    run_dir: Path,
    iteration: int = 1,
    proof_idx: int = 0,
    telemetry: AIITelemetry | None = None,
    task_id: str | None = None,
    task_name: str | None = None,
    task_sequence: int | None = None,
) -> tuple[dict, bool, float]:
    """
    Execute a proof plan using Claude Code agent with Lean verification.

    The agent has access to lean_run_code MCP tool which compiles and
    verifies Lean 4 code with Mathlib support (ring, linarith, etc.).

    Args:
        plan: The proof plan to execute
        context: Context from dependent artifacts
        config: Pipeline configuration
        run_dir: Run output directory
        iteration: Current invention loop iteration number
        proof_idx: Index of this proof within the iteration
        telemetry: Optional AIITelemetry for message logging
        task_id: Task ID for telemetry
        task_name: Task name for telemetry

    Returns:
        (result_dict, status, cost_usd)

        result_dict structure:
        {
            "theorem_statement": "...",
            "proof_successful": true/false,
            "proof_outcome_explanation": "...",
            "lean_code": "...",
            "lean_explanation": "...",
            "verified": true/false
        }
    """
    # Create initializer and finalizer
    initializer = AgentInitializer(telemetry=telemetry, task_id=task_id, task_name=task_name)
    finalizer = AgentFinalizer(telemetry=telemetry, task_id=task_id, task_name=task_name)

    # Setup workspace from template
    workspace_dir = run_dir / (task_id or f"proof_workspace_idx{proof_idx}")
    initializer.setup_workspace(workspace_dir, template_dir=WORKSPACE_TEMPLATE)

    # Start task (must be before any emit_message calls)
    initializer.start_task(sequence=task_sequence)

    # Create callback for summary aggregation (agent suppresses summaries due to expected_files_struct_out_field)
    callback = telemetry.create_callback(task_id, task_name, group="proof") if telemetry and task_id and task_name else None

    # Log execution info
    if telemetry and task_id and task_name:
        telemetry.emit_message("INFO", f"Executing PROOF: {plan.title}", task_name, task_id)
        telemetry.emit_message("INFO", f"Workspace: {workspace_dir}", task_name, task_id)

    # Get proof executor config
    proof_cfg = config.invention_loop.execute.proof
    agent_cfg = proof_cfg.claude_agent
    model = agent_cfg.model
    max_turns = agent_cfg.max_turns
    timeout_seconds = agent_cfg.seq_prompt_timeout

    # Build plan text from typed fields
    plan_text = plan.to_prompt_yaml()

    workspace_str = str(workspace_dir)

    # Build sequential prompts
    prompts = u_prompt.get_all_prompts(
        plan_text=plan_text,
        artifact_pool=artifact_pool,
        dependency_ids=plan.artifact_dependencies,
        workspace_path=workspace_str,
    )

    # Get verification config
    verify_retries = proof_cfg.verify_retries

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
        system_prompt=get_system(),
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
        output_format=ProofArtifact.to_struct_output(),
        # Expected files validation via structured output (auto-retry on missing)
        expected_files_struct_out_field="out_expected_files",
        max_expected_files_retries=verify_retries,
    )

    try:
        # Create and run agent
        agent = Agent(options)
        all_responses: list = []  # Track for cost aggregation

        if telemetry and task_id and task_name:
            telemetry.emit_message("INFO", "Running Claude agent with Lean tool", task_name, task_id)

        # Run multi-step prompts
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
        # POST-VALIDATION: Schema check (with retry)
        # =================================================================
        expected_files = ProofArtifact.get_expected_out_files()
        schema_max_retries = proof_cfg.schema_retries

        for schema_attempt in range(1, schema_max_retries + 2):
            verification = verify_proof_output(
                workspace_dir=workspace_dir,
                expected_files=expected_files,
            )

            if verification.get("schema_errors"):
                for err in verification["schema_errors"][:3]:
                    if telemetry and task_id and task_name:
                        telemetry.emit_message("WARNING", f"Schema: {err}", task_name, task_id)

            if verification.get("valid", False) or schema_attempt > schema_max_retries:
                break

            retry_prompt = build_proof_retry_prompt(
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
        # Read result from proof_out.json (Claude agent creates this file)
        proof_out_file = workspace_dir / "proof_out.json"
        if proof_out_file.exists():
            try:
                with open(proof_out_file, 'r', encoding='utf-8') as f:
                    proof_result = json.load(f)
                # Ensure required fields exist
                proof_result.setdefault("proof_successful", proof_result.get("verified", False))
                proof_result.setdefault("verified", proof_result.get("proof_successful", False))
            except (json.JSONDecodeError, IOError) as e:
                finalizer.end_task_failure(f"Failed to read proof_out.json: {e}", cost=cost)
                raise RuntimeError(f"Failed to read proof_out.json: {e}") from e
        else:
            finalizer.end_task_failure("proof_out.json not found in workspace", cost=cost)
            raise RuntimeError("proof_out.json not found in workspace")

        # Add metadata
        proof_result["theorem_statement"] = informal_proof_draft
        proof_result["workspace_path"] = str(workspace_dir)
        proof_result["cost_usd"] = cost
        proof_result["lemma_count"] = verification.get("lemma_count", 0)

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
            raise RuntimeError(f"Proof executor produced no structured output for {plan.id} — cannot extract title/summary")

        title = title or plan.title.strip()

        # Determine status based on verification
        verified = proof_result.get("verified", False) or proof_result.get("proof_successful", False)
        status = True if verified else False

        proof_result["summary"] = summary

        if telemetry and task_id and task_name:
            schema_ok = verification.get("valid", False)
            status_str = "✓ VERIFIED" if verified else "✗ NOT VERIFIED"
            schema_note = "" if schema_ok else " (schema issues)"
            telemetry.emit_message(
                "SUCCESS" if (verified and schema_ok) else "WARN",
                f"Proof complete: {status_str}{schema_note}, ${cost:.4f}",
                task_name, task_id
            )
        finalizer.end_task_success(cost=cost)

        return proof_result, status, cost

    except asyncio.TimeoutError:
        finalizer.end_task_timeout(timeout_seconds)
        raise

    except Exception as e:
        finalizer.end_task_error(str(e))
        raise
