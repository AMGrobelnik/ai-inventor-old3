"""Python script → Jupyter notebook demo conversion via Claude Agent.

Converts .py artifact scripts into self-contained Jupyter notebooks
with inlined JSON data and GitHub URL loading for Colab compatibility.

Used for: experiment, dataset, evaluation artifact types.
"""

import asyncio
import json
from pathlib import Path

from aii_lib import Agent, AgentOptions, AgentInitializer, AgentFinalizer, AIITelemetry, MessageType, JSONSink
from aii_lib.agent_backend import aggregate_summaries, ExpectedFile

from aii_pipeline.utils import PipelineConfig, rel_path, get_project_root
from aii_pipeline.prompts.steps._4_gen_paper_repo._1b_artifact_demos.schema_code import CodeDemo

# Import demo generation prompts
from aii_pipeline.prompts.steps._4_gen_paper_repo._1b_artifact_demos.u_prompt_code import (
    get_all_prompts as get_all_notebook_prompts,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._1b_artifact_demos.s_prompt_code import get as get_notebook_sysprompt

from .._utils import get_readable_folder_name, build_github_code_mini_demo_data_url


def github_to_colab_url(github_url: str) -> str:
    """Convert GitHub file URL to Google Colab URL.

    Example:
        github.com/user/repo/blob/main/demo_notebooks/exp/code_demo.ipynb
        -> colab.research.google.com/github/user/repo/blob/main/demo_notebooks/exp/code_demo.ipynb
    """
    if "github.com" in github_url:
        return github_url.replace("github.com", "colab.research.google.com/github")
    return github_url


async def convert_to_notebook(
    config: PipelineConfig,
    artifact_id: str,
    artifact_name: str,
    artifact_type: str,
    artifact,
    output_dir: Path,
    telemetry: AIITelemetry,
    task_sequence: int = 0,
    repo_url: str | None = None,
) -> Path | None:
    """Convert Python code to self-contained Jupyter notebook using Claude Agent.

    The agent reads source files from the artifact's workspace_path (in artifact_info)
    and writes output to its own workspace_dir.

    Args:
        config: Pipeline configuration
        artifact_id: Artifact identifier
        artifact_name: Name of the main script (e.g., "method.py") from demo_files
        artifact_type: Type of artifact (experiment, evaluation, dataset, etc.)
        artifact: Artifact object (BaseArtifact) with workspace_path for source files
        output_dir: Output directory
        telemetry: AIITelemetry instance
        task_sequence: Sequence number for parallel task ordering
        repo_url: GitHub repo URL for constructing raw data URLs (for Colab compatibility)

    Returns:
        Path to the created notebook, or None on failure.
    """
    task_id = f"demo_{artifact_id}"
    task_name = task_id

    # Create initializer and finalizer
    initializer = AgentInitializer(telemetry=telemetry, task_id=task_id, task_name=task_name)
    finalizer = AgentFinalizer(telemetry=telemetry, task_id=task_id, task_name=task_name)

    # Create callback for group aggregation
    callback = telemetry.create_callback(task_id, task_name, group="b_gen_artifact_demos")

    # Setup workspace
    workspace_dir = output_dir / "notebook_workspaces" / artifact_id
    initializer.setup_workspace(workspace_dir)

    # Start task with sequence for parallel buffering
    initializer.start_task(sequence=task_sequence)

    try:
        # Get agent config
        agent_cfg = config.gen_paper_repo.gen_artifact_demos.claude_agent

        options = AgentOptions(
            model=agent_cfg.model,
            max_turns=agent_cfg.max_turns,
            agent_timeout=agent_cfg.agent_timeout,
            agent_retries=agent_cfg.agent_retries,
            seq_prompt_timeout=agent_cfg.seq_prompt_timeout,
            seq_prompt_retries=agent_cfg.seq_prompt_retries,
            message_timeout=agent_cfg.message_timeout,
            message_retries=agent_cfg.message_retries,
            cwd=str(workspace_dir),
            system_prompt=get_notebook_sysprompt(),
            continue_seq_item=True,  # Continue conversation between prompts
            mcp_servers=str(get_project_root() / ".mcp.json"),  # ToolUniverse MCP
            # AIITelemetry integration
            telemetry=telemetry,
            run_id=task_id,
            agent_context=task_name,
            # SDK native structured output for title/summary
            output_format=CodeDemo.to_struct_output(),
            # Expected files validation via structured output (auto-retry on missing)
            expected_files_struct_out_field="out_expected_files",
            max_expected_files_retries=2,
        )

        # Get expected output files for this artifact type to include in prompt
        from .._1b_gen_artifact_demos import get_expected_out_files_for_type
        available_files = get_expected_out_files_for_type(artifact_type)

        # Compute folder name (uses sanitized title, same as _2b_deploy_to_repo)
        folder_name = get_readable_folder_name(artifact_id, artifact.title)
        github_code_mini_demo_data_url = build_github_code_mini_demo_data_url(repo_url, folder_name) if repo_url else None

        # Generate prompts — agent reads source files from artifact's workspace_path (in artifact_info), writes to workspace_dir
        demos_cfg = config.gen_paper_repo.gen_artifact_demos
        prompts = get_all_notebook_prompts(
            artifact_name=artifact_name,
            artifact=artifact,
            available_files=available_files,
            repo_url=repo_url,
            github_code_mini_demo_data_url=github_code_mini_demo_data_url,
            workspace_path=str(workspace_dir),
            max_notebook_total_runtime=demos_cfg.max_notebook_total_runtime,
        )

        # Run agent
        agent = Agent(options)
        result = await agent.run(prompts)

        cost = result.total_cost

        # Emit ONE aggregated summary (not individual prompt summaries)
        if result.prompt_results:
            agg = aggregate_summaries(result.prompt_results)
            if agg:
                callback(agg)


        if result.failed:
            err = result.error_message or "unknown error"
            telemetry.emit_message("ERROR", f"Demo agent failed: {err}", task_name, task_id)
            finalizer.end_task_failure(f"Agent failed: {err}", cost=cost)
            return None

        # Check if notebook was created
        notebook_path = workspace_dir / "code_demo.ipynb"
        if not notebook_path.exists():
            # Check for alternative names
            for nb_file in workspace_dir.glob("*.ipynb"):
                notebook_path = nb_file
                break

        # Extract title/summary/expected_files from agent structured output
        demo_title = ""
        demo_summary = ""
        demo_expected_files = {}
        if result.structured_output and isinstance(result.structured_output, dict):
            demo_title = result.structured_output.get("title", "")
            demo_summary = result.structured_output.get("summary", "")
            demo_expected_files = result.structured_output.get("out_expected_files", {})

        if notebook_path and notebook_path.exists():
            # Verify notebook contains the EXACT GitHub URL for Colab compatibility
            github_url_ok = False

            if github_code_mini_demo_data_url:
                try:
                    nb_content = notebook_path.read_text()
                    if github_code_mini_demo_data_url in nb_content:
                        github_url_ok = True
                        telemetry.emit_message("INFO", f"GitHub URL verified: {github_code_mini_demo_data_url}", task_name, task_id)
                    else:
                        telemetry.emit_message("WARNING", f"Notebook missing exact GitHub URL: {github_code_mini_demo_data_url}", task_name, task_id)

                        # Retry: send a follow-up prompt to fix the URL
                        telemetry.emit_message("INFO", "Sending retry prompt to fix GitHub URL...", task_name, task_id)
                        fix_prompt = f"""CRITICAL FIX REQUIRED:

The notebook code_demo.ipynb is missing the correct GitHub URL for Colab compatibility.

The notebook MUST contain this EXACT URL:
GITHUB_DATA_URL = "{github_code_mini_demo_data_url}"

Please edit code_demo.ipynb and ensure the data loading cell contains this URL.
This is required for the notebook to work in Google Colab after deployment."""

                        fix_result = await agent.run([fix_prompt])
                        cost += fix_result.total_cost

                        # Re-check after fix
                        nb_content = notebook_path.read_text()
                        if github_code_mini_demo_data_url in nb_content:
                            github_url_ok = True
                            telemetry.emit_message("SUCCESS", "GitHub URL fixed after retry", task_name, task_id)
                        else:
                            telemetry.emit_message("ERROR", "GitHub URL still missing after retry!", task_name, task_id)

                except Exception as e:
                    telemetry.emit_message("WARNING", f"Could not verify GitHub URL: {e}", task_name, task_id)

            status_msg = f"Notebook created (${cost:.4f})"
            if repo_url and github_url_ok:
                status_msg += " [GitHub URL ✓]"
            elif repo_url:
                status_msg += " [GitHub URL MISSING]"

            telemetry.emit_message("SUCCESS", status_msg, task_name, task_id)
            finalizer.end_task_success(cost=cost)
            return notebook_path, demo_title, demo_summary, demo_expected_files

        telemetry.emit_message("ERROR", "Notebook not created", task_name, task_id)
        finalizer.end_task_failure("No output", cost=cost)
        raise RuntimeError("Demo notebook was not created by the agent")

    except asyncio.TimeoutError:
        telemetry.emit_message("ERROR", f"Timeout after {agent_cfg.agent_timeout}s", task_name, task_id)
        finalizer.end_task_timeout(agent_cfg.agent_timeout)
        raise

    except Exception as e:
        telemetry.emit_message("ERROR", f"Exception: {e}", task_name, task_id)
        finalizer.end_task_error(str(e))
        raise
