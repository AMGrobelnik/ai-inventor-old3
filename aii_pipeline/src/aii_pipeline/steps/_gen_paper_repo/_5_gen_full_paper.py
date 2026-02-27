"""FINAL_GEN_PAPER Step - Generate LaTeX paper and PDF using Claude Agent.

Uses Claude Agent to:
1. Create paper.tex from paper text content
2. Insert figures at appropriate locations
3. Compile to PDF using pdflatex
4. Push paper and figures to GitHub repository

This is the final paper generation step that produces the deliverable outputs.
"""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path

from aii_lib import Agent, AgentOptions, AgentInitializer, AgentFinalizer, AIITelemetry, MessageType, JSONSink, get_model_short, get_tooluniverse_mcp_config
from aii_lib.agent_backend import aggregate_summaries

from aii_pipeline.utils import PipelineConfig, rel_path, get_project_root
from aii_pipeline.prompts.steps._4_gen_paper_repo._2a_viz_gen.schema import Figure
from aii_pipeline.prompts.steps._4_gen_paper_repo._1a_write_papers.schema import PaperText
from aii_pipeline.prompts.steps._4_gen_paper_repo.schema import GistDeployment
from aii_pipeline.prompts.steps._4_gen_paper_repo._5_gen_full_paper.schema import GenPaperRepoOut
from aii_pipeline.prompts.steps._4_gen_paper_repo._5_gen_full_paper.u_prompt import (
    get_latex_filename,
    get_pdf_filename,
    get_figures_folder,
    get as get_latex_user_prompt,
    get_expected_out_files,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._5_gen_full_paper.s_prompt import get as get_latex_system_prompt
from aii_pipeline.prompts.steps._4_gen_paper_repo._5_gen_full_paper.schema import FullPaper
from ._utils_push import push_paper_to_repo


async def generate_paper_with_agent(
    config: PipelineConfig,
    paper: PaperText,
    figures: list[Figure],
    workspace_dir: Path,
    telemetry: AIITelemetry,
) -> tuple[Path | None, Path | None, float]:
    """Generate LaTeX paper and compile to PDF using Claude Agent.

    Args:
        config: Pipeline configuration
        paper: PaperText object with title, abstract, body text, summary
        figures: Figure objects with file paths (for copying + prompt context)
        workspace_dir: Workspace directory for agent
        telemetry: AIITelemetry instance

    Returns:
        Tuple of (tex_path, pdf_path, cost)
    """
    model_short = get_model_short(config.gen_paper_repo.gen_full_paper.claude_agent.model)
    task_id = f"full_paper__{model_short}"
    task_name = task_id

    # Create initializer and finalizer
    initializer = AgentInitializer(telemetry=telemetry, task_id=task_id, task_name=task_name)
    finalizer = AgentFinalizer(telemetry=telemetry, task_id=task_id, task_name=task_name)

    # Create callback for group aggregation
    callback = telemetry.create_callback(task_id, task_name, group="final_gen_paper")

    # Setup workspace
    initializer.setup_workspace(workspace_dir)
    initializer.start_task()

    try:
        # Copy figures to workspace
        figures_dir = workspace_dir / get_figures_folder()
        figures_dir.mkdir(exist_ok=True)

        for fig in figures:
            if fig.figure_path:
                src = Path(fig.figure_path)
                if src.exists():
                    dst = figures_dir / src.name
                    shutil.copy(src, dst)

        # Create copies with workspace-relative paths for the prompt.
        # Do NOT mutate originals — they're shared with figure_pool, result.json, and push.
        prompt_figures = [
            fig.model_copy(update={"figure_path": f"{get_figures_folder()}/{Path(fig.figure_path).name}"})
            if fig.figure_path else fig.model_copy()
            for fig in figures
        ]

        # Get agent config
        agent_cfg = config.gen_paper_repo.gen_full_paper.claude_agent

        # Always connect MCP for aii_web_fetch_grep + dblp tools.
        # When use_aii_web_tools=True: all 3 MCP tools replace built-in WebSearch/WebFetch.
        # When use_aii_web_tools=False: only grep via MCP, built-in WebSearch/WebFetch for the rest.
        mcp_servers = get_tooluniverse_mcp_config(use_aii_server=True)
        if agent_cfg.use_aii_web_tools:
            disallowed_tools = ["WebSearch", "WebFetch"]
        else:
            disallowed_tools = [
                "mcp__aii_tooluniverse__aii_web_search_fast",
                "mcp__aii_tooluniverse__aii_web_fetch_direct",
            ]

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
            system_prompt=get_latex_system_prompt(),
            continue_seq_item=True,  # Continue conversation between prompts
            mcp_servers=mcp_servers,
            disallowed_tools=disallowed_tools,
            # AIITelemetry integration
            telemetry=telemetry,
            run_id=task_id,
            agent_context=task_name,
            # SDK native structured output for title/summary
            output_format=FullPaper.to_struct_output(),
            # Expected files validation via structured output (auto-retry on missing)
            expected_files_struct_out_field="out_expected_files",
            max_expected_files_retries=2,
        )

        # Build prompt (GitHub push is handled by Python code, not agent)
        prompt = get_latex_user_prompt(
            paper=paper,
            figures=prompt_figures,
            workspace_path=str(workspace_dir),
        )

        telemetry.emit_message("INFO", "Starting LaTeX generation and compilation", task_name, task_id)

        # Run agent
        agent = Agent(options)
        result = await agent.run([prompt])

        cost = result.total_cost

        # Emit ONE aggregated summary (not individual prompt summaries)
        if result.prompt_results:
            agg = aggregate_summaries(result.prompt_results)
            if agg:
                callback(agg)


        if result.failed:
            err = result.error_message or "unknown error"
            telemetry.emit_message("ERROR", f"GEN_PAPER agent failed: {err}", task_name, task_id)
            finalizer.end_task_failure(f"Agent failed: {err}", cost=cost)
            raise RuntimeError(f"GEN_PAPER agent failed: {err}")

        # Check output files
        tex_path = workspace_dir / get_latex_filename()
        pdf_path = workspace_dir / get_pdf_filename()

        if pdf_path.exists():
            telemetry.emit_message("SUCCESS", f"PDF generated: {tex_path.name}", task_name, task_id)
            finalizer.end_task_success(cost=cost)
            return tex_path, pdf_path, cost

        elif tex_path.exists():
            telemetry.emit_message("WARNING", "LaTeX created but PDF compilation failed", task_name, task_id)
            finalizer.end_task("Partial", cost=cost)
            return tex_path, None, cost

        else:
            telemetry.emit_message("ERROR", "LaTeX generation failed - no output files", task_name, task_id)
            finalizer.end_task_failure("No output files", cost=cost)
            raise RuntimeError("LaTeX generation produced no output files")

    except asyncio.TimeoutError:
        telemetry.emit_message("ERROR", "GEN_PAPER agent timed out", task_name, task_id)
        finalizer.end_task_timeout(agent_cfg.seq_prompt_timeout)
        raise

    except Exception as e:
        telemetry.emit_message("ERROR", f"GEN_PAPER failed: {e}", task_name, task_id)
        finalizer.end_task_error(str(e))
        raise


async def run_gen_full_paper_module(
    config: PipelineConfig,
    paper: PaperText | None,
    figures: list[Figure],
    gist_deployments: list[GistDeployment] | None = None,
    total_cost: float = 0.0,
    telemetry: AIITelemetry | None = None,
    output_dir: Path | None = None,
    repo_url: str | None = None,
) -> GenPaperRepoOut:
    """
    Run the FINAL_GEN_PAPER step.

    Uses Claude Agent to generate LaTeX from paper text,
    compile to PDF, and push to GitHub.

    Args:
        config: Pipeline configuration
        paper: The paper draft (with text content)
        figures: Figure objects with figure_path set
        gist_deployments: Gist deployment results
        total_cost: Total cost of gen_paper module so far
        telemetry: AIITelemetry instance
        output_dir: Output directory
        repo_url: GitHub repo URL for pushing

    Returns:
        GenPaperRepoOut with final outputs
    """
    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "gen_full_paper_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "gen_full_paper_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.start_module("FINAL_GEN_PAPER")

    try:
        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, "FINAL_GEN_PAPER - Generating LaTeX and PDF with Claude Agent")
        telemetry.emit(MessageType.INFO, "=" * 60)

        if not output_dir:
            output_dir = Path("./gen_paper_output")

        output_dir.mkdir(parents=True, exist_ok=True)
        paper_dir = output_dir / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)

        result = GenPaperRepoOut(
            output_dir=str(output_dir),
            repo_url=repo_url,
            figures=figures,
            gist_deployments=gist_deployments or [],
            metadata={
                "generated_at": datetime.now().isoformat(),
                "module": "gen_paper",
                "total_cost_usd": total_cost,
            }
        )

        if not paper:
            telemetry.emit(MessageType.WARNING, "No paper to process")
            return result

        telemetry.emit(MessageType.INFO, f"   Paper: {paper.id}")
        telemetry.emit(MessageType.INFO, f"   Figures: {len(figures)}")
        if gist_deployments:
            telemetry.emit(MessageType.INFO, f"   Gists: {len(gist_deployments)}")
        if repo_url:
            telemetry.emit(MessageType.INFO, f"   Repo: {repo_url}")

        if not paper.title:
            raise ValueError("Paper has no title — cannot generate paper")

        # Generate paper with Claude Agent (GitHub push handled separately)
        workspace_dir = paper_dir / "workspace"
        tex_path, pdf_path, gen_cost = await generate_paper_with_agent(
            config=config,
            paper=paper,
            figures=figures,
            workspace_dir=workspace_dir,
            telemetry=telemetry,
        )

        total_cost += gen_cost

        # Copy final outputs to paper_dir
        final_tex = None
        final_pdf = None

        if tex_path and tex_path.exists():
            final_tex = paper_dir / get_latex_filename()
            shutil.copy(tex_path, final_tex)
            telemetry.emit(MessageType.INFO, f"   LaTeX: {rel_path(final_tex)}")

        if pdf_path and pdf_path.exists():
            final_pdf = paper_dir / get_pdf_filename()
            shutil.copy(pdf_path, final_pdf)
            telemetry.emit(MessageType.INFO, f"   PDF: {rel_path(final_pdf)}")

        # Copy references.bib if it exists
        final_bib = None
        bib_path = workspace_dir / "references.bib"
        if bib_path.exists():
            final_bib = paper_dir / "references.bib"
            shutil.copy(bib_path, final_bib)
            telemetry.emit(MessageType.INFO, f"   Bibliography: {rel_path(final_bib)}")

        # Copy figures to paper_dir from gen_viz output directory.
        # Uses the known absolute location instead of fig.figure_path which may
        # have been changed to relative paths during agent prompt construction.
        figures_out = paper_dir / get_figures_folder()
        figures_out.mkdir(exist_ok=True)
        gen_viz_figures_dir = output_dir / "figures"
        if gen_viz_figures_dir.exists():
            for fig_file in gen_viz_figures_dir.iterdir():
                if fig_file.is_file():
                    shutil.copy(fig_file, figures_out / fig_file.name)

        # Push paper and figures to GitHub (with proper README update)
        figs_with_path = [f for f in figures if f.figure_path and Path(f.figure_path).exists()]
        telemetry.emit(MessageType.INFO, f"   Push check: repo_url={bool(repo_url)}, tex={bool(final_tex)}, pdf={bool(final_pdf)}")
        telemetry.emit(MessageType.INFO, f"   Figures: {len(figures)} total, {len(figs_with_path)} with valid paths")

        if repo_url and (final_tex or final_pdf):
            push_success = push_paper_to_repo(
                repo_url=repo_url,
                paper_tex_path=final_tex,
                paper_pdf_path=final_pdf if final_pdf and final_pdf.exists() else None,
                paper_bib_path=final_bib,
                figures=figures,
                telemetry=telemetry,
                workspace_dir=workspace_dir,
            )
            if push_success:
                telemetry.emit(MessageType.SUCCESS, f"   Paper pushed to GitHub: {repo_url}")
                # Emit as SYSTEM so it appears in JSONL logs (INFO/SUCCESS are filtered by JSONSink)
                telemetry.emit(MessageType.SYSTEM, f"Paper pushed to GitHub: {repo_url}")
            else:
                telemetry.emit(MessageType.ERROR, "   Paper/figures push to repo FAILED - check logs above")
                telemetry.emit(MessageType.SYSTEM, f"Paper push FAILED for repo: {repo_url}")
        else:
            # Log why push was skipped
            skip_reasons = []
            if not repo_url:
                skip_reasons.append("no repo_url (Track B may not have run)")
            if not final_tex and not final_pdf:
                skip_reasons.append("no tex/pdf files generated")
            telemetry.emit(MessageType.WARNING, f"   Skipping push to repo: {', '.join(skip_reasons)}")
            telemetry.emit(MessageType.SYSTEM, f"Paper push skipped: {', '.join(skip_reasons)}")

        # Update result
        result.paper = paper
        result.metadata["total_cost_usd"] = total_cost
        result.metadata["paper_tex"] = str(final_tex) if final_tex else None
        result.metadata["paper_pdf"] = str(final_pdf) if final_pdf and final_pdf.exists() else None
        result.metadata["repo_url"] = repo_url
        result.metadata["expected_files"] = [f.path for f in get_expected_out_files()]
        result.metadata["llm_provider"] = "claude_agent"
        result.metadata["output_dir"] = str(output_dir) if output_dir else None

        # Save final result (filename must match STEP_OUTPUT_FILES["assemble"] in _4_gen_paper_repo.py)
        result_file = output_dir / "gen_paper_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
        telemetry.emit(MessageType.INFO, f"   Saved result: {rel_path(result_file)}")

        if final_pdf and final_pdf.exists():
            telemetry.emit(MessageType.SUCCESS, f"FINAL_GEN_PAPER complete: {rel_path(final_pdf)}")
        elif final_tex and final_tex.exists():
            telemetry.emit(MessageType.SUCCESS, f"FINAL_GEN_PAPER complete (LaTeX only): {rel_path(final_tex)}")
        else:
            telemetry.emit(MessageType.WARNING, "FINAL_GEN_PAPER complete (no outputs)")

        telemetry.emit_module_summary("FINAL_GEN_PAPER")

        return result

    finally:
        # Always remove module sinks to prevent leak into subsequent modules
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
