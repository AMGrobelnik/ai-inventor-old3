"""Gen Paper Module - Post-loop paper generation.

Takes narrative + artifacts from discovery loop, produces:
- GitHub repo with code artifacts
- Prepared artifacts (notebooks, markdown, Lean playground links)
- Visualizations (image generation)
- Paper draft compiled to LaTeX/PDF

SEQUENTIAL EXECUTION (with concurrent sub-steps):

    Step 1:  create_repo              (quick setup, no LLM)
    Step 2a: write_paper_text  ─┐
    Step 2b: gen_artifact_demos ─┘ concurrent
    Step 3:  gen_viz
    Step 4:  gen_full_paper           (combines paper + figures into LaTeX)
    Step 5:  deploy_to_repo           (push everything to GitHub)

Steps 2a and 2b run concurrently. All other groups run sequentially.
Each step has its own internal parallelism (e.g., gen_demos runs multiple
agents concurrently, gen_viz generates multiple images concurrently).

Uses aii_lib for:
- AIITelemetry: Centralized task tracking and logging
- chat: Simple LLM calls
"""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path

from aii_lib import create_telemetry, AIITelemetry, MessageType, JSONSink

from aii_pipeline.utils import PipelineConfig, rel_path, build_module_output, emit_module_output
from ._invention_loop.pools import load_all_pools

# Step imports
from ._gen_paper_repo._0b_create_repo import run_create_repo_module
from ._gen_paper_repo._1a_write_paper_text import run_write_paper_text_module
from ._gen_paper_repo._1b_gen_artifact_demos import run_gen_artifact_demos_module
from ._gen_paper_repo._2a_gen_viz import run_gen_viz_module
from ._gen_paper_repo._2b_deploy_to_repo import run_deploy_to_repo_module
from ._gen_paper_repo._5_gen_full_paper import run_gen_full_paper_module

from aii_pipeline.prompts.steps._4_gen_paper_repo._2a_viz_gen.schema import Figure
from aii_pipeline.prompts.steps._4_gen_paper_repo._1a_write_papers.schema import PaperText, parse_figures_from_xml
from aii_pipeline.prompts.steps._4_gen_paper_repo.schema import GistDeployment
from aii_pipeline.prompts.steps._4_gen_paper_repo._5_gen_full_paper.schema import GenPaperRepoOut
from aii_pipeline.prompts.steps._4_gen_paper_repo._1b_artifact_demos.schema_code import BaseDemo, AnyDemo
from aii_pipeline.prompts.steps._3_invention_loop.schema import InventionLoopOut
from ._gen_paper_repo.pools import FigurePool, DemoPool


# Step execution order:
# Step 1:  create_repo
# Step 2a: write_paper  ─┐ concurrent
# Step 2b: gen_demos    ─┘
# Step 3:  gen_viz
# Step 4:  gen_full_paper
# Step 5:  deploy_repo (push everything to GitHub)
GEN_PAPER_STEPS = ["create_repo", "write_paper", "gen_demos", "gen_viz", "gen_full_paper", "deploy_repo"]

# Map step names to output files
STEP_OUTPUT_FILES = {
    "write_paper": "write_paper_text_results.json",
    "gen_viz": "gen_viz_results.json",
    "create_repo": "repo_info.json",
    "gen_demos": "prepared_artifacts.json",
    "deploy_repo": "gist_deployments.json",
    "gen_full_paper": "gen_paper_result.json",
}


def detect_last_step(resume_dir: Path) -> str | None:
    """Detect the last completed step from a resume directory."""
    last = None
    for step in GEN_PAPER_STEPS:
        output_file = STEP_OUTPUT_FILES.get(step)
        if output_file and (resume_dir / output_file).exists():
            last = step
    return last


def copy_gen_paper_results(resume_dir: Path, output_dir: Path) -> dict:
    """Copy intermediate result files from resume_dir to output_dir."""
    copied = []
    for step, output_file in STEP_OUTPUT_FILES.items():
        src = resume_dir / output_file
        if src.exists():
            dst = output_dir / output_file
            shutil.copy2(src, dst)
            copied.append(output_file)
    return {"files": copied, "count": len(copied)}


def load_step_result(output_dir: Path, step: str) -> dict | list | None:
    """Load the result of a completed step from its output file."""
    output_file = STEP_OUTPUT_FILES.get(step)
    if not output_file:
        return None
    file_path = output_dir / output_file
    if not file_path.exists():
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def run_gen_paper_module(
    config: PipelineConfig,
    invention_loop_result: "InventionLoopOut",
    run_dir: Path | None = None,
    workspace_dir: Path | None = None,
    telemetry: AIITelemetry | None = None,
    cumulative: dict | None = None,
) -> GenPaperRepoOut:
    """
    Run the full gen_paper module with SEQUENTIAL execution.

    Steps run in order, with concurrent sub-steps within each group:
      1:  create_repo
      2a: write_paper  ─┐ concurrent
      2b: gen_demos    ─┘
      3:  gen_viz
      4:  gen_full_paper
      5:  deploy_repo

    Args:
        config: Pipeline configuration
        invention_loop_result: Result from invention_loop module
        run_dir: Run output directory
        workspace_dir: Workspace directory
        telemetry: AIITelemetry instance (optional, creates local if None)
        cumulative: Cumulative cost tracking dict

    Returns:
        GenPaperRepoOut with all outputs
    """
    # Create output directory
    if run_dir:
        output_dir = run_dir / "4_gen_paper_repo"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = config.init.outputs_directory
        output_dir = Path(f"{output_base}/{timestamp}_gen_paper")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up workspace
    if not workspace_dir:
        workspace_dir = output_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Extract data from invention loop result
    narrative = invention_loop_result.narrative
    hypothesis = invention_loop_result.hypothesis

    # Load pools from invention loop output
    pools_dir = Path(invention_loop_result.pools_dir)
    if not pools_dir.exists():
        raise RuntimeError(f"Pools directory not found: {pools_dir}. Cannot load artifact data.")
    strategy_pool, plan_pool, artifact_pool, narrative_pool = load_all_pools(pools_dir)
    all_artifacts = artifact_pool.get_all()

    # Create gen_paper_repo pools
    figure_pool = FigurePool()
    demo_pool = DemoPool()

    # =========================================================================
    # STEP CONFIGURATION
    # =========================================================================
    pipeline_cfg = config.init.pipeline

    resume_dir = Path(pipeline_cfg.gen_paper_resume_dir) if pipeline_cfg.gen_paper_resume_dir else None
    first_step = pipeline_cfg.gen_paper_first_step
    last_step = pipeline_cfg.gen_paper_last_step
    run_gen_full_paper = pipeline_cfg.gen_paper_run_gen_full_paper

    # Validate step names
    if first_step and first_step not in GEN_PAPER_STEPS:
        raise ValueError(f"Invalid gen_paper_first_step: {first_step}. Must be one of: {GEN_PAPER_STEPS}")
    if last_step and last_step not in GEN_PAPER_STEPS:
        raise ValueError(f"Invalid gen_paper_last_step: {last_step}. Must be one of: {GEN_PAPER_STEPS}")

    # Validate first_step requires resume_dir
    if first_step and first_step != GEN_PAPER_STEPS[0]:
        if not resume_dir:
            raise ValueError(
                f"gen_paper_first_step={first_step} requires gen_paper_resume_dir to be set. "
                f"Steps before '{first_step}' will be skipped and their outputs must be loaded from resume_dir."
            )
        if not resume_dir.exists():
            raise ValueError(f"gen_paper_resume_dir does not exist: {resume_dir}")

    # =========================================================================
    # RESUME HANDLING
    # =========================================================================
    resumed_from = None
    if resume_dir and resume_dir.exists():
        copied = copy_gen_paper_results(resume_dir, output_dir)
        last_completed = detect_last_step(resume_dir)
        resumed_from = {
            "dir": str(resume_dir),
            "copied": copied,
            "last_completed": last_completed,
        }

    # Skip/stop helpers
    # Steps 2a (write_paper) and 2b (gen_demos) are a concurrent group —
    # treat them as a single unit for skip/stop: both share the index of whichever is earlier.
    CONCURRENT_GROUP = {"write_paper", "gen_demos"}

    def _effective_index(step: str) -> int:
        """Get step index, treating concurrent group as one unit."""
        if step in CONCURRENT_GROUP:
            return min(GEN_PAPER_STEPS.index(s) for s in CONCURRENT_GROUP)
        return GEN_PAPER_STEPS.index(step)

    def should_skip(step: str) -> bool:
        """Skip if step is before first_step."""
        if not first_step:
            return False
        return _effective_index(step) < _effective_index(first_step)

    def should_stop(step: str) -> bool:
        """Stop after this step if it matches last_step."""
        if not last_step:
            return False
        # If last_step is in the concurrent group, stop after the whole group
        if last_step in CONCURRENT_GROUP and step in CONCURRENT_GROUP:
            return True
        return last_step == step

    # Use provided telemetry or create local one
    local_telemetry = telemetry is None
    if local_telemetry:
        telemetry = create_telemetry(output_dir, "gen_paper")

    # Add module-specific JSON sink
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "gen_paper_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "gen_paper_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.start_module_group("GEN_PAPER")

    try:
        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, "GEN_PAPER - Post-Loop Paper Generation")
        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, f"   Hypothesis: {hypothesis.get('title', 'N/A')[:50]}")
        telemetry.emit(MessageType.INFO, f"   Artifacts: {len(all_artifacts)}")
        telemetry.emit(MessageType.INFO, f"   Narrative: {narrative.id if narrative else 'N/A'}")
        telemetry.emit(MessageType.INFO, f"   Output: {rel_path(output_dir)}")

        # Log step configuration
        step_range = f"{first_step or 'create_repo'} → {last_step or 'gen_full_paper'}"
        telemetry.emit(MessageType.INFO, f"   Steps: {step_range}")
        if resumed_from:
            telemetry.emit(MessageType.INFO, f"   Resume: {resumed_from['dir']}")
            telemetry.emit(MessageType.INFO, f"   Last completed: {resumed_from['last_completed'] or 'none'}")
        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, "")

        # Initialize result variables
        repo_url = None
        paper_texts = []
        paper = None
        figures: list[Figure] = []
        prepared_artifacts = []
        gist_deployments = []

        # =============================================================
        # STEP 1: CREATE_REPO (quick setup, no LLM)
        # =============================================================
        if should_skip("create_repo"):
            telemetry.emit(MessageType.INFO, "Step 1: CREATE_REPO [SKIPPED - loading from resume]")
            loaded = load_step_result(output_dir, "create_repo")
            if loaded:
                repo_url = loaded.get("repo_url")
                telemetry.emit(MessageType.INFO, f"   Loaded repo_url: {repo_url}")
            else:
                raise RuntimeError(
                    f"create_repo skipped but '{STEP_OUTPUT_FILES['create_repo']}' not found in resume_dir."
                )
        else:
            repo_info = await run_create_repo_module(
                config=config,
                hypothesis=hypothesis,
                telemetry=telemetry,
                output_dir=output_dir,
            )
            repo_url = repo_info.get("repo_url") if repo_info else None

        if should_stop("create_repo"):
            telemetry.emit(MessageType.INFO, "\n[STOPPING after create_repo as configured]")
            return _build_partial_result(
                output_dir=output_dir, repo_url=repo_url,
                telemetry=telemetry, figure_pool=figure_pool, demo_pool=demo_pool,
                stop_reason="create_repo",
            )

        # =============================================================
        # STEP 2a + 2b: WRITE_PAPER + GEN_DEMOS [concurrent]
        # =============================================================
        telemetry.emit(MessageType.INFO, "\n--- Step 2a: write_paper + Step 2b: gen_demos ---")

        async def step_2a_write_paper():
            """Step 2a: Write paper text with viz placeholders."""
            if should_skip("write_paper"):
                telemetry.emit(MessageType.INFO, "Step 2a: WRITE_PAPER [SKIPPED - loading from resume]")
                loaded = load_step_result(output_dir, "write_paper")
                if loaded:
                    items = loaded.get("paper_texts", []) if isinstance(loaded, dict) else loaded
                    paper_texts = [PaperText(**d) if isinstance(d, dict) else d for d in items]
                    telemetry.emit(MessageType.INFO, f"   Loaded {len(paper_texts)} paper texts")
                    return paper_texts
                raise RuntimeError(
                    f"write_paper skipped but '{STEP_OUTPUT_FILES['write_paper']}' not found in resume_dir."
                )
            return await run_write_paper_text_module(
                config=config,
                narrative_pool=narrative_pool,
                artifact_pool=artifact_pool,
                telemetry=telemetry,
                output_dir=output_dir,
                valid_artifact_ids={a.id for a in all_artifacts},
                hypothesis=hypothesis,
            )

        async def step_2b_gen_demos():
            """Step 2b: Generate artifact demos (notebooks, markdown, Lean)."""
            if should_skip("gen_demos"):
                telemetry.emit(MessageType.INFO, "Step 2b: GEN_DEMOS [SKIPPED - loading from resume]")
                loaded = load_step_result(output_dir, "gen_demos")
                if loaded:
                    items = loaded.get("prepared", []) if isinstance(loaded, dict) else loaded
                    from pydantic import TypeAdapter
                    demo_adapter = TypeAdapter(AnyDemo)
                    demos = [demo_adapter.validate_python(d) if isinstance(d, dict) else d for d in items]
                    telemetry.emit(MessageType.INFO, f"   Loaded {len(demos)} prepared artifacts")
                    return demos
                raise RuntimeError(
                    f"gen_demos skipped but '{STEP_OUTPUT_FILES['gen_demos']}' not found in resume_dir."
                )
            return await run_gen_artifact_demos_module(
                config=config,
                artifacts=all_artifacts,
                telemetry=telemetry,
                output_dir=output_dir,
                repo_url=repo_url,
            )

        # Run 2a and 2b concurrently
        step2_results = await asyncio.gather(
            step_2a_write_paper(),
            step_2b_gen_demos(),
            return_exceptions=True,
        )

        # Emit module summaries after BOTH steps complete (avoids interleaving)
        telemetry.emit_module_summary("A_WRITE_PAPER_TEXT")
        telemetry.emit_module_summary("B_GEN_ARTIFACT_DEMOS")

        # Unpack results (handle exceptions)
        if isinstance(step2_results[0], Exception):
            telemetry.emit(MessageType.ERROR, f"2a write_paper failed: {step2_results[0]}")
            paper_texts = []
        else:
            paper_texts = step2_results[0]

        if isinstance(step2_results[1], Exception):
            telemetry.emit(MessageType.ERROR, f"2b gen_demos failed: {step2_results[1]}")
            prepared_artifacts = []
        else:
            prepared_artifacts = step2_results[1]

        # Add demos to pool
        if prepared_artifacts:
            for demo in prepared_artifacts:
                demo_pool.add(demo)

        # Use first paper text
        paper = paper_texts[0] if paper_texts else None
        if paper:
            telemetry.emit(MessageType.INFO, f"\n   Using paper text: {paper.id}")

        if should_stop("write_paper") or should_stop("gen_demos"):
            stop_step = "write_paper" if should_stop("write_paper") else "gen_demos"
            telemetry.emit(MessageType.INFO, f"\n[STOPPING after {stop_step} as configured]")
            return _build_partial_result(
                output_dir=output_dir, repo_url=repo_url, paper=paper,
                figures=[], gist_deployments=[], prepared_artifacts=prepared_artifacts,
                telemetry=telemetry, figure_pool=figure_pool, demo_pool=demo_pool,
                stop_reason=stop_step,
            )

        # =============================================================
        # STEP 3: GEN_VIZ
        # =============================================================
        telemetry.emit(MessageType.INFO, "\n--- Step 3: gen_viz ---")

        # Parse figure specs from paper text XML (needed for gen_viz)
        paper_figures = []
        if paper:
            paper_figures = parse_figures_from_xml(paper.model_dump())
            telemetry.emit(MessageType.INFO, f"   Parsed {len(paper_figures)} figure specs from paper text")

        if should_skip("gen_viz"):
            telemetry.emit(MessageType.INFO, "Step 3: GEN_VIZ [SKIPPED - loading from resume]")
            loaded = load_step_result(output_dir, "gen_viz")
            if loaded:
                loaded_figs = loaded.get("figures", []) if isinstance(loaded, dict) else loaded
                figures = [Figure(**d) if isinstance(d, dict) else d for d in loaded_figs]
                telemetry.emit(MessageType.INFO, f"   Loaded {len(figures)} figures")
            else:
                raise RuntimeError(
                    f"gen_viz skipped but '{STEP_OUTPUT_FILES['gen_viz']}' not found in resume_dir."
                )
        else:
            figures = await run_gen_viz_module(
                config=config,
                figures=paper_figures,
                telemetry=telemetry,
                output_dir=output_dir,
            )

        # Add figures to pool
        if figures:
            figure_pool.add_many(figures)

        if should_stop("gen_viz"):
            telemetry.emit(MessageType.INFO, "\n[STOPPING after gen_viz as configured]")
            return _build_partial_result(
                output_dir=output_dir, repo_url=repo_url, paper=paper,
                figures=figures, prepared_artifacts=prepared_artifacts,
                telemetry=telemetry, figure_pool=figure_pool, demo_pool=demo_pool,
                stop_reason="gen_viz",
            )

        # =============================================================
        # STEP 4: GEN_FULL_PAPER
        # =============================================================
        if should_skip("gen_full_paper"):
            telemetry.emit(MessageType.INFO, "Step 4: GEN_FULL_PAPER [SKIPPED - loading from resume]")
            loaded = load_step_result(output_dir, "gen_full_paper")
            if loaded:
                result = GenPaperRepoOut(**loaded)
                telemetry.emit(MessageType.INFO, f"   Loaded gen_full_paper result")
            else:
                raise RuntimeError(
                    f"gen_full_paper skipped but '{STEP_OUTPUT_FILES['gen_full_paper']}' not found in resume_dir."
                )
        elif not run_gen_full_paper:
            telemetry.emit(MessageType.INFO, "\nStep 4: GEN_FULL_PAPER [SKIPPED - disabled in config]")
            return _build_partial_result(
                output_dir=output_dir, repo_url=repo_url, paper=paper,
                figures=figures, prepared_artifacts=prepared_artifacts,
                telemetry=telemetry, figure_pool=figure_pool, demo_pool=demo_pool,
                stop_reason="gen_full_paper disabled",
            )
        elif not paper:
            telemetry.emit(MessageType.WARNING, "\nStep 4: GEN_FULL_PAPER [SKIPPED - no paper draft available]")
            return _build_partial_result(
                output_dir=output_dir, repo_url=repo_url, paper=paper,
                figures=figures, prepared_artifacts=prepared_artifacts,
                telemetry=telemetry, figure_pool=figure_pool, demo_pool=demo_pool,
                stop_reason="no paper draft",
            )
        else:
            telemetry.emit(MessageType.INFO, "\n--- Step 4: gen_full_paper ---")

            # Check if already done (resume from mid-step)
            gen_full_paper_output = load_step_result(output_dir, "gen_full_paper")
            if gen_full_paper_output and resumed_from:
                telemetry.emit(MessageType.INFO, "Step 4: GEN_FULL_PAPER [SKIPPED - already completed]")
                result = GenPaperRepoOut(**gen_full_paper_output)
            else:
                accumulated_cost = sum(s.get("total_cost", 0.0) or 0.0 for s in telemetry._module_summaries)
                result = await run_gen_full_paper_module(
                    config=config,
                    paper=paper,
                    figures=figures,
                    gist_deployments=gist_deployments,
                    total_cost=accumulated_cost,
                    telemetry=telemetry,
                    output_dir=output_dir,
                    repo_url=repo_url,
                )

        if should_stop("gen_full_paper"):
            telemetry.emit(MessageType.INFO, "\n[STOPPING after gen_full_paper as configured]")
            return _build_partial_result(
                output_dir=output_dir, repo_url=repo_url, paper=paper,
                figures=figures, gist_deployments=gist_deployments,
                prepared_artifacts=prepared_artifacts,
                telemetry=telemetry, figure_pool=figure_pool, demo_pool=demo_pool,
                stop_reason="gen_full_paper",
            )

        # =============================================================
        # STEP 5: DEPLOY_TO_REPO (push everything to GitHub)
        # =============================================================
        telemetry.emit(MessageType.INFO, "\n--- Step 5: deploy_to_repo ---")

        if should_skip("deploy_repo"):
            telemetry.emit(MessageType.INFO, "Step 5: DEPLOY_REPO [SKIPPED - loading from resume]")
            loaded = load_step_result(output_dir, "deploy_repo")
            if loaded:
                items = loaded.get("deployments", []) if isinstance(loaded, dict) else loaded
                gist_deployments = [GistDeployment(**d) if isinstance(d, dict) else d for d in items]
                telemetry.emit(MessageType.INFO, f"   Loaded {len(gist_deployments)} gist deployments")
            else:
                raise RuntimeError(
                    f"deploy_repo skipped but '{STEP_OUTPUT_FILES['deploy_repo']}' not found in resume_dir."
                )
        else:
            # Extract paper paths from gen_full_paper result for README generation
            paper_pdf_path = Path(result.metadata["paper_pdf"]) if result.metadata.get("paper_pdf") else None
            paper_latex_dir = paper_pdf_path.parent if paper_pdf_path else None

            # Diagnostic: log paper path extraction for debugging README paper section
            telemetry.emit(MessageType.INFO, f"   deploy_to_repo: paper_pdf from metadata = {result.metadata.get('paper_pdf')!r}")
            telemetry.emit(MessageType.INFO, f"   deploy_to_repo: paper_pdf_path = {paper_pdf_path} (exists={paper_pdf_path.exists() if paper_pdf_path else 'N/A'})")

            gist_deployments = await run_deploy_to_repo_module(
                config=config,
                artifacts=all_artifacts,
                prepared_artifacts=prepared_artifacts,
                telemetry=telemetry,
                output_dir=output_dir,
                repo_url=repo_url,
                paper_pdf_path=paper_pdf_path,
                paper_latex_dir=paper_latex_dir,
                figures=figures,
            )

        # Propagate gist deployments into final result
        result.gist_deployments = gist_deployments

        # =============================================================
        # FINALIZE
        # =============================================================
        telemetry.emit(MessageType.INFO, "")
        telemetry.emit(MessageType.SUCCESS, "=" * 60)
        telemetry.emit(MessageType.SUCCESS, "GEN_PAPER Complete")
        telemetry.emit(MessageType.SUCCESS, "=" * 60)
        telemetry.emit(MessageType.INFO, f"   Output: {rel_path(output_dir)}")
        telemetry.emit(MessageType.INFO, f"   Paper: {result.metadata.get('paper_pdf', result.metadata.get('paper_tex', 'N/A'))}")
        telemetry.emit(MessageType.INFO, f"   Figures: {len(figures)}")
        if result.gist_deployments:
            telemetry.emit(MessageType.INFO, f"   Gists: {len(result.gist_deployments)}")
        if repo_url:
            telemetry.emit(MessageType.SUCCESS, f"   GitHub Repo: {repo_url}")
        telemetry.emit(MessageType.INFO, f"   Cost: ${result.metadata.get('total_cost_usd', 0.0):.4f}")

        # Emit as SYSTEM so it appears in JSONL logs
        if repo_url:
            telemetry.emit(MessageType.SYSTEM, f"GitHub Repository: {repo_url}")

        # Emit standardized module output
        std_output = build_module_output(
            module="gen_paper_repo",
            outputs=[result],
            cumulative=cumulative or {},
            output_dir=output_dir,
            total_cost_usd=result.metadata.get("total_cost_usd", 0.0),
        )
        emit_module_output(std_output, telemetry, output_dir=output_dir)

        telemetry.emit_module_group_summary("GEN_PAPER")

        # Save pipeline summary
        pipeline_summary = {
            'completed_at': datetime.now().isoformat(),
            'run_dir': str(run_dir) if run_dir else None,
            'workspace_dir': str(workspace_dir) if workspace_dir else None,
            'output_directory': str(output_dir),
            'hypothesis': hypothesis.get('title', 'N/A'),
            'narrative_id': narrative.id if narrative else 'N/A',
            'artifacts_count': len(all_artifacts),
            'paper_generated': result.metadata.get('paper_pdf') is not None or result.metadata.get('paper_tex') is not None,
            'repo_url': repo_url,
        }
        summary_path = output_dir / "pipeline_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_summary, f, indent=2, ensure_ascii=False)
        telemetry.emit(MessageType.INFO, f"   Pipeline summary saved to: {rel_path(summary_path)}")

        # Save pools
        figure_pool.save(output_dir / "figure_pool.json")
        demo_pool.save(output_dir / "demo_pool.json")

        return result

    finally:
        # Remove module sinks to prevent leak into subsequent modules
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        if local_telemetry:
            telemetry.flush()


def _build_partial_result(
    output_dir: Path,
    telemetry: AIITelemetry,
    figure_pool: FigurePool,
    demo_pool: DemoPool,
    stop_reason: str,
    repo_url: str | None = None,
    paper: PaperText | None = None,
    figures: list[Figure] | None = None,
    gist_deployments: list[GistDeployment] | None = None,
    prepared_artifacts: list[BaseDemo] | None = None,
) -> GenPaperRepoOut:
    """Build a partial result when stopping early."""
    telemetry.emit(MessageType.INFO, "")
    telemetry.emit(MessageType.INFO, "=" * 60)
    telemetry.emit(MessageType.INFO, f"GEN_PAPER Complete (stopped: {stop_reason})")
    telemetry.emit(MessageType.INFO, "=" * 60)
    telemetry.emit(MessageType.INFO, f"   Output: {rel_path(output_dir)}")
    if repo_url:
        telemetry.emit(MessageType.INFO, f"   Repo URL: {repo_url}")
    if paper:
        telemetry.emit(MessageType.INFO, f"   Paper: {paper.id}")
    if figures:
        telemetry.emit(MessageType.INFO, f"   Figures: {len(figures)}")
    if gist_deployments:
        telemetry.emit(MessageType.INFO, f"   Gist deployments: {len(gist_deployments)}")

    result = GenPaperRepoOut(
        paper=paper,
        figures=figures or [],
        gist_deployments=gist_deployments or [],
        repo_url=repo_url,
        output_dir=str(output_dir),
        metadata={
            "gen_full_paper_skipped": True,
            "stop_reason": stop_reason,
            "total_cost_usd": sum(s.get("total_cost", 0.0) or 0.0 for s in telemetry._module_summaries),
        },
    )

    # Save pools
    figure_pool.save(output_dir / "figure_pool.json")
    demo_pool.save(output_dir / "demo_pool.json")

    telemetry.emit_module_group_summary("GEN_PAPER")
    return result


async def main():
    """Main function for standalone execution."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python _4_gen_paper_repo.py <invention_loop_result.json>")
        return 1

    config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
    config = PipelineConfig.from_yaml(config_path)

    result_path = Path(sys.argv[1])
    if not result_path.exists():
        print(f"File not found: {result_path}")
        return 1

    with open(result_path, 'r') as f:
        invention_loop_result = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/{timestamp}_gen_paper")
    run_dir.mkdir(parents=True, exist_ok=True)

    result = await run_gen_paper_module(
        config=config,
        invention_loop_result=invention_loop_result,
        run_dir=run_dir,
    )

    if result:
        print(f"Gen paper completed successfully: {result.output_dir}")
        return 0
    return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
