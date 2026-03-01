"""VIZ_GEN Step - Generate visualizations from paper figures.

Takes the paper's figure specs and generates actual figures.

Two backends:
  use_claude_agent=True  → Claude agent with aii_image_gen_nano_banana skill
                           Agent gets workspace + figure spec, uses skill to generate,
                           verifies output, and saves PNG to figures/ directory.
  use_claude_agent=False → Direct Gemini image gen via OpenRouter (free_viz)
                           Pure image output from models like gemini-3-pro-image-preview.

All figures: one per placeholder, no variations, no ranking.
Output is PNG format.

Uses aii_lib for:
- Agent + AgentOptions: Claude Code SDK agent orchestration
- OpenRouterClient: LLM client with generate_image() for pure image output
- AIITelemetry: Task tracking
"""

import asyncio
import io
import json
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

from aii_lib import OpenRouterClient, AIITelemetry, MessageType, JSONSink, get_model_short
from aii_lib.agent_backend import Agent, AgentOptions, aggregate_summaries

from aii_pipeline.utils import PipelineConfig, rel_path
from aii_pipeline.prompts.steps._4_gen_paper_repo._2a_viz_gen.schema import (
    Figure,
    VizFigureOutput,
    get_output_filename,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._2a_viz_gen.u_prompt import get as get_viz_user_prompt
from aii_pipeline.prompts.steps._4_gen_paper_repo._2a_viz_gen.s_prompt import get as get_viz_system_prompt


# =============================================================================
# CLAUDE AGENT BACKEND
# =============================================================================

async def generate_image_viz_agent(
    telemetry: AIITelemetry,
    task_id: str,
    task_name: str,
    figure: Figure,
    model: str,
    max_turns: int,
    cwd: Path,
    figures_dir: Path,
    agent_timeout: int | None = None,
    agent_retries: int = 2,
    seq_prompt_timeout: int | None = None,
    seq_prompt_retries: int = 3,
    message_timeout: int | None = None,
    message_retries: int = 3,
    task_sequence: int = 0,
) -> Figure | None:
    """Generate a figure using Claude agent with aii_image_gen_nano_banana skill.

    The agent receives the figure spec and workspace path, uses the nano_banana
    skill to generate the image, verifies it, and saves the PNG.

    Returns the same Figure with figure_path filled in.
    """
    # Use absolute path to avoid CWD mismatch issues with agent
    abs_cwd = Path(cwd).resolve()

    telemetry.emit_task_start(task_id, task_name, sequence=task_sequence)
    callback = telemetry.create_callback(task_id, task_name, group="a_gen_viz")

    try:
        # Build prompt with workspace path pointing to figures dir
        prompt = get_viz_user_prompt(
            figure_spec=figure,
            workspace_path=str(figures_dir),
        )
        system_prompt = get_viz_system_prompt()

        # Expected output filename
        output_filename = get_output_filename(figure.id, 0)

        options = AgentOptions(
            model=model,
            cwd=abs_cwd,
            max_turns=max_turns,
            permission_mode="bypassPermissions",
            system_prompt=system_prompt,
            # Telemetry integration
            telemetry=telemetry,
            run_id=task_id,
            agent_context=task_id,
            # Agent-level timeouts/retries
            agent_timeout=agent_timeout,
            agent_retries=agent_retries,
            seq_prompt_timeout=seq_prompt_timeout,
            seq_prompt_retries=seq_prompt_retries,
            message_timeout=message_timeout,
            message_retries=message_retries,
            # SDK native structured output for title/summary
            output_format=VizFigureOutput.to_struct_output(),
            # No expected_files validation — _find_generated_image() handles file search
            # across both figures_dir and agent CWD after the run completes.
        )

        agent = Agent(options)
        response = await agent.run(prompt)

        # Emit ONE aggregated summary (not individual prompt summaries)
        if response.prompt_results:
            agg = aggregate_summaries(response.prompt_results)
            if agg:
                callback(agg)

        if response.failed:
            telemetry.emit_message(
                "ERROR",
                f"Agent failed: {response.error_message or 'unknown'}",
                task_name,
                task_id,
            )
            telemetry.emit_task_end(task_id, task_name, f"Failed: agent error")
            return None

        # Check if the agent saved an image file
        # Look for any image file matching the figure ID in the figures dir and agent CWD
        found_path = _find_generated_image(figure.id, figures_dir, abs_cwd)

        if found_path:
            # If found outside figures_dir, copy it there
            if found_path.parent.resolve() != figures_dir.resolve():
                dest = figures_dir / output_filename
                shutil.copy2(found_path, dest)
                found_path = dest

            # Ensure it's named correctly
            if found_path.name != output_filename:
                dest = figures_dir / output_filename
                shutil.copy2(found_path, dest)
                found_path = dest

            figure.figure_path = str(found_path)
            telemetry.emit_message(
                "SUCCESS",
                f"Image saved: {found_path.name} ({found_path.stat().st_size} bytes)",
                task_name,
                task_id,
            )
            telemetry.emit_task_end(task_id, task_name, "Success")
            return figure

        telemetry.emit_message("ERROR", f"No image file found after agent run for {figure.id}", task_name, task_id)
        telemetry.emit_task_end(task_id, task_name, "Failed: no image file")
        return None

    except Exception as e:
        telemetry.emit_message("ERROR", f"Exception: {e}", task_name, task_id)
        telemetry.emit_task_end(task_id, task_name, f"Failed: {e}")
        raise


def _find_generated_image(figure_id: str, *search_dirs: Path) -> Path | None:
    """Search for a generated image file matching the figure ID.

    Search priority:
    1. Exact filename match (e.g. fig_2_v0.png)
    2. Figure ID in filename (e.g. fig_2_chart.png)
    3. Most recently modified image file in the directory (fallback)

    Returns the first match found, or None.
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    fallback_candidates: list[Path] = []

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for f in search_dir.iterdir():
            if not f.is_file() or f.suffix.lower() not in image_extensions:
                continue
            if figure_id in f.stem:
                return f
            fallback_candidates.append(f)

        # Check one level deep (agent might create subfolders)
        for subdir in search_dir.iterdir():
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            for f in subdir.iterdir():
                if not f.is_file() or f.suffix.lower() not in image_extensions:
                    continue
                if figure_id in f.stem:
                    return f
                fallback_candidates.append(f)

    # Fallback: return most recently modified image file
    if fallback_candidates:
        return max(fallback_candidates, key=lambda p: p.stat().st_mtime)

    return None


# =============================================================================
# OPENROUTER (FREE_VIZ) BACKEND
# =============================================================================

async def generate_image_viz_openrouter(
    telemetry: AIITelemetry,
    task_id: str,
    task_name: str,
    figure: Figure,
    model: str,
    api_key: str,
    output_dir: Path,
    llm_timeout: int = 120,
    task_sequence: int = 0,
    image_size: str | None = None,
) -> Figure | None:
    """Generate a figure using direct Gemini image generation via OpenRouter.

    Uses OpenRouterClient.generate_image() with modalities=["image"] to get
    pure image output from models like gemini-3-pro-image-preview.

    The returned JPEG image is converted to PNG for consistency.

    Returns the same Figure with figure_path filled in.

    Args:
        image_size: Gemini image resolution - "1K" (default), "2K" (higher), "4K" (highest)
    """
    telemetry.emit_task_start(task_id, task_name, sequence=task_sequence)
    callback = telemetry.create_callback(task_id, task_name, group="a_gen_viz")

    try:
        prompt = get_viz_user_prompt(figure)
        system_prompt = get_viz_system_prompt()

        async with OpenRouterClient(api_key=api_key, model=model, timeout=llm_timeout) as client:
            image_bytes = await client.generate_image(
                prompt=prompt,
                model=model,
                system=system_prompt,
                image_size=image_size,
                message_callback=callback,
            )

            if image_bytes:
                # Convert to PNG for consistency (API returns JPEG)
                img = Image.open(io.BytesIO(image_bytes))
                png_buffer = io.BytesIO()
                img.save(png_buffer, format='PNG')
                png_bytes = png_buffer.getvalue()

                # Save using schema-defined filename (variation_idx=0 always)
                output_filename = get_output_filename(figure.id, 0)
                output_path = output_dir / output_filename

                with open(output_path, 'wb') as f:
                    f.write(png_bytes)

                telemetry.emit_message("SUCCESS", f"Image saved: {output_filename} ({len(png_bytes)} bytes)", task_name, task_id)
                telemetry.emit_task_end(task_id, task_name, "Success")

                # Return figure with figure_path set
                figure.figure_path = str(output_path)
                return figure

            telemetry.emit_message("ERROR", f"No image returned from {model}", task_name, task_id)
            telemetry.emit_task_end(task_id, task_name, "Failed: no image")
            raise RuntimeError(f"No image returned from {model} for figure {figure.id}")

    except Exception as e:
        telemetry.emit_message("ERROR", f"Exception: {e}", task_name, task_id)
        telemetry.emit_task_end(task_id, task_name, f"Failed: {e}")
        raise


# =============================================================================
# MODULE RUNNER
# =============================================================================

async def run_gen_viz_module(
    config: PipelineConfig,
    figures: list[Figure],
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
) -> list[Figure]:
    """Run the A_GEN_VIZ step.

    Generates one figure per placeholder.
    Backend selected by viz_gen.use_claude_agent config flag.

    Args:
        config: Pipeline configuration
        figures: Figure specs parsed from paper text XML
        telemetry: AIITelemetry instance
        output_dir: Output directory

    Returns:
        List of Figure objects with figure_path filled in.
    """
    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "gen_viz_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "gen_viz_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.start_module("A_GEN_VIZ")

    try:
        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, "A_GEN_VIZ - Generating figures")
        telemetry.emit(MessageType.INFO, "=" * 60)

        if not figures:
            telemetry.emit(MessageType.WARNING, "No figures to generate")
            return []

        gen_paper_cfg = config.gen_paper_repo
        viz_cfg = gen_paper_cfg.viz_gen
        use_claude_agent = viz_cfg.use_claude_agent

        # Create figures output directory
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
        else:
            figures_dir = Path("./figures")
            figures_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================================
        # SETUP BACKEND
        # =====================================================================
        if use_claude_agent:
            claude_cfg = viz_cfg.claude_agent
            max_concurrent = claude_cfg.max_concurrent_agents
            llm_provider = "claude_agent"

            # Dedicated workspace subdir for agent CWD
            step_dir = (output_dir / "_2a_gen_viz").resolve() if output_dir else Path.cwd().resolve()
            agent_cwd = step_dir / "claude_agent"
            agent_cwd.mkdir(parents=True, exist_ok=True)

            telemetry.emit(MessageType.INFO, f"   Provider: claude_agent (nano_banana skill)")
            telemetry.emit(MessageType.INFO, f"   Model: {claude_cfg.model}")
        else:
            api_key = config.api_keys.openrouter
            free_viz_cfg = viz_cfg.free_viz
            max_concurrent = free_viz_cfg.max_concurrent if free_viz_cfg else 10
            llm_provider = "openrouter"

            # Get image gen models
            free_viz_models = [{
                "model": m.model,
                "llm_timeout": m.llm_timeout,
            } for m in free_viz_cfg.models] if free_viz_cfg and free_viz_cfg.models else []
            free_viz_image_size = getattr(free_viz_cfg, 'image_size', None) if free_viz_cfg else None

            if not free_viz_models:
                telemetry.emit(MessageType.ERROR, "No image_gen models configured in viz_gen.free_viz.models")
                return []

            telemetry.emit(MessageType.INFO, f"   Provider: openrouter (free_viz)")
            telemetry.emit(MessageType.INFO, f"   Image gen models: {[m['model'] for m in free_viz_models]}")
            telemetry.emit(MessageType.INFO, f"   Image size: {free_viz_image_size or '1K (default)'}")

        telemetry.emit(MessageType.INFO, f"   Figures: {len(figures)}")
        telemetry.emit(MessageType.INFO, f"   Max concurrent: {max_concurrent}")

        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        # =====================================================================
        # BUILD TASKS
        # =====================================================================
        if use_claude_agent:
            model_short = get_model_short(claude_cfg.model)

            async def run_agent_task(counter: int, figure: Figure):
                # Stagger agent launches by 5s each to avoid init contention
                if counter > 0:
                    await asyncio.sleep(counter * 5)
                async with semaphore:
                    task_id = f"img_viz_{figure.id}__{model_short}"
                    return await generate_image_viz_agent(
                        telemetry=telemetry,
                        task_id=task_id,
                        task_name=task_id,
                        figure=figure,
                        model=claude_cfg.model,
                        max_turns=claude_cfg.max_turns,
                        cwd=agent_cwd,
                        figures_dir=figures_dir,
                        agent_timeout=claude_cfg.agent_timeout,
                        agent_retries=claude_cfg.agent_retries,
                        seq_prompt_timeout=claude_cfg.seq_prompt_timeout,
                        seq_prompt_retries=claude_cfg.seq_prompt_retries,
                        message_timeout=claude_cfg.message_timeout,
                        message_retries=claude_cfg.message_retries,
                        task_sequence=counter,
                    )

            results = await asyncio.gather(
                *[run_agent_task(i, fig) for i, fig in enumerate(figures)],
                return_exceptions=True,
            )
        else:
            # OpenRouter free_viz path
            task_params = []
            for counter, figure in enumerate(figures):
                model_cfg = free_viz_models[counter % len(free_viz_models)]
                model_short = get_model_short(model_cfg["model"])
                task_id = f"img_viz_{figure.id}__{model_short}"
                task_params.append({
                    "task_id": task_id,
                    "task_name": task_id,
                    "figure": figure,
                    "task_counter": counter,
                    "model_cfg": model_cfg,
                })

            async def run_openrouter_task(params: dict):
                async with semaphore:
                    return await generate_image_viz_openrouter(
                        telemetry=telemetry,
                        task_id=params["task_id"],
                        task_name=params["task_name"],
                        figure=params["figure"],
                        model=params["model_cfg"]["model"],
                        api_key=api_key,
                        output_dir=figures_dir,
                        llm_timeout=params["model_cfg"]["llm_timeout"],
                        task_sequence=params["task_counter"],
                        image_size=free_viz_image_size,
                    )

            results = await asyncio.gather(
                *[run_openrouter_task(p) for p in task_params],
                return_exceptions=True,
            )

        # =====================================================================
        # COLLECT RESULTS
        # =====================================================================
        generated_figures: list[Figure] = []
        for r in results:
            if isinstance(r, Exception):
                telemetry.emit(MessageType.WARNING, f"Figure task failed: {r}")
            elif r is not None:
                generated_figures.append(r)

        telemetry.emit(MessageType.SUCCESS, f"A_GEN_VIZ complete: {len(generated_figures)}/{len(figures)} figures generated")

        # Save output
        if output_dir:
            output_file = output_dir / "gen_viz_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "figures": [fig.model_dump() for fig in generated_figures],
                    "mode": "claude_agent" if use_claude_agent else "free_viz",
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "module": "gen_viz",
                        "llm_provider": llm_provider,
                        "total_figures": len(figures),
                        "successful_figures": len(generated_figures),
                        "output_dir": str(output_dir) if output_dir else None,
                    },
                }, f, indent=2, ensure_ascii=False)
            telemetry.emit(MessageType.INFO, f"   Saved to: {rel_path(output_file)}")

        telemetry.emit_module_summary("A_GEN_VIZ")

        return generated_figures

    finally:
        # Always remove module sinks to prevent leak into subsequent modules
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
