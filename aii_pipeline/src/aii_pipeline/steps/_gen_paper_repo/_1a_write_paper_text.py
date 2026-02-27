"""A_WRITE_PAPER_TEXT Step - Generate paper text with inline figures.

Uses Claude agent to generate paper content (title, abstract, paper_text, summary)
with embedded figure specs as inline XML.

Figures are embedded INLINE as XML in the paper_text field.
They are parsed out using parse_figures_from_xml() - NO separate array.

Verification follows the same pattern as experiment/gen_strat:
1. Run agent → get text with inline XML figures
2. Parse figures from XML using parse_figures_from_xml()
3. Verify using verify_figures() (checks IDs, required fields, etc.)
4. If errors → build_figure_retry_prompt() → retry

Artifacts are copied to ./artifacts/ folder in the agent's workspace.
Agent can read mini/preview files from the artifacts.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from aii_lib import AIITelemetry, MessageType, JSONSink, get_model_short, get_tooluniverse_mcp_config
from aii_lib.agent_backend import Agent, AgentOptions, aggregate_summaries

from aii_pipeline.utils import PipelineConfig, rel_path

from aii_pipeline.prompts.steps._4_gen_paper_repo._1a_write_papers.u_prompt import (
    get as get_write_paper_prompt,
    build_figure_retry_prompt,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._1a_write_papers.s_prompt import get as get_write_paper_sysprompt
from aii_pipeline.prompts.steps._4_gen_paper_repo._1a_write_papers.schema import (
    PaperText,
    parse_figures_from_xml,
    verify_figures,
)


def _build_paper_text(
    data: dict,
    draft_idx: int,
) -> PaperText:
    """Build PaperText from parsed data."""
    return PaperText(
        id=f"paper_{draft_idx + 1}",
        title=data.get("title", ""),
        abstract=data.get("abstract", ""),
        paper_text=data.get("paper_text", ""),
        summary=data.get("summary", ""),
    )


async def generate_paper_text(
    telemetry: AIITelemetry,
    task_id: str,
    task_name: str,
    narrative_pool,
    artifact_pool,
    hypothesis: dict | None,
    model: str,
    max_turns: int,
    cwd: Path,
    output_dir: Path,
    draft_idx: int = 0,
    agent_timeout: int | None = None,
    agent_retries: int = 2,
    seq_prompt_timeout: int | None = None,
    seq_prompt_retries: int = 5,
    message_timeout: int | None = None,
    message_retries: int = 3,
    valid_artifact_ids: set[str] | None = None,
    verify_retries: int = 2,
    task_sequence: int = 0,
    mcp_servers: dict | None = None,
    disallowed_tools: list[str] | None = None,
) -> PaperText | None:
    """Generate paper text using Claude agent with structured JSON output.

    Figures are embedded INLINE as XML in text sections.
    After getting the response:
    1. Parse figures from inline XML using parse_figures_from_xml()
    2. Verify using verify_figures()
    3. If errors → send retry prompt with build_figure_retry_prompt()

    Returns:
        PaperText with content sections and figures, or None on failure
    """
    # Use absolute path to avoid CWD mismatch issues with agent
    abs_cwd = Path(cwd).resolve()

    telemetry.emit_task_start(task_id, task_name, sequence=task_sequence)
    callback = telemetry.create_callback(task_id, task_name, group="a_write_paper_text")

    try:
        prompt = get_write_paper_prompt(
            narrative_pool=narrative_pool,
            artifact_pool=artifact_pool,
            hypothesis=hypothesis,
        )
        system_prompt = get_write_paper_sysprompt()

        options = AgentOptions(
            model=model,
            cwd=cwd,
            max_turns=max_turns,
            permission_mode="bypassPermissions",
            system_prompt=system_prompt,
            mcp_servers=mcp_servers or {},
            disallowed_tools=disallowed_tools,
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
            # Structured JSON output (SDK native)
            output_format=PaperText.to_struct_output(),
        )

        agent = Agent(options)
        current_prompt = prompt

        # Collect all prompt results across retry attempts for aggregation
        all_prompt_results = []

        for attempt in range(verify_retries + 1):
            response = await agent.run(current_prompt)

            # Collect prompt results for aggregation (emit once at the end)
            all_prompt_results.extend(response.prompt_results)

            if response.failed:
                telemetry.emit_task_end(task_id, task_name, f"Agent failed: {response.error_message or 'unknown error'}")
                return None

            if not response.structured_output:
                continue

            data = response.structured_output if isinstance(response.structured_output, dict) else {}

            # =================================================================
            # Parse figures from inline XML in text sections
            # =================================================================
            figures = parse_figures_from_xml(data)

            # =================================================================
            # Verify figures (figure types, artifact IDs, etc.)
            # =================================================================
            verification = verify_figures(
                figures=figures,
            )

            if not verification["valid"]:
                # Build error summary for logging
                all_errors = (
                    verification.get("id_errors", []) +
                    verification.get("field_errors", [])
                )
                error_summary = "; ".join(all_errors[:3])
                if len(all_errors) > 3:
                    error_summary += f" ... (+{len(all_errors) - 3} more)"

                if attempt < verify_retries:
                    # Build retry prompt and continue
                    telemetry.emit_message(
                        "WARNING",
                        f"Paper text {draft_idx} figure verification failed (attempt {attempt + 1}/{verify_retries + 1}): {error_summary}",
                        task_name,
                        task_id,
                    )
                    current_prompt = build_figure_retry_prompt(
                        verification=verification,
                    )
                    continue
                else:
                    # Max retries reached - log warning but continue with what we have
                    telemetry.emit_message(
                        "WARNING",
                        f"Paper text {draft_idx} still has figure errors after {verify_retries} retries: {error_summary}",
                        task_name,
                        task_id,
                    )

            # Build and return paper text (figures added to pool by orchestrator)
            draft = _build_paper_text(data, draft_idx)

            # Emit ONE aggregated summary for all attempts
            agg = aggregate_summaries(all_prompt_results)
            if agg:
                callback(agg)

            telemetry.emit_task_end(
                task_id,
                task_name,
                f"Paper text {draft_idx} generated ({len(figures)} figures, {verification['figures_valid']} valid)",
            )
            return draft

        # Emit aggregated summary even on no output
        agg = aggregate_summaries(all_prompt_results)
        if agg:
            callback(agg)

        telemetry.emit_task_end(task_id, task_name, "No output")
        return None

    except Exception as e:
        telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
        raise


async def run_write_paper_text_module(
    config: PipelineConfig,
    narrative_pool,
    artifact_pool,
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
    valid_artifact_ids: set[str] | None = None,
    hypothesis: dict | None = None,
) -> list[PaperText]:
    """Run the A_WRITE_PAPER_TEXT step using Claude agent.

    Returns:
        List of PaperText objects with text content and figures
    """
    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "write_paper_text_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "write_paper_text_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.start_module("A_WRITE_PAPER_TEXT")

    try:
        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, "A_WRITE_PAPER_TEXT - Generating paper text with inline figures")
        telemetry.emit(MessageType.INFO, "=" * 60)

        gen_paper_cfg = config.gen_paper_repo
        write_cfg = gen_paper_cfg.write_paper_text
        claude_cfg = write_cfg.claude_agent

        num_variations = write_cfg.variations
        max_parallel = claude_cfg.max_concurrent_agents

        # Always connect MCP for aii_web_fetch_grep.
        # When use_aii_web_tools=True: all 3 MCP tools replace built-in WebSearch/WebFetch.
        # When use_aii_web_tools=False: only grep via MCP, built-in WebSearch/WebFetch for the rest.
        mcp_servers = get_tooluniverse_mcp_config(use_aii_server=True)
        if claude_cfg.use_aii_web_tools:
            disallowed_tools = ["WebSearch", "WebFetch"]
        else:
            disallowed_tools = [
                "mcp__aii_tooluniverse__aii_web_search_fast",
                "mcp__aii_tooluniverse__aii_web_fetch_direct",
            ]

        # Dedicated workspace subdir within gen_paper_repo, with claude_agent/ as CWD
        step_dir = (output_dir / "_1a_write_paper_text").resolve() if output_dir else Path.cwd().resolve()
        agent_cwd = step_dir / "claude_agent"
        agent_cwd.mkdir(parents=True, exist_ok=True)

        telemetry.emit(MessageType.INFO, f"   Artifacts: {len(artifact_pool.get_all())}")

        verify_retries = write_cfg.verify_retries

        telemetry.emit(MessageType.INFO, f"   Provider: claude_agent")
        telemetry.emit(MessageType.INFO, f"   Model: {claude_cfg.model}")
        telemetry.emit(MessageType.INFO, f"   Variations to generate: {num_variations}")
        telemetry.emit(MessageType.INFO, f"   Artifacts available: {len(artifact_pool.get_all())}")
        if valid_artifact_ids:
            telemetry.emit(MessageType.INFO, f"   Valid artifact IDs for validation: {len(valid_artifact_ids)}")

        # Generate paper texts in parallel with semaphore for concurrency control
        sem = asyncio.Semaphore(max_parallel)

        async def run_task(i: int):
            async with sem:
                model_short = get_model_short(claude_cfg.model)
                task_id = f"paper_text_v{i}__{model_short}"
                task_name = task_id

                return await generate_paper_text(
                    telemetry=telemetry,
                    task_id=task_id,
                    task_name=task_name,
                    narrative_pool=narrative_pool,
                    artifact_pool=artifact_pool,
                    hypothesis=hypothesis,
                    model=claude_cfg.model,
                    max_turns=claude_cfg.max_turns,
                    cwd=agent_cwd,
                    output_dir=output_dir,
                    draft_idx=i,
                    agent_timeout=claude_cfg.agent_timeout,
                    agent_retries=claude_cfg.agent_retries,
                    seq_prompt_timeout=claude_cfg.seq_prompt_timeout,
                    seq_prompt_retries=claude_cfg.seq_prompt_retries,
                    message_timeout=claude_cfg.message_timeout,
                    message_retries=claude_cfg.message_retries,
                    valid_artifact_ids=valid_artifact_ids,
                    verify_retries=verify_retries,
                    task_sequence=i,
                    mcp_servers=mcp_servers,
                    disallowed_tools=disallowed_tools,
                )

        tasks = [run_task(i) for i in range(num_variations)]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                telemetry.emit(MessageType.WARNING, f"WritePaperText failed: {r}")
        paper_texts = [r for r in results if r is not None and not isinstance(r, Exception)]

        telemetry.emit(
            MessageType.SUCCESS,
            f"A_WRITE_PAPER_TEXT complete: {len(paper_texts)}/{num_variations} paper texts"
        )

        # Save results summary
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "write_paper_text_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "paper_texts": [d.model_dump() for d in paper_texts],
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "module": "write_paper_text",
                        "llm_provider": "claude_agent",
                        "model": claude_cfg.model,
                        "output_dir": str(output_dir) if output_dir else None,
                    },
                    "num_variations": num_variations,
                    "successful": len(paper_texts),
                }, f, indent=2, ensure_ascii=False)
            telemetry.emit(MessageType.INFO, f"   Saved to: {rel_path(output_file)}")

        return paper_texts

    finally:
        # Always remove module sinks to prevent leak into subsequent modules
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
