"""NARRATE Step - Generate narratives from artifact pool.

Multiple LLMs generate research narratives that tell a coherent story
using the available artifacts.

Each narrative:
- Weaves artifacts into a coherent research story
- References specific artifacts by ID
- Explains the significance of findings
- Builds toward a conclusion

Supports two backends:
- OpenRouter (default): Uses chat() with structured output
- Claude agent: Uses Agent with SDK native output_format for structured output

Uses aii_lib for:
- OpenRouterClient: LLM calls (OpenRouter backend)
- Agent/AgentOptions: Claude agent calls
- AIITelemetry: Task tracking and logging
"""

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path

from aii_lib import OpenRouterClient, AIITelemetry, MessageType, JSONSink, chat, get_model_short, get_tooluniverse_mcp_config
from aii_lib.agent_backend import Agent, AgentOptions, aggregate_summaries

from aii_pipeline.utils import PipelineConfig, rel_path

from aii_pipeline.prompts.steps._3_invention_loop._4_gen_narr.schema import Narrative
from ._invention_loop.pools import ArtifactPool, NarrativePool
from aii_pipeline.prompts.steps._3_invention_loop._4_gen_narr.s_prompt import get as get_narrate_system_prompt
from aii_pipeline.prompts.steps._3_invention_loop._4_gen_narr.u_prompt import get as get_narrate_prompt


async def gen_narr(
    telemetry: AIITelemetry,
    task_id: str,
    task_name: str,
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    iteration: int,
    reasoning_effort: str = "medium",
    llm_timeout: int = 600,
) -> Narrative | None:
    """Generate narrative using aii_lib chat() wrapper."""
    telemetry.emit_task_start(task_id, task_name)
    callback = telemetry.create_callback(task_id, task_name, group="gen_narr")

    try:
        async with OpenRouterClient(api_key=api_key, model=model, timeout=llm_timeout) as client:
            # Use chat() wrapper from aii_lib - handles tool loop, stats, callbacks
            result = await chat(
                client=client,
                prompt=prompt,
                system=system_prompt,
                reasoning_effort=reasoning_effort,
                response_format=Narrative,
                message_callback=callback,
                timeout=llm_timeout,
            )

            output_text = client.extract_json_from_response(result.response)
            if output_text:
                data = json.loads(output_text)

                narrative = Narrative(
                    id=task_id,
                    title=data.get("title", ""),
                    narrative=data.get("narrative", ""),
                    summary=data.get("summary", ""),
                    artifacts_used=data.get("artifacts_used", []),
                    gaps=data.get("gaps", []),
                )

                telemetry.emit_task_end(task_id, task_name, f"Generated ({len(narrative.narrative)} chars)")
                return narrative

        telemetry.emit_task_end(task_id, task_name, "No output")
        raise RuntimeError(f"Narrative generation produced no output for {model}")

    except asyncio.TimeoutError:
        telemetry.emit_task_end(task_id, task_name, f"Timeout ({llm_timeout}s)")
        raise
    except json.JSONDecodeError as e:
        telemetry.emit_task_end(task_id, task_name, f"JSON parse error: {e}")
        raise
    except Exception as e:
        telemetry.emit(MessageType.ERROR, f"Narrative generation failed for {model}: {e}")
        telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
        raise


async def gen_narr_claude_agent(
    telemetry: AIITelemetry,
    task_id: str,
    task_name: str,
    prompt: str,
    system_prompt: str,
    model: str,
    max_turns: int,
    cwd: Path,
    output_dir: Path,
    iteration: int,
    agent_timeout: int | None = None,
    agent_retries: int = 2,
    seq_prompt_timeout: int | None = None,
    seq_prompt_retries: int = 5,
    message_timeout: int | None = None,
    message_retries: int = 3,
    mcp_servers: dict | None = None,
    disallowed_tools: list[str] | None = None,
) -> Narrative | None:
    """Generate narrative using Claude agent with structured JSON output."""
    telemetry.emit_task_start(task_id, task_name)

    try:
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
            # SDK native structured output
            output_format=Narrative.to_struct_output(),
        )

        agent = Agent(options)
        response = await agent.run(prompt)

        # Store summary for module aggregation (agent already emits to console)
        if response.prompt_results:
            aggregated = aggregate_summaries(response.prompt_results)
            if aggregated:
                telemetry._pending_summaries.append(aggregated)

        if response.failed:
            telemetry.emit_task_end(task_id, task_name, f"Agent failed: {response.error_message or 'unknown error'}")
            return None

        if response.structured_output:
            data = response.structured_output if isinstance(response.structured_output, dict) else response.structured_output

            narrative = Narrative(
                id=task_id,
                title=data.get("title", ""),
                narrative=data.get("narrative", ""),
                summary=data.get("summary", ""),
                artifacts_used=data.get("artifacts_used", []),
                gaps=data.get("gaps", []),
            )

            telemetry.emit_task_end(task_id, task_name, f"Generated ({len(narrative.narrative)} chars)")
            return narrative

        telemetry.emit_task_end(task_id, task_name, "No output")
        return None

    except Exception as e:
        telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
        raise


async def run_gen_narr_module(
    config: PipelineConfig,
    hypothesis: dict,
    artifact_pool: ArtifactPool,
    narrative_pool: NarrativePool,
    iteration: int,
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
    cumulative: dict | None = None,
) -> list[Narrative]:
    """
    Run the NARRATE step.

    Multiple LLMs generate narratives from the artifact pool.
    Narratives are added to the narrative pool.

    Args:
        config: Pipeline configuration
        hypothesis: The research hypothesis
        artifact_pool: Pool of artifacts to narrate
        narrative_pool: Pool to add narratives to
        iteration: Current iteration
        telemetry: AIITelemetry instance for logging
        output_dir: Directory to save outputs

    Returns:
        List of generated narratives
    """
    telemetry.start_module("GEN_NARR")

    telemetry.emit(MessageType.INFO, "=" * 60)
    telemetry.emit(MessageType.INFO, f"GEN_NARR - Generating narratives for iteration {iteration}")
    telemetry.emit(MessageType.INFO, "=" * 60)

    narrate_cfg = config.invention_loop.narrative
    use_claude_agent = narrate_cfg.use_claude_agent

    # Check if narrative generation should start
    start_at = narrate_cfg.start_at_iteration
    if iteration < start_at:
        telemetry.emit(MessageType.INFO, f"Skipping narrative generation (starts at iteration {start_at})")
        return []

    # Configuration
    narratives_per_round = narrate_cfg.narratives_per_round

    # =========================================================================
    # SETUP BACKEND
    # =========================================================================
    # Step subdir within iteration dir (always created regardless of backend)
    if output_dir:
        step_dir = (output_dir / "gen_narr").resolve()
        step_dir.mkdir(parents=True, exist_ok=True)
        output_dir = step_dir

    if use_claude_agent:
        claude_cfg = narrate_cfg.claude_agent
        max_parallel = claude_cfg.max_concurrent_agents

        # Always connect MCP for aii_web_fetch_grep.
        # When use_aii_web_tools=True: all 3 MCP tools replace built-in WebSearch/WebFetch.
        # When use_aii_web_tools=False: only grep via MCP, built-in WebSearch/WebFetch for the rest.
        mcp_servers = get_tooluniverse_mcp_config(use_aii_server=True)
        dblp_block = [
            "mcp__aii_tooluniverse__dblp_bib_search",
            "mcp__aii_tooluniverse__dblp_bib_fetch",
        ]
        if claude_cfg.use_aii_web_tools:
            disallowed_tools = ["WebSearch", "WebFetch"] + dblp_block
        else:
            disallowed_tools = [
                "mcp__aii_tooluniverse__aii_web_search_fast",
                "mcp__aii_tooluniverse__aii_web_fetch_direct",
            ] + dblp_block

        models = [{"model": claude_cfg.model, "model_short": get_model_short(claude_cfg.model)}]
        llm_provider = "claude_agent"
        llm_timeout = claude_cfg.seq_prompt_timeout
    else:
        llm_cfg = narrate_cfg.llm_client
        global_suffix = llm_cfg.suffix
        llm_timeout = llm_cfg.llm_timeout
        max_parallel = None  # No limit for OpenRouter
        llm_provider = "openrouter"

        # Parse models - apply global suffix if not specified per-model
        models = []
        for m in llm_cfg.models:
            models.append({
                "model": m.model,
                "reasoning_effort": m.reasoning_effort,
                "suffix": m.suffix or global_suffix,
            })

    # Add module-specific JSON sink (after backend setup so it uses step_dir when applicable)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "gen_narr_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "gen_narr_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.emit(MessageType.INFO, f"   Provider: {llm_provider}")
    telemetry.emit(MessageType.INFO, f"   Generating {narratives_per_round} narratives")
    telemetry.emit(MessageType.INFO, f"   Models: {[m['model'] for m in models]}")
    telemetry.emit(MessageType.INFO, f"   Artifacts available: {len(artifact_pool.get_all())}")
    telemetry.emit(MessageType.INFO, f"   Timeout: {llm_timeout}s")

    # Build prompt (same for all)
    prompt = get_narrate_prompt(hypothesis, artifact_pool)

    # Build system prompt (emitted per-task right before chat call)
    system_prompt = get_narrate_system_prompt()

    # Build task configs
    task_configs = []
    for i in range(narratives_per_round):
        model_cfg = models[i % len(models)]
        model_name = model_cfg["model"]

        if not use_claude_agent:
            # Ensure model has provider prefix for OpenRouter
            if "/" not in model_name:
                model_name = f"openai/{model_name}"
            model_short = get_model_short(model_name)
        else:
            model_short = model_cfg.get("model_short", model_name)

        task_id = f"narr_v{i+1}_it{iteration}__{model_short}"
        task_configs.append((task_id, model_name, model_cfg.get("reasoning_effort", "medium")))

    # Run tasks (with optional semaphore for Claude agent)
    telemetry.emit(MessageType.INFO, f"   Running {len(task_configs)} narrative generators...")
    sem = asyncio.Semaphore(max_parallel) if max_parallel else None

    async def run_task(task_id: str, model: str, reasoning_effort: str):
        async with sem if sem else nullcontext():
            if use_claude_agent:
                # Per-task CWD so parallel agents don't collide
                task_cwd = (output_dir / task_id) if output_dir else Path.cwd().resolve() / task_id
                task_cwd.mkdir(parents=True, exist_ok=True)
                return task_id, await gen_narr_claude_agent(
                    telemetry=telemetry,
                    task_id=task_id,
                    task_name=task_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=claude_cfg.model,
                    max_turns=claude_cfg.max_turns,
                    cwd=task_cwd,
                    output_dir=output_dir,
                    iteration=iteration,
                    agent_timeout=claude_cfg.agent_timeout,
                    agent_retries=claude_cfg.agent_retries,
                    seq_prompt_timeout=claude_cfg.seq_prompt_timeout,
                    seq_prompt_retries=claude_cfg.seq_prompt_retries,
                    message_timeout=claude_cfg.message_timeout,
                    message_retries=claude_cfg.message_retries,
                    mcp_servers=mcp_servers,
                    disallowed_tools=disallowed_tools,
                )
            else:
                return task_id, await gen_narr(
                    telemetry=telemetry,
                    task_id=task_id,
                    task_name=task_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    api_key=config.api_keys.openrouter,
                    iteration=iteration,
                    reasoning_effort=reasoning_effort,
                    llm_timeout=llm_timeout,
                )

    task_results = await asyncio.gather(*[
        run_task(task_id, model, reasoning_effort)
        for task_id, model, reasoning_effort in task_configs
    ], return_exceptions=True)

    # Convert to dict: task_id -> Narrative | None
    results = {}
    for tr in task_results:
        if isinstance(tr, Exception):
            telemetry.emit(MessageType.WARNING, f"GenNarr task failed: {tr}")
            continue
        task_id, narr = tr
        results[task_id] = narr

    # Filter and add to pool (results is dict: task_id -> Narrative | None)
    narratives = []
    for task_id, narr in results.items():
        if narr and narr.narrative:
            # Add to pool (uses task_id as the id)
            added = narrative_pool.add(
                id=narr.id,
                title=narr.title,
                narrative=narr.narrative,
                summary=narr.summary,
                artifacts_used=narr.artifacts_used,
                gaps=narr.gaps,
            )
            narratives.append(added)

    telemetry.emit(MessageType.SUCCESS, f"NARRATE complete: {len(narratives)} narratives generated")

    # Compute total cost from pending summaries before emit_module_summary clears them
    total_cost_usd = sum(s.get("total_cost", 0.0) for s in telemetry._pending_summaries)

    # Build and emit module output
    from aii_pipeline.utils import build_module_output, emit_module_output
    module_output = build_module_output(
        module="gen_narr",
        iteration=iteration,
        outputs=narratives,
        cumulative=cumulative or {},
        total_cost_usd=total_cost_usd,
        output_dir=output_dir,
        llm_provider=llm_provider,
    )
    emit_module_output(module_output, telemetry, output_dir=output_dir)

    telemetry.emit_module_summary("GEN_NARR")

    # Remove module sinks to prevent leak into subsequent modules
    for sink in module_sinks:
        sink.flush()
        telemetry.remove_sink(sink)

    return narratives
