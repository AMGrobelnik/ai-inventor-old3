#!/usr/bin/env python3
"""
Hypothesis Generation Module - Multi-Model Parallel LLM

Generates research hypotheses using:
- Multiple models (hypos_per_llm × num_models = total hypotheses)
- asyncio.Semaphore for concurrent execution with max_parallel limit
- Structured JSON output (Hypothesis schema)
- Seed prompts from prep_context module
- aii_lib AIITelemetry for sequenced logging

Supports two backends:
- OpenRouter (default): Uses chat() with tool loop
- Claude agent: Uses Agent with SDK native output_format for structured output
"""

import asyncio
import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from aii_lib import (
    MessageType,
    AIITelemetry,
    JSONSink,
    OpenRouterClient,
    chat,
    create_telemetry,
    get_model_short,
    get_tooluniverse_mcp_config,
)
from aii_lib.agent_backend import Agent, AgentOptions, aggregate_summaries
from aii_lib.abilities.tools.utils import get_openrouter_tools

from aii_pipeline.prompts.steps._2_gen_hypo.u_prompt import get as get_gen_hypo_prompt, get_force_output_prompt
from aii_pipeline.prompts.steps._2_gen_hypo.s_prompt import get as get_gen_hypo_sysprompt
from aii_pipeline.prompts.steps._2_gen_hypo.schema import Hypothesis, GenHypoOut
from aii_pipeline.utils import PipelineConfig, rel_path, build_module_output, emit_module_output


async def _run_task_openrouter(
    task_id: str,
    task_name: str,
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    timeout: int,
    reasoning_effort: str | None,
    tools: list[dict] | None,
    telemetry: AIITelemetry,
) -> dict | None:
    """Run a single hypothesis generation task with OpenRouter."""
    telemetry.emit_task_start(task_id, task_name)
    callback = telemetry.create_callback(task_id, task_name, group="gen_hypo")

    try:
        effective_model = OpenRouterClient.resolve_model(model)

        async with OpenRouterClient(
            api_key=api_key,
            model=effective_model,
            timeout=timeout,
        ) as client:
            result = await chat(
                client=client,
                prompt=prompt,
                system=system_prompt,
                tools=tools,
                response_format=Hypothesis,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                message_callback=callback,
                emit_summary=True,
            )

            output_json = client.extract_json_from_response(result.response)
            if output_json:
                telemetry.emit_task_end(task_id, task_name, "OK")
                return json.loads(output_json)

            telemetry.emit_task_end(task_id, task_name, "No output")
            return None

    except Exception as e:
        telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
        raise


async def _run_task_claude_agent(
    task_id: str,
    prompt: str,
    system_prompt: str,
    model: str,
    max_turns: int,
    cwd: Path,
    output_dir: Path,
    mcp_servers: dict | None,
    disallowed_tools: list[str] | None,
    telemetry: AIITelemetry,
    task_sequence: int,
    agent_timeout: int | None = None,
    agent_retries: int = 2,
    seq_prompt_timeout: int | None = None,
    seq_prompt_retries: int = 5,
    message_timeout: int | None = None,
    message_retries: int = 3,
) -> dict | None:
    """Run a single hypothesis generation task with Claude agent."""
    telemetry.emit_task_start(task_id, task_id)

    try:

        options = AgentOptions(
            model=model,
            cwd=cwd,
            max_turns=max_turns,
            permission_mode="bypassPermissions",
            system_prompt=system_prompt,
            mcp_servers=mcp_servers or {},
            disallowed_tools=disallowed_tools,
            # Agent-level timeouts/retries
            agent_timeout=agent_timeout,
            agent_retries=agent_retries,
            seq_prompt_timeout=seq_prompt_timeout,
            seq_prompt_retries=seq_prompt_retries,
            message_timeout=message_timeout,
            message_retries=message_retries,
            # Telemetry integration
            telemetry=telemetry,
            run_id=task_id,
            agent_context=task_id,
            # SDK native structured output
            output_format=Hypothesis.to_struct_output(),
            # Domain-specific force output prompt (used when max_turns exceeded without output)
            force_output_prompt=get_force_output_prompt(),
        )

        agent = Agent(options)
        response = await agent.run(prompt)

        # Store summary for module aggregation (agent already emits to console)
        if response.prompt_results:
            aggregated = aggregate_summaries(response.prompt_results)
            if aggregated:
                telemetry._pending_summaries.append(aggregated)

        if response.structured_output:
            telemetry.emit_task_end(task_id, task_id, "OK")
            return response.structured_output

        telemetry.emit_task_end(task_id, task_id, "No output")
        return None

    except Exception as e:
        telemetry.emit_task_end(task_id, task_id, f"Error: {e}")
        raise


async def run_gen_hypo_module(
    config: PipelineConfig,
    agent_prompts=None,
    run_dir=None,
    telemetry: AIITelemetry | None = None,
    cumulative: dict | None = None,
):
    """
    Run hypothesis generation with parallel LLM calls.

    Uses asyncio.Semaphore for managed concurrent execution.
    Supports OpenRouter (default) or Claude agent backend.

    Args:
        config: Typed pipeline configuration
        agent_prompts: List of prompt lists (one per agent) from prep_context
        run_dir: Output directory
        telemetry: Shared telemetry instance (created by pipeline). If None, creates local one.
    """
    # Create output directory (ensure absolute path)
    if run_dir:
        output_dir = (Path(run_dir) / "2_gen_hypo").resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{config.init.outputs_directory}/{timestamp}_gen_hypo").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use provided telemetry or create local one
    local_telemetry = telemetry is None
    if local_telemetry:
        telemetry = create_telemetry(output_dir, "gen_hypo")

    try:
        s1 = JSONSink(output_dir / "gen_hypo_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "gen_hypo_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks = [s1, s2]
        telemetry.start_module("GEN_HYPO")

        # Get config
        research_direction = config.init.research_direction
        seeded_per_llm = config.gen_hypo.seeded_hypos_per_llm
        unseeded_per_llm = config.gen_hypo.unseeded_hypos_per_llm
        research_grounding = config.gen_hypo.research_grounding
        use_claude_agent = config.gen_hypo.use_claude_agent
        tool_names = ["aii_web_search_fast", "aii_web_fetch_direct", "aii_web_fetch_grep"] if research_grounding else None

        # =====================================================================
        # SETUP BACKEND
        # =====================================================================
        if use_claude_agent:
            claude_cfg = config.gen_hypo.claude_agent
            max_parallel = claude_cfg.max_concurrent_agents

            # Step subdir with claude_agent/ as CWD
            agent_cwd = (output_dir / "claude_agent").resolve()
            agent_cwd.mkdir(parents=True, exist_ok=True)

            # Always connect MCP server for aii_web_fetch_grep.
            # When use_aii_web_tools=True: all 3 MCP tools replace built-in WebSearch/WebFetch.
            # When use_aii_web_tools=False: only grep via MCP, built-in WebSearch/WebFetch for the rest.
            mcp_servers = get_tooluniverse_mcp_config(use_aii_server=True) if research_grounding else {}
            # DBLP tools are never needed for hypothesis generation
            dblp_block = [
                "mcp__aii_tooluniverse__dblp_bib_search",
                "mcp__aii_tooluniverse__dblp_bib_fetch",
            ]
            if claude_cfg.use_aii_web_tools:
                disallowed_tools = ["WebSearch", "WebFetch"] + dblp_block
            else:
                # Block MCP search/fetch so agent uses built-in WebSearch/WebFetch instead
                disallowed_tools = [
                    "mcp__aii_tooluniverse__aii_web_search_fast",
                    "mcp__aii_tooluniverse__aii_web_fetch_direct",
                ] + dblp_block

            models = [{"model": claude_cfg.model, "model_short": get_model_short(claude_cfg.model)}]
            llm_provider = "claude_agent"
        else:
            llm_client = config.gen_hypo.llm_client
            max_parallel = config.gen_hypo.max_parallel
            llm_provider = "openrouter"

            # Convert tool names to OpenRouter format
            or_tools = get_openrouter_tools(tool_names) if tool_names else None

            # Build models list
            models = []
            for m in llm_client.models:
                models.append({
                    "model": m.model,
                    "model_short": get_model_short(m.model),
                    "suffix": m.suffix,
                    "reasoning_effort": m.reasoning_effort,
                })

            if not models:
                telemetry.emit(MessageType.ERROR, "No models configured in gen_hypo.llm_client.models")
                return None

        num_models = len(models)
        total_seeded = seeded_per_llm * num_models
        total_unseeded = unseeded_per_llm * num_models
        total_hypotheses = total_seeded + total_unseeded

        agent_prompts = agent_prompts or [[] for _ in range(total_seeded)]
        while len(agent_prompts) < total_seeded:
            agent_prompts.append([])

        if use_claude_agent:
            if not research_grounding:
                tools_str = "none"
            elif claude_cfg.use_aii_web_tools:
                tools_str = ", ".join(tool_names)
            else:
                tools_str = "WebSearch, WebFetch, aii_web_fetch_grep"
        else:
            tools_str = ", ".join(tool_names) if tool_names else "none"
        telemetry.emit(MessageType.INFO, f"GenHypo - Generating {total_hypotheses} hypotheses ({seeded_per_llm} seeded + {unseeded_per_llm} unseeded per LLM × {num_models} models)")
        telemetry.emit(MessageType.INFO, f"   Provider: {llm_provider} | Tools: {tools_str} | Parallel: {max_parallel or 'unlimited'}")

        system_prompt_seeded = get_gen_hypo_sysprompt(seeded=True)
        system_prompt_unseeded = get_gen_hypo_sysprompt(seeded=False)
        if max_parallel is not None and max_parallel <= 0:
            max_parallel = 1
        sem = asyncio.Semaphore(max_parallel) if max_parallel else None

        # Build task configs
        task_configs = []
        seeded_idx = 0
        task_sequence = 0

        for model_config in models:
            model_name = model_config["model"]
            model_short = model_config.get("model_short", model_name)

            for hypo_idx in range(seeded_per_llm):
                seeded_idx += 1
                task_sequence += 1
                task_id = f"hypo_v{task_sequence}__{model_short}"
                agent_inspiration = agent_prompts[seeded_idx - 1] if (seeded_idx - 1) < len(agent_prompts) else []
                seeds = [{"id": p.get('id', '?'), "prompt": p.get('prompt', '')} for p in agent_inspiration if isinstance(p, dict)]

                task_configs.append({
                    "task_id": task_id,
                    "task_sequence": task_sequence,
                    "prompt": get_gen_hypo_prompt(agent_inspiration, research_direction, web_search=research_grounding),
                    "model": model_name,
                    "model_config": model_config,
                    "is_seeded": True,
                    "seeds": seeds,
                })

        for model_config in models:
            model_name = model_config["model"]
            model_short = model_config.get("model_short", model_name)

            for hypo_idx in range(unseeded_per_llm):
                task_sequence += 1
                task_id = f"hypo_v{task_sequence}__{model_short}"

                task_configs.append({
                    "task_id": task_id,
                    "task_sequence": task_sequence,
                    "prompt": get_gen_hypo_prompt([], research_direction, web_search=research_grounding),
                    "model": model_name,
                    "model_config": model_config,
                    "is_seeded": False,
                    "seeds": [],
                })

        # =====================================================================
        # RUN TASKS
        # =====================================================================
        async def run_task(task_cfg: dict):
            async with sem if sem else nullcontext():
                task_id = task_cfg["task_id"]
                sys_prompt = system_prompt_seeded if task_cfg["is_seeded"] else system_prompt_unseeded

                if use_claude_agent:
                    result = await _run_task_claude_agent(
                        task_id=task_id,
                        prompt=task_cfg["prompt"],
                        system_prompt=sys_prompt,
                        model=claude_cfg.model,
                        max_turns=claude_cfg.max_turns,
                        cwd=agent_cwd,
                        output_dir=output_dir,
                        mcp_servers=mcp_servers,
                        disallowed_tools=disallowed_tools,
                        telemetry=telemetry,
                        task_sequence=task_cfg["task_sequence"],
                        agent_timeout=claude_cfg.agent_timeout,
                        agent_retries=claude_cfg.agent_retries,
                        seq_prompt_timeout=claude_cfg.seq_prompt_timeout,
                        seq_prompt_retries=claude_cfg.seq_prompt_retries,
                        message_timeout=claude_cfg.message_timeout,
                        message_retries=claude_cfg.message_retries,
                    )
                else:
                    model_config = task_cfg["model_config"]
                    result = await _run_task_openrouter(
                        task_id=task_id,
                        task_name=task_id,
                        prompt=task_cfg["prompt"],
                        system_prompt=sys_prompt,
                        model=model_config["model"],
                        api_key=config.api_keys.openrouter,
                        timeout=llm_client.llm_timeout,
                        reasoning_effort=model_config.get("reasoning_effort"),
                        tools=or_tools,
                        telemetry=telemetry,
                    )

                return {
                    "task_id": task_id,
                    "model": task_cfg["model"],
                    "is_seeded": task_cfg["is_seeded"],
                    "seeds": task_cfg["seeds"],
                    "result": result,
                }

        task_results = await asyncio.gather(*[run_task(cfg) for cfg in task_configs], return_exceptions=True)

        # Collect results
        hypotheses = []
        for tr in task_results:
            if isinstance(tr, Exception):
                telemetry.emit(MessageType.WARNING, f"GenHypo task failed: {tr}")
                continue
            if tr["result"]:
                hypotheses.append({
                    "hypothesis_id": tr["task_id"],
                    "model": tr["model"],
                    "is_seeded": tr["is_seeded"],
                    "seeds": tr["seeds"],
                    **tr["result"],
                })

        telemetry.emit(MessageType.SUCCESS, f"GenHypo completed - {len(hypotheses)}/{total_hypotheses} hypotheses generated")

        seeded_success = sum(1 for h in hypotheses if h.get("is_seeded"))
        unseeded_success = sum(1 for h in hypotheses if not h.get("is_seeded"))

        # Build return value (for in-memory pipeline flow)
        module_output = GenHypoOut(
            output_dir=str(output_dir),
            hypotheses=hypotheses,
        )

        # Emit standardized module output
        std_output = build_module_output(
            module="gen_hypo",
            outputs=hypotheses,
            cumulative=cumulative or {},
            output_dir=output_dir,
            llm_provider=llm_provider,
            models=[m["model"] for m in models],
            seeded_hypos_per_llm=seeded_per_llm,
            unseeded_hypos_per_llm=unseeded_per_llm,
            total_seeded=total_seeded,
            total_unseeded=total_unseeded,
            total_hypotheses=total_hypotheses,
            research_grounding=research_grounding,
            tools=tool_names or [],
            successful=len(hypotheses),
            seeded_successful=seeded_success,
            unseeded_successful=unseeded_success,
            research_direction=research_direction,
        )
        emit_module_output(std_output, telemetry, output_dir=output_dir)
        telemetry.emit_module_summary("GEN_HYPO")

        return module_output

    finally:
        # Remove module sinks to prevent leak into subsequent modules
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        # Only flush/close if we created local telemetry
        if local_telemetry:
            telemetry.flush()


async def main():
    """Main function for standalone execution."""
    config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
    config = PipelineConfig.from_yaml(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/{timestamp}_gen_hypo")
    run_dir.mkdir(parents=True, exist_ok=True)

    result = await run_gen_hypo_module(config, agent_prompts=None, run_dir=run_dir)

    if result:
        print(f"Generated {len(result['hypotheses'])} hypotheses")
        return 0
    else:
        print("Hypothesis generation failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
