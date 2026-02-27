"""GEN_STRAT Step - Generate research strategies from multiple LLMs.

Each strategy contains:
- title, objective, rationale: What we're doing and why
- artifact_directions: Artifacts to create THIS iteration (IDs assigned by code after LLM output)
- expected_outcome: What we'll have after this iteration
Each strategy's artifact directions are elaborated into detailed plans.

Supports two backends:
- OpenRouter (default): Uses chat() with structured output
- Claude agent: Uses Agent with SDK native output_format for structured output

Verification + Retry (similar to verify_citations in audit_hypo):
- Verifies: strategy count, valid dependencies
- Retries with conversation continuation if verification fails

Uses aii_lib for:
- OpenRouterClient: LLM calls (OpenRouter backend)
- Agent/AgentOptions: Claude agent calls
- AIITelemetry: Task tracking and logging
"""

import asyncio
import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from aii_lib import OpenRouterClient, AIITelemetry, MessageType, JSONSink, chat, get_model_short, get_tooluniverse_mcp_config
from aii_lib.agent_backend import Agent, AgentOptions

from aii_pipeline.utils import PipelineConfig, rel_path
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.schema import Strategy, ArtifactDirection
from ._invention_loop.pools import ArtifactPool, NarrativePool
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.s_prompt import get as get_gen_strat_system_prompt
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.u_prompt import (
    get as get_gen_strat_prompt,
    build_artifact_retry_prompt,
)
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.schema import (
    Strategies,
    assign_artifact_direction_ids,
    verify_strategies,
)


# =============================================================================
# OPENROUTER BACKEND (with verification + retry)
# =============================================================================

async def gen_strat(
    telemetry: AIITelemetry,
    task_id: str,
    task_name: str,
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    iteration: int,
    existing_artifact_ids: set[str],
    artifact_pool_map: dict[str, str],
    num_strategies: int,
    reasoning_effort: str = "medium",
    suffix: str | None = None,
    llm_timeout: int = 600,
    verify_retries: int = 2,
    min_valid_artifacts: int = 1,
    allowed_artifacts: list[str] | None = None,
    art_limit: int | None = None,
) -> list[dict]:
    """Generate strategies using aii_lib chat() with structured output.

    Includes verification + retry loop similar to cited_args workflow.
    """
    telemetry.emit_task_start(task_id, task_name)
    callback = telemetry.create_callback(task_id, task_name, group="gen_strat")

    try:
        effective_model = f"{model}:{suffix}" if suffix else model

        async with OpenRouterClient(api_key=api_key, model=effective_model, timeout=llm_timeout) as client:
            # Initial generation
            result = await chat(
                client=client,
                prompt=prompt,
                system=system_prompt,
                reasoning_effort=reasoning_effort,
                response_format=Strategies,
                message_callback=callback,
                timeout=llm_timeout,
                emit_summary=False,  # Summary emitted after verification
            )

            messages = result.messages
            conv_stats = result.stats

            output_text = client.extract_json_from_response(result.response)
            output_text = output_text.strip() if output_text else ""

            if not output_text:
                telemetry.emit_task_end(task_id, task_name, "No output")
                return []

            data = json.loads(output_text)
            strategies = data.get("strategies", [])

            # Assign IDs to artifact directions (LLM doesn't generate IDs)
            # Use working copy so IDs accumulate across strategies but existing_artifact_ids stays clean for verification
            id_tracker = set(existing_artifact_ids)
            for s in strategies:
                assign_artifact_direction_ids(s, id_tracker, iteration)

            # Verification + retry loop
            for attempt in range(verify_retries + 1):
                verification = verify_strategies(
                    strategies=strategies,
                    num_expected=num_strategies,
                    existing_artifact_ids=existing_artifact_ids,
                    artifact_pool_map=artifact_pool_map,
                    min_valid_artifacts=min_valid_artifacts,
                    allowed_artifacts=allowed_artifacts,
                    art_limit=art_limit,
                )

                if verification["valid"]:
                    status = f"{len(strategies)} strateg{'ies' if len(strategies) != 1 else 'y'}"
                    if attempt > 0:
                        status += f" (retry {attempt})"
                    telemetry.emit_task_end(task_id, task_name, status)
                    return strategies

                # Log issues
                all_errors = verification["count_errors"] + verification["id_errors"] + verification["dep_errors"] + verification["type_errors"] + verification["limit_errors"]
                for err in all_errors[:5]:  # Limit logging
                    telemetry.emit_message("WARNING", err, task_name, task_id)

                # Log valid artifact count if below minimum
                valid_count = verification.get("valid_artifact_count", 0)
                total_count = verification.get("total_artifact_count", 0)
                if valid_count < min_valid_artifacts:
                    telemetry.emit_message(
                        "WARNING",
                        f"Only {valid_count}/{total_count} valid artifacts (need {min_valid_artifacts})",
                        task_name, task_id
                    )

                # Retry if attempts left
                if attempt < verify_retries:
                    retry_prompt = build_artifact_retry_prompt(
                        verification=verification,
                        num_strategies_requested=num_strategies,
                        min_valid_artifacts=min_valid_artifacts,
                        art_limit=art_limit,
                    )
                    telemetry.emit_message(
                        "RETRY",
                        f"Verification failed ({len(all_errors)} issues), retrying...",
                        task_name, task_id
                    )

                    messages.append({"role": "user", "content": retry_prompt})

                    result = await chat(
                        client=client,
                        messages=messages,
                        reasoning_effort=reasoning_effort,
                        response_format=Strategies,
                        message_callback=callback,
                        timeout=llm_timeout,
                        conversation_stats=conv_stats,
                        emit_summary=False,
                    )

                    messages = result.messages
                    output_text = client.extract_json_from_response(result.response)
                    output_text = output_text.strip() if output_text else ""

                    if output_text:
                        data = json.loads(output_text)
                        strategies = data.get("strategies", [])
                        # Assign IDs - fresh tracker since retry replaces all strategies
                        id_tracker = set(existing_artifact_ids)
                        for s in strategies:
                            assign_artifact_direction_ids(s, id_tracker, iteration)
                    else:
                        telemetry.emit_message("RETRY", "Retry produced no output, keeping previous", task_name, task_id)

            # All retries exhausted - return what we have
            status = f"{len(strategies)} strateg{'ies' if len(strategies) != 1 else 'y'} (invalid)"
            telemetry.emit_task_end(task_id, task_name, status)
            return strategies

    except asyncio.TimeoutError:
        telemetry.emit_task_end(task_id, task_name, f"Timeout ({llm_timeout}s)")
        raise
    except json.JSONDecodeError as e:
        telemetry.emit_task_end(task_id, task_name, f"JSON parse error: {e}")
        raise
    except Exception as e:
        telemetry.emit(MessageType.ERROR, f"Strategy generation failed for {model}: {e}")
        telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
        raise


# =============================================================================
# CLAUDE AGENT BACKEND (with verification + retry)
# =============================================================================

async def gen_strat_claude_agent(
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
    existing_artifact_ids: set[str],
    artifact_pool_map: dict[str, str],
    num_strategies: int,
    agent_timeout: int | None = None,
    agent_retries: int = 2,
    seq_prompt_timeout: int | None = None,
    seq_prompt_retries: int = 5,
    message_timeout: int | None = None,
    message_retries: int = 3,
    verify_retries: int = 2,
    min_valid_artifacts: int = 1,
    allowed_artifacts: list[str] | None = None,
    art_limit: int | None = None,
    mcp_servers: dict | None = None,
    disallowed_tools: list[str] | None = None,
) -> list[dict]:
    """Generate strategies using Claude agent with structured JSON output.

    Includes verification + retry loop with conversation continuation.
    """
    # Use absolute path to avoid CWD mismatch issues with agent
    abs_cwd = Path(cwd).resolve()
    output_file = str(abs_cwd / f"{task_id}.json")

    telemetry.emit_task_start(task_id, task_name)
    callback = telemetry.create_callback(task_id, task_name, group="gen_strat")

    # Track all responses for aggregated summary
    all_responses: list = []

    def _emit_aggregated_summary():
        """Emit aggregated summary from all agent responses."""
        if not all_responses:
            return
        all_summaries = []
        for resp in all_responses:
            for pr in resp.prompt_results:
                if pr.summary_data:
                    all_summaries.append(pr.summary_data)
        if not all_summaries:
            return
        summary = {
            "type": "summary",
            "run_id": task_id,
            "agent_context": task_name,
            "total_cost": sum(s.get("total_cost", 0) for s in all_summaries),
            "token_cost": sum(s.get("token_cost", 0) for s in all_summaries),
            "tool_cost": sum(s.get("tool_cost", 0) for s in all_summaries),
            "model": all_summaries[0].get("model", "claude-agent") if all_summaries else "claude-agent",
            "status": "aggregated",
            "is_aggregated": len(all_summaries) > 1,
            "num_calls": sum(s.get("num_calls", 1) for s in all_summaries),
            "runtime_seconds": sum(s.get("runtime_seconds", 0) for s in all_summaries),
            "input_tokens": sum(s.get("input_tokens", 0) for s in all_summaries),
            "output_tokens": sum(s.get("output_tokens", 0) for s in all_summaries),
        }
        callback(summary)
        # Store for module-level cost aggregation (read by run_gen_strat_module)
        telemetry._pending_summaries.append(summary)

    try:
        options = AgentOptions(
            model=model,
            cwd=cwd,
            max_turns=max_turns,
            permission_mode="bypassPermissions",
            system_prompt=system_prompt,
            continue_seq_item=True,  # Enable conversation continuation for retries
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
            output_format=Strategies.to_struct_output(),
        )

        agent = Agent(options)

        # Initial generation
        response = await agent.run(prompt)
        all_responses.append(response)

        if response.failed:
            _emit_aggregated_summary()
            telemetry.emit_task_end(task_id, task_name, f"Agent failed: {response.error_message or 'unknown error'}")
            return []

        if not response.structured_output:
            _emit_aggregated_summary()
            telemetry.emit_task_end(task_id, task_name, "No output")
            return []

        data = response.structured_output if isinstance(response.structured_output, dict) else response.structured_output
        strategies = data.get("strategies", [])

        # Assign IDs to artifact directions (LLM doesn't generate IDs)
        # Use working copy so IDs accumulate across strategies but existing_artifact_ids stays clean for verification
        id_tracker = set(existing_artifact_ids)
        for s in strategies:
            assign_artifact_direction_ids(s, id_tracker, iteration)

        # Add metadata
        # Verification + retry loop
        for attempt in range(verify_retries + 1):
            verification = verify_strategies(
                strategies=strategies,
                num_expected=num_strategies,
                existing_artifact_ids=existing_artifact_ids,
                artifact_pool_map=artifact_pool_map,
                min_valid_artifacts=min_valid_artifacts,
                allowed_artifacts=allowed_artifacts,
                art_limit=art_limit,
            )

            if verification["valid"]:
                status = f"{len(strategies)} strateg{'ies' if len(strategies) != 1 else 'y'}"
                if attempt > 0:
                    status += f" (retry {attempt})"
                _emit_aggregated_summary()
                telemetry.emit_task_end(task_id, task_name, status)
                return strategies

            # Log issues
            all_errors = verification["count_errors"] + verification["id_errors"] + verification["dep_errors"] + verification["type_errors"] + verification["limit_errors"]
            for err in all_errors[:5]:  # Limit logging
                telemetry.emit_message("WARNING", err, task_name, task_id)

            # Log valid artifact count if below minimum
            valid_count = verification.get("valid_artifact_count", 0)
            total_count = verification.get("total_artifact_count", 0)
            if valid_count < min_valid_artifacts:
                telemetry.emit_message(
                    "WARNING",
                    f"Only {valid_count}/{total_count} valid artifacts (need {min_valid_artifacts})",
                    task_name, task_id
                )

            # Retry if attempts left
            if attempt < verify_retries:
                retry_prompt = build_artifact_retry_prompt(
                    verification=verification,
                    num_strategies_requested=num_strategies,
                    min_valid_artifacts=min_valid_artifacts,
                    art_limit=art_limit,
                )
                telemetry.emit_message(
                    "RETRY",
                    f"Verification failed ({len(all_errors)} issues), retrying...",
                    task_name, task_id
                )

                # Continue conversation with agent
                response = await agent.run(retry_prompt)
                all_responses.append(response)

                if response.structured_output:
                    data = response.structured_output if isinstance(response.structured_output, dict) else response.structured_output
                    strategies = data.get("strategies", [])
                    # Assign IDs - fresh tracker since retry replaces all strategies
                    id_tracker = set(existing_artifact_ids)
                    for s in strategies:
                        assign_artifact_direction_ids(s, id_tracker, iteration)
                else:
                    telemetry.emit_message("RETRY", "Retry produced no output, keeping previous", task_name, task_id)

        # All retries exhausted - return what we have
        _emit_aggregated_summary()
        status = f"{len(strategies)} strateg{'ies' if len(strategies) != 1 else 'y'} (invalid)"
        telemetry.emit_task_end(task_id, task_name, status)
        return strategies

    except Exception as e:
        _emit_aggregated_summary()
        telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
        raise


# =============================================================================
# MAIN STEP
# =============================================================================

async def run_gen_strat_module(
    config: PipelineConfig,
    hypothesis: dict,
    artifact_pool: ArtifactPool,
    narrative_pool: NarrativePool | None,
    iteration: int,
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
    previous_strategies: list[dict] | None = None,
    cumulative: dict | None = None,
) -> list[Strategy]:
    """
    Run the GEN_STRAT step.

    Generates strategies from multiple LLMs. Each strategy contains
    artifact directions that will be elaborated into plans.
    """
    telemetry.start_module("GEN_STRAT")

    telemetry.emit(MessageType.INFO, "=" * 60)
    telemetry.emit(MessageType.INFO, f"GEN_STRAT - Generating strategies for iteration {iteration}")
    telemetry.emit(MessageType.INFO, "=" * 60)

    # Get config
    gen_strat_cfg = config.invention_loop.gen_strat
    max_iterations = config.invention_loop.max_iterations
    allowed_artifacts = config.invention_loop.allowed_artifacts
    strats_per_call = gen_strat_cfg.strats_per_call  # Strategies generated per LLM call
    calls_per_llm = gen_strat_cfg.calls_per_llm  # Parallel calls per model
    max_parallel = gen_strat_cfg.max_concurrent  # Applies to both OpenRouter and Claude agent
    use_claude_agent = gen_strat_cfg.use_claude_agent

    # Verification config
    verify_cfg = gen_strat_cfg.verify_artifacts
    verify_retries = verify_cfg.retry
    min_valid_artifacts = verify_cfg.min_valid_artifacts
    art_limit = gen_strat_cfg.art_limit
    artifact_context_per_type = gen_strat_cfg.artifact_context_per_type

    # =========================================================================
    # SETUP BACKEND
    # =========================================================================
    # Step subdir within iteration dir (always created regardless of backend)
    if output_dir:
        step_dir = (output_dir / "gen_strat").resolve()
        step_dir.mkdir(parents=True, exist_ok=True)
        output_dir = step_dir

    if use_claude_agent:
        claude_cfg = gen_strat_cfg.claude_agent

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
        llm_cfg = gen_strat_cfg.llm_client
        llm_timeout = llm_cfg.llm_timeout
        llm_provider = "openrouter"

        # Parse models from config
        models = [{"model": m.model, "reasoning_effort": m.reasoning_effort, "suffix": m.suffix} for m in llm_cfg.models]

    # Add module-specific JSON sink (after backend setup so it uses step_dir when applicable)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "gen_strat_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "gen_strat_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    openrouter_key = config.api_keys.openrouter

    num_models = len(models)
    total_calls = calls_per_llm * num_models
    total_strategies = strats_per_call * total_calls

    # Get existing artifact IDs and types from the pool (for verification)
    existing_artifact_ids = {a.id for a in artifact_pool.get_all()}
    artifact_pool_map = {a.id: a.type for a in artifact_pool.get_all()}

    model_names = [m["model"] for m in models]
    telemetry.emit(MessageType.INFO, f"   Provider: {llm_provider}")
    telemetry.emit(MessageType.INFO, f"   Models: {model_names}")
    telemetry.emit(MessageType.INFO, f"   Strategies: {total_strategies} ({strats_per_call}/call x {calls_per_llm} calls x {num_models} models)")
    telemetry.emit(MessageType.INFO, f"   Iteration: {iteration} of {max_iterations}")
    telemetry.emit(MessageType.INFO, f"   Narratives available: {len(narrative_pool.get_all()) if narrative_pool else 0}")
    telemetry.emit(MessageType.INFO, f"   Allowed artifacts: {allowed_artifacts if allowed_artifacts else 'all'}")
    telemetry.emit(MessageType.INFO, f"   Verify: retries={verify_retries}, min_valid={min_valid_artifacts}")
    telemetry.emit(MessageType.INFO, f"   Art limit: {art_limit if art_limit else 'none'}")
    telemetry.emit(MessageType.INFO, f"   Timeout: {llm_timeout}s")

    # Build system prompt
    system_prompt = get_gen_strat_system_prompt(allowed_artifacts)

    # Build task configs - calls_per_llm tasks per model
    task_configs = []
    task_sequence = 0

    for model_cfg in models:
        model_name = model_cfg["model"]
        model_short = get_model_short(model_name) if not use_claude_agent else model_cfg.get("model_short", model_name)

        for call_idx in range(calls_per_llm):
            task_sequence += 1
            task_id = f"strat_v{task_sequence}_it{iteration}__{model_short}"

            prompt = get_gen_strat_prompt(
                hypothesis=hypothesis,
                artifact_pool=artifact_pool,
                narrative_pool=narrative_pool,
                current_iteration=iteration,
                max_iterations=max_iterations,
                previous_strategies=previous_strategies,
                allowed_artifacts=allowed_artifacts,
                num_strategies=strats_per_call,
                art_limit=art_limit,
                artifact_context_per_type=artifact_context_per_type,
            )

            task_configs.append((task_id, prompt, model_cfg))

    # Run all generators in parallel (with optional semaphore for Claude agent)
    telemetry.emit(MessageType.INFO, f"   Running {len(task_configs)} strategy generators...")
    sem = asyncio.Semaphore(max_parallel) if max_parallel else None

    async def run_task(task_id: str, prompt: str, model_cfg: dict):
        async with sem if sem else nullcontext():
            if use_claude_agent:
                # Per-task CWD so parallel agents don't collide
                task_cwd = (output_dir / task_id) if output_dir else Path.cwd().resolve() / task_id
                task_cwd.mkdir(parents=True, exist_ok=True)
                return task_id, await gen_strat_claude_agent(
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
                    existing_artifact_ids=existing_artifact_ids,
                    artifact_pool_map=artifact_pool_map,
                    num_strategies=strats_per_call,
                    agent_timeout=claude_cfg.agent_timeout,
                    agent_retries=claude_cfg.agent_retries,
                    seq_prompt_timeout=claude_cfg.seq_prompt_timeout,
                    seq_prompt_retries=claude_cfg.seq_prompt_retries,
                    message_timeout=claude_cfg.message_timeout,
                    message_retries=claude_cfg.message_retries,
                    verify_retries=verify_retries,
                    min_valid_artifacts=min_valid_artifacts,
                    allowed_artifacts=allowed_artifacts,
                    art_limit=art_limit,
                    mcp_servers=mcp_servers,
                    disallowed_tools=disallowed_tools,
                )
            else:
                return task_id, await gen_strat(
                    telemetry=telemetry,
                    task_id=task_id,
                    task_name=task_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model_cfg["model"],
                    api_key=openrouter_key,
                    iteration=iteration,
                    existing_artifact_ids=existing_artifact_ids,
                    artifact_pool_map=artifact_pool_map,
                    num_strategies=strats_per_call,
                    reasoning_effort=model_cfg.get("reasoning_effort", "medium"),
                    suffix=model_cfg.get("suffix"),
                    llm_timeout=llm_timeout,
                    verify_retries=verify_retries,
                    min_valid_artifacts=min_valid_artifacts,
                    allowed_artifacts=allowed_artifacts,
                    art_limit=art_limit,
                )

    task_results = await asyncio.gather(*[
        run_task(task_id, prompt, model_cfg)
        for task_id, prompt, model_cfg in task_configs
    ], return_exceptions=True)

    # Collect all strategy dicts with their source task_id
    all_strategy_items: list[tuple[str, dict]] = []
    for tr in task_results:
        if isinstance(tr, Exception):
            telemetry.emit(MessageType.WARNING, f"GenStrat task failed: {tr}")
            continue
        task_id, strategy_dicts = tr
        if strategy_dicts is None:
            continue
        for s_dict in strategy_dicts:
            all_strategy_items.append((task_id, s_dict))

    # Convert to Strategy objects — use task_id + idx as the id
    all_strategies: list[Strategy] = []
    task_idx_counters: dict[str, int] = {}

    for task_id, s_dict in all_strategy_items:
        try:
            task_idx_counters[task_id] = task_idx_counters.get(task_id, 0) + 1
            idx = task_idx_counters[task_id]
            strategy_id = f"{task_id}_idx{idx}"

            # Convert artifact directions
            directions = []
            for a_dict in s_dict.get("artifact_directions", []):
                direction = ArtifactDirection(
                    id=a_dict.get("id", ""),
                    type=a_dict.get("type", "research"),
                    objective=a_dict.get("objective", ""),
                    approach=a_dict.get("approach", ""),
                    depends_on=a_dict.get("depends_on", []),
                )
                directions.append(direction)

            strategy = Strategy(
                id=strategy_id,
                title=s_dict.get("title", ""),
                objective=s_dict.get("objective", ""),
                rationale=s_dict.get("rationale", ""),
                artifact_directions=directions,
                expected_outcome=s_dict.get("expected_outcome", ""),
            )

            all_strategies.append(strategy)

        except Exception as e:
            telemetry.emit(MessageType.WARNING, f"Failed to create strategy: {e}")
            raise

    telemetry.emit(MessageType.SUCCESS, f"GEN_STRAT complete: {len(all_strategies)} strategies generated")

    # Compute total cost from pending summaries before emit_module_summary clears them
    total_cost_usd = sum(s.get("total_cost", 0.0) for s in telemetry._pending_summaries)

    # Build and emit module output
    from aii_pipeline.utils import build_module_output, emit_module_output
    module_output = build_module_output(
        module="gen_strat",
        iteration=iteration,
        outputs=all_strategies,
        cumulative=cumulative or {},
        total_cost_usd=total_cost_usd,
        output_dir=output_dir,
        llm_provider=llm_provider,
        verify_retries=verify_retries,
        min_valid_artifacts=min_valid_artifacts,
    )
    emit_module_output(module_output, telemetry, output_dir=output_dir)

    telemetry.emit_module_summary("GEN_STRAT")

    # Remove module sinks to prevent leak into subsequent modules
    for sink in module_sinks:
        sink.flush()
        telemetry.remove_sink(sink)

    return all_strategies
