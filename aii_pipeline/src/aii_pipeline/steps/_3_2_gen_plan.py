"""GEN_PLAN Step - Elaborate artifact_directions into detailed plans.

Takes artifact_directions from ALL strategies and elaborates each into
detailed plans. For each (artifact_direction, llm) combination, we generate
`plans_per_strat` plans.

Each artifact type has its own plan schema:
- proof: informal_proof_draft, rationale
- research: research_plan, rationale
- dataset: ideal_dataset_criteria, dataset_search_plan
- experiment: implementation_pseudocode, fallback_plan, dependencies, testing_plan
- evaluation: metrics_descriptions, metrics_justification

Total plans = total_artifact_directions_across_strats × num_models × plans_per_strat

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
from datetime import datetime
from pathlib import Path

from aii_lib import OpenRouterClient, AIITelemetry, MessageType, JSONSink, chat, get_model_short, get_tooluniverse_mcp_config
from aii_lib.agent_backend import Agent, AgentOptions, aggregate_summaries

from aii_pipeline.utils import PipelineConfig, rel_path
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.schema import Strategy, ArtifactDirection
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.schema import BasePlan, PlanType
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.schema import ArtifactType
from ._invention_loop.pools import PlanPool, ArtifactPool
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.s_prompt import (
    get as get_plan_system_prompt,
)
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.u_prompt import (
    get as get_planner_prompt,
)
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.schema import (
    get_plan_schema,
)


async def gen_plan_for_art(
    telemetry: AIITelemetry,
    task_id: str,
    task_name: str,
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    iteration: int,
    artifact_direction: ArtifactDirection,
    reasoning_effort: str = "medium",
    suffix: str | None = None,
    llm_timeout: int = 600,
) -> dict | None:
    """Generate a plan for a single artifact direction using OpenRouter."""
    telemetry.emit_task_start(task_id, task_name)
    callback = telemetry.create_callback(task_id, task_name, group="gen_plan")

    # Get type-specific plan class and its LLM schema
    plan_cls = get_plan_schema(artifact_direction.type)

    try:
        effective_model = f"{model}:{suffix}" if suffix else model

        async with OpenRouterClient(
            api_key=api_key, model=effective_model, timeout=llm_timeout
        ) as client:
            result = await chat(
                client=client,
                prompt=prompt,
                system=system_prompt,
                reasoning_effort=reasoning_effort,
                response_format=plan_cls.plan_output_format()["schema"],
                message_callback=callback,
                timeout=llm_timeout,
            )

            output_text = client.extract_json_from_response(result.response)
            output_text = output_text.strip() if output_text else ""
            if output_text:
                plan = json.loads(output_text)

                # Add metadata (only fields that come from code, not schema)
                plan["type"] = artifact_direction.type
                plan["in_art_direction_id"] = artifact_direction.id
                # Inherit dependencies from artifact direction
                plan["artifact_dependencies"] = artifact_direction.depends_on

                telemetry.emit_task_end(task_id, task_name, "1 plan")
                return plan

        telemetry.emit_task_end(task_id, task_name, "No output")
        return None

    except asyncio.TimeoutError:
        telemetry.emit_task_end(task_id, task_name, f"Timeout ({llm_timeout}s)")
        raise
    except json.JSONDecodeError as e:
        telemetry.emit_task_end(task_id, task_name, f"JSON parse error: {e}")
        raise
    except Exception as e:
        telemetry.emit(
            MessageType.ERROR, f"OpenRouter plan generation failed for {model}: {e}"
        )
        telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
        raise


async def gen_plan_for_art_claude_agent(
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
    artifact_direction: ArtifactDirection,
    agent_timeout: int | None = None,
    agent_retries: int = 2,
    seq_prompt_timeout: int | None = None,
    seq_prompt_retries: int = 5,
    message_timeout: int | None = None,
    message_retries: int = 3,
    mcp_servers: dict | None = None,
    disallowed_tools: list[str] | None = None,
) -> dict | None:
    """Generate a plan for a single artifact direction using Claude agent."""
    telemetry.emit_task_start(task_id, task_name)

    # Get type-specific plan class and its LLM schema
    plan_cls = get_plan_schema(artifact_direction.type)

    try:
        options = AgentOptions(
            model=model,
            cwd=cwd,
            max_turns=max_turns,
            permission_mode="bypassPermissions",
            system_prompt=system_prompt,
            mcp_servers=mcp_servers or {},
            disallowed_tools=disallowed_tools,
            telemetry=telemetry,
            run_id=task_id,
            agent_context=task_id,
            agent_timeout=agent_timeout,
            agent_retries=agent_retries,
            seq_prompt_timeout=seq_prompt_timeout,
            seq_prompt_retries=seq_prompt_retries,
            message_timeout=message_timeout,
            message_retries=message_retries,
            output_format=plan_cls.plan_output_format(),
        )

        agent = Agent(options)
        response = await agent.run(prompt)

        # Store summary for module aggregation (agent already emits to console)
        if response.prompt_results:
            aggregated = aggregate_summaries(response.prompt_results)
            if aggregated:
                telemetry._pending_summaries.append(aggregated)

        if response.failed:
            err = response.error_message or "unknown error"
            telemetry.emit(MessageType.ERROR, f"Plan agent failed for {task_id}: {err}")
            telemetry.emit_task_end(task_id, task_name, f"Agent failed: {err}")
            return None

        if response.structured_output:
            plan = (
                response.structured_output
                if isinstance(response.structured_output, dict)
                else response.structured_output
            )

            # Add metadata (only fields that come from code, not schema)
            plan["type"] = artifact_direction.type
            plan["in_art_direction_id"] = artifact_direction.id
            plan["artifact_dependencies"] = artifact_direction.depends_on

            telemetry.emit_task_end(task_id, task_name, "1 plan")
            return plan

        telemetry.emit_task_end(task_id, task_name, "No output")
        return None

    except Exception as e:
        telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
        raise


def _get_creatable_types(
    artifact_pool: ArtifactPool,
    allowed_artifacts: list[str] | None = None,
) -> list[str]:
    """
    Determine which artifact types can be created in the current iteration.

    Types without dependencies (always creatable if allowed):
    - research, dataset, proof

    Types with dependencies:
    - experiment: requires at least one DATASET artifact in pool
    - evaluation: requires at least one EXPERIMENT artifact in pool
    """
    all_types = ["research", "dataset", "proof", "experiment", "evaluation"]
    allowed = allowed_artifacts if allowed_artifacts else all_types

    creatable = []
    for artifact_type in allowed:
        if artifact_type in ["research", "dataset", "proof"]:
            creatable.append(artifact_type)
        elif artifact_type == "experiment":
            has_dataset = any(
                a.type == ArtifactType.DATASET for a in artifact_pool.get_all()
            )
            if has_dataset:
                creatable.append(artifact_type)
        elif artifact_type == "evaluation":
            has_experiment = any(
                a.type == ArtifactType.EXPERIMENT for a in artifact_pool.get_all()
            )
            if has_experiment:
                creatable.append(artifact_type)

    return creatable


def _create_testing_strategy(
    artifact_pool: ArtifactPool,
    allowed_artifacts: list[str] | None,
    iteration: int,
) -> Strategy:
    """Create a synthetic strategy for testing mode with one artifact direction per creatable type."""
    creatable_types = _get_creatable_types(artifact_pool, allowed_artifacts)

    artifact_directions = []
    for i, artifact_type in enumerate(creatable_types, start=1):
        # Find a valid dependency for types that need them
        depends_on = []
        if artifact_type == "experiment":
            datasets = [a for a in artifact_pool.get_all() if a.type == ArtifactType.DATASET]
            if datasets:
                depends_on = [datasets[0].id]
        elif artifact_type == "evaluation":
            experiments = [a for a in artifact_pool.get_all() if a.type == ArtifactType.EXPERIMENT]
            if experiments:
                depends_on = [experiments[0].id]

        artifact_directions.append(
            ArtifactDirection(
                id=f"test_{artifact_type}_iter{iteration}_idx{i}",
                type=artifact_type,
                objective=f"Test {artifact_type} generation for iteration {iteration}",
                approach=f"Generate a test {artifact_type} to verify the executor works",
                depends_on=depends_on,
            )
        )

    return Strategy(
        id=f"test_strat_it{iteration}__testing",
        title="Testing Strategy",
        objective="Test all artifact types",
        rationale="Synthetic strategy for testing mode",
        artifact_directions=artifact_directions,
        expected_outcome="One artifact of each creatable type",
    )


async def run_gen_plan_module(
    config: PipelineConfig,
    hypothesis: dict,
    plan_pool: PlanPool,
    artifact_pool: ArtifactPool,
    iteration: int,
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
    strategies: list[Strategy] | None = None,
    cumulative: dict | None = None,
) -> list[BasePlan]:
    """
    Run the GEN_PLAN step.

    Takes artifact_directions from ALL strategies and elaborates each into
    detailed plans. All strategies' directions are processed in parallel.

    Args:
        config: Pipeline configuration
        hypothesis: The research hypothesis
        plan_pool: Pool to add plans to
        artifact_pool: Existing artifacts (for context)
        iteration: Current iteration number
        telemetry: Telemetry instance
        output_dir: Directory for output files
        strategies: All strategies with artifact_directions to elaborate

    Returns:
        List of Plan objects added to the pool
    """
    telemetry.start_module("GEN_PLAN")

    telemetry.emit(MessageType.INFO, "=" * 60)
    telemetry.emit(
        MessageType.INFO, f"GEN_PLAN - Elaborating artifact directions for iteration {iteration}"
    )
    telemetry.emit(MessageType.INFO, "=" * 60)

    # Check for testing mode
    testing_mode = config.invention_loop.test_all_artifacts
    allowed_artifacts = config.invention_loop.allowed_artifacts

    if testing_mode:
        telemetry.emit(MessageType.INFO, "   TESTING MODE: Creating synthetic strategy")
        test_strat = _create_testing_strategy(artifact_pool, allowed_artifacts, iteration)
        strategies = [test_strat]
        telemetry.emit(
            MessageType.INFO,
            f"   Created test strategy with {len(test_strat.artifact_directions)} artifact directions",
        )

    if not strategies:
        telemetry.emit(MessageType.ERROR, "No strategies provided - cannot generate plans")
        return []

    # Collect artifact_directions from ALL strategies, tracking which strategy each came from
    artifact_directions = []
    direction_to_strategy: dict[str, str] = {}  # direction_id -> strategy_id
    for strat in strategies:
        for direction in strat.artifact_directions:
            direction_to_strategy[direction.id] = strat.id
        artifact_directions.extend(strat.artifact_directions)

    if not artifact_directions:
        telemetry.emit(MessageType.WARNING, "Strategies have no artifact_directions - skipping gen_plan")
        return []

    gen_plan_cfg = config.invention_loop.gen_plan
    use_claude_agent = gen_plan_cfg.use_claude_agent
    plans_per_strat = gen_plan_cfg.plans_per_strat
    openrouter_key = config.api_keys.openrouter

    # Setup backend
    # Step subdir within iteration dir (always created regardless of backend)
    if output_dir:
        step_dir = (output_dir / "gen_plan").resolve()
        step_dir.mkdir(parents=True, exist_ok=True)
        output_dir = step_dir

    if use_claude_agent:
        claude_cfg = gen_plan_cfg.claude_agent
        max_parallel = claude_cfg.max_concurrent_agents

        # Always connect MCP for aii_web_fetch_grep.
        # When use_aii_web_tools=True: all 3 MCP tools replace built-in WebSearch/WebFetch.
        # When use_aii_web_tools=False: only grep via MCP, built-in WebSearch/WebFetch for the rest.
        mcp_servers = get_tooluniverse_mcp_config(use_aii_server=True)
        dblp_block = [
            "mcp__aii_tooluniverse__dblp_bib_search",
            "mcp__aii_tooluniverse__dblp_bib_fetch",
        ]
        # Disallow Bash — gen_plan is a planner, not an executor.
        # Web research is fine, but code execution belongs in gen_art.
        bash_block = ["Bash"]
        if claude_cfg.use_aii_web_tools:
            disallowed_tools = ["WebSearch", "WebFetch"] + dblp_block + bash_block
        else:
            disallowed_tools = [
                "mcp__aii_tooluniverse__aii_web_search_fast",
                "mcp__aii_tooluniverse__aii_web_fetch_direct",
            ] + dblp_block + bash_block

        models = [
            {"model": claude_cfg.model, "model_short": get_model_short(claude_cfg.model)}
        ]
        llm_provider = "claude_agent"
        llm_timeout = claude_cfg.seq_prompt_timeout
    else:
        llm_cfg = gen_plan_cfg.llm_client
        llm_timeout = llm_cfg.llm_timeout
        max_parallel = None
        llm_provider = "openrouter"
        models = [
            {
                "model": m.model,
                "reasoning_effort": m.reasoning_effort,
                "suffix": m.suffix,
            }
            for m in llm_cfg.models
        ]

    # Add module-specific JSON sink (after backend setup so it uses step_dir when applicable)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "gen_plan_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "gen_plan_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    num_models = len(models)
    num_directions = len(artifact_directions)
    total_tasks = num_directions * num_models * plans_per_strat

    telemetry.emit(MessageType.INFO, f"   Provider: {llm_provider}")
    telemetry.emit(MessageType.INFO, f"   Strategies: {len(strategies)} ({', '.join(s.id for s in strategies)})")
    telemetry.emit(MessageType.INFO, f"   Artifact directions (combined): {num_directions}")
    telemetry.emit(MessageType.INFO, f"   Models: {[m['model'] for m in models]}")
    telemetry.emit(MessageType.INFO, f"   Plans per strat: {plans_per_strat}")
    telemetry.emit(MessageType.INFO, f"   Total tasks: {total_tasks}")
    telemetry.emit(MessageType.INFO, f"   Timeout: {llm_timeout}s")

    # Build task configs: one task per (artifact_direction, model, plan_num) combination
    task_configs = []
    task_counter = 0

    for direction in artifact_directions:
        # Get type-specific system prompt for this artifact type
        system_prompt = get_plan_system_prompt(direction.type)

        for model_cfg in models:
            for plan_num in range(plans_per_strat):
                task_counter += 1
                model_short = get_model_short(model_cfg["model"])
                task_id = f"plan_{direction.id}_v{plan_num + 1}_it{iteration}__{model_short}"

                prompt = get_planner_prompt(
                    hypothesis=hypothesis,
                    artifact_pool=artifact_pool,
                    artifact_direction=direction,
                )

                task_configs.append((task_id, prompt, system_prompt, model_cfg, direction))

    telemetry.emit(MessageType.INFO, f"   Running {len(task_configs)} planners...")

    # Run all planners in parallel
    sem = asyncio.Semaphore(max_parallel) if max_parallel else None

    async def run_task(
        task_id: str,
        prompt: str,
        system_prompt: str,
        model_cfg: dict,
        direction: ArtifactDirection,
    ):
        async with sem if sem else nullcontext():
            if use_claude_agent:
                # Per-task CWD so parallel agents don't collide
                task_cwd = (output_dir / task_id) if output_dir else Path.cwd().resolve() / task_id
                task_cwd.mkdir(parents=True, exist_ok=True)
                return task_id, direction.id, await gen_plan_for_art_claude_agent(
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
                    artifact_direction=direction,
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
                return task_id, direction.id, await gen_plan_for_art(
                    telemetry=telemetry,
                    task_id=task_id,
                    task_name=task_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model_cfg["model"],
                    api_key=openrouter_key,
                    iteration=iteration,
                    artifact_direction=direction,
                    reasoning_effort=model_cfg.get("reasoning_effort", "medium"),
                    suffix=model_cfg.get("suffix"),
                    llm_timeout=llm_timeout,
                )

    task_results = await asyncio.gather(
        *[
            run_task(task_id, prompt, system_prompt, model_cfg, direction)
            for task_id, prompt, system_prompt, model_cfg, direction in task_configs
        ],
        return_exceptions=True,
    )

    # Assemble Plan objects from task results — idx suffix for ordering visibility in logs
    all_plans: list[BasePlan] = []
    existing_artifact_ids = {a.id for a in artifact_pool.get_all()}
    plan_idx = 0

    for tr in task_results:
        if isinstance(tr, Exception):
            telemetry.emit(MessageType.WARNING, f"GenPlan task failed: {tr}")
            continue
        task_id, direction_id, p_dict = tr
        if p_dict is None:
            continue
        try:
            plan_idx += 1
            p_type = PlanType(p_dict.get("type", "research"))

            # Filter out non-existent artifact dependencies
            raw_deps = p_dict.get("artifact_dependencies", [])
            valid_deps = [d for d in raw_deps if d in existing_artifact_ids]
            dropped = [d for d in raw_deps if d not in existing_artifact_ids]
            if dropped:
                telemetry.emit(
                    MessageType.INFO,
                    f"Dropped invalid deps from '{p_dict.get('title', 'Untitled')[:30]}': {dropped}",
                )

            plan_id = f"{task_id}_idx{plan_idx}"

            # Build concrete subclass with flat kwargs
            metadata_keys = {"type", "in_art_direction_id", "in_strat_id", "artifact_dependencies"}
            content_fields = {k: v for k, v in p_dict.items() if k not in metadata_keys}

            cls = get_plan_schema(p_type.value)
            plan = cls(
                id=plan_id,
                artifact_dependencies=valid_deps,
                in_art_direction_id=p_dict.get("in_art_direction_id"),
                in_strat_id=direction_to_strategy.get(direction_id),
                **content_fields,
            )

            plan = plan_pool.add(plan)
            all_plans.append(plan)

        except Exception as e:
            telemetry.emit(MessageType.WARNING, f"Failed to create plan: {e}")
            raise

    telemetry.emit(
        MessageType.SUCCESS, f"GEN_PLAN complete: {len(all_plans)} plans generated"
    )

    # Compute total cost from pending summaries before emit_module_summary clears them
    total_cost_usd = sum(s.get("total_cost", 0.0) for s in telemetry._pending_summaries)

    # Build and emit module output
    from aii_pipeline.utils import build_module_output, emit_module_output
    module_output = build_module_output(
        module="gen_plan",
        iteration=iteration,
        outputs=all_plans,
        cumulative=cumulative or {},
        total_cost_usd=total_cost_usd,
        output_dir=output_dir,
        llm_provider=llm_provider,
    )
    emit_module_output(module_output, telemetry, output_dir=output_dir)

    telemetry.emit_module_summary("GEN_PLAN")

    # Remove module sinks to prevent leak into subsequent modules
    for sink in module_sinks:
        sink.flush()
        telemetry.remove_sink(sink)

    return all_plans
