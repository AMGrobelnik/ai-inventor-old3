"""RANK_PLAN Step - Per-artifact-direction Swiss-BT tournament ranking of plans.

Plans are grouped by in_art_direction_id and ranked independently within each group.
The best plan for each artifact_direction is selected for execution.
This means we select exactly one plan per artifact_direction in the strategy.

Uses SwissBTRanker for sample-efficient tournament ranking:
- Round 1: Random pairs (bootstrap initial estimates)
- Round 2+: Swiss-style pairing (adjacent ratings compete)
- Scoring: Bradley-Terry MLE (principled, order-invariant)
- Early stop: When top item's confidence interval separates

Supports two backends (consistent with rank_hypo and rank_strat):
- OpenRouter (default): Uses SwissBTRanker's built-in LLM calls
- Claude agent: Uses SwissBTRanker with use_claude_agent flag
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from aii_lib import AIITelemetry, MessageType, JSONSink, SwissBTRanker

from aii_pipeline.utils import PipelineConfig, rel_path

from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.schema import Strategy
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.schema import BasePlan
from aii_pipeline.prompts.steps.optional.ranking_schemas import RankingResult
from aii_pipeline.steps._invention_loop.pools import PlanPool, ArtifactPool
from aii_pipeline.prompts.steps.optional.rank_plan.s_prompt import get as get_rank_plan_system_prompt
from aii_pipeline.prompts.steps.optional.rank_plan.u_prompt import get as get_rank_plan_prompt
from aii_pipeline.prompts.steps.optional.ranking_schemas import PairwisePreference, PairwisePreferenceSimple


# ============================================================================
# Per-artifact-plan ranking helper
# ============================================================================

async def rank_plans_for_art(
    plans: list[BasePlan],
    direction_id: str,
    config: PipelineConfig,
    artifact_pool: ArtifactPool,
    strategy: Strategy,
    telemetry: AIITelemetry,
    iteration: int,
    output_dir: Path | None = None,
) -> tuple[list[BasePlan], dict[str, float], dict[str, float], list, dict]:
    """
    Rank plans for a single artifact_direction using SwissBTRanker.

    Args:
        plans: List of plans for this artifact_direction
        direction_id: The in_art_direction_id being ranked
        config: Pipeline configuration
        artifact_pool: Existing artifacts for context
        strategy: The winning strategy for context
        telemetry: Telemetry instance
        iteration: Current iteration
        output_dir: Output directory for Claude agent workspace

    Returns:
        Tuple of (ranked_plans, win_rates, bt_scores, all_comparisons, metadata)
    """
    n = len(plans)

    # If only 1 plan, no ranking needed
    if n == 1:
        return (
            plans,
            {plans[0].id: 1.0},
            {plans[0].id: 1.0},
            [],
            {"converged": True, "convergence_reason": "single_item", "rounds_completed": 0},
        )

    rank_cfg = config.invention_loop.rank_plan

    # Tournament parameters (same pattern as rank_strat)
    max_rounds = rank_cfg.max_rounds
    bootstrap_opponents = rank_cfg.bootstrap_opponents_per_item
    votes_per_pair_per_llm = rank_cfg.votes_per_pair_per_llm
    early_stop_win_prob = rank_cfg.early_stop_win_prob
    include_justification = rank_cfg.include_justification
    use_claude_agent = rank_cfg.use_claude_agent

    # =========================================================================
    # SETUP BACKEND (Claude agent or OpenRouter)
    # =========================================================================
    if use_claude_agent:
        # Claude agent path
        claude_cfg = rank_cfg.claude_agent

        # Create workspace for agent
        if output_dir:
            agent_cwd = (output_dir / f"claude_agent_{direction_id}").resolve()
        else:
            agent_cwd = Path(f"./claude_agent_{direction_id}").resolve()
        agent_cwd.mkdir(parents=True, exist_ok=True)

        # Single "model" for display
        models = [{"model": claude_cfg.model}]
        api_keys = {}
        llm_timeout = 300
        swap_testing = False  # Claude agent handles its own comparisons
        llm_provider = "claude_agent"
    else:
        # OpenRouter path
        api_keys = config.api_keys.model_dump()
        llm_cfg = rank_cfg.llm_client
        llm_timeout = llm_cfg.llm_timeout
        swap_testing = rank_cfg.swap_test_per_pair_per_llm
        llm_provider = "openrouter"

        models = []
        for m in llm_cfg.models:
            models.append({
                "model": m.model,
                "reasoning_effort": m.reasoning_effort,
                "suffix": m.suffix or llm_cfg.suffix,
            })

        if not models:
            # No ranking - return as-is
            return (
                plans,
                {p.id: 0.5 for p in plans},
                {p.id: 1.0 for p in plans},
                [],
                {"converged": False, "convergence_reason": "no_models", "rounds_completed": 0},
            )

    # Create ranker - same for both paths
    ranker = SwissBTRanker(
        telemetry=telemetry,
        api_keys=api_keys,
        models=models,
        max_rounds=max_rounds,
        bootstrap_opponents_per_hypo=bootstrap_opponents,
        votes_per_pair_per_llm=votes_per_pair_per_llm,
        early_stop_win_prob=early_stop_win_prob,
        llm_timeout=llm_timeout,
        unit_id_prefix=f"prop_{direction_id}",
        swap_testing=swap_testing,
    )

    # Prompt builder receives indices and looks up plan data
    def build_pairwise_prompt(idx_a: int, idx_b: int) -> str:
        return get_rank_plan_prompt(
            plans[idx_a],
            plans[idx_b],
            artifact_pool,
            strategy,
        )

    # Pass indices [0, 1, 2, ...] - prompt builder uses them to look up data
    plan_indices = list(range(n))
    system_prompt = get_rank_plan_system_prompt()
    response_schema = PairwisePreference if include_justification else PairwisePreferenceSimple

    # Run ranking - use_claude_agent flag determines backend
    if use_claude_agent:
        result = await ranker.rank(
            items_to_rank=plan_indices,
            build_pairwise_prompt=build_pairwise_prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            module_name=f"RANK_PLAN_{direction_id}",
            use_claude_agent=True,
            claude_model=claude_cfg.model,
            claude_max_turns=claude_cfg.max_turns,
            cwd=agent_cwd,
            output_dir=output_dir,
        )
    else:
        result = await ranker.rank(
            items_to_rank=plan_indices,
            build_pairwise_prompt=build_pairwise_prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            module_name=f"RANK_PLAN_{direction_id}",
        )

    if not result:
        # Ranking failed - return as-is
        return (
            plans,
            {p.id: 0.5 for p in plans},
            {p.id: 1.0 for p in plans},
            [],
            {"converged": False, "convergence_reason": "ranking_failed", "rounds_completed": 0},
        )

    # Map results back - ru.unit is the index
    ranked_plans = []
    win_rates = {}
    bt_scores = {}

    for ru in result.ranked_units:
        idx = ru.unit
        plan = plans[idx]
        unit_id = f"prop_{direction_id}_{idx + 1}"
        ranked_plans.append(plan)
        win_rates[plan.id] = ru.win_rate
        bt_scores[plan.id] = result.bt_scores.get(unit_id, 1.0)

    metadata = {
        "converged": result.converged,
        "convergence_reason": result.convergence_reason,
        "rounds_completed": result.rounds_completed,
        "llm_provider": llm_provider,
    }

    return ranked_plans, win_rates, bt_scores, result.all_comparisons, metadata


# ============================================================================
# Main ranking function
# ============================================================================

async def run_rank_plan_module(
    config: PipelineConfig,
    plan_pool: PlanPool,
    artifact_pool: ArtifactPool,
    strategy: Strategy,
    iteration: int,
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
) -> RankingResult:
    """
    Run the RANK_PLAN step using Swiss-BT tournament method.

    Plans are grouped by in_art_direction_id and ranked independently within each group.
    The best plan for each artifact_direction is selected for execution.

    Uses SwissBTRanker with either:
    - OpenRouter (default): Built-in LLM comparison calls
    - Claude agent: Built-in Claude agent comparison via use_claude_agent flag

    Args:
        config: Pipeline configuration
        plan_pool: Pool with plans to rank
        artifact_pool: For context building
        strategy: The winning strategy for context
        iteration: Current iteration
        telemetry: AIITelemetry instance for logging
        output_dir: Directory to save outputs

    Returns:
        RankingResult with ranked plans and ELO updates
    """
    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "rank_plan_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "rank_plan_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.start_module("RANK_PLAN")

    telemetry.emit(MessageType.INFO, "=" * 60)
    telemetry.emit(MessageType.INFO, f"RANK_PLAN - Ranking plans per artifact_direction for iteration {iteration}")
    telemetry.emit(MessageType.INFO, "=" * 60)

    # Get config
    rank_cfg = config.invention_loop.rank_plan
    testing_mode = config.invention_loop.test_all_artifacts
    use_claude_agent = rank_cfg.use_claude_agent
    llm_provider = "claude_agent" if use_claude_agent else "openrouter"

    # Get plans for this iteration
    all_plans = plan_pool.get_by_iteration(iteration)
    n_total = len(all_plans)

    if n_total == 0:
        telemetry.emit(MessageType.WARNING, "No pending plans to rank")
        telemetry.emit_module_summary("RANK_PLAN")
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        return RankingResult(
            ranked_items=[],
            bt_updates={},
            win_rates={},
            all_judgments=[],
        )

    # Group plans by in_art_direction_id
    plans_by_direction: dict[str, list[BasePlan]] = defaultdict(list)
    for p in all_plans:
        direction_id = p.in_art_direction_id or "unknown"
        plans_by_direction[direction_id].append(p)

    num_directions = len(plans_by_direction)
    telemetry.emit(MessageType.INFO, f"   Total plans: {n_total}")
    telemetry.emit(MessageType.INFO, f"   Artifact directions: {num_directions}")
    telemetry.emit(MessageType.INFO, f"   Backend: {llm_provider}")
    for direction_id, props in plans_by_direction.items():
        telemetry.emit(MessageType.INFO, f"     {direction_id}: {len(props)} plans")

    # Rank plans for each artifact_direction independently
    all_win_rates: dict[str, float] = {}
    all_bt_scores: dict[str, float] = {}
    all_comparisons: list = []
    selected_plans: list[BasePlan] = []
    ranked_by_direction: dict[str, list[BasePlan]] = {}
    all_metadata: dict[str, dict] = {}

    for direction_id, direction_plans in plans_by_direction.items():
        telemetry.emit(MessageType.INFO, f"   Ranking {len(direction_plans)} plans for {direction_id}...")

        ranked, win_rates, bt_scores, comparisons, metadata = await rank_plans_for_art(
            plans=direction_plans,
            direction_id=direction_id,
            config=config,
            artifact_pool=artifact_pool,
            strategy=strategy,
            telemetry=telemetry,
            iteration=iteration,
            output_dir=output_dir,
        )

        # Store results
        ranked_by_direction[direction_id] = ranked
        all_win_rates.update(win_rates)
        all_bt_scores.update(bt_scores)
        all_comparisons.extend(comparisons)
        all_metadata[direction_id] = metadata

        # Log result
        if ranked:
            best = ranked[0]
            telemetry.emit(
                MessageType.INFO,
                f"     Best: {best.id} (BT={bt_scores.get(best.id, 1.0):.3f}, WR={win_rates.get(best.id, 0.5):.1%})"
            )
            telemetry.emit(
                MessageType.INFO,
                f"     Convergence: {metadata.get('convergence_reason', 'unknown')} after {metadata.get('rounds_completed', 0)} rounds"
            )

        # Select best plan for this artifact_direction
        if ranked:
            if testing_mode:
                # Testing mode: select ALL plans
                selected_plans.extend(ranked)
            else:
                # Normal mode: select only the best one
                selected_plans.append(ranked[0])

    if testing_mode:
        telemetry.emit(
            MessageType.INFO,
            f"   TESTING MODE: Selected all {len(selected_plans)} plans for execution"
        )
    else:
        telemetry.emit(
            MessageType.SUCCESS,
            f"RANK_PLAN complete: {len(selected_plans)} plans selected (1 per artifact_direction)"
        )

    # Build ranked plans data for output
    ranked_plans_data = []
    for direction_id, ranked in ranked_by_direction.items():
        for i, p in enumerate(ranked):
            ranked_plans_data.append({
                "plan_id": p.id,
                "in_art_direction_id": direction_id,
                "rank_within_direction": i + 1,
                "plan": p.model_dump(),
                "win_rate": all_win_rates.get(p.id, 0.5),
                "bt_score": all_bt_scores.get(p.id, 1.0),
                "selected": p in selected_plans,
            })

    # Build result
    ranking_result = RankingResult(
        ranked_items=[p.id for p in selected_plans],
        bt_updates={p.id: all_bt_scores.get(p.id, 1.0) for p in all_plans},
        win_rates=all_win_rates,
        all_judgments=[],
    )

    # Save comprehensive output
    if output_dir:
        output_file = output_dir / "rank_plan_output.json"
        # Convert ComparisonResult objects to dicts
        all_comparisons_dicts = []
        for c in all_comparisons:
            if hasattr(c, 'unit_a_id'):
                all_comparisons_dicts.append({
                    "unit_a_id": c.unit_a_id,
                    "unit_b_id": c.unit_b_id,
                    "preferred": c.preferred,
                    "model": c.model if hasattr(c, 'model') else None,
                    "provider": c.provider if hasattr(c, 'provider') else None,
                    "justification": c.justification if hasattr(c, 'justification') else None,
                    "error": c.error if hasattr(c, 'error') else None,
                })

        output_result = {
            "selected_plans": [
                d for d in ranked_plans_data if d["selected"]
            ],
            "ranked_by_direction": {
                direction_id: [
                    d for d in ranked_plans_data if d["in_art_direction_id"] == direction_id
                ]
                for direction_id in ranked_by_direction.keys()
            },
            "all_comparisons": all_comparisons_dicts,
            "win_rates": all_win_rates,
            "bt_scores": all_bt_scores,
            "per_direction_metadata": all_metadata,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "module": "rank_plan",
                "iteration": iteration,
                "llm_provider": llm_provider,
                "num_plans": n_total,
                "num_artifact_directions": num_directions,
                "num_selected": len(selected_plans),
                "testing_mode": testing_mode,
            }
        }
        telemetry.emit(MessageType.MODULE_OUTPUT, json.dumps(output_result, indent=2, ensure_ascii=False))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_result, f, indent=2, ensure_ascii=False)
        telemetry.emit(MessageType.INFO, f"   Results saved to: {rel_path(output_file)}")

    telemetry.emit_module_summary("RANK_PLAN")

    # Remove module sinks to prevent leak into subsequent modules
    for sink in module_sinks:
        sink.flush()
        telemetry.remove_sink(sink)

    return ranking_result
