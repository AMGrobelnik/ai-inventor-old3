"""RANK_STRAT Step - Swiss-Style Tournament Ranking of Strategies.

Sample-efficient ranking via iterative rounds:
- Round 1: Random pairs (bootstrap initial estimates)
- Round 2+: Swiss-style pairing (adjacent ratings compete)
- Scoring: Bradley-Terry MLE (principled, order-invariant)
- Early stop: When top item's confidence interval separates

Uses indices [0, 1, 2, ...] to look up strategy data.

Supports two backends:
- OpenRouter (default): Uses SwissBTRanker's built-in LLM calls
- Claude agent: Uses SwissBTRanker with use_claude_agent flag
"""

import json
from datetime import datetime
from pathlib import Path

from aii_lib import MessageType, SwissBTRanker, AIITelemetry, JSONSink
from aii_pipeline.prompts.steps.optional.ranking_schemas import PairwisePreference, PairwisePreferenceSimple
from aii_pipeline.prompts.steps.optional.rank_strat.u_prompt import get as get_pairwise_comparison
from aii_pipeline.prompts.steps.optional.rank_strat.s_prompt import get as get_rank_strat_sysprompt
from aii_pipeline.utils import PipelineConfig, rel_path
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.schema import Strategy
from aii_pipeline.steps._invention_loop.pools import ArtifactPool


async def run_rank_strat_module(
    config: PipelineConfig,
    strategies: list[Strategy],
    artifact_pool: ArtifactPool,
    iteration: int,
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
) -> Strategy | None:
    """
    Run strategy ranking using Swiss-BT tournament method.

    Args:
        config: Pipeline configuration
        strategies: List of strategies to rank
        iteration: Current iteration number
        telemetry: AIITelemetry instance
        output_dir: Directory to save outputs

    Returns:
        The winning strategy, or None if ranking failed
    """
    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "rank_strat_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "rank_strat_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.start_module("RANK_STRAT")

    telemetry.emit(MessageType.INFO, "=" * 60)
    telemetry.emit(MessageType.INFO, f"RANK_STRAT - Ranking {len(strategies)} strategies")
    telemetry.emit(MessageType.INFO, "=" * 60)

    if not strategies:
        telemetry.emit(MessageType.ERROR, "No strategies provided for ranking")
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        return None

    if len(strategies) == 1:
        telemetry.emit(MessageType.INFO, "Only one strategy - auto-selecting")
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        return strategies[0]

    # Config
    rank_cfg = config.invention_loop.rank_strat
    max_iterations = config.invention_loop.max_iterations
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
        agent_cwd = (output_dir / "claude_agent").resolve() if output_dir else Path("./claude_agent").resolve()
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
        llm_client = rank_cfg.llm_client
        llm_timeout = llm_client.llm_timeout
        swap_testing = rank_cfg.swap_test_per_pair_per_llm
        llm_provider = "openrouter"

        models = []
        for m in llm_client.models:
            models.append({
                "model": m.model,
                "reasoning_effort": m.reasoning_effort,
                "suffix": m.suffix or llm_client.suffix
            })

        if not models:
            telemetry.emit(MessageType.ERROR, "No models configured in rank_strat.llm_client.models")
            for sink in module_sinks:
                sink.flush()
                telemetry.remove_sink(sink)
            return None

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
        unit_id_prefix="strat",
        swap_testing=swap_testing,
    )

    # Build prompt builder - receives indices
    def build_pairwise_prompt(idx_a: int, idx_b: int) -> str:
        return get_pairwise_comparison(
            strategy_a=strategies[idx_a].model_dump(),
            strategy_b=strategies[idx_b].model_dump(),
            artifact_pool=artifact_pool,
            current_iteration=iteration,
            max_iterations=max_iterations,
        )

    # Pass indices [0, 1, 2, ...] - prompt builder uses them to look up data
    strategy_indices = list(range(len(strategies)))

    system_prompt = get_rank_strat_sysprompt()

    # Choose schema based on config
    response_schema = PairwisePreference if include_justification else PairwisePreferenceSimple

    # Run ranking - use_claude_agent flag determines backend
    if use_claude_agent:
        result = await ranker.rank(
            items_to_rank=strategy_indices,
            build_pairwise_prompt=build_pairwise_prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            module_name="RANK_STRAT",
            use_claude_agent=True,
            claude_model=claude_cfg.model,
            claude_max_turns=claude_cfg.max_turns,
            cwd=agent_cwd,
            output_dir=output_dir,
        )
    else:
        result = await ranker.rank(
            items_to_rank=strategy_indices,
            build_pairwise_prompt=build_pairwise_prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            module_name="RANK_STRAT",
        )

    if not result:
        telemetry.emit(MessageType.ERROR, "RANK_STRAT ranking failed — ranker returned no result")
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        return None

    # Map results back - ru.unit is the index
    ranked_strategies = []
    for ru in result.ranked_units:
        idx = ru.unit
        strategy = strategies[idx]
        ranked_strategies.append({
            "strategy_id": strategy.id,
            "strategy": strategy.model_dump(),
            "win_rate": ru.win_rate,
            "bt_score": result.bt_scores.get(f"strat_{idx + 1}", 1.0),
            "elo_rating": ru.elo_rating,
        })

    best_strategy = strategies[result.ranked_units[0].unit]
    telemetry.emit(MessageType.SUCCESS, f"Selected: {best_strategy.id} (BT={ranked_strategies[0]['bt_score']:.3f}, WR={ranked_strategies[0]['win_rate']:.1%})")
    telemetry.emit(MessageType.INFO, f"Convergence: {result.convergence_reason} after {result.rounds_completed} rounds")

    all_comparisons = [
        {
            "unit_a_id": c.unit_a_id,
            "unit_b_id": c.unit_b_id,
            "preferred": c.preferred,
            "model": c.model,
            "provider": c.provider,
            "justification": c.justification,
            "error": c.error,
        }
        for c in result.all_comparisons
    ]

    output_result = {
        'selected_strategy': best_strategy.model_dump(),
        'ranked_strategies': ranked_strategies,
        'all_comparisons': all_comparisons,
        'win_rates': result.win_rates,
        'bt_scores': result.bt_scores,
        'rounds_completed': result.rounds_completed,
        'converged': result.converged,
        'convergence_reason': result.convergence_reason,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'module': 'rank_strat',
            'llm_provider': llm_provider,
            'iteration': iteration,
            'num_strategies': len(strategies),
            'max_rounds': max_rounds,
            **result.metadata,
        }
    }

    telemetry.emit(MessageType.MODULE_OUTPUT, json.dumps(output_result, indent=2, ensure_ascii=False))

    # Save output
    if output_dir:
        output_file = output_dir / "rank_strat_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_result, f, indent=2, ensure_ascii=False)
        telemetry.emit(MessageType.INFO, f"Results saved to: {rel_path(output_file)}")

    telemetry.emit_module_summary("RANK_STRAT")

    # Remove module sinks to prevent leak into subsequent modules
    for sink in module_sinks:
        sink.flush()
        telemetry.remove_sink(sink)

    return best_strategy
