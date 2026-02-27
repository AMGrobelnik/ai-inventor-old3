"""RANK_NARR Step - Swiss-BT tournament ranking of narratives with gap extraction.

New narratives compete against the existing narrative pool.
Judge sees: Narrative A vs Narrative B (no other context).
Judge also identifies: What's MISSING from each narrative?

Outputs:
- BT score updates for all narratives involved (persistent)
- Derived artifact BT scores (avg BT of narratives using each artifact)
- Gap extraction -> seeds for next PROPOSE
- Convergence signal

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
from datetime import datetime
from pathlib import Path

from aii_lib import AIITelemetry, MessageType, JSONSink, SwissBTRanker

from aii_pipeline.utils import PipelineConfig, rel_path

from aii_pipeline.prompts.steps._3_invention_loop._4_gen_narr.schema import Narrative
from aii_pipeline.prompts.steps.optional.ranking_schemas import RankingResult, Gap, IterationStats
from aii_pipeline.steps._invention_loop.pools import NarrativePool, ArtifactPool
from aii_pipeline.prompts.steps.optional.rank_narr.s_prompt import get as get_rank_narr_system_prompt
from aii_pipeline.prompts.steps.optional.rank_narr.u_prompt import get as get_rank_narr_prompt
from aii_pipeline.prompts.steps.optional.rank_narr.schema import (
    PairwisePreference,
    PairwisePreferenceSimple,
)


# ============================================================================
# Gap extraction from comparison results
# ============================================================================

def extract_gaps_from_comparisons(all_comparisons: list) -> list[Gap]:
    """Extract and aggregate gaps from SwissBTRanker comparison results."""
    gap_counts: dict[str, Gap] = {}

    for comp in all_comparisons:
        if hasattr(comp, 'error') and comp.error:
            continue
        if not hasattr(comp, 'raw_response') or not comp.raw_response:
            continue

        for gap_key in ["gap_a", "gap_b"]:
            gap_text = comp.raw_response.get(gap_key)
            if gap_text and len(gap_text) > 10:
                # Simple deduplication by lowercased text
                key = gap_text.lower()[:100]
                if key not in gap_counts:
                    gap_counts[key] = Gap(
                        description=gap_text,
                        frequency=0,
                        source_narratives=[],
                    )
                gap_counts[key].frequency += 1

    # Return sorted by frequency
    return sorted(gap_counts.values(), key=lambda g: g.frequency, reverse=True)


# ============================================================================
# Main ranking function
# ============================================================================

async def run_rank_narr_module(
    config: PipelineConfig,
    narrative_pool: NarrativePool,
    artifact_pool: ArtifactPool,
    new_narratives: list[Narrative],
    iteration: int,
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
) -> tuple[RankingResult, IterationStats, bool]:
    """
    Run the RANK_NARR step using Swiss-BT tournament method.

    Performs tournament-style ranking on all narratives (new + existing).
    Uses SwissBTRanker for sample-efficient comparisons.
    Also extracts gaps from judge feedback.

    Uses SwissBTRanker with either:
    - OpenRouter (default): Built-in LLM comparison calls
    - Claude agent: Built-in Claude agent comparison via use_claude_agent flag

    Args:
        config: Pipeline configuration
        narrative_pool: Pool of all narratives
        artifact_pool: Pool of artifacts (for correlation update)
        new_narratives: Newly generated narratives to rank
        iteration: Current iteration
        telemetry: AIITelemetry instance for logging
        output_dir: Directory to save outputs

    Returns:
        Tuple of (RankingResult, IterationStats, should_stop)
    """
    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "rank_narr_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "rank_narr_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.start_module("RANK_NARR")

    telemetry.emit(MessageType.INFO, "=" * 60)
    telemetry.emit(MessageType.INFO, f"RANK_NARR - Ranking narratives for iteration {iteration}")
    telemetry.emit(MessageType.INFO, "=" * 60)

    # Get config
    eval_cfg = config.invention_loop.rank_narr

    # Tournament parameters (same pattern as rank_strat)
    max_rounds = eval_cfg.max_rounds
    bootstrap_opponents = eval_cfg.bootstrap_opponents_per_item
    votes_per_pair_per_llm = eval_cfg.votes_per_pair_per_llm
    early_stop_win_prob = eval_cfg.early_stop_win_prob
    include_justification = eval_cfg.include_justification
    use_claude_agent = eval_cfg.use_claude_agent

    # =========================================================================
    # SETUP BACKEND (Claude agent or OpenRouter)
    # =========================================================================
    if use_claude_agent:
        # Claude agent path
        claude_cfg = eval_cfg.claude_agent

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
        llm_cfg = eval_cfg.llm_client
        llm_timeout = llm_cfg.llm_timeout
        swap_testing = eval_cfg.swap_test_per_pair_per_llm
        llm_provider = "openrouter"

        models = []
        for m in llm_cfg.models:
            models.append({
                "model": m.model,
                "reasoning_effort": m.reasoning_effort,
                "suffix": m.suffix or llm_cfg.suffix,
            })

        if not models:
            telemetry.emit(MessageType.ERROR, "No models configured in invention_loop.rank_narr.llm_client.models")
            telemetry.emit_module_summary("RANK_NARR")
            for sink in module_sinks:
                sink.flush()
                telemetry.remove_sink(sink)
            return RankingResult(
                ranked_items=[],
                bt_updates={},
                win_rates={},
                all_judgments=[],
            ), IterationStats(iteration=iteration), False

    if not new_narratives:
        telemetry.emit(MessageType.INFO, "No new narratives to rank")
        telemetry.emit_module_summary("RANK_NARR")
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        return RankingResult(
            ranked_items=[],
            bt_updates={},
            win_rates={},
            all_judgments=[],
        ), IterationStats(iteration=iteration), False

    # Get all narratives to rank (new + existing)
    all_narratives = narrative_pool.get_all()
    n = len(all_narratives)

    if n < 2:
        telemetry.emit(MessageType.WARNING, "Not enough narratives to rank")
        telemetry.emit_module_summary("RANK_NARR")
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        return RankingResult(
            ranked_items=[narr.id for narr in all_narratives],
            bt_updates={narr.id: 1.0 for narr in all_narratives},
            win_rates={narr.id: 0.5 for narr in all_narratives},
            all_judgments=[],
        ), IterationStats(iteration=iteration), False

    telemetry.emit(MessageType.INFO, f"   Total narratives: {n} (new: {len(new_narratives)})")
    telemetry.emit(MessageType.INFO, f"   Backend: {llm_provider}")
    telemetry.emit(MessageType.INFO, f"   Max rounds: {max_rounds}")
    telemetry.emit(MessageType.INFO, f"   Bootstrap opponents: {bootstrap_opponents}")
    telemetry.emit(MessageType.INFO, f"   Votes per pair: {votes_per_pair_per_llm}")
    telemetry.emit(MessageType.INFO, f"   Early stop prob: {early_stop_win_prob}")
    if not use_claude_agent:
        telemetry.emit(MessageType.INFO, f"   Models: {[m['model'] for m in models]}")
        telemetry.emit(MessageType.INFO, f"   Timeout: {llm_timeout}s")

    # Create SwissBTRanker - same for both paths
    ranker = SwissBTRanker(
        telemetry=telemetry,
        api_keys=api_keys,
        models=models,
        max_rounds=max_rounds,
        bootstrap_opponents_per_hypo=bootstrap_opponents,
        votes_per_pair_per_llm=votes_per_pair_per_llm,
        early_stop_win_prob=early_stop_win_prob,
        llm_timeout=llm_timeout,
        unit_id_prefix="narr",
        swap_testing=swap_testing,
    )

    # Prompt builder receives indices and looks up narrative data
    def build_pairwise_prompt(idx_a: int, idx_b: int) -> str:
        return get_rank_narr_prompt(all_narratives[idx_a], all_narratives[idx_b])

    # Pass indices [0, 1, 2, ...] - prompt builder uses them to look up data
    narrative_indices = list(range(n))
    system_prompt = get_rank_narr_system_prompt()
    # Both schemas include gap_a/gap_b for gap extraction; difference is justification field
    response_schema = PairwisePreference if include_justification else PairwisePreferenceSimple

    # Run ranking - use_claude_agent flag determines backend
    if use_claude_agent:
        result = await ranker.rank(
            items_to_rank=narrative_indices,
            build_pairwise_prompt=build_pairwise_prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            module_name=f"RANK_NARR iter {iteration}",
            use_claude_agent=True,
            claude_model=claude_cfg.model,
            claude_max_turns=claude_cfg.max_turns,
            cwd=agent_cwd,
            output_dir=output_dir,
        )
    else:
        result = await ranker.rank(
            items_to_rank=narrative_indices,
            build_pairwise_prompt=build_pairwise_prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            module_name=f"RANK_NARR iter {iteration}",
        )

    if not result:
        telemetry.emit(MessageType.ERROR, "RANK_NARR ranking failed — ranker returned no result")
        telemetry.emit_module_summary("RANK_NARR")
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        return RankingResult(
            ranked_items=[narr.id for narr in all_narratives],
            bt_updates={narr.id: 1.0 for narr in all_narratives},
            win_rates={narr.id: 0.5 for narr in all_narratives},
            all_judgments=[],
        ), IterationStats(iteration=iteration), False

    # Map results back - ru.unit is the index
    win_rates = {}
    bt_scores = {}
    for ru in result.ranked_units:
        idx = ru.unit
        narr = all_narratives[idx]
        unit_id = f"narr_{idx + 1}"
        win_rates[narr.id] = ru.win_rate
        bt_scores[narr.id] = result.bt_scores.get(unit_id, 1.0)

    # Log convergence
    telemetry.emit(
        MessageType.INFO,
        f"   Convergence: {result.convergence_reason} after {result.rounds_completed} rounds"
    )
    if result.ranked_units:
        best = all_narratives[result.ranked_units[0].unit]
        telemetry.emit(
            MessageType.INFO,
            f"   Best: {best.id} (BT={bt_scores.get(best.id, 1.0):.3f}, WR={win_rates.get(best.id, 0.5):.1%})"
        )

    # Extract gaps from comparison results
    gaps = extract_gaps_from_comparisons(result.all_comparisons)

    telemetry.emit(MessageType.INFO, f"   Extracted {len(gaps)} gaps")
    for gap in gaps[:3]:
        telemetry.emit(MessageType.INFO, f"     - {gap.description[:60]}... (freq: {gap.frequency})")

    # Best narrative is last in iteration (after ranking reorder)
    iter_narrs = narrative_pool.get_by_iteration(iteration)
    best_narr_id = iter_narrs[-1].id if iter_narrs else ""

    iter_stats = IterationStats(
        iteration=iteration,
        narratives_produced=[n.id for n in new_narratives],
        best_narrative_id=best_narr_id,
    )

    # Log top narratives
    telemetry.emit(MessageType.INFO, "")
    telemetry.emit(MessageType.INFO, "Top Narratives (by BT score):")
    ranked_narrs = sorted(all_narratives, key=lambda n: bt_scores.get(n.id, 1.0), reverse=True)
    for narr in ranked_narrs[:5]:
        telemetry.emit(MessageType.INFO, f"  [{narr.id}] BT={bt_scores.get(narr.id, 1.0):.3f} | {narr.narrative[:50]}...")

    # Check convergence (would need to track stats_history in main loop)
    should_stop = False

    telemetry.emit(MessageType.SUCCESS, f"RANK_NARR complete: {len(gaps)} gaps extracted")

    # Build ranked narratives list with full data
    ranked_narratives_data = []
    for ru in result.ranked_units:
        idx = ru.unit
        narr = all_narratives[idx]
        ranked_narratives_data.append({
            "narrative_id": narr.id,
            "win_rate": ru.win_rate,
            "bt_score": bt_scores.get(narr.id, 1.0),
            "rank": ru.rank if hasattr(ru, 'rank') else 0,
        })

    # Build result with raw BT scores
    bt_updates = {narr.id: bt_scores.get(narr.id, 1.0) for narr in all_narratives}
    ranking_result = RankingResult(
        ranked_items=[all_narratives[ru.unit].id for ru in result.ranked_units],
        bt_updates=bt_updates,
        win_rates=win_rates,
        all_judgments=[],
        gaps_extracted=gaps,
    )

    # Save comprehensive output
    if output_dir:
        output_file = output_dir / "rank_narr_output.json"
        # Convert ComparisonResult objects to dicts
        all_comparisons_dicts = []
        for c in result.all_comparisons:
            if hasattr(c, 'unit_a_id'):
                all_comparisons_dicts.append({
                    "unit_a_id": c.unit_a_id,
                    "unit_b_id": c.unit_b_id,
                    "preferred": c.preferred if hasattr(c, 'preferred') else None,
                    "model": c.model if hasattr(c, 'model') else None,
                    "provider": c.provider if hasattr(c, 'provider') else None,
                    "justification": c.justification if hasattr(c, 'justification') else None,
                    "gap_a": c.raw_response.get("gap_a") if hasattr(c, 'raw_response') and c.raw_response else None,
                    "gap_b": c.raw_response.get("gap_b") if hasattr(c, 'raw_response') and c.raw_response else None,
                    "error": c.error if hasattr(c, 'error') else None,
                })

        output_result = {
            "iteration": iteration,
            "num_narratives": n,
            "num_new_narratives": len(new_narratives),
            "ranked_narratives": ranked_narratives_data,
            "gaps_extracted": [g.model_dump() for g in gaps],
            "all_comparisons": all_comparisons_dicts,
            "win_rates": win_rates,
            "bt_scores": bt_scores,
            "top_narratives": [
                {"id": narr.id, "bt_score": bt_scores.get(narr.id, 1.0)}
                for narr in sorted(all_narratives, key=lambda n: bt_scores.get(n.id, 1.0), reverse=True)[:10]
            ],
            "iteration_stats": iter_stats.model_dump(),
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "module": "rank_narr",
                "llm_provider": llm_provider,
                "converged": result.converged,
                "convergence_reason": result.convergence_reason,
                "rounds_completed": result.rounds_completed,
            }
        }
        telemetry.emit(MessageType.MODULE_OUTPUT, json.dumps(output_result, indent=2, ensure_ascii=False))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_result, f, indent=2, ensure_ascii=False)
        telemetry.emit(MessageType.INFO, f"   Results saved to: {rel_path(output_file)}")

    telemetry.emit_module_summary("RANK_NARR")

    # Remove module sinks to prevent leak into subsequent modules
    for sink in module_sinks:
        sink.flush()
        telemetry.remove_sink(sink)

    return ranking_result, iter_stats, should_stop
