#!/usr/bin/env python3
"""
Hypothesis Ranking Module - Swiss-Style Tournament with Bradley-Terry Scoring

Sample-efficient ranking via iterative rounds:
- Round 1: Random pairs (bootstrap initial estimates)
- Round 2+: Swiss-style pairing (adjacent ratings compete)
- Scoring: Bradley-Terry MLE (principled, order-invariant)
- Early stop: When top item's confidence interval separates

Uses indices [0, 1, 2, ...] to look up hypothesis + audit data.

Supports two backends:
- OpenRouter (default): Uses SwissBTRanker's built-in LLM calls
- Claude agent: Uses SwissBTRanker with use_claude_agent flag
"""

import json
from datetime import datetime
from pathlib import Path

from aii_lib import (
    MessageType,
    SwissBTRanker,
    AIITelemetry,
    JSONSink,
    create_telemetry,
)
from aii_lib.telemetry import logger
from aii_pipeline.prompts.steps.optional.ranking_schemas import PairwisePreference, PairwisePreferenceSimple
from aii_pipeline.prompts.steps.optional.rank_hypo.u_prompt import get as get_pairwise_comparison
from aii_pipeline.prompts.steps.optional.rank_hypo.s_prompt import get as get_rank_hypo_sysprompt
from aii_pipeline.utils import PipelineConfig, rel_path


async def run_rank_hypo_module(
    config: PipelineConfig,
    audited_hypotheses=None,
    run_dir=None,
    workspace_dir=None,
    telemetry: AIITelemetry = None,
):
    """Run hypothesis ranking using Swiss-BT tournament method.

    Uses SwissBTRanker with either:
    - OpenRouter (default): Built-in LLM comparison calls
    - Claude agent: Built-in Claude agent comparison via use_claude_agent flag
    """
    if run_dir:
        output_dir = run_dir / "4_rank_hypo"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{config.init.outputs_directory}/{timestamp}_rank_hypo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use provided telemetry or create local one
    local_telemetry = telemetry is None
    if local_telemetry:
        telemetry = create_telemetry(output_dir, "rank_hypo")

    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    s1 = JSONSink(output_dir / "rank_hypo_pipeline_messages.jsonl")
    s2 = JSONSink(output_dir / "rank_hypo_pipeline_messages_sequenced.jsonl", sequenced=True)
    telemetry.add_sink(s1)
    telemetry.add_sink(s2)
    module_sinks.extend([s1, s2])

    try:

        telemetry.start_module("RANK_HYPO")

        if not audited_hypotheses:
            telemetry.emit(MessageType.ERROR, "No audited hypotheses provided for ranking")
            for sink in module_sinks:
                sink.flush()
                telemetry.remove_sink(sink)
            return None

        # Extract parallel arrays - all indexed by position [0, 1, 2, ...]
        hypothesis_ids = [ah.get('hypothesis_id', f"hypo_{i+1}") for i, ah in enumerate(audited_hypotheses)]
        hypotheses = [ah.get('hypothesis', {}) for ah in audited_hypotheses]
        audits = [ah.get('audit', {}) for ah in audited_hypotheses]
        num_hypotheses = len(hypotheses)

        # Config
        rank_cfg = config.rank_hypo
        max_rounds = rank_cfg.max_rounds
        bootstrap_opponents_per_hypo = rank_cfg.bootstrap_opponents_per_hypo
        votes_per_pair_per_llm = rank_cfg.votes_per_pair_per_llm
        early_stop_win_prob = rank_cfg.early_stop_win_prob
        include_justification = rank_cfg.include_justification
        use_claude_agent = rank_cfg.use_claude_agent

        # =====================================================================
        # SETUP BACKEND (Claude agent or OpenRouter)
        # =====================================================================
        if use_claude_agent:
            # Claude agent path
            claude_cfg = rank_cfg.claude_agent

            # Create workspace for agent
            agent_cwd = (output_dir / "claude_agent").resolve()
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
                telemetry.emit(MessageType.ERROR, "No models configured in rank_hypo.models")
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
            bootstrap_opponents_per_hypo=bootstrap_opponents_per_hypo,
            votes_per_pair_per_llm=votes_per_pair_per_llm,
            early_stop_win_prob=early_stop_win_prob,
            llm_timeout=llm_timeout,
            unit_id_prefix="hypo",
            swap_testing=swap_testing,
        )

        # Prompt builder receives indices and looks up hypothesis + audit
        def build_pairwise_prompt(idx_a: int, idx_b: int) -> str:
            return get_pairwise_comparison(
                hypothesis_a=hypotheses[idx_a],
                hypothesis_b=hypotheses[idx_b],
                audit_a=audits[idx_a],
                audit_b=audits[idx_b],
            )

        # Pass indices [0, 1, 2, ...] - prompt builder uses them to look up data
        hypothesis_indices = list(range(num_hypotheses))
        system_prompt = get_rank_hypo_sysprompt()
        response_schema = PairwisePreference if include_justification else PairwisePreferenceSimple

        # Run ranking - use_claude_agent flag determines backend
        if use_claude_agent:
            result = await ranker.rank(
                items_to_rank=hypothesis_indices,
                build_pairwise_prompt=build_pairwise_prompt,
                system_prompt=system_prompt,
                response_schema=response_schema,
                module_name="RANK_HYPO",
                use_claude_agent=True,
                claude_model=claude_cfg.model,
                claude_max_turns=claude_cfg.max_turns,
                cwd=agent_cwd,
                output_dir=output_dir,
            )
        else:
            result = await ranker.rank(
                items_to_rank=hypothesis_indices,
                build_pairwise_prompt=build_pairwise_prompt,
                system_prompt=system_prompt,
                response_schema=response_schema,
                module_name="RANK_HYPO",
            )

        if not result:
            telemetry.emit(MessageType.ERROR, "RANK_HYPO ranking failed — ranker returned no result")
            for sink in module_sinks:
                sink.flush()
                telemetry.remove_sink(sink)
            return None

        # Map results back - ru.unit is the index
        ranked_hypotheses = []
        for ru in result.ranked_units:
            idx = ru.unit
            if idx < 0 or idx >= num_hypotheses:
                logger.warning(f"Ranker returned out-of-bounds index {idx} (num_hypotheses={num_hypotheses}), skipping")
                continue
            unit_id = f"hypo_{idx + 1}"
            ranked_hypotheses.append({
                "hypothesis_id": hypothesis_ids[idx],
                "hypothesis": hypotheses[idx],
                "audit": audits[idx],
                "win_rate": ru.win_rate,
                "bt_score": result.bt_scores.get(unit_id, 1.0),
                "elo_rating": ru.elo_rating,
            })

        ranked_hypotheses.sort(key=lambda x: x.get("bt_score", 1.0), reverse=True)
        best_hypothesis = ranked_hypotheses[0]
        telemetry.emit(MessageType.SUCCESS, f"Selected: {best_hypothesis['hypothesis_id']} (BT={best_hypothesis['bt_score']:.3f}, WR={best_hypothesis['win_rate']:.1%})")
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
            'selected_hypothesis': best_hypothesis,
            'ranked_hypotheses': ranked_hypotheses,
            'all_comparisons': all_comparisons,
            'win_rates': result.win_rates,
            'bt_scores': result.bt_scores,
            'rounds_completed': result.rounds_completed,
            'converged': result.converged,
            'convergence_reason': result.convergence_reason,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'module': 'rank_hypo',
                'llm_provider': llm_provider,
                'num_hypotheses': num_hypotheses,
                'max_rounds': max_rounds,
                'output_dir': str(output_dir),
                **result.metadata,
            }
        }

        telemetry.emit(MessageType.MODULE_OUTPUT, json.dumps(output_result, indent=2, ensure_ascii=False))

        output_file = output_dir / "rank_hypo_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_result, f, indent=2, ensure_ascii=False)

        telemetry.emit(MessageType.INFO, f"Results saved to: {rel_path(output_dir)}")
        telemetry.emit_module_summary("RANK_HYPO")

        # Remove module sinks to prevent leak into subsequent modules
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)

        return {
            'output_dir': str(output_dir),
            'selected_hypothesis': best_hypothesis,
            'ranked_hypotheses': ranked_hypotheses,
            'win_rates': result.win_rates,
            'bt_scores': result.bt_scores,
            'rounds_completed': result.rounds_completed,
            'converged': result.converged,
            'metadata': output_result['metadata'],
            'module_summary': result.module_summary,
        }

    finally:
        # Only flush/close if we created local telemetry
        if local_telemetry:
            telemetry.flush()


async def main():
    """Main function for standalone execution."""
    import asyncio

    config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
    config = PipelineConfig.from_yaml(config_path)

    sample_audited = [
        {
            "hypothesis_id": "hypo_1",
            "hypothesis": {"title": "Test 1", "hypothesis": "Test hypothesis 1"},
            "audit": {"feasibility": {"positive_args": [], "negative_args": []}, "novelty": {"positive_args": [], "negative_args": []}},
        },
        {
            "hypothesis_id": "hypo_2",
            "hypothesis": {"title": "Test 2", "hypothesis": "Test hypothesis 2"},
            "audit": {"feasibility": {"positive_args": [], "negative_args": []}, "novelty": {"positive_args": [], "negative_args": []}},
        },
        {
            "hypothesis_id": "hypo_3",
            "hypothesis": {"title": "Test 3", "hypothesis": "Test hypothesis 3"},
            "audit": {"feasibility": {"positive_args": [], "negative_args": []}, "novelty": {"positive_args": [], "negative_args": []}},
        },
    ]

    result = await run_rank_hypo_module(config, audited_hypotheses=sample_audited)
    if result:
        print("Hypothesis ranking completed")
        return 0
    return 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)
