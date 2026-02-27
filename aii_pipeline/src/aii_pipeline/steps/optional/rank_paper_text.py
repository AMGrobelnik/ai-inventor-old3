"""PAPER_TEXT_RANK Step - Swiss-BT tournament ranking of paper text variations.

Uses SwissBTRanker for sample-efficient tournament ranking:
- Round 1: Random pairs (bootstrap initial estimates)
- Round 2+: Swiss-style pairing (adjacent ratings compete)
- Scoring: Bradley-Terry MLE (principled, order-invariant)
- Early stop: When top item's confidence interval separates

Supports two backends (consistent with rank_hypo and rank_strat):
- OpenRouter (default): Uses SwissBTRanker's built-in LLM calls
- Claude agent: Uses SwissBTRanker with use_claude_agent flag

Uses aii_lib for:
- SwissBTRanker: Swiss-style tournament ranking
- AIITelemetry: Task tracking
"""

import json
from datetime import datetime
from pathlib import Path

from aii_lib import AIITelemetry, MessageType, JSONSink, SwissBTRanker

from aii_pipeline.utils import PipelineConfig, rel_path
from aii_pipeline.prompts.steps._4_gen_paper_repo._1a_write_papers.schema import PaperText
from aii_pipeline.prompts.steps.optional.rank_paper.u_prompt import get as get_paper_compare_prompt
from aii_pipeline.prompts.steps.optional.rank_paper.s_prompt import get as get_rank_paper_sysprompt
from aii_pipeline.prompts.steps.optional.ranking_schemas import PairwisePreference, PairwisePreferenceSimple


async def run_rank_paper_text_module(
    config: PipelineConfig,
    paper_texts: list[PaperText],
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
) -> list[PaperText]:
    """
    Run the A_RANK_PAPER_TEXT step using Swiss-BT tournament method.

    Ranks paper texts and returns them sorted by BT/ELO score.

    Uses SwissBTRanker with either:
    - OpenRouter (default): Built-in LLM comparison calls
    - Claude agent: Built-in Claude agent comparison via use_claude_agent flag

    Args:
        config: Pipeline configuration
        paper_texts: All generated paper texts
        telemetry: AIITelemetry instance
        output_dir: Output directory

    Returns:
        List of PaperText objects sorted by score (best first)
    """
    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "rank_paper_text_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "rank_paper_text_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.start_module("A_RANK_PAPER_TEXT")

    telemetry.emit(MessageType.INFO, "=" * 60)
    telemetry.emit(MessageType.INFO, "A_RANK_PAPER_TEXT - Ranking paper texts")
    telemetry.emit(MessageType.INFO, "=" * 60)

    if not paper_texts:
        telemetry.emit(MessageType.WARNING, "No papers to rank")
        telemetry.emit_module_summary("A_RANK_PAPER_TEXT")
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        return []

    if len(paper_texts) == 1:
        telemetry.emit(MessageType.INFO, "   Only one paper text, selecting as winner")
        telemetry.emit_module_summary("A_RANK_PAPER_TEXT")
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        return paper_texts

    gen_paper_cfg = config.gen_paper_repo
    rank_cfg = gen_paper_cfg.rank_paper_text

    # Tournament parameters
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
            telemetry.emit(MessageType.ERROR, "No models configured for ranking")
            telemetry.emit_module_summary("A_RANK_PAPER_TEXT")
            for sink in module_sinks:
                sink.flush()
                telemetry.remove_sink(sink)
            return paper_texts

    telemetry.emit(MessageType.INFO, f"   Papers to rank: {len(paper_texts)}")
    telemetry.emit(MessageType.INFO, f"   Backend: {llm_provider}")
    telemetry.emit(MessageType.INFO, f"   Max rounds: {max_rounds}")
    telemetry.emit(MessageType.INFO, f"   Bootstrap opponents: {bootstrap_opponents}")
    if not use_claude_agent:
        telemetry.emit(MessageType.INFO, f"   Judge models: {[m['model'] for m in models]}")

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
        unit_id_prefix="paper",
        swap_testing=swap_testing,
    )

    # Prompt builder receives indices and looks up paper data
    def build_pairwise_prompt(idx_a: int, idx_b: int) -> str:
        return get_paper_compare_prompt(paper_texts[idx_a], paper_texts[idx_b])

    # Pass indices [0, 1, 2, ...] - prompt builder uses them to look up data
    paper_indices = list(range(len(paper_texts)))
    system_prompt = get_rank_paper_sysprompt()
    response_schema = PairwisePreference if include_justification else PairwisePreferenceSimple

    # Run ranking - use_claude_agent flag determines backend
    if use_claude_agent:
        result = await ranker.rank(
            items_to_rank=paper_indices,
            build_pairwise_prompt=build_pairwise_prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            module_name="A_RANK_PAPER_TEXT",
            use_claude_agent=True,
            claude_model=claude_cfg.model,
            claude_max_turns=claude_cfg.max_turns,
            cwd=agent_cwd,
            output_dir=output_dir,
        )
    else:
        result = await ranker.rank(
            items_to_rank=paper_indices,
            build_pairwise_prompt=build_pairwise_prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            module_name="A_RANK_PAPER_TEXT",
        )

    # Build ranked list
    ranked_papers = []
    if result and result.ranked_units:
        for i, ru in enumerate(result.ranked_units):
            idx = ru.unit
            paper = paper_texts[idx]
            ranked_papers.append(paper)

        unit_id = f"paper_{result.ranked_units[0].unit + 1}"
        winner_bt = result.bt_scores.get(unit_id, 1.0)
        telemetry.emit(MessageType.INFO, f"   Winner: {ranked_papers[0].id} (BT: {winner_bt:.3f})")
        telemetry.emit(
            MessageType.INFO,
            f"   Convergence: {result.convergence_reason} after {result.rounds_completed} rounds"
        )
    else:
        # Fallback - return in original order
        ranked_papers = paper_texts

    telemetry.emit(MessageType.SUCCESS, f"A_RANK_PAPER_TEXT complete: {len(ranked_papers)} papers ranked")

    # Build all_comparisons for logging
    all_comparisons = []
    if result:
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
        "winner": ranked_papers[0].model_dump() if ranked_papers else None,
        "all_paper_texts": [d.model_dump() for d in ranked_papers],
        "ranking": [
            {"id": p.id}
            for p in ranked_papers
        ] if ranked_papers else [],
        "all_comparisons": all_comparisons,
        "win_rates": result.win_rates if result else {},
        "bt_scores": result.bt_scores if result else {},
        "rounds_completed": result.rounds_completed if result else 0,
        "converged": result.converged if result else False,
        "convergence_reason": result.convergence_reason if result else "no_result",
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "module": "rank_paper_text",
            "llm_provider": llm_provider,
            "output_dir": str(output_dir) if output_dir else None,
            **(result.metadata if result else {}),
        },
    }

    telemetry.emit(MessageType.MODULE_OUTPUT, json.dumps(output_result, indent=2, ensure_ascii=False))

    # Save output
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "rank_paper_text_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_result, f, indent=2, ensure_ascii=False)
        telemetry.emit(MessageType.INFO, f"   Saved to: {rel_path(output_file)}")

    telemetry.emit_module_summary("A_RANK_PAPER_TEXT")

    # Remove module sinks to prevent leak into subsequent modules
    for sink in module_sinks:
        sink.flush()
        telemetry.remove_sink(sink)

    return ranked_papers
