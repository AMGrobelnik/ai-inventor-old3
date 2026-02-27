"""VIZ_RANK Step - Swiss-BT tournament ranking of generated figure variations.

Uses SwissBTRanker for sample-efficient tournament ranking:
- Round 1: Random pairs (bootstrap initial estimates)
- Round 2+: Swiss-style pairing (adjacent ratings compete)
- Scoring: Bradley-Terry MLE (principled, order-invariant)
- Early stop: When top item's confidence interval separates

Multimodal judges compare images/charts directly with paper context.

Supports two backends (consistent with rank_hypo and rank_strat):
- OpenRouter (default): Uses SwissBTRanker's built-in LLM calls
- Claude agent: Uses SwissBTRanker with use_claude_agent flag

Uses aii_lib for:
- SwissBTRanker: Swiss-style tournament ranking
- AIITelemetry: Task tracking
"""

import asyncio
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from aii_lib import AIITelemetry, MessageType, JSONSink, SwissBTRanker

from aii_pipeline.utils import PipelineConfig, rel_path
from aii_pipeline.prompts.steps._4_gen_paper_repo._1a_write_papers.schema import PaperText
from aii_pipeline.prompts.steps._4_gen_paper_repo._2a_viz_gen.schema import Figure
from aii_pipeline.prompts.steps.optional.rank_viz.u_prompt import get as get_viz_compare_prompt
from aii_pipeline.prompts.steps.optional.rank_viz.s_prompt import get as get_rank_viz_sysprompt
from aii_pipeline.prompts.steps.optional.ranking_schemas import PairwisePreference, PairwisePreferenceSimple


def _extract_paper_context(paper: PaperText, placeholder: Figure) -> str | None:
    """Extract surrounding text context for a figure placeholder from the paper.

    Looks for the figure placeholder in the paper text and extracts
    surrounding paragraphs for context.
    """
    content = paper.paper_text
    if not content:
        return None

    # Look for the figure reference in paper content
    # Pattern: <figure id="fig_XXX"> XML tag
    pattern = rf'<figure\s+id=["\']?{re.escape(placeholder.id)}["\']?\s*>'
    match = re.search(pattern, content, re.IGNORECASE)

    if match:
        # Extract ~300 chars before and after the reference
        start = max(0, match.start() - 300)
        end = min(len(content), match.end() + 300)
        return content[start:end]

    # Fallback: return first 500 chars
    return content[:500]


async def run_rank_viz_module(
    config: PipelineConfig,
    winning_paper: PaperText | None,
    figure_placeholders: list[Figure],
    figure_results: list[Figure],
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
) -> list[Figure]:
    """
    Run the A_RANK_VIZ step using Swiss-BT tournament method.

    Ranks variations for each figure using paper context for better judgment.

    Uses SwissBTRanker with either:
    - OpenRouter (default): Built-in LLM comparison calls
    - Claude agent: Built-in Claude agent comparison via use_claude_agent flag

    Args:
        config: Pipeline configuration
        winning_paper: Winning paper draft (for paper context text)
        figure_placeholders: Figure specs parsed from paper text XML
        figure_results: All generated figure variations
        telemetry: AIITelemetry instance
        output_dir: Output directory

    Returns:
        List of winning Figure objects (one per figure)
    """
    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "rank_viz_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "rank_viz_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.start_module("A_RANK_VIZ")

    telemetry.emit(MessageType.INFO, "=" * 60)
    telemetry.emit(MessageType.INFO, "A_RANK_VIZ - Ranking figure variations with paper context")
    telemetry.emit(MessageType.INFO, "=" * 60)

    if not figure_results:
        telemetry.emit(MessageType.WARNING, "No figures to rank")
        telemetry.emit_module_summary("A_RANK_VIZ")
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        return []

    placeholders = figure_placeholders or []

    gen_paper_cfg = config.gen_paper_repo
    rank_cfg = gen_paper_cfg.rank_viz

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
        api_keys = {}
        llm_timeout = 300
        swap_testing = False
        llm_provider = "claude_agent"
        models = [{"model": claude_cfg.model}]
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
            telemetry.emit_module_summary("A_RANK_VIZ")
            for sink in module_sinks:
                sink.flush()
                telemetry.remove_sink(sink)
            return []

    telemetry.emit(MessageType.INFO, f"   Figures to rank: {len(set(r.id for r in figure_results))}")
    telemetry.emit(MessageType.INFO, f"   Total variations: {len(figure_results)}")
    telemetry.emit(MessageType.INFO, f"   Backend: {llm_provider}")
    telemetry.emit(MessageType.INFO, f"   Max rounds: {max_rounds}")
    telemetry.emit(MessageType.INFO, f"   Paper context: {'Yes' if winning_paper else 'No'}")
    if not use_claude_agent:
        telemetry.emit(MessageType.INFO, f"   Judge models: {[m['model'] for m in models]}")

    # Group results by figure ID
    results_by_figure: dict[str, list[Figure]] = {}
    for result in figure_results:
        if result.id not in results_by_figure:
            results_by_figure[result.id] = []
        results_by_figure[result.id].append(result)

    # Get placeholders by ID for context
    placeholders_by_id = {p.id: p for p in placeholders}

    # Create shared semaphore for global comparison concurrency control across all figures
    max_concurrent_comp = rank_cfg.max_concurrent_comp
    shared_comparison_semaphore = asyncio.Semaphore(max_concurrent_comp) if max_concurrent_comp else None

    # Helper function to rank a single figure's variations
    async def rank_single_figure(fig_id: str, variations: list[Figure]) -> Figure | None:
        """Rank variations for a single figure and return the winner."""
        if len(variations) <= 1:
            # Only one variation, it wins by default
            return variations[0] if variations else None

        placeholder = placeholders_by_id.get(fig_id)
        if not placeholder:
            # Create a minimal placeholder from figure result
            telemetry.emit(MessageType.WARNING, f"No placeholder found for figure {fig_id}, using result metadata")
            placeholder = Figure(
                id=fig_id,
                title=variations[0].title if variations else fig_id,
                description=variations[0].description if variations else "",
            )

        # Get paper context for this figure
        paper_context = None
        if winning_paper:
            paper_context = _extract_paper_context(winning_paper, placeholder)

        telemetry.emit(MessageType.INFO, f"   Ranking {len(variations)} variations for {fig_id}")

        # Setup agent workspace if using Claude agent
        agent_cwd = None
        if use_claude_agent:
            step_dir = (output_dir / f"_4a_rank_viz_{fig_id}").resolve() if output_dir else Path(f"./_4a_rank_viz_{fig_id}").resolve()
            agent_cwd = step_dir / "claude_agent"
            agent_cwd.mkdir(parents=True, exist_ok=True)

        # Create SwissBTRanker with shared comparison semaphore for global concurrency control
        ranker = SwissBTRanker(
            telemetry=telemetry,
            api_keys=api_keys,
            models=models,
            max_rounds=max_rounds,
            bootstrap_opponents_per_hypo=bootstrap_opponents,
            votes_per_pair_per_llm=votes_per_pair_per_llm,
            early_stop_win_prob=early_stop_win_prob,
            llm_timeout=llm_timeout,
            unit_id_prefix=f"viz_{fig_id}",
            swap_testing=swap_testing,
            comparison_semaphore=shared_comparison_semaphore,
            ensemble_strategy=rank_cfg.ensemble_strategy,
        )

        # Prompt builder receives indices and looks up variation data
        def build_pairwise_prompt(idx_a: int, idx_b: int) -> str | list:
            fig_a = variations[idx_a]
            fig_b = variations[idx_b]

            # For Claude agent: copy figures to workspace and tell agent to read them
            if use_claude_agent and agent_cwd:
                local_path_a = None
                local_path_b = None

                # Copy figure A to workspace
                if fig_a.figure_path and Path(fig_a.figure_path).exists():
                    dest_a = agent_cwd / f"figure_a_{idx_a}.png"
                    shutil.copy(fig_a.figure_path, dest_a)
                    local_path_a = f"./figure_a_{idx_a}.png"

                # Copy figure B to workspace
                if fig_b.figure_path and Path(fig_b.figure_path).exists():
                    dest_b = agent_cwd / f"figure_b_{idx_b}.png"
                    shutil.copy(fig_b.figure_path, dest_b)
                    local_path_b = f"./figure_b_{idx_b}.png"

                return get_viz_compare_prompt(
                    placeholder,
                    fig_a,
                    fig_b,
                    paper_context=paper_context,
                    use_claude_agent=True,
                    local_path_a=local_path_a,
                    local_path_b=local_path_b,
                )

            # For OpenRouter: embed images as base64
            return get_viz_compare_prompt(placeholder, fig_a, fig_b, paper_context=paper_context)

        # Pass indices [0, 1, 2, ...] - prompt builder uses them to look up data
        variation_indices = list(range(len(variations)))
        system_prompt = get_rank_viz_sysprompt()
        response_schema = PairwisePreference if include_justification else PairwisePreferenceSimple

        # Run ranking
        if use_claude_agent:
            result = await ranker.rank(
                items_to_rank=variation_indices,
                build_pairwise_prompt=build_pairwise_prompt,
                system_prompt=system_prompt,
                response_schema=response_schema,
                module_name=f"VIZ_RANK_{fig_id}",
                use_claude_agent=True,
                claude_model=claude_cfg.model,
                claude_max_turns=claude_cfg.max_turns,
                cwd=agent_cwd,
                output_dir=output_dir,
            )
        else:
            result = await ranker.rank(
                items_to_rank=variation_indices,
                build_pairwise_prompt=build_pairwise_prompt,
                system_prompt=system_prompt,
                response_schema=response_schema,
                module_name=f"VIZ_RANK_{fig_id}",
            )

        # Get winner (highest rank)
        if result and result.ranked_units:
            winner_ru = result.ranked_units[0]
            idx = winner_ru.unit
            winner = variations[idx]
            unit_id = f"viz_{fig_id}_{idx + 1}"
            bt_score = result.bt_scores.get(unit_id, 1.0)
            telemetry.emit(MessageType.INFO, f"   Winner for {fig_id}: {winner.figure_path} (BT: {bt_score:.3f})")
            return winner

        return None

    # Rank all figures in PARALLEL (comparison-level concurrency controlled by SwissBTRanker)
    max_concurrent_comp = rank_cfg.max_concurrent_comp
    telemetry.emit(MessageType.INFO, f"   Running {len(results_by_figure)} figure rankings (max {max_concurrent_comp} concurrent comparisons)...")

    ranking_tasks = [
        rank_single_figure(fig_id, variations)
        for fig_id, variations in results_by_figure.items()
    ]
    # Use return_exceptions=True to prevent one failing task from cancelling others
    ranking_results = await asyncio.gather(*ranking_tasks, return_exceptions=True)

    # Collect winners (filter out None results and exceptions)
    winners: list[Figure] = []
    for r in ranking_results:
        if isinstance(r, Exception):
            telemetry.emit(MessageType.WARNING, f"Ranking task failed: {r}")
        elif r is not None:
            winners.append(r)

    telemetry.emit(MessageType.SUCCESS, f"A_RANK_VIZ complete: {len(winners)} winners selected")

    # Save output
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "rank_viz_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "winners": [w.model_dump() for w in winners],
                "all_results": [r.model_dump() for r in figure_results],
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "module": "rank_viz",
                    "llm_provider": llm_provider,
                    "output_dir": str(output_dir) if output_dir else None,
                },
            }, f, indent=2, ensure_ascii=False)
        telemetry.emit(MessageType.INFO, f"   Saved to: {rel_path(output_file)}")

    telemetry.emit_module_summary("A_RANK_VIZ")

    # Remove module sinks to prevent leak into subsequent modules
    for sink in module_sinks:
        sink.flush()
        telemetry.remove_sink(sink)

    return winners
