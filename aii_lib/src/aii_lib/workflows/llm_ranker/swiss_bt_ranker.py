"""
Swiss-Style Tournament with Bradley-Terry Scoring.

Sample-efficient ranking via iterative rounds:
- Round 1: Random pairs (bootstrap initial estimates)
- Round 2+: Swiss-style pairing (adjacent ratings compete)
- Scoring: Bradley-Terry MLE (principled, order-invariant)
- Early stop: When top item's confidence interval separates

Features:
- Multiple LLM models vote on each matchup
- OpenRouter provider support via chat() wrapper
- Pydantic schema for structured JSON responses
- Bradley-Terry scoring with bootstrap confidence intervals
- Parallel execution within rounds, sequential between rounds
- Model-specific configs (reasoning_effort, suffix)

Literature basis:
- Bradley-Terry: LMSYS Chatbot Arena standard
- Swiss pairing: Chess tournaments, Arena-Lite
- Sample efficiency: O(n log n) vs O(n²) exhaustive
"""

import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Generic, TypeVar

from aii_lib.telemetry import logger
from pydantic import BaseModel

from ...telemetry import AIITelemetry
from ...llm_backend import OpenRouterClient
from ...llm_backend.tool_loop import chat
from ...utils import get_model_short
from ...agent_backend import Agent, AgentOptions

from .k_random_ranker import ModelConfig, ComparisonResult, RankedUnit
from .ranking_diagnostics import RankingDiagnostics, compute_diagnostics, format_diagnostics


T = TypeVar("T")

# Bradley-Terry constants
BT_INITIAL_STRENGTH = 1.0
BT_CONVERGENCE_THRESHOLD = 1e-6
BT_MAX_ITERATIONS = 100


def win_prob_to_gap(win_prob: float) -> float:
    """Convert win probability threshold to BT score gap.

    Given P(top beats 2nd) = win_prob, calculate the BT score gap needed.

    Math: win_prob = s_top / (s_top + s_second)
    With s_second normalized to 1.0:
        gap = s_top - 1 = win_prob / (1 - win_prob) - 1

    Examples:
        0.60 → gap 0.50
        0.70 → gap 1.33
        0.75 → gap 2.00
        0.80 → gap 3.00
    """
    if win_prob <= 0.5:
        return 0.0
    if win_prob >= 1.0:
        return float('inf')
    return win_prob / (1 - win_prob) - 1


def gap_to_win_prob(gap: float) -> float:
    """Convert BT score gap to win probability.

    Inverse of win_prob_to_gap.

    Math: With s_top = 1 + gap and s_second = 1:
        win_prob = (1 + gap) / (2 + gap)
    """
    if gap <= 0:
        return 0.5
    return (1 + gap) / (2 + gap)


@dataclass
class SwissBTRankResult(Generic[T]):
    """Complete result of Swiss-BT ranking workflow."""
    ranked_units: list[RankedUnit[T]]
    bt_scores: dict[str, float]  # Bradley-Terry strength parameters
    bt_confidence_intervals: dict[str, tuple[float, float]]  # 95% CI per unit
    win_rates: dict[str, float]
    all_comparisons: list[ComparisonResult]
    rounds_completed: int
    converged: bool
    convergence_reason: str
    metadata: dict
    module_summary: dict | None = None
    diagnostics: "RankingDiagnostics | None" = None  # Judge quality diagnostics


@dataclass
class RoundStats:
    """Statistics for a single round."""
    round_num: int
    num_pairs: int
    comparisons_run: int
    successful: int
    failed: int
    duration_ms: float
    top_unit_id: str
    top_bt_score: float
    second_bt_score: float
    gap: float


class SwissBTRanker(Generic[T]):
    """
    Swiss-Style Tournament with Bradley-Terry Scoring.

    Sample-efficient ranking via iterative rounds:
    1. Round 1: Random pairs to bootstrap initial BT estimates
    2. Round 2+: Swiss-style pairing (pair by current BT score)
    3. After each round: Update BT scores via MLE
    4. Stop when: max_rounds reached OR top item separates

    Each comparison randomly selects one LLM from the provided list.

    Example:
        ranker = SwissBTRanker(
            telemetry=telemetry,
            api_keys={"openrouter": "sk-or-..."},
            models=["openai/gpt-5-mini", "google/gemini-3-flash"],
            max_rounds=5,
            bootstrap_opponents_per_hypo=3,
        )
        result = await ranker.rank(...)
    """

    def __init__(
        self,
        telemetry: AIITelemetry,
        api_keys: dict,
        models: list[dict | str | ModelConfig],
        *,
        max_rounds: int = 5,
        bootstrap_opponents_per_hypo: int = 3,
        votes_per_pair_per_llm: int = 10,
        early_stop_win_prob: float = 0.75,
        llm_timeout: int = 180,
        unit_id_prefix: str = "unit",
        swap_testing: bool = True,
        max_concurrent_comparisons: int | None = None,
        comparison_semaphore: asyncio.Semaphore | None = None,
        ensemble_strategy: str | None = None,
    ):
        """
        Args:
            telemetry: AIITelemetry instance for logging
            api_keys: Dict with "openrouter" key
            models: List of model configs
            max_rounds: Maximum tournament rounds (default 5)
            bootstrap_opponents_per_hypo: K random opponents per unit in round 1 (default 3)
            votes_per_pair_per_llm: Each LLM judges each pair this many times
            early_stop_win_prob: Stop when P(#1 beats #2) >= this (default 0.75)
            llm_timeout: Timeout per comparison in seconds
            unit_id_prefix: Prefix for unit IDs (e.g., "hypo", "prop")
            swap_testing: If True, run both (A,B) and (B,A) orderings and only count
                consistent votes. Inconsistent votes (position bias) become ties.
            max_concurrent_comparisons: Max LLM comparisons running in parallel (semaphore).
                If None, all comparisons run concurrently (no limit). Ignored if
                comparison_semaphore is provided.
            comparison_semaphore: Optional external semaphore for concurrency control.
                Use this to share a semaphore across multiple ranker instances (e.g.,
                for global concurrency control across all figure rankings).
            ensemble_strategy: Multi-model ensemble strategy. None = independent votes
                (default). "or" = if ANY model prefers A, ensemble says A (optimistic).
                "majority" = majority vote wins. Requires 2+ models.
        """
        self.telemetry = telemetry
        self.api_keys = api_keys
        self.models = self._parse_models(models)
        self.max_rounds = max_rounds
        self.bootstrap_opponents_per_hypo = bootstrap_opponents_per_hypo
        self.votes_per_pair_per_llm = votes_per_pair_per_llm
        self.early_stop_win_prob = early_stop_win_prob
        self.llm_timeout = llm_timeout
        self.unit_id_prefix = unit_id_prefix
        self.swap_testing = swap_testing
        self.max_concurrent_comparisons = max_concurrent_comparisons
        self.ensemble_strategy = ensemble_strategy
        if ensemble_strategy and len(self.models) < 2:
            logger.warning(f"Ensemble strategy '{ensemble_strategy}' requires 2+ models, got {len(self.models)}. Disabling ensemble.")
            self.ensemble_strategy = None
        # Use external semaphore if provided, otherwise create one from max_concurrent_comparisons
        if comparison_semaphore is not None:
            self._comparison_semaphore = comparison_semaphore
        elif max_concurrent_comparisons is not None:
            self._comparison_semaphore = asyncio.Semaphore(max_concurrent_comparisons)
        else:
            self._comparison_semaphore = None
        self._task_counter = 0

    def _parse_models(self, raw_models: list) -> list[ModelConfig]:
        """Parse models config - can be strings, dicts, or ModelConfig."""
        models = []
        for m in raw_models:
            if isinstance(m, str):
                models.append(ModelConfig(model=m))
            elif isinstance(m, ModelConfig):
                models.append(m)
            elif isinstance(m, dict):
                models.append(ModelConfig(
                    model=m["model"],
                    reasoning_effort=m.get("reasoning_effort"),
                    suffix=m.get("suffix"),
                ))
            else:
                raise ValueError(f"Invalid model config type: {type(m)}")
        return models

    def _next_task_id(self) -> int:
        self._task_counter += 1
        return self._task_counter

    def _unique_suffix(self) -> str:
        """Generate unique suffix from timestamp (minute+second+millisecond)."""
        t = time.time()
        ms = int((t % 1) * 1000)
        lt = time.localtime(t)
        return f"{lt.tm_min:02d}{lt.tm_sec:02d}{ms:03d}"

    def _unit_id(self, idx: int) -> str:
        """Generate unit ID from index."""
        return f"{self.unit_id_prefix}_{idx + 1}"

    def get_model_strs(self) -> list[str]:
        """Get display strings for models with suffixes."""
        model_strs = []
        for m in self.models:
            name = m.model
            if m.suffix:
                name += f":{m.suffix}"
            model_strs.append(name)
        return model_strs

    @staticmethod
    def generate_swiss_pairs(
        n: int,
        bt_scores: dict[str, float],
        unit_ids: list[str],
        confidence_intervals: dict[str, tuple[float, float]] | None = None,
    ) -> list[tuple[int, int]]:
        """
        Generate Swiss-style pairs: sort by BT score, chain adjacent.

        If confidence_intervals provided, only pairs with overlapping CIs are included
        (adaptive mode - skip comparisons where ranking is already confident).

        Args:
            n: Number of units
            bt_scores: Current Bradley-Terry scores by unit_id
            unit_ids: List of unit IDs in index order
            confidence_intervals: Optional CIs for adaptive filtering

        Returns:
            List of (a_idx, b_idx) pairs
        """
        if n < 2:
            return []

        # Sort indices by BT score (descending)
        sorted_indices = sorted(
            range(n),
            key=lambda i: bt_scores.get(unit_ids[i], BT_INITIAL_STRENGTH),
            reverse=True,
        )

        # Chain adjacent: 0v1, 1v2, 2v3, ... (each faces above and below)
        pairs: list[tuple[int, int]] = []
        skipped = 0

        for i in range(n - 1):
            a_idx = sorted_indices[i]      # Higher ranked
            b_idx = sorted_indices[i + 1]  # Lower ranked

            # Adaptive filtering: skip if CIs don't overlap
            if confidence_intervals is not None:
                a_id = unit_ids[a_idx]
                b_id = unit_ids[b_idx]
                a_ci = confidence_intervals.get(a_id, (0.0, 10.0))
                b_ci = confidence_intervals.get(b_id, (0.0, 10.0))

                # CIs overlap if lower item's upper bound >= higher item's lower bound
                # a_ci[0] is lower bound of higher-ranked item
                # b_ci[1] is upper bound of lower-ranked item
                if b_ci[1] < a_ci[0]:
                    # No overlap - confident that a > b, skip this pair
                    skipped += 1
                    continue

            # Randomize A/B position to avoid position bias
            if random.random() < 0.5:
                pairs.append((a_idx, b_idx))
            else:
                pairs.append((b_idx, a_idx))

        if skipped > 0:
            logger.info(f"   Adaptive: skipped {skipped} pairs (CIs don't overlap)")

        return pairs

    @staticmethod
    def generate_bootstrap_pairs(n: int, k: int) -> list[tuple[int, int]]:
        """
        Generate K random opponent pairs for each unit (bootstrap round).

        Same logic as KRandomRanker - each unit gets K comparisons
        against randomly selected opponents.

        Args:
            n: Number of units
            k: Number of random opponents per unit

        Returns:
            List of (a_idx, b_idx) pairs
        """
        if n < 2:
            return []

        pairs: list[tuple[int, int]] = []

        for i in range(n):
            possible_opponents = [j for j in range(n) if j != i]

            if len(possible_opponents) >= k:
                opponents = random.sample(possible_opponents, k)
            else:
                opponents = random.choices(possible_opponents, k=k)

            for j in opponents:
                # Randomize A/B position to avoid position bias
                if random.random() < 0.5:
                    pairs.append((i, j))
                else:
                    pairs.append((j, i))

        return pairs

    def calculate_bradley_terry(
        self,
        comparisons: list[ComparisonResult],
        n: int,
    ) -> dict[str, float]:
        """
        Calculate Bradley-Terry scores via MLE.

        Uses iterative algorithm that converges to global maximum.
        P(i beats j) = strength_i / (strength_i + strength_j)

        Args:
            comparisons: All comparison results
            n: Number of units

        Returns:
            Dict of unit_id -> BT strength (normalized to sum=n)
        """
        unit_ids = [self._unit_id(i) for i in range(n)]

        # Initialize strengths
        strengths = {uid: BT_INITIAL_STRENGTH for uid in unit_ids}

        # Count wins and games
        wins: dict[str, float] = {uid: 0.0 for uid in unit_ids}
        games_against: dict[str, dict[str, int]] = {
            uid: {other: 0 for other in unit_ids if other != uid}
            for uid in unit_ids
        }

        for comp in comparisons:
            if comp.error or not comp.preferred:
                continue
            if comp.preferred not in ["A", "B"]:
                continue

            a_id, b_id = comp.unit_a_id, comp.unit_b_id
            if a_id not in wins or b_id not in wins:
                continue

            games_against[a_id][b_id] = games_against[a_id].get(b_id, 0) + 1
            games_against[b_id][a_id] = games_against[b_id].get(a_id, 0) + 1

            if comp.preferred == "A":
                wins[a_id] += 1.0
            else:
                wins[b_id] += 1.0

        # Iterative MLE (guaranteed to converge for BT)
        for iteration in range(BT_MAX_ITERATIONS):
            old_strengths = strengths.copy()

            for uid in unit_ids:
                total_wins = wins[uid]
                if total_wins == 0:
                    continue

                denominator = 0.0
                for other_id, num_games in games_against[uid].items():
                    if num_games > 0:
                        denominator += num_games / (strengths[uid] + strengths[other_id])

                if denominator > 0:
                    strengths[uid] = total_wins / denominator

            # Normalize to sum = n
            total = sum(strengths.values())
            if total > 0:
                for uid in strengths:
                    strengths[uid] = (strengths[uid] / total) * n

            # Check convergence
            max_change = max(
                abs(strengths[uid] - old_strengths[uid])
                for uid in unit_ids
            )
            if max_change < BT_CONVERGENCE_THRESHOLD:
                break

        return strengths

    def calculate_win_rates(
        self,
        comparisons: list[ComparisonResult],
        n: int,
    ) -> dict[str, float]:
        """Calculate simple win rates from comparison results."""
        wins = {self._unit_id(i): 0.0 for i in range(n)}
        games = {self._unit_id(i): 0 for i in range(n)}

        for comp in comparisons:
            if comp.error or not comp.preferred:
                continue
            if comp.preferred not in ["A", "B"]:
                continue

            games[comp.unit_a_id] = games.get(comp.unit_a_id, 0) + 1
            games[comp.unit_b_id] = games.get(comp.unit_b_id, 0) + 1

            if comp.preferred == "A":
                wins[comp.unit_a_id] = wins.get(comp.unit_a_id, 0) + 1.0
            else:
                wins[comp.unit_b_id] = wins.get(comp.unit_b_id, 0) + 1.0

        win_rates = {}
        for uid in wins:
            if games.get(uid, 0) > 0:
                win_rates[uid] = wins[uid] / games[uid]
            else:
                win_rates[uid] = 0.5

        return win_rates

    def calculate_bootstrap_ci(
        self,
        comparisons: list[ComparisonResult],
        n: int,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
    ) -> dict[str, tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for BT scores.

        Resamples comparisons with replacement, computes BT scores for each
        resample, and returns percentile-based confidence intervals.

        Args:
            comparisons: All comparison results
            n: Number of units
            n_bootstrap: Number of bootstrap iterations (default 1000)
            ci_level: Confidence level (default 0.95 for 95% CI)

        Returns:
            Dict of unit_id -> (lower_bound, upper_bound)
        """
        unit_ids = [self._unit_id(i) for i in range(n)]

        # Filter to valid comparisons only
        valid_comparisons = [
            c for c in comparisons
            if not c.error and c.preferred in ["A", "B"]
        ]

        if len(valid_comparisons) < 3:
            # Not enough data for meaningful bootstrap
            return {uid: (0.0, 2.0) for uid in unit_ids}

        # Collect bootstrap BT scores
        bootstrap_scores: dict[str, list[float]] = {uid: [] for uid in unit_ids}

        for _ in range(n_bootstrap):
            # Resample with replacement
            resampled = random.choices(valid_comparisons, k=len(valid_comparisons))

            # Compute BT scores on resample
            bt_scores = self.calculate_bradley_terry(resampled, n)

            for uid in unit_ids:
                bootstrap_scores[uid].append(bt_scores.get(uid, 1.0))

        # Compute percentiles
        alpha = (1 - ci_level) / 2
        lower_pct = alpha * 100
        upper_pct = (1 - alpha) * 100

        confidence_intervals = {}
        for uid in unit_ids:
            scores = sorted(bootstrap_scores[uid])
            lower_idx = int(lower_pct / 100 * len(scores))
            upper_idx = int(upper_pct / 100 * len(scores)) - 1
            lower_idx = max(0, min(lower_idx, len(scores) - 1))
            upper_idx = max(0, min(upper_idx, len(scores) - 1))
            confidence_intervals[uid] = (scores[lower_idx], scores[upper_idx])

        return confidence_intervals

    async def _run_comparison(
        self,
        unit_a_id: str,
        unit_b_id: str,
        prompt: str | list,
        system_prompt: str,
        response_schema: type[BaseModel],
        model_cfg: ModelConfig,
        group: str,
        round_num: int,
    ) -> ComparisonResult:
        """Run pairwise comparison using chat() wrapper with OpenRouter."""
        model_short = get_model_short(model_cfg.model)
        task_num = self._next_task_id()
        task_id = f"r{round_num}_cmp_{unit_a_id}_{unit_b_id}_{model_short}_{self._unique_suffix()}"
        task_name = f"r{round_num}-{self.unit_id_prefix}-cmp{task_num}__{model_short}"

        self.telemetry.emit_task_start(task_id, task_name)
        callback = self.telemetry.create_callback(task_id, task_name, group=group)

        try:
            effective_model = OpenRouterClient.resolve_model(model_cfg.model, model_cfg.suffix)

            async with OpenRouterClient(
                api_key=self.api_keys.get('openrouter'),
                model=effective_model,
                timeout=self.llm_timeout,
            ) as client:
                result = await chat(
                    client=client,
                    prompt=prompt,
                    system=system_prompt,
                    tools=None,
                    response_format=response_schema,
                    message_callback=callback,
                    reasoning_effort=model_cfg.reasoning_effort,
                    timeout=self.llm_timeout,
                )

                output_json = client.extract_json_from_response(result.response)
                if output_json:
                    result_dict = json.loads(output_json)
                    preferred = result_dict.get("preferred")
                    justification = result_dict.get("justification")

                    self.telemetry.emit_task_end(task_id, task_name, f"Preferred: {preferred or 'N/A'}")

                    return ComparisonResult(
                        unit_a_id=unit_a_id,
                        unit_b_id=unit_b_id,
                        preferred=preferred,
                        model=model_cfg.model,
                        provider="openrouter",
                        justification=justification,
                        raw_response=result_dict,
                    )
                else:
                    self.telemetry.emit_task_end(task_id, task_name, "Empty response")
                    raise RuntimeError(f"Empty response from {model_cfg.model} for comparison {unit_a_id} vs {unit_b_id}")

        except asyncio.TimeoutError:
            self.telemetry.emit_task_end(task_id, task_name, f"Timeout ({self.llm_timeout}s)")
            raise

        except Exception as e:
            logger.exception(f"Comparison failed ({unit_a_id} vs {unit_b_id}, {model_cfg.model})")
            self.telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
            raise

    async def _run_swap_tested_comparison(
        self,
        orig_a_idx: int,
        orig_b_idx: int,
        items_to_rank: list[T],
        build_pairwise_prompt: Callable[[T, T], str | list],
        system_prompt: str,
        response_schema: type[BaseModel],
        model_cfg: ModelConfig,
        group: str,
        round_num: int,
    ) -> ComparisonResult:
        """Run both (A,B) and (B,A) orderings and check consistency.

        Consistency logic:
        - Normal (A,B): A=orig_a, B=orig_b
        - Swapped (B,A): A=orig_b, B=orig_a
        - If normal picks "A" and swapped picks "B" → both chose orig_a → CONSISTENT
        - If normal picks "B" and swapped picks "A" → both chose orig_b → CONSISTENT
        - If normal picks "A" and swapped picks "A" → position bias → INCONSISTENT
        - If normal picks "B" and swapped picks "B" → position bias → INCONSISTENT

        Returns single ComparisonResult with winner, or error if inconsistent.
        """
        unit_a_id = self._unit_id(orig_a_idx)
        unit_b_id = self._unit_id(orig_b_idx)

        # Run normal order (A=orig_a, B=orig_b)
        prompt_normal = build_pairwise_prompt(items_to_rank[orig_a_idx], items_to_rank[orig_b_idx])
        result_normal = await self._run_comparison(
            unit_a_id, unit_b_id,
            prompt_normal, system_prompt, response_schema,
            model_cfg, group, round_num,
        )

        # Run swapped order (A=orig_b, B=orig_a)
        prompt_swapped = build_pairwise_prompt(items_to_rank[orig_b_idx], items_to_rank[orig_a_idx])
        result_swapped = await self._run_comparison(
            unit_b_id, unit_a_id,  # Note: swapped IDs
            prompt_swapped, system_prompt, response_schema,
            model_cfg, group, round_num,
        )

        # Check for errors
        if result_normal.error or result_swapped.error:
            return ComparisonResult(
                unit_a_id=unit_a_id,
                unit_b_id=unit_b_id,
                preferred=None,
                model=model_cfg.model,
                provider="openrouter",
                error=f"Swap test error: normal={result_normal.error}, swapped={result_swapped.error}",
            )

        normal_vote = result_normal.preferred
        swapped_vote = result_swapped.preferred

        # Check consistency: votes should differ (both point to same hypothesis)
        if normal_vote != swapped_vote:
            # CONSISTENT - determine winner
            if normal_vote == "A":
                winner = "A"  # orig_a won (normal picked A, swapped picked B)
            else:
                winner = "B"  # orig_b won (normal picked B, swapped picked A)

            return ComparisonResult(
                unit_a_id=unit_a_id,
                unit_b_id=unit_b_id,
                preferred=winner,
                model=model_cfg.model,
                provider="openrouter",
                justification=f"[Swap-tested] Normal: {normal_vote}, Swapped: {swapped_vote}. "
                              f"Normal justification: {result_normal.justification}",
            )
        else:
            # INCONSISTENT - position bias detected
            return ComparisonResult(
                unit_a_id=unit_a_id,
                unit_b_id=unit_b_id,
                preferred=None,
                model=model_cfg.model,
                provider="openrouter",
                error=f"Position bias: both votes picked '{normal_vote}' regardless of order",
            )

    async def _run_comparison_claude_agent(
        self,
        unit_a_id: str,
        unit_b_id: str,
        prompt: str | list,
        system_prompt: str,
        response_schema: type[BaseModel],
        round_num: int,
        model: str,
        max_turns: int,
        cwd: Path,
        output_dir: Path | None,
        group: str = "rank",
    ) -> ComparisonResult:
        """Run pairwise comparison using Claude agent."""
        task_num = self._next_task_id()
        task_id = f"r{round_num}_cmp_{unit_a_id}_{unit_b_id}_claude_{self._unique_suffix()}"
        task_name = f"r{round_num}-{self.unit_id_prefix}-cmp{task_num}__claude-{model}"

        self.telemetry.emit_task_start(task_id, task_name)
        callback = self.telemetry.create_callback(task_id, task_name, group=group)
        json_log_path = str(output_dir / f"{task_id}_messages.jsonl") if output_dir else None

        try:
            options = AgentOptions(
                model=model,
                cwd=cwd,
                max_turns=max_turns,
                permission_mode="bypassPermissions",
                system_prompt=system_prompt,
                mcp_servers={},  # No tools for comparison
                json_log_path=json_log_path,
                # Telemetry integration
                telemetry=self.telemetry,
                run_id=task_id,
                agent_context=task_name,
                # Structured JSON output (SDK native)
                output_format=response_schema.to_struct_output(),
            )

            agent = Agent(options)
            # Handle different prompt formats: str, list[str], or list[dict] (messages)
            if isinstance(prompt, str):
                prompt_text = prompt
            elif isinstance(prompt, list):
                if prompt and isinstance(prompt[0], dict):
                    # Messages format: extract content from each message
                    prompt_text = "\n".join(m.get("content", "") for m in prompt if isinstance(m, dict))
                else:
                    # List of strings
                    prompt_text = "\n".join(str(p) for p in prompt)
            else:
                prompt_text = str(prompt)
            response = await agent.run(prompt_text)

            # Emit agent summaries through callback for group aggregation
            for pr in response.prompt_results:
                if pr.summary_data:
                    callback(pr.summary_data)

            if response.structured_output:
                result_dict = response.structured_output if isinstance(response.structured_output, dict) else response.structured_output
                preferred = result_dict.get("preferred")
                justification = result_dict.get("justification")

                self.telemetry.emit_task_end(task_id, task_name, f"Preferred: {preferred or 'N/A'}")

                return ComparisonResult(
                    unit_a_id=unit_a_id,
                    unit_b_id=unit_b_id,
                    preferred=preferred,
                    model=f"claude-{model}",
                    provider="claude_agent",
                    justification=justification,
                    raw_response=result_dict,
                )
            else:
                self.telemetry.emit_task_end(task_id, task_name, "Empty response")
                raise RuntimeError(f"Empty response from claude-{model} for comparison {unit_a_id} vs {unit_b_id}")

        except Exception as e:
            logger.exception(f"Claude agent comparison failed ({unit_a_id} vs {unit_b_id})")
            self.telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
            raise

    def _merge_ensemble_results(
        self,
        results: list[ComparisonResult],
    ) -> list[ComparisonResult]:
        """Merge per-model results into ensemble verdicts using self.ensemble_strategy.

        Groups results by (unit_a_id, unit_b_id) pair, then applies the strategy:
        - "or": If ANY model prefers A → ensemble=A. Only B if ALL prefer B.
        - "majority": Majority vote wins. Ties → None (error).

        Returns one ComparisonResult per pair.
        """
        from collections import defaultdict

        # Group by pair (normalize key so (a,b) and (b,a) don't split)
        groups: dict[tuple[str, str], list[ComparisonResult]] = defaultdict(list)
        for r in results:
            key = (r.unit_a_id, r.unit_b_id)
            groups[key].append(r)

        merged: list[ComparisonResult] = []
        strategy = self.ensemble_strategy

        for (unit_a_id, unit_b_id), group in groups.items():
            valid = [r for r in group if not r.error and r.preferred in ("A", "B")]
            models_used = [r.model for r in group]

            if not valid:
                merged.append(ComparisonResult(
                    unit_a_id=unit_a_id,
                    unit_b_id=unit_b_id,
                    preferred=None,
                    model=f"ensemble:{strategy}",
                    provider="ensemble",
                    error=f"All {len(group)} model votes failed",
                ))
                continue

            a_votes = [r for r in valid if r.preferred == "A"]
            b_votes = [r for r in valid if r.preferred == "B"]

            if strategy == "or":
                # OR: if ANY model says A, ensemble says A
                if a_votes:
                    preferred = "A"
                    justification = (
                        f"[ensemble:or] {len(a_votes)}/{len(valid)} models preferred A. "
                        f"Models: {models_used}. "
                        f"A justification: {a_votes[0].justification}"
                    )
                else:
                    preferred = "B"
                    justification = (
                        f"[ensemble:or] All {len(b_votes)} models preferred B (unanimous). "
                        f"Models: {models_used}. "
                        f"B justification: {b_votes[0].justification}"
                    )
            elif strategy == "majority":
                if len(a_votes) > len(b_votes):
                    preferred = "A"
                elif len(b_votes) > len(a_votes):
                    preferred = "B"
                else:
                    preferred = None  # Tie
                justification = (
                    f"[ensemble:majority] A={len(a_votes)}, B={len(b_votes)}. "
                    f"Models: {models_used}"
                )
            else:
                raise ValueError(f"Unknown ensemble strategy: {strategy}")

            merged.append(ComparisonResult(
                unit_a_id=unit_a_id,
                unit_b_id=unit_b_id,
                preferred=preferred,
                model=f"ensemble:{strategy}",
                provider="ensemble",
                justification=justification,
                error=None if preferred else f"Tie in {strategy} vote",
            ))

        return merged

    async def _run_round(
        self,
        round_num: int,
        pairs: list[tuple[int, int]],
        items_to_rank: list[T],
        build_pairwise_prompt: Callable[[T, T], str | list],
        system_prompt: str,
        response_schema: type[BaseModel],
        group: str,
        *,
        use_claude_agent: bool = False,
        claude_model: str = "claude-sonnet-4-5",
        claude_max_turns: int = 100,
        cwd: Path | None = None,
        output_dir: Path | None = None,
    ) -> list[ComparisonResult]:
        """Run all comparisons for a single round in parallel.

        All rounds: All LLMs judge each pair, votes_per_pair_per_llm times each.
        If swap_testing enabled, each vote runs both (A,B) and (B,A) orderings.
        If use_claude_agent, uses Claude agent instead of OpenRouter (single comparison per pair).
        Concurrency controlled by max_concurrent_comparisons (semaphore shared across rounds).
        """
        async def run_with_semaphore(coro):
            """Wrap coroutine with semaphore control."""
            if self._comparison_semaphore is not None:
                async with self._comparison_semaphore:
                    return await coro
            return await coro

        all_tasks = []

        for orig_a_idx, orig_b_idx in pairs:
            # Claude agent path - single comparison per pair (no multi-model voting)
            if use_claude_agent:
                # Randomize A/B position
                if random.random() < 0.5:
                    a_idx, b_idx = orig_a_idx, orig_b_idx
                else:
                    a_idx, b_idx = orig_b_idx, orig_a_idx

                unit_a_id = self._unit_id(a_idx)
                unit_b_id = self._unit_id(b_idx)
                prompt = build_pairwise_prompt(items_to_rank[a_idx], items_to_rank[b_idx])

                all_tasks.append(run_with_semaphore(self._run_comparison_claude_agent(
                    unit_a_id=unit_a_id,
                    unit_b_id=unit_b_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_schema=response_schema,
                    round_num=round_num,
                    model=claude_model,
                    max_turns=claude_max_turns,
                    cwd=cwd or Path.cwd(),
                    output_dir=output_dir,
                    group=group,
                )))
                continue

            # Default OpenRouter path - multi-model voting
            # When ensemble is active, fix A/B position per pair so all models
            # judge the same ordering (their votes get merged afterward).
            if self.ensemble_strategy:
                if random.random() < 0.5:
                    a_idx, b_idx = orig_a_idx, orig_b_idx
                else:
                    a_idx, b_idx = orig_b_idx, orig_a_idx
                unit_a_id = self._unit_id(a_idx)
                unit_b_id = self._unit_id(b_idx)
                prompt = build_pairwise_prompt(items_to_rank[a_idx], items_to_rank[b_idx])

                for model_cfg in self.models:
                    all_tasks.append(run_with_semaphore(self._run_comparison(
                        unit_a_id, unit_b_id,
                        prompt, system_prompt, response_schema,
                        model_cfg, group, round_num,
                    )))
            else:
                for model_cfg in self.models:
                    for _ in range(self.votes_per_pair_per_llm):
                        if self.swap_testing:
                            # Run both orderings and check consistency
                            all_tasks.append(run_with_semaphore(self._run_swap_tested_comparison(
                                orig_a_idx, orig_b_idx,
                                items_to_rank, build_pairwise_prompt,
                                system_prompt, response_schema,
                                model_cfg, group, round_num,
                            )))
                        else:
                            # Simple mode: randomize position once
                            if random.random() < 0.5:
                                a_idx, b_idx = orig_a_idx, orig_b_idx
                            else:
                                a_idx, b_idx = orig_b_idx, orig_a_idx

                            unit_a_id = self._unit_id(a_idx)
                            unit_b_id = self._unit_id(b_idx)
                            prompt = build_pairwise_prompt(items_to_rank[a_idx], items_to_rank[b_idx])

                            all_tasks.append(run_with_semaphore(self._run_comparison(
                                unit_a_id, unit_b_id,
                                prompt, system_prompt, response_schema,
                                model_cfg, group, round_num,
                            )))

        results: list[ComparisonResult] = await asyncio.gather(*all_tasks)

        # Apply ensemble merging if configured
        if self.ensemble_strategy and not use_claude_agent:
            results = self._merge_ensemble_results(results)

        return results

    async def rank(
        self,
        items_to_rank: list[T],
        build_pairwise_prompt: Callable[[T, T], str | list],
        system_prompt: str,
        response_schema: type[BaseModel],
        *,
        group: str = "rank",
        module_name: str = "RANK",
        # Claude agent params
        use_claude_agent: bool = False,
        claude_model: str = "claude-sonnet-4-5",
        claude_max_turns: int = 100,
        cwd: Path | None = None,
        output_dir: Path | None = None,
    ) -> SwissBTRankResult[T] | None:
        """
        Run Swiss-style tournament with Bradley-Terry scoring.

        Args:
            items_to_rank: List of items to compare
            build_pairwise_prompt: Function(item_a, item_b) -> comparison prompt
            system_prompt: System prompt for LLM comparisons
            response_schema: Pydantic model with "preferred" field ("A" or "B")
            group: AIITelemetry group name
            module_name: Name for logging
            use_claude_agent: If True, use Claude agent instead of OpenRouter
            claude_model: Claude model name (sonnet, opus, etc.)
            claude_max_turns: Max turns for Claude agent
            cwd: Working directory for Claude agent
            output_dir: Output directory for logs

        Returns:
            SwissBTRankResult with ranked units and convergence info
        """
        n = len(items_to_rank)
        if n < 2:
            logger.warning(f"Need at least 2 units to rank, got {n}")
            return None

        self._task_counter = 0
        model_strs = self.get_model_strs()
        unit_ids = [self._unit_id(i) for i in range(n)]

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  {module_name} - LLM PAIRWISE RANKING")
        logger.info("=" * 70)
        logger.info("")
        logger.info("CONFIGURATION")
        logger.info("-" * 40)
        logger.info(f"  Items to rank: {n}")
        logger.info(f"  LLM judges: {model_strs}")
        logger.info(f"  Max rounds: {self.max_rounds}")
        logger.info("")
        logger.info("VOTING SETUP")
        logger.info("-" * 40)
        swap_info = " (tests both orderings, discards inconsistent)" if self.swap_testing else ""
        logger.info(f"  Swap testing: {self.swap_testing}{swap_info}")
        logger.info(f"  Votes per pair per LLM: {self.votes_per_pair_per_llm}")
        logger.info("")
        logger.info("ROUND STRUCTURE")
        logger.info("-" * 40)
        logger.info(f"  Round 1 (bootstrap): Each item faces {self.bootstrap_opponents_per_hypo} random opponent(s)")
        logger.info(f"  Round 2+ (Swiss): Adjacent-ranked items compete (1v2, 2v3, ...)")
        logger.info(f"  Early stop: When P(#1 beats #2) >= {self.early_stop_win_prob:.0%}")
        logger.info("")
        logger.info("=" * 70)

        # Initialize
        bt_scores = {uid: BT_INITIAL_STRENGTH for uid in unit_ids}
        all_comparisons: list[ComparisonResult] = []
        round_stats: list[RoundStats] = []
        round_snapshots: list[dict] = []  # Store full ranking after each round
        converged = False
        convergence_reason = "max_rounds"

        for round_num in range(1, self.max_rounds + 1):
            logger.info("")
            logger.info("-" * 50)
            round_type = "BOOTSTRAP" if round_num == 1 else "SWISS"
            logger.info(f"  ROUND {round_num}/{self.max_rounds} ({round_type})")
            logger.info("-" * 50)

            # Generate pairs
            if round_num == 1:
                pairs = self.generate_bootstrap_pairs(n, self.bootstrap_opponents_per_hypo)
                num_comparisons = len(pairs) * len(self.models) * self.votes_per_pair_per_llm
                logger.info(f"  Pairs: {len(pairs)} (random matchups)")
            else:
                # Swiss chain pairing: each item faces neighbors in current ranking
                pairs = self.generate_swiss_pairs(n, bt_scores, unit_ids)
                num_comparisons = len(pairs) * len(self.models) * self.votes_per_pair_per_llm
                logger.info(f"  Pairs: {len(pairs)} (adjacent ranks compete)")
            logger.info(f"  API calls: {num_comparisons} ({len(pairs)} pairs × {len(self.models)} LLMs × {self.votes_per_pair_per_llm} votes)")

            if not pairs:
                logger.warning("   No pairs generated, skipping round")
                continue

            # Run round
            start_time = datetime.now()
            round_results = await self._run_round(
                round_num=round_num,
                pairs=pairs,
                items_to_rank=items_to_rank,
                build_pairwise_prompt=build_pairwise_prompt,
                system_prompt=system_prompt,
                response_schema=response_schema,
                group=group,
                use_claude_agent=use_claude_agent,
                claude_model=claude_model,
                claude_max_turns=claude_max_turns,
                cwd=cwd,
                output_dir=output_dir,
            )
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            all_comparisons.extend(round_results)

            # Update BT scores
            bt_scores = self.calculate_bradley_terry(all_comparisons, n)

            # Get top two for convergence check
            sorted_by_bt = sorted(
                unit_ids,
                key=lambda uid: bt_scores.get(uid, 0),
                reverse=True,
            )
            top_id = sorted_by_bt[0]
            second_id = sorted_by_bt[1] if len(sorted_by_bt) > 1 else sorted_by_bt[0]
            top_score = bt_scores[top_id]
            second_score = bt_scores[second_id]
            gap = top_score - second_score

            # Track round stats
            successful = len([r for r in round_results if not r.error])
            stats = RoundStats(
                round_num=round_num,
                num_pairs=len(pairs),
                comparisons_run=len(round_results),
                successful=successful,
                failed=len(round_results) - successful,
                duration_ms=duration_ms,
                top_unit_id=top_id,
                top_bt_score=top_score,
                second_bt_score=second_score,
                gap=gap,
            )
            round_stats.append(stats)

            # Calculate win probability for interpretable output
            current_win_prob = gap_to_win_prob(gap)

            failed = len(round_results) - successful
            failed_info = f", {failed} failed" if failed > 0 else ""
            logger.info(f"  Completed: {successful} votes{failed_info} ({duration_ms/1000:.1f}s)")

            # Calculate current ranking state
            round_win_rates = self.calculate_win_rates(all_comparisons, n)

            # Count wins/losses per unit
            win_counts = {uid: 0 for uid in unit_ids}
            loss_counts = {uid: 0 for uid in unit_ids}
            for comp in all_comparisons:
                if comp.error or not comp.preferred:
                    continue
                if comp.preferred == "A":
                    win_counts[comp.unit_a_id] += 1
                    loss_counts[comp.unit_b_id] += 1
                else:
                    win_counts[comp.unit_b_id] += 1
                    loss_counts[comp.unit_a_id] += 1

            # Store snapshot for end summary
            round_snapshots.append({
                'round_num': round_num,
                'sorted_units': list(sorted_by_bt),
                'bt_scores': dict(bt_scores),
                'win_rates': dict(round_win_rates),
                'win_counts': dict(win_counts),
                'loss_counts': dict(loss_counts),
                'gap': gap,
                'win_prob': current_win_prob,
            })

            # Show ranking table after this round
            logger.info("")
            logger.info(f"  Current standings (BT=Bradley-Terry score, WR=win rate):")
            logger.info(f"  {'Rank':<5} {'ID':<10} {'BT':<8} {'WR':<7} {'W-L':<7}")
            logger.info(f"  {'-'*5} {'-'*10} {'-'*8} {'-'*7} {'-'*7}")
            for rank, uid in enumerate(sorted_by_bt, 1):
                bt = bt_scores.get(uid, 1.0)
                wr = round_win_rates.get(uid, 0.5)
                w = win_counts.get(uid, 0)
                l = loss_counts.get(uid, 0)
                logger.info(f"  {rank:<5} {uid:<10} {bt:<8.2f} {wr:<7.0%} {w}-{l}")

            # Early stop check
            logger.info("")
            logger.info(f"  Confidence: P(#{1} beats #{2}) = {current_win_prob:.0%} (need {self.early_stop_win_prob:.0%} to stop early)")

            if round_num >= 2 and current_win_prob >= self.early_stop_win_prob:
                converged = True
                convergence_reason = f"win_prob ({current_win_prob:.0%} >= {self.early_stop_win_prob:.0%})"
                logger.info(f"  >> EARLY STOP: {top_id} is the clear winner ({current_win_prob:.0%} confidence)")
                break

        # Final calculations
        win_rates = self.calculate_win_rates(all_comparisons, n)

        # Build ranked units (sorted by BT score)
        ranked_units: list[RankedUnit[T]] = []
        for i in range(n):
            unit_id = self._unit_id(i)
            ranked_units.append(RankedUnit(
                unit_id=unit_id,
                unit=items_to_rank[i],
                win_rate=round(win_rates.get(unit_id, 0.5), 3),
                elo_rating=round(bt_scores.get(unit_id, 1.0) * 1000, 1),  # Scale for display
            ))

        ranked_units.sort(key=lambda x: bt_scores.get(x.unit_id, 0), reverse=True)

        for i, ru in enumerate(ranked_units):
            ru.rank = i + 1

        # Log tournament summary
        logger.info("")
        logger.info("")
        logger.info("=" * 70)
        logger.info("  TOURNAMENT COMPLETE")
        logger.info("=" * 70)
        logger.info("")
        logger.info("SUMMARY")
        logger.info("-" * 40)
        total_successful = len([c for c in all_comparisons if not c.error])
        total_failed = len(all_comparisons) - total_successful
        failed_info = f" ({total_failed} failed)" if total_failed > 0 else ""
        logger.info(f"  Rounds: {len(round_stats)}/{self.max_rounds}")
        logger.info(f"  Comparisons: {total_successful}{failed_info}")
        if converged:
            logger.info(f"  Result: Early stop - clear winner found")
        else:
            logger.info(f"  Result: Completed all rounds")
        logger.info("")

        # Round-by-round progression (compact)
        logger.info("ROUND PROGRESSION")
        logger.info("-" * 40)
        logger.info(f"  {'Rnd':<4} {'Type':<10} {'Votes':<6} {'Leader':<10} {'P(1>2)':<8} {'Time':<6}")
        logger.info(f"  {'-'*4} {'-'*10} {'-'*6} {'-'*10} {'-'*8} {'-'*6}")
        for s in round_stats:
            wp = gap_to_win_prob(s.gap)
            round_type = "Bootstrap" if s.round_num == 1 else "Swiss"
            logger.info(
                f"  {s.round_num:<4} {round_type:<10} {s.successful:<6} {s.top_unit_id:<10} "
                f"{wp:<8.0%} {s.duration_ms/1000:<.1f}s"
            )
        logger.info("")

        # Final rankings with titles
        logger.info("FINAL RANKINGS")
        logger.info("-" * 40)
        logger.info(f"  {'#':<3} {'ID':<10} {'BT':<8} {'WR':<7} {'Title'}")
        logger.info(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*7} {'-'*35}")
        for ru in ranked_units:
            title = "N/A"
            if isinstance(ru.unit, dict):
                title = ru.unit.get('title', ru.unit.get('name', 'N/A'))
            elif hasattr(ru.unit, 'title'):
                title = ru.unit.title
            elif isinstance(ru.unit, int):
                title = f"idx={ru.unit}"
            bt = bt_scores.get(ru.unit_id, 1.0)
            logger.info(f"  {ru.rank:<3} {ru.unit_id:<10} {bt:<8.2f} {ru.win_rate:<7.0%} {title[:35]}")
        logger.info("")

        # Compute bootstrap confidence intervals
        logger.info("CONFIDENCE INTERVALS (95%)")
        logger.info("-" * 40)
        logger.info("  Computing bootstrap CIs (1000 resamples)...")
        bt_confidence_intervals = self.calculate_bootstrap_ci(all_comparisons, n)

        logger.info(f"  {'#':<3} {'ID':<10} {'BT':<8} {'95% CI':<16} {'Notes'}")
        logger.info(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*16} {'-'*20}")
        prev_ci = None
        for ru in ranked_units:
            bt = bt_scores.get(ru.unit_id, 1.0)
            ci = bt_confidence_intervals.get(ru.unit_id, (0.0, 2.0))
            notes = ""
            if prev_ci is not None:
                # Check if current CI overlaps with previous (higher ranked)
                if ci[1] >= prev_ci[0]:
                    notes = "overlaps above"
            logger.info(f"  {ru.rank:<3} {ru.unit_id:<10} {bt:<8.2f} [{ci[0]:.2f}-{ci[1]:.2f}]     {notes}")
            prev_ci = ci
        logger.info("")
        logger.info("=" * 70)

        # Compute and log diagnostics
        diagnostics = compute_diagnostics(all_comparisons)
        diag_text = format_diagnostics(diagnostics, models=model_strs)
        for line in diag_text.split("\n"):
            logger.info(line)

        # Build metadata
        successful = len([c for c in all_comparisons if not c.error])
        metadata = {
            'ranking_method': 'swiss_bradley_terry',
            'num_units': n,
            'max_rounds': self.max_rounds,
            'rounds_completed': len(round_stats),
            'bootstrap_opponents_per_hypo': self.bootstrap_opponents_per_hypo,
            'votes_per_pair_per_llm': self.votes_per_pair_per_llm,
            'early_stop_win_prob': self.early_stop_win_prob,
            'models': model_strs,
            'model_selection': 'random_per_comparison',
            'total_comparisons': len(all_comparisons),
            'successful_comparisons': successful,
            'failed_comparisons': len(all_comparisons) - successful,
            'converged': converged,
            'convergence_reason': convergence_reason,
            'round_stats': [
                {
                    'round': s.round_num,
                    'pairs': s.num_pairs,
                    'successful': s.successful,
                    'top_id': s.top_unit_id,
                    'top_bt': round(s.top_bt_score, 3),
                    'gap': round(s.gap, 3),
                    'duration_ms': round(s.duration_ms, 1),
                }
                for s in round_stats
            ],
        }

        return SwissBTRankResult(
            ranked_units=ranked_units,
            bt_scores={k: round(v, 4) for k, v in bt_scores.items()},
            bt_confidence_intervals=bt_confidence_intervals,
            win_rates=win_rates,
            all_comparisons=all_comparisons,
            rounds_completed=len(round_stats),
            converged=converged,
            convergence_reason=convergence_reason,
            metadata=metadata,
            module_summary=None,  # No longer tracked at group level
            diagnostics=diagnostics,
        )


__all__ = [
    "SwissBTRanker",
    "SwissBTRankResult",
    "RoundStats",
    "BT_INITIAL_STRENGTH",
    "BT_CONVERGENCE_THRESHOLD",
    "BT_MAX_ITERATIONS",
    "win_prob_to_gap",
    "gap_to_win_prob",
]
