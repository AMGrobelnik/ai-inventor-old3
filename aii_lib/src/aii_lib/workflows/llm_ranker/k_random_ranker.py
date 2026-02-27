"""
K-Random Opponents Pairwise Ranking with Multiple LLMs.

Single-round ranking where each unit faces K randomly selected opponents.
All comparisons run in parallel, results aggregated by win rate.

Features:
- Multiple LLM models vote on each matchup
- OpenRouter provider support via chat() wrapper
- Pydantic schema for structured JSON responses
- Win rate (primary) + ELO (tiebreaker) ranking
- Parallel execution with telemetry integration
- Model-specific configs (reasoning_effort, suffix)

Used by: rank_hypo, rank_prop, rank_narr in aii_pipeline
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Generic, TypeVar

from aii_lib.telemetry import logger, MessageType
from pydantic import BaseModel

from ...telemetry import AIITelemetry
from ...llm_backend import OpenRouterClient
from ...llm_backend.tool_loop import chat
from ...utils import get_model_short
from .ranking_diagnostics import RankingDiagnostics, compute_diagnostics, format_diagnostics


T = TypeVar("T")

# ELO Constants (used for tiebreaker)
INITIAL_ELO = 1500
RATING_SENSITIVITY = 32


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    model: str
    reasoning_effort: str | None = None
    suffix: str | None = None  # e.g., ":nitro", ":floor" for OpenRouter


@dataclass
class ComparisonResult:
    """Result of a single pairwise comparison."""
    unit_a_id: str  # Unit shown as "A" in prompt
    unit_b_id: str  # Unit shown as "B" in prompt
    preferred: str | None  # "A", "B", or None for error/tie
    model: str
    provider: str
    justification: str | None = None
    error: str | None = None
    raw_response: dict | None = None

    @property
    def pair_key(self) -> tuple[str, str]:
        """Canonical pair identifier (sorted) for grouping comparisons of same pair."""
        return tuple(sorted([self.unit_a_id, self.unit_b_id]))

    @property
    def winner_id(self) -> str | None:
        """ID of the winning unit, or None if no valid preference."""
        if self.preferred == "A":
            return self.unit_a_id
        elif self.preferred == "B":
            return self.unit_b_id
        return None


@dataclass
class RankedUnit(Generic[T]):
    """A ranked unit with scores."""
    unit_id: str
    unit: T
    win_rate: float
    elo_rating: float
    rank: int = 0
    ci_lower: float = 0.0  # 95% CI lower bound for win rate
    ci_upper: float = 1.0  # 95% CI upper bound for win rate


@dataclass
class KRandomRankResult(Generic[T]):
    """Complete result of K-Random ranking workflow."""
    ranked_units: list[RankedUnit[T]]
    win_rates: dict[str, float]
    elo_ratings: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]  # unit_id -> (lower, upper)
    all_comparisons: list[ComparisonResult]
    metadata: dict
    diagnostics: RankingDiagnostics | None = None
    module_summary: dict | None = None




class KRandomRanker(Generic[T]):
    """
    K-Random Opponents Pairwise Ranking with Multiple LLMs.

    Single-round ranking method:
    1. For each unit, select K random opponents
    2. Run all comparisons in parallel (num_models x num_units x K)
    3. Aggregate votes across all models
    4. Calculate win rate and ELO from results
    5. Rank by win rate (primary), ELO (tiebreaker)

    Example:
        ranker = KRandomRanker(
            telemetry=telemetry,
            api_keys={"openrouter": "sk-or-..."},
            models=[
                {"model": "openai/gpt-4.1-mini"},
                {"model": "anthropic/claude-sonnet-4", "suffix": "nitro"},
            ],
            comparisons_per_unit_per_llm=3,
        )
        result = await ranker.rank(
            items_to_rank=hypotheses,
            build_pairwise_prompt=lambda a, b: f"Compare A vs B...",
            system_prompt="You are a research evaluator...",
            response_schema=PairwisePreference,
        )
    """

    def __init__(
        self,
        telemetry: AIITelemetry,
        api_keys: dict,
        models: list[dict | str | ModelConfig],
        *,
        comparisons_per_unit_per_llm: int = 3,
        llm_timeout: int = 180,
        unit_id_prefix: str = "unit",
        initial_elo: float = INITIAL_ELO,
        rating_sensitivity: float = RATING_SENSITIVITY,
        swap_testing: bool = False,
        max_concurrent_comparisons: int | None = None,
        comparison_semaphore: asyncio.Semaphore | None = None,
    ):
        """
        Args:
            telemetry: AIITelemetry instance for logging
            api_keys: Dict with "openrouter" key
            models: List of model configs (str, dict, or ModelConfig)
            comparisons_per_unit_per_llm: K random opponents per unit per LLM
            llm_timeout: Timeout per comparison in seconds
            unit_id_prefix: Prefix for unit IDs (e.g., "hypo", "prop")
            initial_elo: Starting ELO score
            rating_sensitivity: ELO K-factor (higher = faster rating changes)
            swap_testing: If True, run both (A,B) and (B,A) orderings and only
                         count vote if both agree (filters position bias)
            max_concurrent_comparisons: Max LLM comparisons running in parallel.
                If None, all comparisons run concurrently (no limit). Ignored if
                comparison_semaphore is provided.
            comparison_semaphore: Optional external semaphore for concurrency control.
                Use this to share a semaphore across multiple ranker instances.
        """
        self.telemetry = telemetry
        self.api_keys = api_keys
        self.models = self._parse_models(models)
        self.k = comparisons_per_unit_per_llm
        self.llm_timeout = llm_timeout
        self.unit_id_prefix = unit_id_prefix
        self.initial_elo = initial_elo
        self.rating_sensitivity = rating_sensitivity
        self.swap_testing = swap_testing
        self.max_concurrent_comparisons = max_concurrent_comparisons
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

    def _unit_id(self, idx: int, item=None) -> str:
        """Generate unit ID from index, or use item directly if it's a string."""
        if item is not None and isinstance(item, str):
            return item
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
    def generate_k_random_pairs(n: int, k: int) -> list[tuple[int, int]]:
        """
        Generate K random opponent comparisons for each unit.

        Each unit gets K comparisons:
        - If enough opponents exist (n-1 >= k): sample K unique opponents
        - If not enough opponents (n-1 < k): sample with replacement

        Args:
            n: Number of units
            k: Number of comparisons per unit

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

    async def _run_comparison(
        self,
        unit_a_id: str,
        unit_b_id: str,
        prompt: str | list,
        system_prompt: str,
        response_schema: type[BaseModel],
        model_cfg: ModelConfig,
        group: str,
    ) -> ComparisonResult:
        """Run pairwise comparison using chat() wrapper with OpenRouter.

        Supports multimodal prompts (list of content blocks with text/images)."""
        model_short = get_model_short(model_cfg.model)
        task_num = self._next_task_id()
        task_id = f"cmp_{unit_a_id}_{unit_b_id}_{model_short}_{self._unique_suffix()}"
        task_name = f"{self.unit_id_prefix}-cmp{task_num}__{model_short}"

        self.telemetry.emit_task_start(task_id, task_name)
        callback = self.telemetry.create_callback(task_id, task_name, group=group)

        try:
            # Apply suffix (e.g., :nitro or :floor) if specified
            effective_model = OpenRouterClient.resolve_model(model_cfg.model, model_cfg.suffix)

            async with OpenRouterClient(
                api_key=self.api_keys.get('openrouter'),
                model=effective_model,
                timeout=self.llm_timeout,
            ) as client:
                # Use chat() wrapper - no tools, just structured output
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

                    # Warn if preferred is not a valid value
                    if preferred not in ("A", "B"):
                        logger.warning(
                            f"Invalid preferred value '{preferred}' from {model_cfg.model}. "
                            f"Expected 'A' or 'B'. Raw response: {result_dict}"
                        )

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
                    return ComparisonResult(
                        unit_a_id=unit_a_id,
                        unit_b_id=unit_b_id,
                        preferred=None,
                        model=model_cfg.model,
                        provider="openrouter",
                        error="Empty response",
                    )

        except asyncio.TimeoutError:
            self.telemetry.emit_task_end(task_id, task_name, f"Timeout ({self.llm_timeout}s)")
            raise

        except Exception as e:
            self.telemetry.emit_task_end(task_id, task_name, f"Error: {e}")
            raise

    async def _run_swap_tested_comparison(
        self,
        orig_a_idx: int,
        orig_b_idx: int,
        items_to_rank: list,
        build_pairwise_prompt,
        system_prompt: str,
        response_schema: type[BaseModel],
        model_cfg: ModelConfig,
        group: str,
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
        # Use item as unit_id if it's a string
        unit_a_id = self._unit_id(orig_a_idx, items_to_rank[orig_a_idx])
        unit_b_id = self._unit_id(orig_b_idx, items_to_rank[orig_b_idx])

        # Run normal order (A=orig_a, B=orig_b)
        prompt_normal = build_pairwise_prompt(items_to_rank[orig_a_idx], items_to_rank[orig_b_idx])
        result_normal = await self._run_comparison(
            unit_a_id, unit_b_id,
            prompt_normal, system_prompt, response_schema,
            model_cfg, group,
        )

        # Run swapped order (A=orig_b, B=orig_a)
        prompt_swapped = build_pairwise_prompt(items_to_rank[orig_b_idx], items_to_rank[orig_a_idx])
        result_swapped = await self._run_comparison(
            unit_b_id, unit_a_id,  # Note: swapped IDs
            prompt_swapped, system_prompt, response_schema,
            model_cfg, group,
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

    def calculate_win_rates(
        self,
        comparisons: list[ComparisonResult],
        n: int,
        unit_ids: list[str] | None = None,
    ) -> dict[str, float]:
        """Calculate win rates from pairwise comparison results."""
        if unit_ids is None:
            unit_ids = [self._unit_id(i) for i in range(n)]
        wins = {uid: 0.0 for uid in unit_ids}
        games = {uid: 0 for uid in unit_ids}

        for comp in comparisons:
            if comp.error:
                continue

            if not comp.preferred or comp.preferred not in ["A", "B"]:
                continue

            games[comp.unit_a_id] += 1
            games[comp.unit_b_id] += 1

            if comp.preferred == "A":
                wins[comp.unit_a_id] += 1.0
            else:
                wins[comp.unit_b_id] += 1.0

        # Calculate win rates
        win_rates = {}
        for unit_id in wins:
            if games[unit_id] > 0:
                win_rates[unit_id] = wins[unit_id] / games[unit_id]
            else:
                win_rates[unit_id] = 0.5  # No games = neutral

        return win_rates

    def calculate_elo_ratings(
        self,
        comparisons: list[ComparisonResult],
        n: int,
        unit_ids: list[str] | None = None,
    ) -> dict[str, float]:
        """Calculate ELO ratings from pairwise comparison results (tiebreaker only)."""
        if unit_ids is None:
            unit_ids = [self._unit_id(i) for i in range(n)]
        ratings = {uid: self.initial_elo for uid in unit_ids}

        # Shuffle comparisons to reduce order bias
        shuffled = list(comparisons)
        random.shuffle(shuffled)

        for comp in shuffled:
            if comp.error:
                continue

            if not comp.preferred or comp.preferred not in ["A", "B"]:
                continue

            ra = ratings[comp.unit_a_id]
            rb = ratings[comp.unit_b_id]

            ea = 1 / (1 + 10 ** ((rb - ra) / 400))
            eb = 1 / (1 + 10 ** ((ra - rb) / 400))

            if comp.preferred == "A":
                sa, sb = 1, 0
            else:
                sa, sb = 0, 1

            ratings[comp.unit_a_id] = ra + self.rating_sensitivity * (sa - ea)
            ratings[comp.unit_b_id] = rb + self.rating_sensitivity * (sb - eb)

        return ratings

    def calculate_bootstrap_ci(
        self,
        comparisons: list[ComparisonResult],
        n: int,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        unit_ids: list[str] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for win rates.

        Resamples comparisons with replacement, computes win rates for each
        resample, and returns percentile-based confidence intervals.

        Args:
            comparisons: All comparison results
            n: Number of units
            n_bootstrap: Number of bootstrap iterations (default 1000)
            ci_level: Confidence level (default 0.95 for 95% CI)
            unit_ids: List of unit IDs (if None, generates from index)

        Returns:
            Dict of unit_id -> (lower_bound, upper_bound)
        """
        if unit_ids is None:
            unit_ids = [self._unit_id(i) for i in range(n)]

        # Filter to valid comparisons only
        valid_comparisons = [
            c for c in comparisons
            if not c.error and c.preferred in ["A", "B"]
        ]

        if len(valid_comparisons) < 3:
            # Not enough data for meaningful bootstrap
            return {uid: (0.0, 1.0) for uid in unit_ids}

        # Collect bootstrap win rates
        bootstrap_rates: dict[str, list[float]] = {uid: [] for uid in unit_ids}

        for _ in range(n_bootstrap):
            # Resample with replacement
            resampled = random.choices(valid_comparisons, k=len(valid_comparisons))

            # Compute win rates on resample
            win_rates = self.calculate_win_rates(resampled, n, unit_ids=unit_ids)

            for uid in unit_ids:
                bootstrap_rates[uid].append(win_rates.get(uid, 0.5))

        # Compute percentiles
        alpha = (1 - ci_level) / 2
        lower_pct = alpha * 100
        upper_pct = (1 - alpha) * 100

        confidence_intervals = {}
        for uid in unit_ids:
            rates = sorted(bootstrap_rates[uid])
            lower_idx = int(lower_pct / 100 * len(rates))
            upper_idx = int(upper_pct / 100 * len(rates)) - 1
            lower_idx = max(0, min(lower_idx, len(rates) - 1))
            upper_idx = max(0, min(upper_idx, len(rates) - 1))
            confidence_intervals[uid] = (rates[lower_idx], rates[upper_idx])

        return confidence_intervals

    async def rank(
        self,
        items_to_rank: list[T],
        build_pairwise_prompt: Callable[[T, T], str | list],
        system_prompt: str,
        response_schema: type[BaseModel],
        *,
        group: str = "rank",
        module_name: str = "RANK",
    ) -> KRandomRankResult[T] | None:
        """
        Run pairwise ranking on a list of items.

        For each item, K random opponents are selected. Each matchup is sent to
        all configured LLMs for voting. Results are aggregated into win rates
        and ELO scores.

        Args:
            items_to_rank: List of items to compare. Can be objects or indices.
                If using indices [0, 1, 2, ...], your build_pairwise_prompt
                should look up external data by index.
            build_pairwise_prompt: Function(item_a, item_b) -> comparison prompt.
                Receives pairs from items_to_rank for each matchup.
            system_prompt: System prompt for LLM comparisons.
            response_schema: Pydantic model with "preferred" field ("A" or "B").
            group: AIITelemetry group name.
            module_name: Name for logging.

        Returns:
            KRandomRankResult where RankedUnit.unit contains items from items_to_rank.
            Returns None if ranking fails.
        """
        n = len(items_to_rank)
        if n < 2:
            logger.warning(f"Need at least 2 units to rank, got {n}")
            return None

        # Reset task counter
        self._task_counter = 0

        # Build unit IDs - use item directly if string, otherwise generate from prefix+index
        unit_ids = [self._unit_id(i, items_to_rank[i]) for i in range(n)]

        # Generate K-random opponent pairs
        pairs = self.generate_k_random_pairs(n, self.k)
        votes_per_pair = len(self.models)
        llm_calls_per_vote = 2 if self.swap_testing else 1
        total_llm_calls = len(pairs) * votes_per_pair * llm_calls_per_vote
        model_strs = self.get_model_strs()

        self.telemetry.emit(MessageType.INFO, "=" * 60)
        self.telemetry.emit(MessageType.INFO, f"{module_name} - K-Random Opponents Ranking")
        self.telemetry.emit(MessageType.INFO, f"   Units: {n}")
        self.telemetry.emit(MessageType.INFO, f"   Comparisons per unit (K): {self.k}")
        self.telemetry.emit(MessageType.INFO, f"   Matchups: {len(pairs)} ({n} x {self.k})")
        self.telemetry.emit(MessageType.INFO, f"   Models: {len(self.models)} -> {model_strs}")
        self.telemetry.emit(MessageType.INFO, f"   Swap testing: {self.swap_testing}")
        self.telemetry.emit(MessageType.INFO, f"   Timeout: {self.llm_timeout}s per comparison")
        self.telemetry.emit(MessageType.INFO, f"   Total LLM calls: {total_llm_calls}")
        self.telemetry.emit(MessageType.INFO, "=" * 60)

        # Helper to wrap coroutines with semaphore control
        async def run_with_semaphore(coro):
            if self._comparison_semaphore is not None:
                async with self._comparison_semaphore:
                    return await coro
            return await coro

        # Create all comparison tasks (wrapped with semaphore if configured)
        all_tasks = []
        for a_idx, b_idx in pairs:
            # Each model votes on this matchup
            for model_cfg in self.models:
                if self.swap_testing:
                    # Run both orderings and check consistency
                    all_tasks.append(run_with_semaphore(self._run_swap_tested_comparison(
                        a_idx, b_idx,
                        items_to_rank, build_pairwise_prompt,
                        system_prompt, response_schema,
                        model_cfg, group,
                    )))
                else:
                    # Simple mode: just run once
                    unit_a_id = unit_ids[a_idx]
                    unit_b_id = unit_ids[b_idx]
                    prompt = build_pairwise_prompt(items_to_rank[a_idx], items_to_rank[b_idx])
                    all_tasks.append(run_with_semaphore(self._run_comparison(
                        unit_a_id, unit_b_id,
                        prompt, system_prompt, response_schema,
                        model_cfg, group,
                    )))

        # Run ALL comparisons in parallel (concurrency limited by semaphore if configured)
        self.telemetry.emit(MessageType.INFO, f"   Running {len(all_tasks)} comparisons in parallel...")
        start_time = datetime.now()
        all_comparisons: list[ComparisonResult] = await asyncio.gather(*all_tasks)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate win rates, ELO, and confidence intervals
        win_rates = self.calculate_win_rates(all_comparisons, n, unit_ids=unit_ids)
        elo_ratings = self.calculate_elo_ratings(all_comparisons, n, unit_ids=unit_ids)
        confidence_intervals = self.calculate_bootstrap_ci(all_comparisons, n, unit_ids=unit_ids)

        # Build ranked units list with confidence intervals
        ranked_units: list[RankedUnit[T]] = []
        for i in range(n):
            unit_id = unit_ids[i]
            ci = confidence_intervals.get(unit_id, (0.0, 1.0))
            ranked_units.append(RankedUnit(
                unit_id=unit_id,
                unit=items_to_rank[i],
                win_rate=round(win_rates[unit_id], 3),
                elo_rating=round(elo_ratings[unit_id], 1),
                ci_lower=round(ci[0], 3),
                ci_upper=round(ci[1], 3),
            ))

        # Sort by win rate (primary), then ELO (tiebreaker)
        ranked_units.sort(key=lambda x: (x.win_rate, x.elo_rating), reverse=True)

        # Assign ranks
        for i, ru in enumerate(ranked_units):
            ru.rank = i + 1

        # Compute diagnostics
        diagnostics = compute_diagnostics(all_comparisons)
        diag_text = format_diagnostics(diagnostics, models=model_strs)
        self.telemetry.emit(MessageType.INFO, diag_text)

        # Log leaderboard
        self.telemetry.emit(MessageType.INFO, "")
        self.telemetry.emit(MessageType.INFO, "=" * 70)
        self.telemetry.emit(MessageType.INFO, "  LEADERBOARD")
        self.telemetry.emit(MessageType.INFO, "=" * 70)
        self.telemetry.emit(MessageType.INFO, "")
        self.telemetry.emit(MessageType.INFO, "  Rank  Unit                    Win Rate    95% CI           ELO")
        self.telemetry.emit(MessageType.INFO, "  " + "-" * 66)
        for ru in ranked_units:
            # Try to get a title/name for display
            title = "N/A"
            if isinstance(ru.unit, str):
                title = ru.unit
            elif isinstance(ru.unit, dict):
                title = ru.unit.get('title', ru.unit.get('name', 'N/A'))
            elif hasattr(ru.unit, 'title'):
                title = ru.unit.title
            elif isinstance(ru.unit, int):
                title = f"idx={ru.unit}"
            title = title[:22]  # Truncate for display
            ci_str = f"[{ru.ci_lower:.0%}-{ru.ci_upper:.0%}]"
            self.telemetry.emit(
                MessageType.INFO,
                f"  {ru.rank:>3}.  {title:<22}  {ru.win_rate:>6.1%}    {ci_str:<14}  {ru.elo_rating:.0f}"
            )
        self.telemetry.emit(MessageType.INFO, "")
        self.telemetry.emit(MessageType.INFO, "=" * 70)

        # Build metadata
        successful = len([c for c in all_comparisons if not c.error])
        metadata = {
            'ranking_method': 'k_random_opponents',
            'num_units': n,
            'comparisons_per_unit_per_llm': self.k,
            'unique_matchups': len(pairs),
            'models': model_strs,
            'swap_testing': self.swap_testing,
            'total_comparisons': len(all_comparisons),
            'successful_comparisons': successful,
            'failed_comparisons': len(all_comparisons) - successful,
            'duration_ms': duration_ms,
        }

        return KRandomRankResult(
            ranked_units=ranked_units,
            win_rates=win_rates,
            elo_ratings=elo_ratings,
            confidence_intervals=confidence_intervals,
            all_comparisons=all_comparisons,
            metadata=metadata,
            diagnostics=diagnostics,
        )


__all__ = [
    "KRandomRanker",
    "KRandomRankResult",
    "RankedUnit",
    "ComparisonResult",
    "ModelConfig",
    "INITIAL_ELO",
    "RATING_SENSITIVITY",
]
