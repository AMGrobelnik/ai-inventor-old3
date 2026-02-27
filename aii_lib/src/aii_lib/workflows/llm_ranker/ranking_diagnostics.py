"""
Ranking Diagnostics - Measure LLM judge quality and consistency.

Computes:
1. Cross-model agreement: Do different models agree on same pairs?
2. Self-consistency: Does same model give same answer on repeated comparisons?
3. Position bias: Does showing unit as "A" vs "B" affect outcome?
4. Transitivity violations: If A>B and B>C, is A>C?
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .k_random_ranker import ComparisonResult


@dataclass
class PairStats:
    """Statistics for a single pair across all comparisons."""
    pair_key: tuple[str, str]
    total_comparisons: int = 0
    first_wins: int = 0  # pair_key[0] wins
    second_wins: int = 0  # pair_key[1] wins
    errors: int = 0
    # Track by position
    first_as_a_wins: int = 0  # pair_key[0] shown as A, and wins
    first_as_a_total: int = 0  # pair_key[0] shown as A
    first_as_b_wins: int = 0  # pair_key[0] shown as B, and wins
    first_as_b_total: int = 0  # pair_key[0] shown as B
    # Track by model
    model_votes: dict = field(default_factory=dict)  # model -> {"first": n, "second": n}


@dataclass
class ModelStats:
    """Statistics for a single model across all comparisons."""
    model: str
    total_comparisons: int = 0
    successful: int = 0
    errors: int = 0
    # Self-consistency: for pairs this model judged multiple times
    consistent_pairs: int = 0  # Pairs where all votes agreed
    inconsistent_pairs: int = 0  # Pairs where votes disagreed
    total_repeated_pairs: int = 0
    # Position bias
    voted_a: int = 0  # Times this model voted for position A
    voted_b: int = 0  # Times this model voted for position B
    # Swap testing
    swap_discarded: int = 0  # Votes discarded due to position bias in swap test


@dataclass
class RankingDiagnostics:
    """Complete diagnostic report for ranking quality."""
    # Overall
    total_comparisons: int = 0
    successful_comparisons: int = 0
    error_rate: float = 0.0

    # Cross-model agreement
    cross_model_agreement: float = 0.0  # Overall agreement rate
    model_agreement_matrix: dict = field(default_factory=dict)  # model_a -> model_b -> agreement%
    pairs_with_unanimous_agreement: int = 0
    pairs_with_disagreement: int = 0
    most_contested_pairs: list = field(default_factory=list)  # (pair_key, agreement%)

    # Self-consistency (per model)
    model_self_consistency: dict = field(default_factory=dict)  # model -> consistency%
    overall_self_consistency: float = 0.0

    # Position bias
    overall_position_a_rate: float = 0.5  # Rate of voting for A position
    model_position_bias: dict = field(default_factory=dict)  # model -> a_rate
    pair_position_bias: list = field(default_factory=list)  # (pair_key, a_rate, b_rate) for biased pairs

    # Swap testing (position bias filtering)
    swap_test_total: int = 0  # Total swap-tested vote pairs
    swap_test_discarded: int = 0  # Vote pairs discarded due to position bias
    swap_test_discard_rate: float = 0.0  # Overall discard rate
    model_swap_discard_rate: dict = field(default_factory=dict)  # model -> discard rate

    # Transitivity
    transitivity_violations: list = field(default_factory=list)  # [(A, B, C), ...] where A>B, B>C but C>A
    transitivity_violation_rate: float = 0.0

    # Per-model stats
    model_stats: dict = field(default_factory=dict)  # model -> ModelStats

    # Per-pair stats
    pair_stats: dict = field(default_factory=dict)  # pair_key -> PairStats


def compute_diagnostics(comparisons: list[ComparisonResult]) -> RankingDiagnostics:
    """Compute all diagnostics from comparison results."""
    diag = RankingDiagnostics()

    if not comparisons:
        return diag

    # Group comparisons by pair and by model
    by_pair: dict[tuple, list[ComparisonResult]] = defaultdict(list)
    by_model: dict[str, list[ComparisonResult]] = defaultdict(list)
    by_pair_model: dict[tuple, dict[str, list[ComparisonResult]]] = defaultdict(lambda: defaultdict(list))

    for comp in comparisons:
        pair_key = comp.pair_key
        by_pair[pair_key].append(comp)
        by_model[comp.model].append(comp)
        by_pair_model[pair_key][comp.model].append(comp)

    diag.total_comparisons = len(comparisons)
    diag.successful_comparisons = len([c for c in comparisons if not c.error and c.preferred in ("A", "B")])
    diag.error_rate = 1 - (diag.successful_comparisons / diag.total_comparisons) if diag.total_comparisons else 0

    # 1. Compute pair stats and position bias
    _compute_pair_stats(diag, by_pair)

    # 2. Compute model stats and self-consistency
    _compute_model_stats(diag, by_model, by_pair_model)

    # 3. Compute cross-model agreement
    _compute_cross_model_agreement(diag, by_pair_model)

    # 4. Compute transitivity violations
    _compute_transitivity(diag, by_pair)

    return diag


def _compute_pair_stats(diag: RankingDiagnostics, by_pair: dict):
    """Compute per-pair statistics including position bias."""
    total_a_votes = 0
    total_b_votes = 0
    biased_pairs = []

    for pair_key, comps in by_pair.items():
        ps = PairStats(pair_key=pair_key)
        first_id, second_id = pair_key

        for comp in comps:
            ps.total_comparisons += 1

            if comp.error or comp.preferred not in ("A", "B"):
                ps.errors += 1
                continue

            winner = comp.winner_id
            if winner is None:
                # Safety check - shouldn't happen if preferred is A or B
                ps.errors += 1
                continue

            # Track who was shown as A
            if comp.unit_a_id == first_id:
                # first was shown as A
                ps.first_as_a_total += 1
                if winner == first_id:
                    ps.first_wins += 1
                    ps.first_as_a_wins += 1
                    total_a_votes += 1
                else:
                    ps.second_wins += 1
                    total_b_votes += 1
            else:
                # first was shown as B
                ps.first_as_b_total += 1
                if winner == first_id:
                    ps.first_wins += 1
                    ps.first_as_b_wins += 1
                    total_b_votes += 1
                else:
                    ps.second_wins += 1
                    total_a_votes += 1

            # Track by model
            if comp.model not in ps.model_votes:
                ps.model_votes[comp.model] = {"first": 0, "second": 0}
            if winner == first_id:
                ps.model_votes[comp.model]["first"] += 1
            else:
                ps.model_votes[comp.model]["second"] += 1

        diag.pair_stats[pair_key] = ps

        # Check for position bias in this pair
        if ps.first_as_a_total > 0 and ps.first_as_b_total > 0:
            a_rate = ps.first_as_a_wins / ps.first_as_a_total if ps.first_as_a_total else 0
            b_rate = ps.first_as_b_wins / ps.first_as_b_total if ps.first_as_b_total else 0
            # Significant bias if rates differ by more than 25%
            if abs(a_rate - b_rate) > 0.25:
                biased_pairs.append((pair_key, a_rate, b_rate))

    # Overall position bias
    total_valid = total_a_votes + total_b_votes
    diag.overall_position_a_rate = total_a_votes / total_valid if total_valid else 0.5
    diag.pair_position_bias = sorted(biased_pairs, key=lambda x: abs(x[1] - x[2]), reverse=True)[:10]


def _compute_model_stats(diag: RankingDiagnostics, by_model: dict, by_pair_model: dict):
    """Compute per-model statistics including self-consistency and swap testing."""
    for model, comps in by_model.items():
        ms = ModelStats(model=model)
        ms.total_comparisons = len(comps)
        ms.successful = len([c for c in comps if not c.error and c.preferred in ("A", "B")])
        ms.errors = ms.total_comparisons - ms.successful

        for comp in comps:
            # Count swap testing discards (error contains "Position bias:")
            if comp.error and "Position bias:" in str(comp.error):
                ms.swap_discarded += 1
                diag.swap_test_discarded += 1
                diag.swap_test_total += 1
            elif comp.justification and "[Swap-tested]" in str(comp.justification):
                # Successful swap-tested vote
                diag.swap_test_total += 1

            if comp.error or comp.preferred not in ("A", "B"):
                continue
            if comp.preferred == "A":
                ms.voted_a += 1
            else:
                ms.voted_b += 1

        diag.model_stats[model] = ms

        # Position bias per model
        total_votes = ms.voted_a + ms.voted_b
        diag.model_position_bias[model] = ms.voted_a / total_votes if total_votes else 0.5

        # Swap discard rate per model
        swap_total = ms.swap_discarded + len([c for c in comps if c.justification and "[Swap-tested]" in str(c.justification)])
        diag.model_swap_discard_rate[model] = ms.swap_discarded / swap_total if swap_total else 0.0

    # Overall swap discard rate
    diag.swap_test_discard_rate = diag.swap_test_discarded / diag.swap_test_total if diag.swap_test_total else 0.0

    # Self-consistency: for each pair, did same model give same answer multiple times?
    # NOTE: We compare winner_id (actual unit), not preferred (position A/B),
    # because the same pair can be shown in different orders.
    total_consistent = 0
    total_inconsistent = 0

    for pair_key, model_comps in by_pair_model.items():
        for model, comps in model_comps.items():
            valid_comps = [c for c in comps if not c.error and c.preferred and c.winner_id]
            if len(valid_comps) < 2:
                continue

            # Check if all votes agree on the WINNER (not position)
            winners = [c.winner_id for c in valid_comps]
            if len(set(winners)) == 1:
                total_consistent += 1
                if model in diag.model_stats:
                    diag.model_stats[model].consistent_pairs += 1
            else:
                total_inconsistent += 1
                if model in diag.model_stats:
                    diag.model_stats[model].inconsistent_pairs += 1

            if model in diag.model_stats:
                diag.model_stats[model].total_repeated_pairs += 1

    # Overall self-consistency
    total_repeated = total_consistent + total_inconsistent
    diag.overall_self_consistency = total_consistent / total_repeated if total_repeated else 1.0

    # Per-model self-consistency
    for model, ms in diag.model_stats.items():
        if ms.total_repeated_pairs > 0:
            diag.model_self_consistency[model] = ms.consistent_pairs / ms.total_repeated_pairs
        else:
            diag.model_self_consistency[model] = 1.0


def _compute_cross_model_agreement(diag: RankingDiagnostics, by_pair_model: dict):
    """Compute agreement between different models."""
    models = list(diag.model_stats.keys())
    if len(models) < 2:
        diag.cross_model_agreement = 1.0
        return

    # Initialize agreement matrix
    agreement_counts = defaultdict(lambda: defaultdict(lambda: {"agree": 0, "total": 0}))

    pairs_unanimous = 0
    pairs_disagreement = 0
    pair_agreement_rates = []

    for pair_key, model_comps in by_pair_model.items():
        # Get majority vote per model for this pair
        model_majority = {}
        for model, comps in model_comps.items():
            valid_comps = [c for c in comps if not c.error and c.preferred]
            if not valid_comps:
                continue
            # Count votes
            a_votes = sum(1 for c in valid_comps if c.preferred == "A")
            b_votes = len(valid_comps) - a_votes
            # First unit in pair_key wins if majority voted for it
            first_id = pair_key[0]
            first_wins = sum(1 for c in valid_comps if c.winner_id == first_id)
            model_majority[model] = "first" if first_wins > len(valid_comps) / 2 else "second"

        if len(model_majority) < 2:
            continue

        # Check if all models agree
        votes = list(model_majority.values())
        if len(set(votes)) == 1:
            pairs_unanimous += 1
            pair_agreement_rates.append((pair_key, 1.0))
        else:
            pairs_disagreement += 1
            # Calculate agreement rate for this pair
            agree_count = max(votes.count("first"), votes.count("second"))
            pair_agreement_rates.append((pair_key, agree_count / len(votes)))

        # Update pairwise agreement matrix
        for m1, m2 in combinations(model_majority.keys(), 2):
            if model_majority[m1] == model_majority[m2]:
                agreement_counts[m1][m2]["agree"] += 1
                agreement_counts[m2][m1]["agree"] += 1
            agreement_counts[m1][m2]["total"] += 1
            agreement_counts[m2][m1]["total"] += 1

    diag.pairs_with_unanimous_agreement = pairs_unanimous
    diag.pairs_with_disagreement = pairs_disagreement

    # All pairs with agreement rates (sorted lowest to highest)
    diag.most_contested_pairs = sorted(pair_agreement_rates, key=lambda x: x[1])

    # Build agreement matrix
    for m1 in models:
        diag.model_agreement_matrix[m1] = {}
        for m2 in models:
            if m1 == m2:
                diag.model_agreement_matrix[m1][m2] = 1.0
            elif agreement_counts[m1][m2]["total"] > 0:
                diag.model_agreement_matrix[m1][m2] = (
                    agreement_counts[m1][m2]["agree"] / agreement_counts[m1][m2]["total"]
                )
            else:
                diag.model_agreement_matrix[m1][m2] = None

    # Overall cross-model agreement
    total_agree = sum(
        agreement_counts[m1][m2]["agree"]
        for m1 in models for m2 in models if m1 < m2
    )
    total_pairs = sum(
        agreement_counts[m1][m2]["total"]
        for m1 in models for m2 in models if m1 < m2
    )
    diag.cross_model_agreement = total_agree / total_pairs if total_pairs else 1.0


def _compute_transitivity(diag: RankingDiagnostics, by_pair: dict):
    """Check for transitivity violations: A>B, B>C but C>A."""
    # Build preference graph: edges point from loser to winner
    # edge (A, B) with weight w means A beat B w times net
    net_wins = {}  # (winner, loser) -> net win count

    for pair_key, comps in by_pair.items():
        first_id, second_id = pair_key
        valid_comps = [c for c in comps if not c.error and c.preferred]

        first_wins = sum(1 for c in valid_comps if c.winner_id == first_id)
        second_wins = len(valid_comps) - first_wins

        if first_wins > second_wins:
            net_wins[(first_id, second_id)] = first_wins - second_wins
        elif second_wins > first_wins:
            net_wins[(second_id, first_id)] = second_wins - first_wins

    # Find all units
    units = set()
    for (a, b) in net_wins:
        units.add(a)
        units.add(b)
    units = list(units)

    # Check all triplets for transitivity violations
    violations = []
    checked_triplets = 0

    for a, b, c in combinations(units, 3):
        # Check all orderings
        for x, y, z in [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]:
            # Check if x > y > z but z > x
            x_beats_y = (x, y) in net_wins
            y_beats_z = (y, z) in net_wins
            z_beats_x = (z, x) in net_wins

            if x_beats_y and y_beats_z and z_beats_x:
                # Normalize to avoid duplicates
                violation = tuple(sorted([x, y, z]))
                if violation not in [tuple(sorted(v)) for v in violations]:
                    violations.append((x, y, z))

        checked_triplets += 1

    diag.transitivity_violations = violations[:20]  # Limit to 20

    # Violation rate = violations / possible triplets
    n_triplets = len(list(combinations(units, 3))) if len(units) >= 3 else 0
    diag.transitivity_violation_rate = len(violations) / n_triplets if n_triplets else 0.0


def format_diagnostics(diag: RankingDiagnostics, models: list[str] | None = None) -> str:
    """Format diagnostics as human-readable string for logging."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  LLM JUDGE RANKING DIAGNOSTICS")
    lines.append("=" * 70)
    lines.append("")

    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    error_pct = f"({diag.error_rate:.1%} failed)" if diag.error_rate > 0 else ""
    lines.append(f"  Comparisons: {diag.successful_comparisons}/{diag.total_comparisons} successful {error_pct}")
    lines.append("")

    # Swap testing - most important, show first
    if diag.swap_test_total > 0:
        lines.append("SWAP TESTING")
        lines.append("-" * 40)
        lines.append("  Each pair is shown in both orders (A,B) and (B,A).")
        lines.append("  If the judge picks inconsistently, the vote is discarded.")
        lines.append("")
        kept = diag.swap_test_total - diag.swap_test_discarded
        lines.append(f"  Vote pairs: {diag.swap_test_total} tested, {kept} kept, {diag.swap_test_discarded} discarded")
        lines.append(f"  Discard rate: {diag.swap_test_discard_rate:.1%}")
        if diag.swap_test_discard_rate > 0.3:
            lines.append("  WARNING: High discard rate indicates unreliable judges")
        lines.append("")
        lines.append("  By model:")
        for model, rate in sorted(diag.model_swap_discard_rate.items(), key=lambda x: x[1], reverse=True):
            ms = diag.model_stats.get(model)
            discarded = ms.swap_discarded if ms else 0
            short_name = model.split("/")[-1][:30]
            indicator = " [unreliable]" if rate > 0.3 else ""
            lines.append(f"    {short_name:<30}: {rate:>5.1%} discarded ({discarded} pairs){indicator}")
        lines.append("")

    # Self-consistency
    has_repeated = any(ms.total_repeated_pairs > 0 for ms in diag.model_stats.values())
    if has_repeated:
        lines.append("SELF-CONSISTENCY")
        lines.append("-" * 40)
        lines.append("  When a model votes on the same pair multiple times,")
        lines.append("  does it give the same answer? (100% = perfectly consistent)")
        lines.append("")
        lines.append(f"  Overall: {diag.overall_self_consistency:.1%}")
        for model, rate in sorted(diag.model_self_consistency.items(), key=lambda x: x[1]):
            ms = diag.model_stats.get(model)
            n = ms.total_repeated_pairs if ms else 0
            if n > 0:
                short_name = model.split("/")[-1][:30]
                lines.append(f"    {short_name:<30}: {rate:>5.1%} ({n} pairs with repeated votes)")
        lines.append("")

    # Cross-model agreement
    if len(diag.model_stats) > 1:
        lines.append("CROSS-MODEL AGREEMENT")
        lines.append("-" * 40)
        lines.append("  Do different LLM judges agree on which option is better?")
        lines.append("")
        lines.append(f"  Overall agreement: {diag.cross_model_agreement:.1%}")
        lines.append(f"  Unanimous pairs: {diag.pairs_with_unanimous_agreement}")
        lines.append(f"  Contested pairs: {diag.pairs_with_disagreement}")
        if diag.most_contested_pairs:
            lines.append("")
            lines.append("  Most contested (models disagreed):")
            for pair, rate in diag.most_contested_pairs[:3]:
                lines.append(f"    {pair[0]} vs {pair[1]}: {rate:.0%} agreement")
        lines.append("")

        # Agreement matrix
        actual_models = list(diag.model_agreement_matrix.keys()) if diag.model_agreement_matrix else []
        if len(actual_models) > 1:
            lines.append("  Model agreement matrix:")
            short_models = [m.split("/")[-1][:12] for m in actual_models]
            header = "  " + " " * 14 + "".join(f"{sm:<14}" for sm in short_models)
            lines.append(header)
            for m1 in actual_models:
                sm1 = m1.split("/")[-1][:12]
                row = f"  {sm1:<14}"
                for m2 in actual_models:
                    val = diag.model_agreement_matrix.get(m1, {}).get(m2)
                    if val is None:
                        row += f"{'--':<14}"
                    elif m1 == m2:
                        row += f"{'---':<14}"
                    else:
                        row += f"{val:.0%}".ljust(14)
                lines.append(row)
            lines.append("")

    # Position bias (only if no swap testing, since swap testing handles this)
    if diag.swap_test_total == 0:
        lines.append("POSITION BIAS")
        lines.append("-" * 40)
        lines.append("  Does the judge favor the first (A) or second (B) option?")
        lines.append("  50% = no bias, >60% or <40% = biased")
        lines.append("")
        lines.append(f"  Overall A-preference: {diag.overall_position_a_rate:.1%}")
        for model, rate in sorted(diag.model_position_bias.items(), key=lambda x: abs(x[1] - 0.5), reverse=True):
            bias_indicator = ""
            if rate > 0.6:
                bias_indicator = " [A-biased]"
            elif rate < 0.4:
                bias_indicator = " [B-biased]"
            short_name = model.split("/")[-1][:30]
            lines.append(f"    {short_name:<30}: {rate:>5.1%}{bias_indicator}")
        lines.append("")

    # Transitivity
    lines.append("TRANSITIVITY")
    lines.append("-" * 40)
    lines.append("  If A>B and B>C, is A>C? Violations suggest noisy judgments.")
    lines.append("")
    lines.append(f"  Violation rate: {diag.transitivity_violation_rate:.1%}")
    if diag.transitivity_violations:
        lines.append(f"  Violations found: {len(diag.transitivity_violations)}")
        lines.append("  Examples (circular preferences):")
        for v in diag.transitivity_violations[:3]:
            lines.append(f"    {v[0]} > {v[1]} > {v[2]} > {v[0]}")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


__all__ = [
    "RankingDiagnostics",
    "PairStats",
    "ModelStats",
    "compute_diagnostics",
    "format_diagnostics",
]
