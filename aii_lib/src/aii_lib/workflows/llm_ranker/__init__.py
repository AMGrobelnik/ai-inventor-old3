"""
LLM Ranker - Pairwise comparison ranking using LLM judges.

Swiss-style tournament with Bradley-Terry scoring for efficient ranking.
"""

from .k_random_ranker import (
    KRandomRanker,
    KRandomRankResult,
    RankedUnit,
    ComparisonResult,
    ModelConfig,
    INITIAL_ELO,
    RATING_SENSITIVITY,
)
from .swiss_bt_ranker import (
    SwissBTRanker,
    SwissBTRankResult,
    RoundStats,
    BT_INITIAL_STRENGTH,
    BT_CONVERGENCE_THRESHOLD,
)
from .ranking_diagnostics import (
    RankingDiagnostics,
    PairStats,
    ModelStats,
    compute_diagnostics,
    format_diagnostics,
)

__all__ = [
    # K-Random Ranker (single-round, win rate based)
    "KRandomRanker",
    "KRandomRankResult",
    "RankedUnit",
    "ComparisonResult",
    "ModelConfig",
    "INITIAL_ELO",
    "RATING_SENSITIVITY",
    # Swiss-BT Ranker (multi-round, Bradley-Terry scoring)
    "SwissBTRanker",
    "SwissBTRankResult",
    "RoundStats",
    "BT_INITIAL_STRENGTH",
    "BT_CONVERGENCE_THRESHOLD",
    # Ranking Diagnostics
    "RankingDiagnostics",
    "PairStats",
    "ModelStats",
    "compute_diagnostics",
    "format_diagnostics",
]
