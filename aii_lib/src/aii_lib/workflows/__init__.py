"""
Workflows - Multi-step orchestrations using base operations.

Workflows combine base operations (chat, chat_tools, agent) with
telemetry to implement higher-level patterns.
"""

from .llm_ranker import (
    # K-Random Ranker
    KRandomRanker,
    KRandomRankResult,
    RankedUnit,
    ComparisonResult,
    ModelConfig,
    INITIAL_ELO,
    RATING_SENSITIVITY,
    # Swiss-BT Ranker
    SwissBTRanker,
    SwissBTRankResult,
    RoundStats,
    BT_INITIAL_STRENGTH,
    BT_CONVERGENCE_THRESHOLD,
    # Diagnostics
    RankingDiagnostics,
    PairStats,
    ModelStats,
    compute_diagnostics,
    format_diagnostics,
)
from .research_workflow import (
    research_workflow,
    ResearchWorkflowConfig,
    ResearchWorkflowResult,
    RESEARCH_TOOLS,
)
from .cited_args import (
    # Workflow
    CitedArgsConfig,
    CitedArgsResult,
    ClaudeAgentConfig,
    generate_cited_argument,
    cap_results,
    collect_verified_arguments,
    # Citation extraction
    extract_citations_with_url,
    extract_quotes_only,
    # URL fetching
    FetchResult,
    fetch_url_content,
    # Quote matching
    find_exact_match,
    extract_words,
)
from .gen_kg import (
    GenKGConfig,
    GenKGResult,
    generate_kg_triples,
    verify_wikipedia_urls,
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
    # Research Workflow (agnostic)
    "research_workflow",
    "ResearchWorkflowConfig",
    "ResearchWorkflowResult",
    "RESEARCH_TOOLS",
    # Cited Args - Workflow
    "CitedArgsConfig",
    "CitedArgsResult",
    "ClaudeAgentConfig",
    "generate_cited_argument",
    "cap_results",
    "collect_verified_arguments",
    # Cited Args - Citation extraction
    "extract_citations_with_url",
    "extract_quotes_only",
    # Cited Args - URL fetching
    "FetchResult",
    "fetch_url_content",
    # Cited Args - Quote matching
    "find_exact_match",
    "extract_words",
    # Gen KG - Knowledge Graph Generation
    "GenKGConfig",
    "GenKGResult",
    "generate_kg_triples",
    "verify_wikipedia_urls",
]
