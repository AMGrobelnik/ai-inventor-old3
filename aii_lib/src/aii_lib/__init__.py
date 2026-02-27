"""
aii_lib - GenAI toolkit for ai-inventor pipeline.

Structure:
├── kit.py              - GenAIKit (main API)
├── types.py            - Core types (GenAIRun, TokenUsage)
├── telemetry/          - Unified logging system
├── pools/              - BasePool, pool formatting, ID utilities
├── utils/              - General utilities (ForkingServer, call_server, server_available)
├── workflows/          - Multi-step orchestrations (EloRanker)
├── llm_backend/        - LLM clients (OpenAI, Anthropic, Gemini, OpenRouter)
├── agent_backend/      - Claude Agent SDK wrapper
└── tools/              - Custom ToolUniverse tools

Lazy imports:
- Subpackages (utils, pools, telemetry, etc.) can be imported directly without
  triggering the full import chain: `from aii_lib.utils import call_server`
- Top-level imports are loaded lazily on first access
"""

import importlib
import sys
from typing import TYPE_CHECKING

# These lightweight modules are always available (no heavy deps)
from .types import GenAIRun, TokenUsage

# For type checking, import everything statically
if TYPE_CHECKING:
    from .kit import GenAIKit
    from .telemetry import (
        AIITelemetry,
        create_telemetry,
        ConsoleSink,
        JSONSink,
        MessageType,
        TelemetryMessage,
        Colors,
        logger,
    )
    from .pools import (
        BasePool,
        TYPE_ABBREVS,
        get_type_abbrev,
        parse_iteration,
    )
    from .utils import (
        call_server,
        server_available,
        cleanup_run_caches,
        get_model_short,
        LLMPromptModel,
        LLMStructOutModel,
        ClaudeAgentToLLMStructOut,
        ClaudeAgentToLLMStructOutResult,
        get_tooluniverse_mcp_config,
    )
    from .workflows import (
        KRandomRanker,
        KRandomRankResult,
        SwissBTRanker,
        SwissBTRankResult,
        RoundStats,
        RankedUnit,
        ComparisonResult,
        ModelConfig,
        INITIAL_ELO,
        RATING_SENSITIVITY,
        BT_INITIAL_STRENGTH,
        BT_CONVERGENCE_THRESHOLD,
        RankingDiagnostics,
        PairStats,
        ModelStats,
        compute_diagnostics,
        format_diagnostics,
        research_workflow,
        ResearchWorkflowConfig,
        ResearchWorkflowResult,
        RESEARCH_TOOLS,
        CitedArgsConfig,
        CitedArgsResult,
        ClaudeAgentConfig,
        generate_cited_argument,
        cap_results,
        collect_verified_arguments,
        extract_citations_with_url,
        extract_quotes_only,
        FetchResult,
        fetch_url_content,
        find_exact_match,
        extract_words,
        GenKGConfig,
        GenKGResult,
        generate_kg_triples,
        verify_wikipedia_urls,
    )
    from .abilities.tools.utils import (
        get_openrouter_tools,
        execute_tool_calls,
        WebCache,
    )
    from .llm_backend import (
        OpenAIClient,
        AnthropicClient,
        GeminiClient,
        OpenRouterClient,
        ConversationStats,
        chat,
        ToolLoopResult,
    )
    from .agent_backend import (
        Agent,
        SequentialAgent,
        AgentOptions,
        ExpectedFile,
        AgentResponse,
        SessionType,
    )
    from .agent_backend.utils import (
        AgentInitializer,
        AgentFinalizer,
    )


# Lazy import mapping: attribute name -> (module, attribute)
_LAZY_IMPORTS = {
    # kit
    "GenAIKit": (".kit", "GenAIKit"),
    # telemetry
    "AIITelemetry": (".telemetry", "AIITelemetry"),
    "create_telemetry": (".telemetry", "create_telemetry"),
    "ConsoleSink": (".telemetry", "ConsoleSink"),
    "JSONSink": (".telemetry", "JSONSink"),
    "MessageType": (".telemetry", "MessageType"),
    "TelemetryMessage": (".telemetry", "TelemetryMessage"),
    "Colors": (".telemetry", "Colors"),
    "logger": (".telemetry", "logger"),
    # pools - BasePool
    "BasePool": (".pools", "BasePool"),
    "TYPE_ABBREVS": (".pools", "TYPE_ABBREVS"),
    "get_type_abbrev": (".pools", "get_type_abbrev"),
    "parse_iteration": (".pools", "parse_iteration"),
    # utils
    "call_server": (".utils", "call_server"),
    "server_available": (".utils", "server_available"),
    "cleanup_run_caches": (".utils", "cleanup_run_caches"),
    "get_model_short": (".utils", "get_model_short"),
    "LLMPromptModel": (".prompts", "LLMPromptModel"),
    "LLMStructOutModel": (".prompts", "LLMStructOutModel"),
    "ClaudeAgentToLLMStructOut": (".utils", "ClaudeAgentToLLMStructOut"),
    "ClaudeAgentToLLMStructOutResult": (".utils", "ClaudeAgentToLLMStructOutResult"),
    "get_tooluniverse_mcp_config": (".utils", "get_tooluniverse_mcp_config"),
    # workflows - K-Random Ranker
    "KRandomRanker": (".workflows", "KRandomRanker"),
    "KRandomRankResult": (".workflows", "KRandomRankResult"),
    "RankedUnit": (".workflows", "RankedUnit"),
    "ComparisonResult": (".workflows", "ComparisonResult"),
    "ModelConfig": (".workflows", "ModelConfig"),
    "INITIAL_ELO": (".workflows", "INITIAL_ELO"),
    "RATING_SENSITIVITY": (".workflows", "RATING_SENSITIVITY"),
    # workflows - Swiss-BT Ranker
    "SwissBTRanker": (".workflows", "SwissBTRanker"),
    "SwissBTRankResult": (".workflows", "SwissBTRankResult"),
    "RoundStats": (".workflows", "RoundStats"),
    "BT_INITIAL_STRENGTH": (".workflows", "BT_INITIAL_STRENGTH"),
    "BT_CONVERGENCE_THRESHOLD": (".workflows", "BT_CONVERGENCE_THRESHOLD"),
    # workflows - Ranking Diagnostics
    "RankingDiagnostics": (".workflows", "RankingDiagnostics"),
    "PairStats": (".workflows", "PairStats"),
    "ModelStats": (".workflows", "ModelStats"),
    "compute_diagnostics": (".workflows", "compute_diagnostics"),
    "format_diagnostics": (".workflows", "format_diagnostics"),
    # workflows - Research OR
    "research_workflow": (".workflows", "research_workflow"),
    "ResearchWorkflowConfig": (".workflows", "ResearchWorkflowConfig"),
    "ResearchWorkflowResult": (".workflows", "ResearchWorkflowResult"),
    "RESEARCH_TOOLS": (".workflows", "RESEARCH_TOOLS"),
    # workflows - Cited Args
    "CitedArgsConfig": (".workflows", "CitedArgsConfig"),
    "CitedArgsResult": (".workflows", "CitedArgsResult"),
    "ClaudeAgentConfig": (".workflows", "ClaudeAgentConfig"),
    "generate_cited_argument": (".workflows", "generate_cited_argument"),
    "cap_results": (".workflows", "cap_results"),
    "collect_verified_arguments": (".workflows", "collect_verified_arguments"),
    "extract_citations_with_url": (".workflows", "extract_citations_with_url"),
    "extract_quotes_only": (".workflows", "extract_quotes_only"),
    "FetchResult": (".workflows", "FetchResult"),
    "fetch_url_content": (".workflows", "fetch_url_content"),
    "find_exact_match": (".workflows", "find_exact_match"),
    "extract_words": (".workflows", "extract_words"),
    # workflows - Gen KG
    "GenKGConfig": (".workflows", "GenKGConfig"),
    "GenKGResult": (".workflows", "GenKGResult"),
    "generate_kg_triples": (".workflows", "generate_kg_triples"),
    "verify_wikipedia_urls": (".workflows", "verify_wikipedia_urls"),
    # tools
    "get_openrouter_tools": (".abilities.tools.utils", "get_openrouter_tools"),
    "execute_tool_calls": (".abilities.tools.utils", "execute_tool_calls"),
    "WebCache": (".abilities.tools.utils", "WebCache"),
    # llm_backend
    "OpenAIClient": (".llm_backend", "OpenAIClient"),
    "AnthropicClient": (".llm_backend", "AnthropicClient"),
    "GeminiClient": (".llm_backend", "GeminiClient"),
    "OpenRouterClient": (".llm_backend", "OpenRouterClient"),
    "ConversationStats": (".llm_backend", "ConversationStats"),
    "chat": (".llm_backend", "chat"),
    "ToolLoopResult": (".llm_backend", "ToolLoopResult"),
    # agent_backend
    "Agent": (".agent_backend", "Agent"),
    "SequentialAgent": (".agent_backend", "SequentialAgent"),
    "AgentOptions": (".agent_backend", "AgentOptions"),
    "ExpectedFile": (".agent_backend", "ExpectedFile"),
    "AgentResponse": (".agent_backend", "AgentResponse"),
    "SessionType": (".agent_backend", "SessionType"),
    # agent utilities
    "AgentInitializer": (".agent_backend.utils", "AgentInitializer"),
    "AgentFinalizer": (".agent_backend.utils", "AgentFinalizer"),
}


def __getattr__(name: str):
    """Lazy import handler for top-level attributes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package="aii_lib")
        value = getattr(module, attr_name)
        # Cache the imported value
        globals()[name] = value
        return value
    raise AttributeError(f"module 'aii_lib' has no attribute {name!r}")


__all__ = [
    # Core types (always loaded)
    "GenAIRun",
    "TokenUsage",
    # Main API
    "GenAIKit",
    # AIITelemetry
    "AIITelemetry",
    "create_telemetry",
    "ConsoleSink",
    "JSONSink",
    "MessageType",
    "TelemetryMessage",
    "Colors",
    "logger",
    # Pools - BasePool
    "BasePool",
    "TYPE_ABBREVS",
    "get_type_abbrev",
    "parse_iteration",
    # Utils
    "call_server",
    "server_available",
    "cleanup_run_caches",
    "get_model_short",
    "LLMPromptModel",
    "LLMStructOutModel",
    "ClaudeAgentToLLMStructOut",
    "ClaudeAgentToLLMStructOutResult",
    "get_tooluniverse_mcp_config",
    # Workflows - K-Random Ranker (single-round, win rate based)
    "KRandomRanker",
    "KRandomRankResult",
    "RankedUnit",
    "ComparisonResult",
    "ModelConfig",
    "INITIAL_ELO",
    "RATING_SENSITIVITY",
    # Workflows - Swiss-BT Ranker (multi-round, Bradley-Terry scoring)
    "SwissBTRanker",
    "SwissBTRankResult",
    "RoundStats",
    "BT_INITIAL_STRENGTH",
    "BT_CONVERGENCE_THRESHOLD",
    # Workflows - Ranking Diagnostics
    "RankingDiagnostics",
    "PairStats",
    "ModelStats",
    "compute_diagnostics",
    "format_diagnostics",
    # Workflows - Research OR (OpenRouter + tools)
    "research_workflow",
    "ResearchWorkflowConfig",
    "ResearchWorkflowResult",
    "RESEARCH_TOOLS",
    # Workflows - Cited Args OR Workflow
    "CitedArgsConfig",
    "CitedArgsResult",
    "ClaudeAgentConfig",
    "generate_cited_argument",
    "cap_results",
    "collect_verified_arguments",
    # Workflows - Cited Args OR Citation extraction
    "extract_citations_with_url",
    "extract_quotes_only",
    # Workflows - Cited Args OR URL fetching
    "FetchResult",
    "fetch_url_content",
    # Workflows - Cited Args OR Quote matching
    "find_exact_match",
    "extract_words",
    # Workflows - Gen KG (Knowledge Graph Generation)
    "GenKGConfig",
    "GenKGResult",
    "generate_kg_triples",
    "verify_wikipedia_urls",
    # Tool utilities
    "get_openrouter_tools",
    "execute_tool_calls",
    "WebCache",
    # LLM Clients
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "OpenRouterClient",
    "ConversationStats",
    "chat",
    "ToolLoopResult",
    # Agent
    "Agent",
    "SequentialAgent",
    "AgentOptions",
    "ExpectedFile",
    "AgentResponse",
    "SessionType",
    # Agent utilities
    "AgentInitializer",
    "AgentFinalizer",
]
