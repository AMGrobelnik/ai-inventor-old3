"""
Typed pipeline configuration using Pydantic.

Usage:
    from aii_pipeline.utils import PipelineConfig, rel_path

    # Load from YAML file
    config = PipelineConfig.from_yaml("config.yaml")

    # Access with typed attributes (no .get() chains!)
    api_key = config.api_keys.openrouter
    model = config.gen_hypo.llm_client.model
    timeout = config.gen_hypo.llm_client.llm_timeout

    # Still works with raw dict access if needed
    config.raw["custom_key"]

    # Use rel_path for logging
    rel_path("/home/user/projects/ai-inventor/runs/foo")  # -> "runs/foo"
"""

import os
import yaml
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Path Utilities
# =============================================================================

def get_project_root() -> Path:
    """Get the ai-inventor project root directory."""
    # From aii_pipeline/src/aii_pipeline/utils/pipeline_config.py -> ai-inventor
    return Path(__file__).parent.parent.parent.parent.parent


def rel_path(path) -> str:
    """Convert path to be relative to ai-inventor directory for logging."""
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.relative_to(get_project_root()))
    except ValueError:
        # If path is not relative to project root, return as-is
        return str(path)


# =============================================================================
# Shared LLM Config Models
# =============================================================================

class ModelEntry(BaseModel):
    """Single model entry in a multi-model config."""
    model: str
    reasoning_effort: str = "medium"
    suffix: str = ""

    model_config = ConfigDict(extra="allow")


class MultiModelLLMClientConfig(BaseModel):
    """LLM client config with multiple models."""
    client: str = "openrouter"
    llm_timeout: int = 600
    suffix: str = ""
    models: list[ModelEntry] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class GenHypoModelEntry(ModelEntry):
    """Model entry for gen_hypo with research-specific settings."""
    max_tool_iterations: int = 100  # Per-model tool iteration limit


class GenHypoLLMClientConfig(BaseModel):
    """LLM client config for gen_hypo with multiple models."""
    client: str = "openrouter"
    llm_timeout: int = 1200
    models: list[GenHypoModelEntry] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class NoveltyModelEntry(ModelEntry):
    """Model entry for novelty audits with web search settings."""
    max_tool_iterations: int = 20  # Per-model tool iteration limit for web search


class NoveltyLLMClientConfig(BaseModel):
    """LLM client config for novelty audits with multiple models and web search."""
    client: str = "openrouter"
    llm_timeout: int = 300
    suffix: str = ""
    models: list[NoveltyModelEntry] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


# =============================================================================
# API Keys
# =============================================================================

class APIKeysConfig(BaseModel):
    """API keys for all providers. Reads from os.environ (populated by .env via load_dotenv)."""
    openai: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openrouter: str = Field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    anthropic: str = Field(default_factory=lambda: os.environ.get("UNUSED_ANTHROPIC_API_KEY", ""))
    gemini: str = Field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    serper: str = Field(default_factory=lambda: os.environ.get("SERPER_API_KEY", ""))
    leanexplore: str = Field(default_factory=lambda: os.environ.get("LEANEXPLORE_API_KEY", ""))
    huggingface: str = Field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))

    model_config = ConfigDict(extra="allow")


# =============================================================================
# AIITelemetry
# =============================================================================

class TelemetryConfig(BaseModel):
    """AIITelemetry settings."""
    console_msg_truncate: int | None = 5000
    log_messages: bool = True

    model_config = ConfigDict(extra="allow")


# =============================================================================
# Tools
# =============================================================================

class ToolConfig(BaseModel):
    """Single tool configuration."""
    max_concurrent: int = 100
    cache_ttl_hours: float = 10.0

    model_config = ConfigDict(extra="allow")


class ToolsConfig(BaseModel):
    """All tool configurations."""
    aii_web_search_fast: ToolConfig = Field(default_factory=ToolConfig)
    aii_web_fetch_direct: ToolConfig = Field(default_factory=ToolConfig)
    aii_web_fetch_grep: ToolConfig = Field(default_factory=ToolConfig)

    model_config = ConfigDict(extra="allow")


# =============================================================================
# Init / Pipeline
# =============================================================================

class PipelineControlConfig(BaseModel):
    """Pipeline execution control."""
    first_step: str = "seed_hypo"
    last_step: str = "gen_paper_repo"
    # Checkpoint directories for resuming from specific modules
    seed_hypo_out_dir: str | None = None
    gen_hypo_out_dir: str | None = None
    audit_hypo_out_dir: str | None = None
    invention_loop_out_dir: str | None = None
    gen_paper_repo_out_dir: str | None = None
    # Invention loop iteration control (only applies when running invention_loop module)
    invention_loop_resume_dir: str | None = None  # Resume from this previous run's 3_invention_loop dir (loads pools)
    invention_loop_first_step: str | None = None  # "auto" (detect from resume), or step name (gen_strat, gen_plan, gen_art, gen_narr)
    invention_loop_last_step: str | None = None  # Stop after this step (if before first_step, completes iteration then runs next up to this step)
    # Gen paper repo step control (sequential with concurrent sub-steps)
    # Steps: 1:create_repo → 2a:write_paper+2b:gen_demos → 3:gen_viz → 4:gen_full_paper → 5:deploy_repo
    gen_paper_resume_dir: str | None = None  # Resume from this output dir (loads completed step results)
    gen_paper_first_step: str | None = None  # Start from this step (skip earlier steps, load from resume_dir)
    gen_paper_last_step: str | None = None  # Stop after this step
    gen_paper_run_gen_full_paper: bool = True  # Run final gen_full_paper step


    model_config = ConfigDict(extra="allow")


class InitConfig(BaseModel):
    """Initialization settings."""
    research_direction: str
    run_name: str = ""
    outputs_directory: str = "runs"
    pipeline: PipelineControlConfig = Field(default_factory=PipelineControlConfig)

    model_config = ConfigDict(extra="allow")


# =============================================================================
# Seed Hypo (Hypothesis Seed Generation)
# =============================================================================

class InventionKGSelTopicsConfig(BaseModel):
    """Topic selection config for invention_kg."""
    topics: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class InventionKGGetPapersConfig(BaseModel):
    """Paper fetching config for invention_kg."""
    email: str = ""
    papers_per_year: int = 10
    year_range: dict = Field(default_factory=lambda: {"start": 2020, "end": 2025})
    sort_by: str = "cited_by_count"

    model_config = ConfigDict(extra="allow")


class InventionKGGetTriplesClaudeAgentConfig(BaseModel):
    """Claude agent config for get_triples step."""
    model: str = "claude-haiku-4-5"  # claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-6
    max_turns: int = 100
    agent_timeout: int | None = None  # Total timeout for entire agent (None = no limit)
    agent_retries: int = 2  # Retries for entire agent session
    seq_prompt_timeout: int = 600  # 10 min timeout per prompt
    seq_prompt_retries: int = 5
    message_timeout: int | None = None  # Per-message timeout in seconds (None = no timeout)
    message_retries: int = 3  # Max fork+resume attempts for message-level timeouts

    model_config = ConfigDict(extra="allow")


class InventionKGGetTriplesConfig(BaseModel):
    """Triple extraction config for invention_kg."""
    max_papers: int = -1  # -1 = all
    max_concurrent_agents: int = 10  # Max concurrent agent processes
    stagger_delay: float = 2.0
    url_verification_retries: int = 2  # Retries for Wikipedia URL verification
    min_valid_urls: int = 0  # Min valid URLs before restructuring vs searching again
    claude_agent: InventionKGGetTriplesClaudeAgentConfig = Field(
        default_factory=InventionKGGetTriplesClaudeAgentConfig
    )

    model_config = ConfigDict(extra="allow")


class InventionKGBlindSpotsConfig(BaseModel):
    """Blind spots config for gen_hypo_seeds."""
    min_shared_concepts: int = 1
    max_similarity: float = 1.0
    entity_types: list[str] = Field(default_factory=lambda: ["method", "concept"])

    model_config = ConfigDict(extra="allow")


class InventionKGGenHypoSeedsConfig(BaseModel):
    """Hypothesis seed generation config for invention_kg."""
    blind_spots: InventionKGBlindSpotsConfig = Field(default_factory=InventionKGBlindSpotsConfig)

    model_config = ConfigDict(extra="allow")


class InventionKGGenGraphConfig(BaseModel):
    """Graph generation config for invention_kg."""
    temporal_windows: list[list[int]] = Field(default_factory=lambda: [[2018, 2020], [2021, 2023], [2024, 2025]])

    model_config = ConfigDict(extra="allow")


class InventionKGVizGraphConfig(BaseModel):
    """Graph visualization config for invention_kg."""
    port: int = 9020

    model_config = ConfigDict(extra="allow")


class InventionKGConfig(BaseModel):
    """Invention KG pipeline configuration.

    Runs kg pipeline from first_step to last_step.
    Set *_out_dir for the step BEFORE first_step to resume from that output.
    """
    first_step: str = "sel_topics"
    last_step: str = "gen_hypo_prompt"

    # Output dirs for resuming from any step (set the one before first_step)
    sel_topics_out_dir: str = ""
    get_papers_out_dir: str = ""
    clean_papers_out_dir: str = ""
    get_triples_out_dir: str = ""
    add_wikidata_out_dir: str = ""
    link_to_papers_out_dir: str = ""
    gen_hypo_seeds_out_dir: str = ""
    gen_hypo_prompt_out_dir: str = ""
    gen_graphs_out_dir: str = ""

    sel_topics: InventionKGSelTopicsConfig = Field(default_factory=InventionKGSelTopicsConfig)
    get_papers: InventionKGGetPapersConfig = Field(default_factory=InventionKGGetPapersConfig)
    get_triples: InventionKGGetTriplesConfig = Field(default_factory=InventionKGGetTriplesConfig)
    gen_hypo_seeds: InventionKGGenHypoSeedsConfig = Field(default_factory=InventionKGGenHypoSeedsConfig)
    gen_graph: InventionKGGenGraphConfig = Field(default_factory=InventionKGGenGraphConfig)
    viz_graph: InventionKGVizGraphConfig = Field(default_factory=InventionKGVizGraphConfig)

    model_config = ConfigDict(extra="allow")


class SeedHypoSamplingConfig(BaseModel):
    """Sampling configuration for hypothesis seeds."""
    sel_topics: str | list[str] = "auto"
    research_dir_topic_match_k: int = 4
    seed_sampling_pool: int = 20
    topics_per_agent: int = 2
    seeds_per_topic: int = 1

    model_config = ConfigDict(extra="allow")


class SeedHypoConfig(BaseModel):
    """Seed hypo module configuration.

    Steps: gen_seeds → sample_seeds
    Currently supports invention_kg for gen_seeds. Future methods can be added.

    Set *_out_dir for the step BEFORE first_step to resume from that point.
    """
    first_step: str = "gen_seeds"  # gen_seeds | sample_seeds
    last_step: str = "sample_seeds"  # gen_seeds | sample_seeds

    # Output dirs for resuming
    invention_kg_seed_out_dir: str = ""  # Resume from gen_seeds output (for sample_seeds)
    sample_seeds_out_dir: str = ""  # Resume from sample_seeds output (skip both steps)

    invention_kg: InventionKGConfig = Field(default_factory=InventionKGConfig)
    sampling: SeedHypoSamplingConfig = Field(default_factory=SeedHypoSamplingConfig)

    model_config = ConfigDict(extra="allow")


# =============================================================================
# Gen Hypo
# =============================================================================

class ClaudeAgentConfig(BaseModel):
    """Claude agent configuration for structured output."""
    model: str = "claude-sonnet-4-5"  # claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-6
    max_turns: int = 100
    agent_timeout: int | None = None  # Timeout for entire agent run in seconds (None = no timeout)
    agent_retries: int = 2  # Max retry attempts for entire agent on failure/timeout
    seq_prompt_timeout: int | None = None  # Timeout per prompt in seconds (None = no timeout)
    seq_prompt_retries: int = 5  # Max retry attempts per prompt on failure/timeout
    message_timeout: int | None = None  # Per-message timeout in seconds (None = no timeout)
    message_retries: int = 3  # Max fork+resume attempts for message-level timeouts
    max_concurrent_agents: int = 20  # Max parallel agent executions
    use_aii_web_tools: bool = False  # True: MCP aii_web_search_fast/aii_web_fetch_direct; False: built-in WebSearch/WebFetch

    model_config = ConfigDict(extra="allow")


class ClaudeAgentConfigNoConc(BaseModel):
    """Claude agent configuration without concurrency (used inside dimension configs)."""
    model: str = "claude-sonnet-4-5"  # claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-6
    max_turns: int = 100
    agent_timeout: int | None = None  # Timeout for entire agent run in seconds (None = no timeout)
    agent_retries: int = 2  # Max retry attempts for entire agent on failure/timeout
    seq_prompt_timeout: int | None = None  # Timeout per prompt in seconds (None = no timeout)
    seq_prompt_retries: int = 5  # Max retry attempts per prompt on failure/timeout
    message_timeout: int | None = None  # Per-message timeout in seconds (None = no timeout)
    message_retries: int = 3  # Max fork+resume attempts for message-level timeouts
    use_aii_web_tools: bool = False  # True: MCP aii_web_search_fast/aii_web_fetch_direct; False: built-in WebSearch/WebFetch

    model_config = ConfigDict(extra="allow")


class GenHypoConfig(BaseModel):
    """Hypothesis generation module configuration."""
    seeded_hypos_per_llm: int = 1  # Seeded hypotheses per model (with inspiration seeds)
    unseeded_hypos_per_llm: int = 1  # Unseeded hypotheses per model (no seeds)
    research_grounding: bool = True
    max_parallel: int | None = None  # None = unlimited concurrency
    llm_client: GenHypoLLMClientConfig = Field(default_factory=GenHypoLLMClientConfig)

    # Claude agent mode (experimental)
    use_claude_agent: bool = False
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)

    model_config = ConfigDict(extra="allow")

    @property
    def total_hypos_per_llm(self) -> int:
        """Total hypotheses per LLM (seeded + unseeded)."""
        return self.seeded_hypos_per_llm + self.unseeded_hypos_per_llm


# =============================================================================
# Audit Hypo
# =============================================================================

class VerifyCitationsConfig(BaseModel):
    """Citation verification settings."""
    parallel_fetch: bool = False
    retry: int = 2
    min_valid_citations: int = 5  # Minimum valid citations to restructure (vs search again)

    model_config = ConfigDict(extra="allow")


class NoveltyConfig(BaseModel):
    """Novelty audit configuration with multi-model support."""
    num_positive_per_llm: int = 1  # Positive args per LLM (total = this × num_models)
    num_negative_per_llm: int = 1  # Negative args per LLM (total = this × num_models)
    cap_mode: str = "equal"
    llm_client: NoveltyLLMClientConfig = Field(default_factory=NoveltyLLMClientConfig)

    # Claude agent mode (experimental)
    use_claude_agent: bool = False
    claude_agent: ClaudeAgentConfigNoConc = Field(default_factory=ClaudeAgentConfigNoConc)

    model_config = ConfigDict(extra="allow")


class FeasibilityConfig(BaseModel):
    """Feasibility audit configuration with multi-model support."""
    num_positive_per_llm: int = 1  # Positive args per LLM (total = this × num_models)
    num_negative_per_llm: int = 1  # Negative args per LLM (total = this × num_models)
    cap_mode: str = "equal"
    llm_client: MultiModelLLMClientConfig = Field(default_factory=MultiModelLLMClientConfig)

    # Claude agent mode (experimental)
    use_claude_agent: bool = False
    claude_agent: ClaudeAgentConfigNoConc = Field(default_factory=ClaudeAgentConfigNoConc)

    model_config = ConfigDict(extra="allow")


class AuditHypoConfig(BaseModel):
    """Audit hypothesis module configuration."""
    max_concurrent_audits: int = 200
    verify_citations: VerifyCitationsConfig = Field(default_factory=VerifyCitationsConfig)
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    feasibility: FeasibilityConfig = Field(default_factory=FeasibilityConfig)

    model_config = ConfigDict(extra="allow")


# =============================================================================
# invention loop
# =============================================================================

class VerifyArtifactsConfig(BaseModel):
    """Artifact verification settings for strategy generation.

    Verifies (always enforced):
    1. Dependencies reference only existing artifacts (not other artifacts in same strategy)
    2. Dependency types follow rules (e.g., experiments require datasets)
    3. Required number of strategies is generated
    4. Minimum number of valid artifacts across all strategies

    Note: Artifact IDs are assigned by code after LLM output, so ID uniqueness is guaranteed.
    """
    retry: int = 2  # Number of retry attempts if validation fails
    min_valid_artifacts: int = 1  # Minimum valid artifacts per LLM call (across strategies in that call)

    model_config = ConfigDict(extra="allow")


class GenStratConfig(BaseModel):
    """Strategy generation settings."""
    strats_per_call: int = 2  # Strategies generated per LLM call
    calls_per_llm: int = 1  # Parallel calls per model (total tasks = calls_per_llm × num_models)
    art_limit: int | None = None  # Max artifact directions per strategy (included in prompt + enforced with retry)
    max_concurrent: int | None = None  # Max parallel tasks (applies to both OpenRouter and Claude agent)
    artifact_context_per_type: int = 10  # Max artifacts shown per type in LLM prompt context
    use_claude_agent: bool = False  # Use Claude agent instead of OpenRouter
    llm_client: MultiModelLLMClientConfig = Field(default_factory=MultiModelLLMClientConfig)
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)
    verify_artifacts: VerifyArtifactsConfig = Field(default_factory=VerifyArtifactsConfig)

    model_config = ConfigDict(extra="allow")


class GenPlanConfig(BaseModel):
    """Plan generation settings.

    Gen_plan takes artifact_directions from ALL strategies and elaborates each into
    detailed plans. For each (artifact_direction, llm) combination, we generate
    `plans_per_strat` plans.

    Total plans = total_artifact_directions_across_strats × num_models × plans_per_strat
    """
    plans_per_strat: int = 1  # Plans to generate per (artifact_direction, llm) combination
    use_claude_agent: bool = False  # Use Claude agent instead of OpenRouter
    llm_client: MultiModelLLMClientConfig = Field(default_factory=MultiModelLLMClientConfig)
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)

    model_config = ConfigDict(extra="allow")


class ResearchExecuteConfig(BaseModel):
    """Research executor config (OpenRouter-based or Claude agent)."""
    use_claude_agent: bool = False  # Use Claude agent instead of OpenRouter
    model: str = "gpt-5-mini"
    reasoning_effort: str = "medium"
    suffix: str | None = None  # OpenRouter provider suffix (e.g., "nitro")
    max_tool_iterations: int = 5  # Max web search/fetch cycles before forcing output
    llm_timeout: int = 300  # Timeout for OpenRouter LLM calls
    claude_agent: ClaudeAgentConfigNoConc = Field(default_factory=ClaudeAgentConfigNoConc)
    # Post-execution validation retries
    verify_retries: int = 2         # Retries for missing expected files (max_expected_files_retries)
    schema_retries: int = 1         # Retries for schema validation errors (post-execution)

    model_config = ConfigDict(extra="allow")


class AgentExecuteConfig(BaseModel):
    """Agent executor config with nested claude_agent."""
    claude_agent: ClaudeAgentConfigNoConc = Field(default_factory=ClaudeAgentConfigNoConc)
    # Post-execution validation retries
    verify_retries: int = 2         # Retries for missing expected files (max_expected_files_retries)
    schema_retries: int = 1         # Retries for schema validation errors (post-execution)
    min_examples: int = 50          # Minimum examples in full_data_out.json / full_experiment_out.json
    # Proof-specific settings
    max_informal_loops: int = 3
    max_formal_loops: int = 5
    # Dataset-specific settings
    dataset_max_size: str = "300MB" # Max dataset size (shown in prompt, e.g. "300MB", "1GB")
    dataset_search_tool_cap: int = 50    # Max keyword searches in TODO 1
    dataset_chosen_for_preview_cap: int = 25   # Max datasets chosen for preview in TODO 2
    dataset_chosen_for_download_cap: int = 15  # Max datasets chosen for download in TODO 3
    dataset_chosen_final_cap: int = 10  # Max final datasets (caps plan's target_num_datasets)

    model_config = ConfigDict(extra="allow")


class ExecuteConfig(BaseModel):
    """Execute module configuration."""
    max_concurrent_artifacts: int = 5  # Max artifacts executing in parallel (semaphore)
    research: ResearchExecuteConfig = Field(default_factory=ResearchExecuteConfig)
    experiment: AgentExecuteConfig = Field(default_factory=AgentExecuteConfig)
    dataset: AgentExecuteConfig = Field(default_factory=AgentExecuteConfig)
    evaluation: AgentExecuteConfig = Field(default_factory=AgentExecuteConfig)
    proof: AgentExecuteConfig = Field(default_factory=AgentExecuteConfig)

    model_config = ConfigDict(extra="allow")


class NarrativeConfig(BaseModel):
    """Narrative generation settings."""
    start_at_iteration: int = 2
    narratives_per_round: int = 4
    use_claude_agent: bool = False  # Use Claude agent instead of OpenRouter
    llm_client: MultiModelLLMClientConfig = Field(default_factory=MultiModelLLMClientConfig)
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)

    model_config = ConfigDict(extra="allow")


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""
    patience: int = 2

    model_config = ConfigDict(extra="allow")


class InventionLoopConfig(BaseModel):
    """invention loop module configuration."""
    max_iterations: int = 3
    test_all_artifacts: bool = False  # Testing mode: propose creates 1 of each type
    allowed_artifacts: list[str] = Field(default_factory=list)
    # Strategy phase
    gen_strat: GenStratConfig = Field(default_factory=GenStratConfig)
    # Plan phase
    gen_plan: GenPlanConfig = Field(default_factory=GenPlanConfig)
    # Execution phase
    execute: ExecuteConfig = Field(default_factory=ExecuteConfig)
    # Narrative phase
    narrative: NarrativeConfig = Field(default_factory=NarrativeConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)

    model_config = ConfigDict(extra="allow")


# =============================================================================
# Gen Paper
# =============================================================================

class CreateRepoConfig(BaseModel):
    """Create repo step configuration."""
    enabled: bool = True

    model_config = ConfigDict(extra="allow")


class DeployGistsConfig(BaseModel):
    """Deploy gists step configuration."""
    enabled: bool = True

    model_config = ConfigDict(extra="allow")


class VerifyVizConfig(BaseModel):
    """Visualization output verification settings."""
    max_retries: int = 2  # Number of retry attempts if output file doesn't exist

    model_config = ConfigDict(extra="allow")


class FreeVizModelEntry(BaseModel):
    """Model entry for direct image generation."""
    model: str  # OpenRouter model ID
    llm_timeout: int = 120

    model_config = ConfigDict(extra="allow")


class FreeVizConfig(BaseModel):
    """Image generation model configuration with multi-model support."""
    client: str = "openrouter"
    max_concurrent: int = 10  # Max concurrent figure generations
    models: list[FreeVizModelEntry] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class VizGenConfig(BaseModel):
    """Visualization generation step configuration.

    Two backends:
      use_claude_agent=True  → Claude agent with aii_image_gen_nano_banana skill
      use_claude_agent=False → Direct Gemini image gen via OpenRouter (free_viz)
    One figure per placeholder, no variations, no ranking.
    """
    use_claude_agent: bool = False  # True: Claude agent + nano_banana skill; False: direct Gemini via OpenRouter
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)
    free_viz: FreeVizConfig = Field(default_factory=FreeVizConfig)
    verify_viz: VerifyVizConfig = Field(default_factory=VerifyVizConfig)

    model_config = ConfigDict(extra="allow")


class WritePaperTextConfig(BaseModel):
    """Write paper text step configuration with multi-model support."""
    variations: int = 1  # Number of paper drafts to generate
    use_claude_agent: bool = False  # Use Claude agent instead of OpenRouter
    verify_retries: int = 2  # Retries for missing expected files (max_expected_files_retries)
    llm_client: MultiModelLLMClientConfig = Field(default_factory=MultiModelLLMClientConfig)
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)

    model_config = ConfigDict(extra="allow")


class GenArtifactDemosConfig(BaseModel):
    """Generate artifact demos step configuration."""
    enabled: bool = True
    max_notebook_total_runtime: int = 600  # Max seconds for notebook execution via nbconvert
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)

    model_config = ConfigDict(extra="allow")


class GenFullPaperConfig(BaseModel):
    """Generate full paper step configuration."""
    claude_agent: ClaudeAgentConfigNoConc = Field(default_factory=lambda: ClaudeAgentConfigNoConc(
        model="claude-sonnet-4-5", max_turns=100, agent_timeout=900, seq_prompt_timeout=600
    ))

    model_config = ConfigDict(extra="allow")


class GenPaperConfig(BaseModel):
    """Gen paper repo module configuration.

    Steps:
        1:  create_repo
        2a: write_paper_text  ─┐ concurrent
        2b: gen_artifact_demos ─┘
        3:  gen_viz
        4:  gen_full_paper
        5:  deploy_to_repo
    """
    # Step 1: Create repo
    create_repo: CreateRepoConfig = Field(default_factory=CreateRepoConfig)

    # Step 2a: Write paper text (concurrent with 2b)
    write_paper_text: WritePaperTextConfig = Field(default_factory=WritePaperTextConfig)
    # Step 2b: Generate artifact demos (concurrent with 2a)
    gen_artifact_demos: GenArtifactDemosConfig = Field(default_factory=GenArtifactDemosConfig)

    # Step 3: Generate visualizations
    viz_gen: VizGenConfig = Field(default_factory=VizGenConfig)

    # Step 4: Generate full paper (LaTeX/PDF)
    gen_full_paper: GenFullPaperConfig = Field(default_factory=GenFullPaperConfig)

    # Step 5: Deploy to GitHub repo
    deploy_to_repo: DeployGistsConfig = Field(default_factory=DeployGistsConfig)

    model_config = ConfigDict(extra="allow")


# =============================================================================
# Main Pipeline Config
# =============================================================================

class PipelineConfig(BaseModel):
    """
    Main pipeline configuration with typed access.

    Usage:
        config = PipelineConfig.from_yaml("config.yaml")
        config.api_keys.openrouter
        config.gen_hypo.llm_client.model
    """
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    init: InitConfig = Field(default_factory=InitConfig)
    seed_hypo: SeedHypoConfig = Field(default_factory=SeedHypoConfig)
    gen_hypo: GenHypoConfig = Field(default_factory=GenHypoConfig)
    audit_hypo: AuditHypoConfig = Field(default_factory=AuditHypoConfig)
    invention_loop: InventionLoopConfig = Field(default_factory=InventionLoopConfig)
    gen_paper_repo: GenPaperConfig = Field(default_factory=GenPaperConfig)

    # Keep raw dict for any custom/unknown keys
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_yaml(cls, config_path: str | Path, overrides: dict | None = None) -> "PipelineConfig":
        """Load configuration from YAML file with optional overrides.

        Args:
            config_path: Path to YAML config file
            overrides: Optional dict of overrides (supports nested keys)
        """
        config_path = Path(config_path)
        if not config_path.is_absolute():
            # Try current directory first, then package directory
            if not config_path.exists():
                package_dir = Path(__file__).parent.parent
                project_root = package_dir.parent.parent
                config_path = project_root / config_path.name

        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Apply overrides if provided
        if overrides:
            raw_config = cls._deep_merge(raw_config, overrides)

        instance = cls.model_validate(raw_config)
        instance.raw = raw_config
        return instance

    @classmethod
    def from_dict(cls, config_dict: dict, overrides: dict | None = None) -> "PipelineConfig":
        """Load configuration from dict with optional overrides."""
        if overrides:
            config_dict = cls._deep_merge(config_dict, overrides)
        instance = cls.model_validate(config_dict)
        instance.raw = config_dict
        return instance

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep merge override dict into base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PipelineConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def with_overrides(self, overrides: dict) -> "PipelineConfig":
        """Return new config with overrides applied."""
        merged = self._deep_merge(self.raw, overrides)
        return PipelineConfig.from_dict(merged)

    def to_dict(self) -> dict:
        """Export configuration as dict."""
        return self.model_dump(exclude={"raw"})


# Convenience alias
Config = PipelineConfig
