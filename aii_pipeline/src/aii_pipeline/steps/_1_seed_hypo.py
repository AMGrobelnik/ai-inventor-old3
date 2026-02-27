#!/usr/bin/env python3
"""
_1_seed_hypo.py - Hypothesis Seed Generation Module

Two-step process: gen_seeds → sample_seeds

gen_seeds:
  If invention_kg.resume_file is set: loads seeds from that file
  Otherwise: runs the invention_kg pipeline (first_step → last_step)

  The invention_kg pipeline steps:
  1. sel_topics     - Select topics from OpenAlex
  2. get_papers     - Fetch papers for topics
  3. clean_papers   - Extract minimal paper data
  4. get_triples    - Extract triples using agents
  5. add_wikidata   - Enrich triples with Wikidata
  6. link_to_papers - Combine papers + triples
  7. gen_hypo_seeds - Generate blind spots/breakthroughs
  8. gen_hypo_prompt - Format seeds as prompts
  9. gen_graphs     - Generate co-occurrence/ontology graphs

sample_seeds:
  1. Select topics (BM25 match to research_direction or manual list)
  2. Build sampling pool per topic (top N by score_percentile)
  3. Assign topics to agents (round-robin)
  4. Sample prompts for each agent from their topics' pools

Uses aii_lib AIITelemetry for sequenced logging.
"""

import json
import random
from datetime import datetime
from pathlib import Path

from aii_lib import create_telemetry, AIITelemetry, MessageType, JSONSink
from aii_pipeline.utils import PipelineConfig, rel_path, build_module_output, emit_module_output
from aii_pipeline.prompts.steps._1_seed_hypo.schema import SeedHypoOut


# Seed hypo steps (high-level)
SEED_HYPO_STEPS = ["gen_seeds", "sample_seeds"]

# Valid step names for invention_kg pipeline (order matches execution)
KG_STEPS = [
    "sel_topics", "get_papers", "clean_papers", "get_triples",
    "add_wikidata", "link_to_papers", "gen_hypo_seeds",
    "gen_hypo_prompt", "gen_graphs"
]


def load_hypothesis_prompts_from_file(
    file_path: str,
    telemetry: AIITelemetry,
) -> tuple[list[dict], str]:
    """
    Load hypothesis prompts from a JSON file.

    Args:
        file_path: Path to hypo_seed_prompts.json
        telemetry: AIITelemetry instance

    Returns:
        Tuple of (prompts list, source path string)
    """
    prompts_file = Path(file_path)
    if not prompts_file.is_absolute():
        # Navigate to project root: _1_seed_hypo.py -> steps/ -> aii_pipeline/ -> src/ -> aii_pipeline/ -> project_root
        prompts_file = Path(__file__).parent.parent.parent.parent.parent / file_path

    if not prompts_file.exists():
        telemetry.emit(MessageType.WARNING, f"⚠️  Hypothesis prompts file not found: {prompts_file}")
        return [], ""

    with open(prompts_file, 'r') as f:
        prompts = json.load(f)

    telemetry.emit(MessageType.INFO, f"   Loaded {len(prompts)} prompts from: {prompts_file}")
    return prompts, str(prompts_file)


def load_hypothesis_prompts_from_kg(
    kg_data_dir: Path,
    telemetry: AIITelemetry,
) -> tuple[list[dict], str]:
    """
    Load hypothesis prompts from invention_kg output directory.

    Args:
        kg_data_dir: Path to invention_kg data directory (contains hypo_seed_prompts.json)
        telemetry: AIITelemetry instance

    Returns:
        Tuple of (prompts list, source path string)
    """
    prompts_file = kg_data_dir / "hypo_seed_prompts.json"

    if not prompts_file.exists():
        telemetry.emit(MessageType.WARNING, f"⚠️  Prompts file not found: {prompts_file}")
        return [], ""

    with open(prompts_file, 'r') as f:
        prompts = json.load(f)

    telemetry.emit(MessageType.INFO, f"   Loaded {len(prompts)} prompts from: {prompts_file}")
    return prompts, str(prompts_file)


def match_topics_bm25(
    research_direction: str,
    available_topics: list[str],
    top_k: int = 4
) -> list[str]:
    """
    Match research area to most similar topics using BM25.

    Args:
        research_direction: The research area query string
        available_topics: List of available topic names
        top_k: Number of top matching topics to return

    Returns:
        List of matched topic names (most similar first)
    """
    import bm25s

    if not available_topics:
        return []

    # Tokenize topics (corpus)
    corpus_tokens = bm25s.tokenize(available_topics, stopwords="en")

    # Build BM25 index
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    # Tokenize query
    query_tokens = bm25s.tokenize([research_direction], stopwords="en")

    # Search
    results, _ = retriever.retrieve(query_tokens, corpus=available_topics, k=min(top_k, len(available_topics)))

    # Extract matched topics
    matched = results[0].tolist()
    return matched


def get_prompts_by_topic(prompts: list[dict]) -> dict[str, list[dict]]:
    """
    Group prompts by topic and sort each group by score_percentile.
    """
    topic_to_prompts: dict[str, list[dict]] = {}

    for p in prompts:
        for topic in p.get('topics', []):
            if topic not in topic_to_prompts:
                topic_to_prompts[topic] = []
            topic_to_prompts[topic].append(p)

    # Sort each topic's prompts by score_percentile descending
    for topic in topic_to_prompts:
        topic_to_prompts[topic].sort(
            key=lambda x: x.get('score_percentile', x.get('score', 0)),
            reverse=True
        )

    return topic_to_prompts


def build_sampling_pools(
    prompts: list[dict],
    selected_topics: list[str],
    pool_size: int = 10
) -> dict[str, list[dict]]:
    """
    Build sampling pools for selected topics.

    Takes top-k prompts where selected topic is blind (what it's missing).
    Ranked by score_percentile.

    Returns:
        Dict mapping selected_topic -> list of prompts
    """
    pools = {}

    for sel_topic in selected_topics:
        # Get prompts where selected topic is blind (what it's missing)
        sel_blind = [p for p in prompts if p.get('blind_topic') == sel_topic]
        sel_blind.sort(key=lambda x: x.get('score_percentile', 0), reverse=True)
        pools[sel_topic] = sel_blind[:pool_size]

    return pools


def assign_topics_to_agents(
    selected_topics: list[str],
    num_agents: int,
    topics_per_agent: int
) -> list[list[str]]:
    """Assign topics to agents via round-robin distribution."""
    if not selected_topics:
        return [[] for _ in range(num_agents)]

    shuffled_topics = selected_topics.copy()
    random.shuffle(shuffled_topics)

    agent_topics = [[] for _ in range(num_agents)]
    topic_idx = 0

    for agent_idx in range(num_agents):
        for _ in range(topics_per_agent):
            # Round-robin: if more agent slots than topics, wrap around and reuse topics
            if topic_idx >= len(shuffled_topics):
                topic_idx = 0
            agent_topics[agent_idx].append(shuffled_topics[topic_idx])
            topic_idx += 1

    return agent_topics


def sample_seeds_for_agents(
    pools: dict[str, list[dict]],
    agent_topics: list[list[str]],
    seeds_per_topic: int = 2
) -> list[list[dict]]:
    """Sample seeds for each agent from their assigned topics' pools."""
    agent_seeds = []

    for topics in agent_topics:
        sampled = []
        for topic in topics:
            pool = pools.get(topic, [])
            if pool:
                n_select = min(seeds_per_topic, len(pool))
                selected = random.sample(pool, n_select)
                for s in selected:
                    if s not in sampled:
                        sampled.append(s)
        agent_seeds.append(sampled)

    return agent_seeds


def _kg_step_name_to_number(step_name: str) -> int:
    """Convert kg step name to step number (1-indexed)."""
    try:
        return KG_STEPS.index(step_name) + 1
    except ValueError:
        raise ValueError(f"Invalid KG step name '{step_name}', valid steps: {KG_STEPS}")


async def run_invention_kg_pipeline(
    config: PipelineConfig,
    output_dir: Path,
    telemetry: AIITelemetry,
) -> tuple[list[dict], str, Path]:
    """
    Run the invention_kg pipeline to generate hypothesis seeds.

    Args:
        config: Pipeline configuration
        output_dir: Directory to store outputs
        telemetry: AIITelemetry instance

    Returns:
        Tuple of (prompts list, source description, kg_data_dir)
    """
    from aii_pipeline.steps._seed_hypo.invention_kg.config import create_config
    from aii_pipeline.steps._seed_hypo.invention_kg.pipeline import run_pipeline as run_kg_pipeline

    kg_cfg = config.seed_hypo.invention_kg

    # Convert step names to numbers
    start_step = _kg_step_name_to_number(kg_cfg.first_step)
    end_step = _kg_step_name_to_number(kg_cfg.last_step)

    telemetry.emit(MessageType.INFO, "🔬 Running invention_kg pipeline...")
    telemetry.emit(MessageType.INFO, f"   Steps: {kg_cfg.first_step} ({start_step}) → {kg_cfg.last_step} ({end_step})")

    # Create invention_kg config from main pipeline config
    kg_config = create_config({
        # Resume dirs (set the one BEFORE first_step)
        'sel_topics_out_dir': kg_cfg.sel_topics_out_dir,
        'get_papers_out_dir': kg_cfg.get_papers_out_dir,
        'clean_papers_out_dir': kg_cfg.clean_papers_out_dir,
        'get_triples_out_dir': kg_cfg.get_triples_out_dir,
        'add_wikidata_out_dir': kg_cfg.add_wikidata_out_dir,
        'link_to_papers_out_dir': kg_cfg.link_to_papers_out_dir,
        'gen_hypo_seeds_out_dir': kg_cfg.gen_hypo_seeds_out_dir,
        'gen_hypo_prompt_out_dir': kg_cfg.gen_hypo_prompt_out_dir,
        'sel_topics': {'topics': kg_cfg.sel_topics.topics} if kg_cfg.sel_topics.topics else {},
        'get_papers': {
            'email': kg_cfg.get_papers.email,
            'papers_per_year': kg_cfg.get_papers.papers_per_year,
            'year_range': kg_cfg.get_papers.year_range,
            'sort_by': kg_cfg.get_papers.sort_by,
        },
        'get_triples': {
            'max_papers': kg_cfg.get_triples.max_papers,
            'max_concurrent_agents': kg_cfg.get_triples.max_concurrent_agents,
            'stagger_delay': kg_cfg.get_triples.stagger_delay,
            'url_verification_retries': kg_cfg.get_triples.url_verification_retries,
            'min_valid_urls': kg_cfg.get_triples.min_valid_urls,
            'claude_agent': {
                'model': kg_cfg.get_triples.claude_agent.model,
                'max_turns': kg_cfg.get_triples.claude_agent.max_turns,
                'agent_timeout': kg_cfg.get_triples.claude_agent.agent_timeout,
                'agent_retries': kg_cfg.get_triples.claude_agent.agent_retries,
                'seq_prompt_timeout': kg_cfg.get_triples.claude_agent.seq_prompt_timeout,
                'seq_prompt_retries': kg_cfg.get_triples.claude_agent.seq_prompt_retries,
            },
        },
        'gen_hypo_seeds': {
            'blind_spots': {
                'min_shared_concepts': kg_cfg.gen_hypo_seeds.blind_spots.min_shared_concepts,
                'max_similarity': kg_cfg.gen_hypo_seeds.blind_spots.max_similarity,
                'entity_types': kg_cfg.gen_hypo_seeds.blind_spots.entity_types,
            }
        },
        'gen_graph': {
            'temporal_windows': kg_cfg.gen_graph.temporal_windows,
        },
        'viz_graph': {
            'port': kg_cfg.viz_graph.port,
        },
    })

    # Run pipeline with telemetry (now async)
    result = await run_kg_pipeline(
        kg_config,
        start_step=start_step,
        end_step=end_step,
        telemetry=telemetry,
    )

    if not result or not result.get('success'):
        telemetry.emit(MessageType.ERROR, "❌ invention_kg pipeline failed")
        return [], "", output_dir

    # Get prompts from result
    prompts = result.get('prompts', [])
    kg_data_dir = Path(result.get('output_dir', output_dir))

    # Copy to output directory
    output_prompts = output_dir / "hypo_seed_prompts.json"
    with open(output_prompts, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    source = f"invention_kg pipeline"
    telemetry.emit(MessageType.SUCCESS, f"✅ Generated {len(prompts)} prompts from {source}")

    return prompts, source, kg_data_dir


async def run_seed_hypo_module(
    config: PipelineConfig,
    input_text=None,
    run_dir=None,
    workspace_dir=None,
    telemetry: AIITelemetry | None = None,
    cumulative: dict | None = None,
):
    """
    Run the hypothesis seed generation module.

    Steps: gen_seeds → sample_seeds
    - gen_seeds: If resume_file is set, load from file; otherwise run invention_kg pipeline
    - sample_seeds: Sample seeds for agents based on research direction

    Args:
        config: Typed pipeline configuration
        input_text: Optional input text (not used)
        run_dir: Optional run directory for outputs
        workspace_dir: Optional workspace directory
        telemetry: Optional external telemetry (from pipeline)

    Returns:
        Dictionary with agent_prompts and metadata
    """
    # Create output directory
    if run_dir:
        output_dir = run_dir / "1_seed_hypo"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{config.init.outputs_directory}/{timestamp}_seed_hypo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use provided telemetry or create local one
    local_telemetry = telemetry is None
    if local_telemetry:
        telemetry = create_telemetry(output_dir, "seed_hypo")

    try:
        # Get config
        research_direction = config.init.research_direction
        seed_hypo_cfg = config.seed_hypo
        kg_cfg = seed_hypo_cfg.invention_kg
        sampling_cfg = seed_hypo_cfg.sampling

        # Determine which steps to run
        first_step = seed_hypo_cfg.first_step
        last_step = seed_hypo_cfg.last_step
        run_gen_seeds = first_step == "gen_seeds"
        run_sample_seeds = last_step == "sample_seeds"

        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, "🌱 SEED_HYPO - Hypothesis Seed Generation")
        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, f"   Research Area: {research_direction}")
        telemetry.emit(MessageType.INFO, f"   Steps: {first_step} → {last_step}")
        telemetry.emit(MessageType.INFO, f"   Output: {rel_path(output_dir)}")
        telemetry.emit(MessageType.INFO, "=" * 60)

        # Add module-specific JSON sink for messages file
        s1 = JSONSink(output_dir / "seed_hypo_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "seed_hypo_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks = [s1, s2]

        # Start module group (SEED_HYPO wraps GEN_SEEDS and sample_seeds)
        telemetry.start_module_group("SEED_HYPO")

        all_prompts = []
        hypo_source = ""
        kg_data_dir = None

        # Step 1: gen_seeds (uses module GROUP to nest INVENTION_KG module group inside)
        if run_gen_seeds:
            telemetry.start_module_group("GEN_SEEDS")

            # Check if we should load from existing output dir
            if kg_cfg.gen_hypo_prompt_out_dir:
                # Load from gen_hypo_prompt output directory
                telemetry.emit(MessageType.INFO, "\n💡 Loading seeds from gen_hypo_prompt_out_dir...")
                prompts_file = Path(kg_cfg.gen_hypo_prompt_out_dir)
                if not prompts_file.is_absolute():
                    # Navigate to project root: steps/ -> aii_pipeline/ -> src/ -> aii_pipeline/ -> project_root
                    prompts_file = Path(__file__).parent.parent.parent.parent.parent / prompts_file
                # Look for prompts file (hypo_seed_prompts.json, blind_spot_prompts.json, etc.)
                prompts_files = [
                    "hypo_seed_prompts.json",
                    "blind_spot_prompts.json",
                    "_1_hypo_seed_prompts.json",
                ]
                found_file = None
                for fname in prompts_files:
                    if (prompts_file / fname).exists():
                        found_file = prompts_file / fname
                        break

                if found_file:
                    all_prompts, hypo_source = load_hypothesis_prompts_from_file(
                        file_path=str(found_file),
                        telemetry=telemetry,
                    )
                else:
                    telemetry.emit(MessageType.ERROR, f"❌ No prompts file found in {prompts_file}")
                    return None
            else:
                # Run invention_kg pipeline
                telemetry.emit(MessageType.INFO, "\n🔬 Running invention_kg pipeline...")
                all_prompts, hypo_source, kg_data_dir = await run_invention_kg_pipeline(
                    config=config,
                    output_dir=output_dir,
                    telemetry=telemetry,
                )

            telemetry.emit_module_group_summary("GEN_SEEDS")
        else:
            # Skip gen_seeds - load from invention_kg_seed_out_dir
            seed_out_dir = seed_hypo_cfg.invention_kg_seed_out_dir
            if seed_out_dir:
                telemetry.emit(MessageType.INFO, "\n💡 Loading seeds from invention_kg_seed_out_dir...")
                prompts_file = Path(seed_out_dir)
                if not prompts_file.is_absolute():
                    # Navigate to project root: steps/ -> aii_pipeline/ -> src/ -> aii_pipeline/ -> project_root
                    prompts_file = Path(__file__).parent.parent.parent.parent.parent / prompts_file
                if (prompts_file / "hypo_seed_prompts.json").exists():
                    all_prompts, hypo_source = load_hypothesis_prompts_from_file(
                        file_path=str(prompts_file / "hypo_seed_prompts.json"),
                        telemetry=telemetry,
                    )
                else:
                    telemetry.emit(MessageType.ERROR, f"❌ No prompts file found in {prompts_file}")
                    return None
            else:
                telemetry.emit(MessageType.WARNING, "⚠️  Skipping gen_seeds but no invention_kg_seed_out_dir set")

        # Handle case with no prompts
        if not all_prompts:
            telemetry.emit(MessageType.WARNING, "⚠️  No hypothesis prompts available - continuing without")
            if config.gen_hypo.use_claude_agent:
                num_agents = config.gen_hypo.seeded_hypos_per_llm
            else:
                num_agents = config.gen_hypo.seeded_hypos_per_llm * len(config.gen_hypo.llm_client.models)
            module_output = SeedHypoOut(
                output_dir=str(output_dir),
                agent_prompts=[[] for _ in range(num_agents)],
                agent_topics=[[] for _ in range(num_agents)],
            )
            std_output = build_module_output(
                module="seed_hypo",
                outputs=[module_output],
                cumulative=cumulative or {},
                output_dir=output_dir,
                research_direction=research_direction,
                hypo_source=hypo_source,
                first_step=first_step,
                last_step=last_step,
            )
            emit_module_output(std_output, telemetry, output_dir=output_dir)

            telemetry.emit_module_group_summary("SEED_HYPO")
            return module_output

        # Copy prompts file to output directory if not already there
        output_prompts = output_dir / "hypo_seed_prompts.json"
        if not output_prompts.exists():
            with open(output_prompts, 'w', encoding='utf-8') as f:
                json.dump(all_prompts, f, indent=2, ensure_ascii=False)

        # Step 2: sample_seeds
        if run_sample_seeds:
            telemetry.emit_task_start("sample_seeds", "Sample Seeds")

            # Extract available topics from prompts
            available_topics = set()
            for p in all_prompts:
                available_topics.update(p.get('topics', []))
            available_topics = sorted(available_topics)
            telemetry.emit(MessageType.INFO, f"\n   Available topics ({len(available_topics)}): {available_topics}")

            # Select topics (BM25 or manual)
            telemetry.emit(MessageType.INFO, "\n📋 Selecting topics for sampling...")
            sel_topics_cfg = sampling_cfg.sel_topics

            if sel_topics_cfg == 'auto':
                # BM25 match
                selected_topics = match_topics_bm25(
                    research_direction,
                    available_topics,
                    top_k=sampling_cfg.research_dir_topic_match_k,
                )
                telemetry.emit(MessageType.INFO, f"   BM25 matched {len(selected_topics)} topics: {selected_topics}")
            else:
                # Manual list
                selected_topics = [t for t in sel_topics_cfg if t in available_topics]
                telemetry.emit(MessageType.INFO, f"   Manual selection: {selected_topics}")

            if not selected_topics:
                telemetry.emit(MessageType.WARNING, "   No topics selected - using all available")
                selected_topics = list(available_topics)

            # Build sampling pools
            telemetry.emit(MessageType.INFO, "\n🎯 Building sampling pools...")
            pools = build_sampling_pools(all_prompts, selected_topics, pool_size=sampling_cfg.seed_sampling_pool)
            for topic, pool in pools.items():
                telemetry.emit(MessageType.INFO, f"   {topic}: {len(pool)} seeds in pool")

            # Assign topics to agents
            telemetry.emit(MessageType.INFO, "\n👥 Assigning topics to agents...")
            if config.gen_hypo.use_claude_agent:
                num_agents = config.gen_hypo.seeded_hypos_per_llm
            else:
                num_agents = config.gen_hypo.seeded_hypos_per_llm * len(config.gen_hypo.llm_client.models)
            agent_topics = assign_topics_to_agents(selected_topics, num_agents, sampling_cfg.topics_per_agent)
            for i, topics in enumerate(agent_topics):
                telemetry.emit(MessageType.INFO, f"   Agent {i+1}: {topics}")

            # Sample seeds for each agent
            telemetry.emit(MessageType.INFO, "\n🎲 Sampling seeds for agents...")
            agent_prompts = sample_seeds_for_agents(pools, agent_topics, sampling_cfg.seeds_per_topic)

            total_unique = len(set(p['id'] for prompts in agent_prompts for p in prompts))
            telemetry.emit(MessageType.SUCCESS, f"\n✅ Sampled seeds for {num_agents} agents")
            telemetry.emit(MessageType.INFO, f"   Config: {sampling_cfg.topics_per_agent} topics/agent × {sampling_cfg.seeds_per_topic} seeds/topic")
            telemetry.emit(MessageType.INFO, f"   Total unique seeds: {total_unique}")

            for i, seeds in enumerate(agent_prompts):
                seed_ids = [p.get('id', '?')[:50] for p in seeds]
                telemetry.emit(MessageType.INFO, f"   Agent {i+1} ({len(seeds)} seeds): {seed_ids}")

            telemetry.emit_task_end("sample_seeds", "Sample Seeds", "OK")
        else:
            # Skip sample_seeds - just return the prompts without sampling
            telemetry.emit(MessageType.INFO, "\n⏭️  Skipping sample_seeds step")
            selected_topics = []
            pools = {}
            agent_topics = []
            agent_prompts = []
            num_agents = 0

        # Build module output
        module_output = SeedHypoOut(
            output_dir=str(output_dir),
            agent_prompts=agent_prompts,
            agent_topics=agent_topics,
            selected_topics=selected_topics,
            pools={t: [p['id'] for p in pool] for t, pool in pools.items()},
            all_hypo_prompts=all_prompts,
        )

        # Emit standardized module output
        std_output = build_module_output(
            module="seed_hypo",
            outputs=[module_output],
            cumulative=cumulative or {},
            output_dir=output_dir,
            research_direction=research_direction,
            total_hypo_prompts=len(all_prompts),
            num_agents=num_agents,
            topics_per_agent=sampling_cfg.topics_per_agent,
            seeds_per_topic=sampling_cfg.seeds_per_topic,
            seed_sampling_pool=sampling_cfg.seed_sampling_pool,
            research_dir_topic_match_k=sampling_cfg.research_dir_topic_match_k,
            hypo_source=hypo_source,
            first_step=first_step,
            last_step=last_step,
            kg_data_dir=str(kg_data_dir) if kg_data_dir else None,
        )
        emit_module_output(std_output, telemetry, output_dir=output_dir)
        telemetry.emit_module_group_summary("SEED_HYPO")

        return module_output

    finally:
        # Remove module sinks to prevent leak into subsequent modules
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
        # Only flush/close if we created local telemetry
        if local_telemetry:
            telemetry.flush()


async def main():
    """Main function for standalone execution."""
    config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
    config = PipelineConfig.from_yaml(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"{config.init.outputs_directory}/{timestamp}_seed_hypo")
    run_dir.mkdir(parents=True, exist_ok=True)

    result = await run_seed_hypo_module(config, run_dir=run_dir)
    if result:
        print("Hypothesis seed generation completed successfully")
        return 0
    else:
        print("Hypothesis seed generation failed")
        return 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)
