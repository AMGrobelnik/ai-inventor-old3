#!/usr/bin/env python3
"""
AI Inventor - Pipeline Orchestration
Core pipeline orchestration logic for executing research modules in sequence.
"""

import json
from pathlib import Path

import yaml

from aii_lib.telemetry import logger
from aii_lib.utils import ensure_server_running

from aii_lib import create_telemetry, AIITelemetry, MessageType, cleanup_run_caches
from aii_pipeline.utils import PipelineConfig, init_cumulative

# Import module runners
from aii_pipeline.steps._1_seed_hypo import run_seed_hypo_module
from aii_pipeline.steps._2_gen_hypo import run_gen_hypo_module
from aii_pipeline.steps._3_invention_loop import run_invention_loop_module
from aii_pipeline.steps._4_gen_paper_repo import run_gen_paper_module

from aii_pipeline.prompts.steps._1_seed_hypo.schema import SeedHypoOut
from aii_pipeline.prompts.steps._2_gen_hypo.schema import GenHypoOut
from aii_pipeline.prompts.steps._3_invention_loop.schema import InventionLoopOut

# Load ability server config
_ABILITY_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "aii_lib/src/aii_lib/abilities/ability_server/server_config.yaml"
_ability_config = {}
if _ABILITY_CONFIG_PATH.exists():
    with open(_ABILITY_CONFIG_PATH) as f:
        _ability_config = yaml.safe_load(f) or {}

ABILITY_SERVER_PORT = _ability_config.get("server", {}).get("port", 8100)


def _ensure_ability_server_running() -> bool:
    """Ensure the ability server is running. Called automatically at pipeline start."""
    return ensure_server_running(port=ABILITY_SERVER_PORT, log_func=logger.info)


def cleanup_cache(run_dir: Path | str) -> None:
    """Clean up all cache directories created during this run."""
    result = cleanup_run_caches(run_dir, clear_venv=True, clear_hf=True)
    if result["removed"]:
        logger.info(f"🧹 Cleaning up caches: {len(result['removed'])} items")
        for item in result["removed"]:
            logger.debug(f"  - {item}")
        logger.success(f"✅ Removed {result['total_size_mb']:.1f} MB of cached data")


async def run_pipeline(config: PipelineConfig, run_dir=None, workspace_dir=None):
    """Run the pipeline from first_step to last_step with optional checkpoint loading."""
    pipeline_cfg = config.init.pipeline

    # Always ensure ability server is running at pipeline start
    _ensure_ability_server_running()

    # Define the complete pipeline sequence
    # Note: init is no longer a module - it runs automatically above
    # Note: sel_data/gen_sol/eval_sol removed - now handled by invention_loop units
    pipeline_sequence = ['seed_hypo', 'gen_hypo', 'invention_loop', 'gen_paper_repo']

    # Get first and last modules
    first_step = (pipeline_cfg.first_step or 'seed_hypo').strip()
    last_step = (pipeline_cfg.last_step or 'gen_paper_repo').strip()

    # Validate module names
    if first_step not in pipeline_sequence:
        logger.error(f"❌ Invalid first_step: '{first_step}'")
        logger.error(f"Valid modules: {', '.join(pipeline_sequence)}")
        return None

    if last_step not in pipeline_sequence:
        logger.error(f"❌ Invalid last_step: '{last_step}'")
        logger.error(f"Valid modules: {', '.join(pipeline_sequence)}")
        return None

    # Get module indices
    start_index = pipeline_sequence.index(first_step)
    end_index = pipeline_sequence.index(last_step)

    # Validate ordering
    if start_index > end_index:
        logger.error(f"❌ Invalid configuration: first_step '{first_step}' comes after last_step '{last_step}'")
        return None

    # Track results
    results = {}

    # Create cumulative state dict (flows through all modules)
    cumulative = init_cumulative()

    # Create shared telemetry instance for entire pipeline run
    from aii_lib.telemetry import JSONSink
    telemetry = create_telemetry(run_dir, "pipeline")
    root_sink = JSONSink(run_dir / "pipeline_messages_sequenced.jsonl", sequenced=True)
    telemetry.add_sink(root_sink)
    telemetry.start_module("PIPELINE")

    # Load checkpoint data based on first_step
    current_input = None

    if first_step == 'gen_hypo':
        # Need seed_hypo output directory with seed_hypo_output.json
        seed_hypo_out_dir = (pipeline_cfg.seed_hypo_out_dir or '').strip()
        if seed_hypo_out_dir:
            seed_hypo_path = Path(seed_hypo_out_dir)
            if not seed_hypo_path.exists():
                logger.error(f"❌ seed_hypo_out_dir does not exist: {seed_hypo_out_dir}")
                telemetry.emit(MessageType.ERROR, f"seed_hypo_out_dir does not exist: {seed_hypo_out_dir}")
                return None
            if not seed_hypo_path.is_dir():
                logger.error(f"❌ seed_hypo_out_dir must be a directory: {seed_hypo_out_dir}")
                telemetry.emit(MessageType.ERROR, f"seed_hypo_out_dir must be a directory: {seed_hypo_out_dir}")
                return None

            # Load seed_hypo_output.json
            seed_hypo_result_file = seed_hypo_path / "seed_hypo_output.json"
            if not seed_hypo_result_file.exists():
                logger.error(f"❌ seed_hypo_output.json not found in {seed_hypo_out_dir}")
                telemetry.emit(MessageType.ERROR, f"seed_hypo_output.json not found in {seed_hypo_out_dir}")
                return None

            try:
                with open(seed_hypo_result_file, 'r', encoding='utf-8') as f:
                    seed_hypo_result = json.load(f)
            except json.JSONDecodeError as e:
                logger.exception(f"❌ Failed to parse {seed_hypo_result_file}: {e}")
                telemetry.emit(MessageType.ERROR, f"Failed to parse {seed_hypo_result_file}: {e}")
                raise

            results['seed_hypo'] = SeedHypoOut(**seed_hypo_result)
            logger.info(f"📄 Loaded {len(results['seed_hypo'].agent_prompts)} agent prompts from: {seed_hypo_result_file}")
        else:
            logger.info(f"📄 No seed_hypo_out_dir set - gen_hypo will run without seed prompts")

    elif first_step == 'invention_loop':
        # Need gen_hypo output directory with gen_hypo_output.json
        gen_hypo_out_dir = (pipeline_cfg.gen_hypo_out_dir or '').strip()
        if not gen_hypo_out_dir:
            logger.error(f"Starting from invention_loop requires gen_hypo_out_dir to be set")
            telemetry.emit(MessageType.ERROR, "Starting from invention_loop requires gen_hypo_out_dir to be set")
            return None
        gen_hypo_path = Path(gen_hypo_out_dir)
        if not gen_hypo_path.exists():
            logger.error(f"gen_hypo_out_dir does not exist: {gen_hypo_out_dir}")
            telemetry.emit(MessageType.ERROR, f"gen_hypo_out_dir does not exist: {gen_hypo_out_dir}")
            return None

        gen_hypo_result_file = gen_hypo_path / "gen_hypo_output.json"
        if not gen_hypo_result_file.exists():
            logger.error(f"gen_hypo_output.json not found in {gen_hypo_out_dir}")
            telemetry.emit(MessageType.ERROR, f"gen_hypo_output.json not found in {gen_hypo_out_dir}")
            return None

        try:
            with open(gen_hypo_result_file, 'r', encoding='utf-8') as f:
                gen_hypo_result = json.load(f)
        except json.JSONDecodeError as e:
            logger.exception(f"Failed to parse {gen_hypo_result_file}: {e}")
            telemetry.emit(MessageType.ERROR, f"Failed to parse {gen_hypo_result_file}: {e}")
            raise

        # Map 'outputs' → 'hypotheses' (build_module_output writes 'outputs', GenHypoOut expects 'hypotheses')
        if 'outputs' in gen_hypo_result and 'hypotheses' not in gen_hypo_result:
            gen_hypo_result['hypotheses'] = gen_hypo_result.pop('outputs')
        results['gen_hypo'] = GenHypoOut(**gen_hypo_result)
        logger.info(f"Loaded {len(results['gen_hypo'].hypotheses)} hypotheses from: {gen_hypo_result_file}")

    elif first_step == 'gen_paper_repo':
        # Need invention_loop output directory with invention_loop_result.json
        invention_loop_out_dir = (pipeline_cfg.invention_loop_out_dir or '').strip()
        if not invention_loop_out_dir:
            logger.error(f"❌ Starting from gen_paper_repo requires invention_loop_out_dir to be set")
            telemetry.emit(MessageType.ERROR, "Starting from gen_paper_repo requires invention_loop_out_dir to be set")
            return None
        invention_loop_path = Path(invention_loop_out_dir)
        if not invention_loop_path.exists():
            logger.error(f"❌ invention_loop_out_dir does not exist: {invention_loop_out_dir}")
            telemetry.emit(MessageType.ERROR, f"invention_loop_out_dir does not exist: {invention_loop_out_dir}")
            return None
        if not invention_loop_path.is_dir():
            logger.error(f"❌ invention_loop_out_dir must be a directory: {invention_loop_out_dir}")
            telemetry.emit(MessageType.ERROR, f"invention_loop_out_dir must be a directory: {invention_loop_out_dir}")
            return None

        # Load invention_loop_result.json
        invention_loop_result_file = invention_loop_path / "invention_loop_result.json"
        if not invention_loop_result_file.exists():
            logger.error(f"❌ invention_loop_result.json not found in {invention_loop_out_dir}")
            telemetry.emit(MessageType.ERROR, f"invention_loop_result.json not found in {invention_loop_out_dir}")
            return None

        try:
            with open(invention_loop_result_file, 'r', encoding='utf-8') as f:
                invention_loop_result = json.load(f)
        except json.JSONDecodeError as e:
            logger.exception(f"❌ Failed to parse {invention_loop_result_file}: {e}")
            telemetry.emit(MessageType.ERROR, f"Failed to parse {invention_loop_result_file}: {e}")
            raise

        # Store in results so gen_paper_repo can access it
        results['invention_loop'] = InventionLoopOut(**invention_loop_result)
        logger.info(f"📄 Loaded invention_loop result from: {invention_loop_result_file}")
        narrative = results['invention_loop'].narrative
        pools_dir = results['invention_loop'].pools_dir
        logger.info(f"   Narrative: {narrative.id if narrative else 'N/A'}, Pools: {pools_dir}")

    # Log execution plan
    modules_to_run = pipeline_sequence[start_index:end_index+1]
    logger.info(f"🎯 Will run modules: {' → '.join(modules_to_run)}")

    # Run modules in sequence from start to end
    for i in range(start_index, end_index + 1):
        current_module = pipeline_sequence[i]
        # Use same descriptions for consistency
        checkpoint_descriptions = {
            'seed_hypo': 'SeedHypo (knowledge graph -> hypothesis seeds)',
            'gen_hypo': 'GenHypo (hypothesis generation)',
            'invention_loop': 'InventionLoop (iterative invention)',
            'gen_paper_repo': 'GenPaperRepo (paper + repo generation)',
        }
        module_desc = checkpoint_descriptions.get(current_module, current_module)
        logger.info(f"📦 Running module: {module_desc}")

        if current_module == "seed_hypo":
            result = await run_seed_hypo_module(config, input_text=current_input, run_dir=run_dir, workspace_dir=workspace_dir, telemetry=telemetry, cumulative=cumulative)
            if not result:
                logger.error(f"Module {module_desc} failed")
                telemetry.emit(MessageType.ERROR, f"Module {module_desc} failed")
                return None
            results[current_module] = result

        elif current_module == "gen_hypo":
            # Get inspiration prompts from seed_hypo if available
            seed_hypo_result = results.get('seed_hypo')
            agent_prompts = seed_hypo_result.agent_prompts if seed_hypo_result else []
            result = await run_gen_hypo_module(
                config,
                agent_prompts=agent_prompts,
                run_dir=run_dir,
                telemetry=telemetry,
                cumulative=cumulative,
            )
            if not result:
                logger.error(f"Module {module_desc} failed")
                telemetry.emit(MessageType.ERROR, f"Module {module_desc} failed")
                return None
            results[current_module] = result

        elif current_module == "invention_loop":
            # Get hypotheses from gen_hypo (use first hypothesis)
            gen_hypo_result = results.get('gen_hypo')
            hypotheses = gen_hypo_result.hypotheses if gen_hypo_result else []
            if not hypotheses:
                logger.error("invention_loop requires at least one hypothesis from gen_hypo")
                telemetry.emit(MessageType.ERROR, "invention_loop requires at least one hypothesis from gen_hypo")
                return None

            hypothesis = hypotheses[0]

            result = await run_invention_loop_module(
                config,
                hypothesis=hypothesis,
                run_dir=run_dir,
                workspace_dir=workspace_dir,
                telemetry=telemetry,
                cumulative=cumulative,
            )
            if not result:
                logger.error(f"Module {module_desc} failed")
                telemetry.emit(MessageType.ERROR, f"Module {module_desc} failed")
                return None
            results[current_module] = result

        elif current_module == "gen_paper_repo":
            # Get invention loop result (contains narrative, all_artifacts, metadata)
            invention_loop_result = results.get('invention_loop')
            if not invention_loop_result:
                logger.error("gen_paper_repo requires invention_loop result")
                telemetry.emit(MessageType.ERROR, "gen_paper_repo requires invention_loop result")
                return None

            result = await run_gen_paper_module(
                config=config,
                invention_loop_result=invention_loop_result,
                run_dir=run_dir,
                workspace_dir=workspace_dir,
                telemetry=telemetry,
                cumulative=cumulative,
            )
            if not result:
                logger.error(f"Module {module_desc} failed")
                telemetry.emit(MessageType.ERROR, f"Module {module_desc} failed")
                return None
            results[current_module] = result
            # Paper.pdf is generated

        else:
            logger.error(f"Module {current_module} not implemented")
            telemetry.emit(MessageType.ERROR, f"Module {current_module} not implemented")
            return None

    # Flush telemetry (writes JSON log if configured)
    telemetry.flush()

    logger.info("Pipeline completed")

    # Re-emit all module summaries for end-of-run recap
    telemetry.emit_all_module_summaries()

    # Emit pipeline-level summary at the very end
    telemetry.emit_pipeline_summary()

    # Clean up root sink
    root_sink.flush()
    telemetry.remove_sink(root_sink)

    return results


