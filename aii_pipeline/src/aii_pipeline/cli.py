#!/usr/bin/env python3
"""
CLI interface for aii_pipeline.
Handles command-line argument parsing, configuration overrides, and entry point setup.
"""

import asyncio
import argparse
import os
import random
import string
import sys
from datetime import datetime
from pathlib import Path

from aii_lib.telemetry import logger

from aii_pipeline.utils import PipelineConfig, rel_path
from aii_pipeline.pipeline import run_pipeline, cleanup_cache


def auto_cast_value(value):
    """Auto-cast string values to appropriate types (bool, int, float, or string)."""
    if isinstance(value, bool):
        return value

    if not isinstance(value, str):
        return value

    # Try boolean
    if value.lower() in ('true', 'yes', '1'):
        return True
    if value.lower() in ('false', 'no', '0'):
        return False

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


def parse_cli_overrides(args_list):
    """
    Parse CLI arguments with dot notation into nested dictionary.

    Examples:
        --init.research_direction "ML"
            → {'init': {'research_direction': 'ML'}}

        --gen_hypo.openai_settings.model "gpt-4"
            → {'gen_hypo': {'openai_settings': {'model': 'gpt-4'}}}

        --sel_hypo.claude_code_settings.max_turns 500
            → {'sel_hypo': {'claude_code_settings': {'max_turns': 500}}}

    Args:
        args_list: List of command-line arguments (sys.argv[1:])

    Returns:
        dict: Nested dictionary with configuration overrides
    """
    overrides = {}

    i = 0
    while i < len(args_list):
        arg = args_list[i]

        # Check if it's a CLI flag (starts with --)
        if arg.startswith('--'):
            key_path = arg[2:]  # Remove '--' prefix

            # Check if value is embedded with = (e.g., --key=value)
            if '=' in key_path:
                key_path, value = key_path.split('=', 1)
                i += 1
            # Otherwise, get the value from next argument
            elif i + 1 < len(args_list) and not args_list[i + 1].startswith('--'):
                value = args_list[i + 1]
                i += 2
            else:
                # Boolean flag (no value provided, treat as True)
                value = True
                i += 1

            # Parse key path (e.g., "init.research_direction" → ['init', 'research_direction'])
            keys = key_path.split('.')

            # Build nested dictionary
            dotted_key = key_path
            current = overrides
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                elif not isinstance(current[key], dict):
                    raise ValueError(f"Conflicting CLI overrides: '{dotted_key}' conflicts with a parent key")
                current = current[key]

            # Set the final value with type inference
            final_key = keys[-1]
            current[final_key] = auto_cast_value(value)
        else:
            # Not a flag, skip
            i += 1

    return overrides


def setup_argparser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='AI Inventor - Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Override research area
  aii_pipeline --init.research_direction "Multi-LLM Agent Systems"

  # Override multiple settings
  aii_pipeline --init.research_direction "ML"

  # Override OpenAI model
  aii_pipeline --gen_hypo.openai_settings.model "gpt-4"

  # Set module range
  aii_pipeline --init.pipeline.first_step "gen_hypo" --init.pipeline.last_step "gen_paper_repo"

  # Resume from checkpoint
  aii_pipeline --init.pipeline.first_step "invention_loop" --init.pipeline.gen_hypo_out_dir "runs/my_run/2_gen_hypo"

  # Override hypothesis seeds config
  aii_pipeline --prep_context.hypothesis_seeds.research_dir_topic_match_k 3 --prep_context.hypothesis_seeds.seed_sampling_pool 10

Any config parameter from config.yaml can be overridden using dot notation: --section.subsection.key value
        """
    )
    return parser


async def main():
    """Main CLI entry point with configuration loading and pipeline execution."""
    # Load .env file from project root (does NOT override existing env vars)
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    load_dotenv(env_path)

    # Parse command-line arguments dynamically using dot notation
    parser = setup_argparser()

    # Don't define specific arguments - parse everything dynamically
    args, unknown = parser.parse_known_args()

    # Parse CLI overrides from all arguments
    cli_overrides = parse_cli_overrides(sys.argv[1:])

    # Load configuration from YAML with overrides
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    try:
        config = PipelineConfig.from_yaml(config_path, overrides=cli_overrides)
        logger.info(f"📋 Config loaded from: {rel_path(config_path)}")

        # Initialize aii_lib global config
        from aii_lib.config import aii_config
        aii_config.init_from_pipeline_config(config)

        # Hardware detection now handled by aii_get_hardware skill at runtime
    except FileNotFoundError:
        logger.error(f"❌ Config file not found: {config_path}")
        return 1
    except Exception as e:
        logger.error("❌ Error loading config", exc=e)
        return 1

    if cli_overrides:
        logger.info("📝 CLI overrides applied:")
        for key, value in cli_overrides.items():
            logger.info(f"   {key}: {value}")

    # Preflight: start servers, then run health check (informational only)
    scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"

    import subprocess
    logger.info("🔧 Starting servers...")
    start_result = subprocess.run(
        ["bash", str(scripts_dir / "start.sh")],
        cwd=str(scripts_dir.parent),
    )
    if start_result.returncode != 0:
        logger.error("❌ Failed to start servers")
        return 1

    logger.info("🔍 Running health check...")
    subprocess.run(
        ["bash", str(scripts_dir / "healthcheck.sh")],
        cwd=str(scripts_dir.parent),
    )

    # Pipeline start
    logger.success("🚀 AI Inventor - Pipeline Runner")

    # Create shared run directory with project subdirectory
    # Use PIPELINE_TIMESTAMP env var if provided, otherwise generate new timestamp
    timestamp = os.environ.get('PIPELINE_TIMESTAMP')
    if timestamp:
        logger.info(f"📅 Using provided run name: {timestamp}")
    else:
        # Check for run_name in config
        run_name = (config.init.run_name or '').strip()

        if run_name:
            # Use custom run name
            output_base = config.init.outputs_directory
            base_run_dir = Path(f"{output_base}/{run_name}")

            # Check if folder exists, if so append unique 3-letter ID
            if base_run_dir.exists():
                # Generate unique 3-letter ID
                attempts = 0
                max_attempts = 1000
                while attempts < max_attempts:
                    unique_id = ''.join(random.choices(string.ascii_lowercase, k=3))
                    timestamp = f"{run_name}_{unique_id}"
                    test_dir = Path(f"{output_base}/{timestamp}")
                    if not test_dir.exists():
                        break
                    attempts += 1

                if attempts >= max_attempts:
                    logger.error(f"❌ Failed to generate unique run name after {max_attempts} attempts")
                    return 1

                logger.info(f"📅 Using run name with unique ID: {timestamp} (folder '{run_name}' already exists)")
            else:
                timestamp = run_name
                logger.info(f"📅 Using run name: {timestamp}")
        else:
            # Fallback: generate run name with filesystem-safe timestamp (no colons)
            iso_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp = f"run__{iso_timestamp}"
            logger.info(f"📅 Generated new run name: {timestamp}")

    output_base = config.init.outputs_directory
    # Resolve relative outputs_directory against aii_pipeline/ directory (where config.yaml lives)
    if not Path(output_base).is_absolute():
        config_dir = Path(__file__).parent.parent.parent  # aii_pipeline/
        output_base = str(config_dir / output_base)
    run_dir = Path(f"{output_base}/{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up run-specific HuggingFace cache directory
    hf_cache_dir = run_dir / ".hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(hf_cache_dir)
    os.environ['HF_DATASETS_CACHE'] = str(hf_cache_dir / 'datasets')
    os.environ['TRANSFORMERS_CACHE'] = str(hf_cache_dir / 'transformers')

    logger.info(f"🗂️ Run-specific HuggingFace cache in {rel_path(run_dir)}")

    # Create workspace directory for AI model's working space
    workspace_dir = run_dir / "model_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Created run directory: {rel_path(run_dir)}")
    logger.info(f"📂 Created model workspace directory: {rel_path(workspace_dir)}")

    # Run the pipeline
    try:
        result = await run_pipeline(config, run_dir=run_dir, workspace_dir=workspace_dir)

        if result:
            logger.success("🎉 Pipeline completed successfully!")
            exit_code = 0
        else:
            logger.error("💥 Pipeline failed!")
            exit_code = 1
    finally:
        # Always clean up all caches at the end
        cleanup_cache(run_dir)

    return exit_code


def cli_main():
    """Synchronous entry point for console script."""
    exit_code = asyncio.run(main())
    exit(exit_code)


if __name__ == "__main__":
    cli_main()
