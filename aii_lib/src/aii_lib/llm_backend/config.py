"""Configuration loader for LLM Clients."""

import os
from pathlib import Path
import yaml
from aii_lib.telemetry import logger

_config = None


def load_config(config_path: str | Path | None = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, searches for config.yaml in:
            1. Current working directory
            2. llm_clients package directory

    Returns:
        Configuration dictionary
    """
    global _config

    if config_path:
        path = Path(config_path)
    else:
        # Search order
        search_paths = [
            Path(__file__).parent / "default_config.yaml",  # src/llm_clients/default_config.yaml (package default)
            Path.cwd() / "llm_backend" / "default_config.yaml",
            Path.cwd() / "default_config.yaml",
        ]

        path = None
        for p in search_paths:
            if p.exists():
                path = p
                break

        if not path:
            logger.warning("No default_config.yaml found")
            return {}

    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}

    with open(path, 'r') as f:
        _config = yaml.safe_load(f)

    logger.info(f"Loaded config from: {path}")
    return _config


def get_config() -> dict:
    """Get cached config or load it."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_openai_config() -> dict:
    """Get OpenAI-specific configuration."""
    config = get_config()
    return config.get('openai', {})


def get_anthropic_config() -> dict:
    """Get Anthropic-specific configuration."""
    config = get_config()
    return config.get('anthropic', {})


def get_gemini_config() -> dict:
    """Get Gemini-specific configuration."""
    config = get_config()
    return config.get('gemini', {})


def get_openrouter_config() -> dict:
    """Get OpenRouter-specific configuration."""
    config = get_config()
    return config.get('openrouter', {})
