#!/usr/bin/env python3
"""
Configuration manager for Building Blocks Knowledge Graph pipeline.
"""

from pathlib import Path
from typing import Dict, Any

from aii_pipeline.steps._seed_hypo.invention_kg.log import setup_logging


class Config:
    """Configuration manager for the pipeline."""

    def __init__(self, config_dict: Dict[str, Any] = None, auto_setup_logging: bool = True):
        """
        Create configuration from dict.

        Args:
            config_dict: Configuration dictionary (defaults to empty dict)
            auto_setup_logging: Automatically setup logging from config (default: True)
        """
        self._config = config_dict or {}

        # Automatically setup logging
        if auto_setup_logging:
            self.setup_logging()

    def get(self, *keys, default=None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            *keys: Keys to traverse (e.g., 'get_papers', 'email')
            default: Default value if key not found

        Returns:
            Configuration value

        Examples:
            config.get('get_papers', 'email')  # Returns email from get_papers section
            config.get('pipeline', 'name')     # Returns pipeline name
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Get top-level configuration section."""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if top-level key exists."""
        return key in self._config

    @property
    def base_dir(self) -> Path:
        """Get base directory for runs (aii_pipeline/runs/)."""
        from .constants import RUNS_DIR
        return RUNS_DIR

    @property
    def module_dir(self) -> Path:
        """Get module directory (invention_kg root)."""
        return Path(__file__).parent

    @property
    def raw_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary."""
        return self._config

    def resolve_path(self, *path_parts: str) -> Path:
        """
        Resolve path relative to base directory.

        Args:
            *path_parts: Path components

        Returns:
            Resolved absolute path
        """
        return self.base_dir.joinpath(*path_parts)

    def setup_logging(self) -> None:
        """Setup logging from configuration."""
        log_config = self.get('logging', default={})

        level = log_config.get('level', 'INFO')
        log_dir = log_config.get('log_dir', 'logs')
        rotation = log_config.get('rotation', '100 MB')

        # Resolve log_dir relative to base directory
        log_dir_path = self.resolve_path(log_dir)

        setup_logging(
            level=level,
            log_dir=log_dir_path,
            rotation=rotation
        )


# Global config instance
_config: Config | None = None


def create_config(config_dict: Dict[str, Any] = None) -> Config:
    """
    Create global configuration from dict.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Config instance
    """
    global _config
    _config = Config(config_dict)
    return _config


def get_config() -> Config:
    """
    Get global configuration instance.

    Returns:
        Config instance

    Raises:
        RuntimeError: If config not yet loaded
    """
    if _config is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _config
