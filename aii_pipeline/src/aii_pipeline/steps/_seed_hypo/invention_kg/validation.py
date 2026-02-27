#!/usr/bin/env python3
"""
Configuration validation for Building Blocks Knowledge Graph Pipeline.

This module validates config.yaml parameters to catch errors early
before pipeline execution begins.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from aii_pipeline.steps._seed_hypo.invention_kg.constants import (
    DEFAULT_MAX_PAPERS,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_STAGGER_DELAY,
    DEFAULT_VIZ_PORT
)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_year_range(year_range: Dict[str, int]) -> List[str]:
    """
    Validate year_range configuration.

    Args:
        year_range: Dictionary with 'start' and 'end' keys

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    current_year = datetime.now().year

    if 'start' not in year_range:
        errors.append("year_range.start is required")
        return errors

    if 'end' not in year_range:
        errors.append("year_range.end is required")
        return errors

    start = year_range['start']
    end = year_range['end']

    # Validate range
    if not isinstance(start, int) or not isinstance(end, int):
        errors.append(f"year_range must contain integers, got start={type(start).__name__}, end={type(end).__name__}")
        return errors

    if start > end:
        errors.append(f"year_range.start ({start}) must be <= year_range.end ({end})")

    if start < 1900:
        errors.append(f"year_range.start ({start}) is too early (minimum: 1900)")

    if end > current_year + 1:
        errors.append(f"year_range.end ({end}) is in the future (maximum: {current_year + 1})")

    if end - start > 100:
        errors.append(f"year_range span ({end - start} years) is too large (maximum: 100 years)")

    return errors


def validate_get_papers_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate get_papers section of config.

    Args:
        config: get_papers configuration dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Validate email
    if 'email' not in config or not config['email']:
        errors.append("get_papers.email is required for OpenAlex API")
    elif '@' not in config['email']:
        errors.append(f"get_papers.email appears invalid: {config['email']}")

    # Validate year_range
    if 'year_range' in config:
        errors.extend(validate_year_range(config['year_range']))
    else:
        errors.append("get_papers.year_range is required")

    # Validate papers_per_year
    papers_per_year = config.get('papers_per_year')
    if papers_per_year is None:
        errors.append("get_papers.papers_per_year is required")
    elif not isinstance(papers_per_year, int):
        errors.append(f"get_papers.papers_per_year must be integer, got {type(papers_per_year).__name__}")
    elif papers_per_year <= 0:
        errors.append(f"get_papers.papers_per_year must be > 0, got {papers_per_year}")
    elif papers_per_year > 10000:
        errors.append(f"get_papers.papers_per_year is very large ({papers_per_year}), consider using <= 10000")

    # Validate sort_by
    valid_sort_options = ['cited_by_count', 'publication_date', 'relevance_score']
    sort_by = config.get('sort_by')
    if sort_by and sort_by not in valid_sort_options:
        errors.append(f"get_papers.sort_by must be one of {valid_sort_options}, got '{sort_by}'")

    return errors


def validate_get_triples_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate get_triples section of config.

    Args:
        config: get_triples configuration dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Validate max_papers
    max_papers = config.get('max_papers', DEFAULT_MAX_PAPERS)
    if not isinstance(max_papers, int):
        errors.append(f"get_triples.max_papers must be integer, got {type(max_papers).__name__}")
    elif max_papers == 0:
        errors.append("get_triples.max_papers cannot be 0 (use -1 for unlimited)")
    elif max_papers < -1:
        errors.append(f"get_triples.max_papers must be >= -1, got {max_papers}")

    # Validate max_concurrent_agents
    max_concurrent = config.get('max_concurrent_agents', DEFAULT_MAX_CONCURRENT)
    if not isinstance(max_concurrent, int):
        errors.append(f"get_triples.max_concurrent_agents must be integer, got {type(max_concurrent).__name__}")
    elif max_concurrent <= 0:
        errors.append(f"get_triples.max_concurrent_agents must be > 0, got {max_concurrent}")
    elif max_concurrent > 50:
        errors.append(f"get_triples.max_concurrent_agents is very high ({max_concurrent}), may overwhelm system")

    # Validate stagger_delay
    stagger_delay = config.get('stagger_delay', DEFAULT_STAGGER_DELAY)
    if not isinstance(stagger_delay, (int, float)):
        errors.append(f"get_triples.stagger_delay must be numeric, got {type(stagger_delay).__name__}")
    elif stagger_delay < 0:
        errors.append(f"get_triples.stagger_delay must be >= 0, got {stagger_delay}")

    # Validate paths exist (warnings, not errors)
    agent_cwd_template = config.get('agent_cwd_template')
    if agent_cwd_template:
        # Path is relative to invention_kg/, will be validated at runtime
        pass

    config_file = config.get('config_file')
    if config_file:
        # Path is relative to invention_kg/, will be validated at runtime
        pass

    return errors


def validate_viz_graph_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate viz_graph section of config.

    Args:
        config: viz_graph configuration dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Validate port
    port = config.get('port', DEFAULT_VIZ_PORT)
    if not isinstance(port, int):
        errors.append(f"viz_graph.port must be integer, got {type(port).__name__}")
    elif port < 1024:
        errors.append(f"viz_graph.port ({port}) requires root privileges, use >= 1024")
    elif port > 65535:
        errors.append(f"viz_graph.port ({port}) is invalid, must be <= 65535")

    return errors


def validate_config(config_dict: Dict[str, Any]) -> List[str]:
    """
    Validate entire pipeline configuration.

    Args:
        config_dict: Full configuration dictionary

    Returns:
        List of error messages (empty if valid)

    Raises:
        ConfigValidationError: If validation fails
    """
    all_errors = []

    # Validate resume configuration
    resume = config_dict.get('resume')
    if resume is not None and not isinstance(resume, bool):
        all_errors.append(f"resume must be boolean, got {type(resume).__name__}")

    run_id = config_dict.get('run_id')
    if run_id and not isinstance(run_id, str):
        all_errors.append(f"run_id must be string, got {type(run_id).__name__}")

    # Validate each section
    if 'get_papers' in config_dict:
        all_errors.extend(validate_get_papers_config(config_dict['get_papers']))

    if 'get_triples' in config_dict:
        all_errors.extend(validate_get_triples_config(config_dict['get_triples']))

    if 'viz_graph' in config_dict:
        all_errors.extend(validate_viz_graph_config(config_dict['viz_graph']))

    return all_errors


def validate_and_raise(config_dict: Dict[str, Any]) -> None:
    """
    Validate configuration and raise exception if invalid.

    Args:
        config_dict: Full configuration dictionary

    Raises:
        ConfigValidationError: If validation fails with detailed error messages
    """
    errors = validate_config(config_dict)
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  ❌ {err}" for err in errors)
        raise ConfigValidationError(error_msg)
