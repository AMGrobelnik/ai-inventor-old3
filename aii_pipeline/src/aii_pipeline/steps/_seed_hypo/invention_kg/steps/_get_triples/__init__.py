"""Triple extraction module."""
from typing import Optional

from aii_lib import AIITelemetry

from .get_triple import get_triples_for_paper
from .logging import setup_logging, set_telemetry as set_logging_telemetry
from .resume import get_completed_paper_indices, set_telemetry as set_resume_telemetry
from .display import (
    create_progress_tracker,
    print_summary,
    print_header,
    print_completion,
    console,
)
from .config import (
    load_pipeline_config,
    load_agent_config,
    load_papers_from_directory,
    validate_paths,
    set_telemetry as set_config_telemetry,
)


def set_telemetry(telemetry: Optional[AIITelemetry]) -> None:
    """Set telemetry instance for all submodules."""
    set_logging_telemetry(telemetry)
    set_resume_telemetry(telemetry)
    set_config_telemetry(telemetry)


__all__ = [
    "get_triples_for_paper",
    "setup_logging",
    "get_completed_paper_indices",
    "create_progress_tracker",
    "print_summary",
    "print_header",
    "print_completion",
    "console",
    "load_pipeline_config",
    "load_agent_config",
    "load_papers_from_directory",
    "validate_paths",
    "set_telemetry",
]
