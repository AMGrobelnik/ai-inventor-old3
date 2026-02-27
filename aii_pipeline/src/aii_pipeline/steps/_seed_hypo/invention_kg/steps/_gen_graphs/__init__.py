#!/usr/bin/env python3
"""
Graph generation helper modules.

This package provides helper functions for _9_gen_graphs.py
"""

from typing import Optional
from aii_lib import AIITelemetry

from .load_papers import load_all_papers, load_papers_by_year
from ._cooccurrence import generate_cooccurrence_graph
from ._ontology import generate_ontology_graph
from ._semantic import generate_semantic_graph
from ._blind_spots import generate_blind_spots_graph

# Import set_telemetry from each submodule
from .load_papers import set_telemetry as set_load_papers_telemetry
from ._cooccurrence import set_telemetry as set_cooccurrence_telemetry
from ._ontology import set_telemetry as set_ontology_telemetry
from ._semantic import set_telemetry as set_semantic_telemetry
from ._blind_spots import set_telemetry as set_blind_spots_telemetry
from ._common import set_telemetry as set_common_telemetry
from ._paper_concept import set_telemetry as set_paper_concept_telemetry
from ._relation_typed import set_telemetry as set_relation_typed_telemetry


def set_telemetry(telemetry: Optional[AIITelemetry]) -> None:
    """Set telemetry instance for all submodules."""
    set_load_papers_telemetry(telemetry)
    set_cooccurrence_telemetry(telemetry)
    set_ontology_telemetry(telemetry)
    set_semantic_telemetry(telemetry)
    set_blind_spots_telemetry(telemetry)
    set_common_telemetry(telemetry)
    set_paper_concept_telemetry(telemetry)
    set_relation_typed_telemetry(telemetry)


__all__ = [
    "load_all_papers",
    "load_papers_by_year",
    "generate_cooccurrence_graph",
    "generate_ontology_graph",
    "generate_semantic_graph",
    "generate_blind_spots_graph",
    "set_telemetry",
]
