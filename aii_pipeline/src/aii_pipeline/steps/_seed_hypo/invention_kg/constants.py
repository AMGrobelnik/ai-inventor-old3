#!/usr/bin/env python3
"""
Constants for Invention Knowledge Graph Pipeline.

This module defines all magic strings, directory names, and default values
used throughout the pipeline to ensure consistency and maintainability.

Note: "Triples" refers to the extracted (paper, relation, concept) tuples
from research papers. Relations include: uses (existing work), proposes (novel contributions).
"""

from pathlib import Path

# ============================================================================
# Base Directory
# ============================================================================
# Base directory for pipeline runs (aii_pipeline/runs/)
# Structure: runs/<run_id>/1_seed_hypo/<step_name>/
RUNS_DIR = Path(__file__).parents[5] / "runs"  # aii_pipeline/runs/
SEED_HYPO_SUBDIR = "1_seed_hypo"  # Subfolder within each run for this module

# Legacy: local base dir for module-level files (prompts, configs)
BASE_DIR = Path(__file__).parent.resolve()  # invention_kg/

# ============================================================================
# Step Directory Names
# ============================================================================
# Directory names for each pipeline step's output
STEP_1_SEL_TOPICS = '_1_sel_topics'        # Selected topics from OpenAlex
STEP_2_PAPERS = '_2_papers'                 # Raw papers from OpenAlex
STEP_3_PAPERS_CLEAN = '_3_papers_clean'     # Cleaned paper data
STEP_4_TRIPLES = '_4_triples'               # Triple extraction runs
STEP_5_WIKIDATA = '_5_wikidata'             # Triples enriched with Wikidata
STEP_6_PAPER_TRIPLES = '_6_paper_triples'   # Combined papers + enriched triples
STEP_7_HYPO_SEEDS = '_7_hypo_seeds'         # Hypothesis seeds (blind spots, breakthroughs)
STEP_8_SEED_PROMPT = '_8_seed_prompt'       # Generated seed prompts from blind spots
STEP_9_GRAPHS = '_9_graphs'                 # All graph types


# ============================================================================
# Graph Paths
# ============================================================================
# Base directory for graph files
GRAPH_BASE_DIR = 'data/_9_graphs'

# Graph subdirectories
GRAPH_COOCCURRENCE_DIR = 'cooccurrence'
GRAPH_ONTOLOGY_DIR = 'ontology'
GRAPH_BIPARTITE_DIR = 'bipartite'
GRAPH_RELATIONS_DIR = 'relations'
GRAPH_DERIVED_DIR = 'derived'

# Default graph files
DEFAULT_COOCCURRENCE_FILE = f'{GRAPH_BASE_DIR}/{GRAPH_COOCCURRENCE_DIR}/all.json'
DEFAULT_ONTOLOGY_FILE = f'{GRAPH_BASE_DIR}/{GRAPH_ONTOLOGY_DIR}/full.json'
DEFAULT_CROSS_TOPIC_FILE = f'{GRAPH_BASE_DIR}/{GRAPH_DERIVED_DIR}/cross_topic.json'

# ============================================================================
# Agent Configuration Defaults
# ============================================================================
DEFAULT_AGENT_CWD_TEMPLATE = 'agent_cwd'
DEFAULT_AGENT_CONFIG_FILE = 'prompts/triples_config.yaml'

# Processing defaults
DEFAULT_MAX_PAPERS = -1  # -1 means process all papers
DEFAULT_MAX_CONCURRENT = 10
DEFAULT_STAGGER_DELAY = 2  # seconds

# ============================================================================
# Visualization Defaults
# ============================================================================
DEFAULT_VIZ_DIR = 'steps/_10_viz_graphs'
DEFAULT_VIZ_PORT = 8000

# ============================================================================
# Logging Defaults
# ============================================================================
DEFAULT_LOG_LEVEL = 'DEBUG'
DEFAULT_LOG_DIR = 'log/logs'
DEFAULT_LOG_ROTATION = '100 MB'

# ============================================================================
# ANSI Color Codes
# ============================================================================
class Colors:
    """ANSI color codes for terminal output."""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    END = "\033[0m"

    @classmethod
    def blue(cls, text: str) -> str:
        """Wrap text in blue color."""
        return f"{cls.BLUE}{text}{cls.END}"

    @classmethod
    def green(cls, text: str) -> str:
        """Wrap text in green color."""
        return f"{cls.GREEN}{text}{cls.END}"

    @classmethod
    def yellow(cls, text: str) -> str:
        """Wrap text in yellow color."""
        return f"{cls.YELLOW}{text}{cls.END}"

    @classmethod
    def cyan(cls, text: str) -> str:
        """Wrap text in cyan color."""
        return f"{cls.CYAN}{text}{cls.END}"

    @classmethod
    def red(cls, text: str) -> str:
        """Wrap text in red color."""
        return f"{cls.RED}{text}{cls.END}"
