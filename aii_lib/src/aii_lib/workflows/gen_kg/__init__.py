"""Knowledge graph generation workflow with Wikipedia URL verification.

Generates knowledge graph triples from research papers with URL verification and retry.
"""

from .workflow import (
    GenKGConfig,
    GenKGResult,
    generate_kg_triples,
)
from .verify import verify_wikipedia_urls

__all__ = [
    "GenKGConfig",
    "GenKGResult",
    "generate_kg_triples",
    "verify_wikipedia_urls",
]
