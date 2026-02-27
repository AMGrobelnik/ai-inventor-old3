"""Two Pools for gen_paper_repo - Figure and Demo.

Each pool manages an in-memory collection of items with:
- Query/filter methods
- save(path) / load(path) for single-JSON persistence

Follows the same pattern as _invention_loop/pools.py.

BasePool lives in aii_lib.pools.
"""

import json
from pathlib import Path

from aii_lib.pools import BasePool
from pydantic import TypeAdapter

from aii_pipeline.prompts.steps._4_gen_paper_repo._2a_viz_gen.schema import Figure
from aii_pipeline.prompts.steps._4_gen_paper_repo._1b_artifact_demos.schema_code import (
    BaseDemo, AnyDemo,
)


# ============================================================================
# Figure Pool
# ============================================================================

class FigurePool(BasePool[Figure]):
    """Pool of figures for paper generation."""

    _item_key = "figures"
    _item_cls = Figure

    def add(self, figure: Figure) -> Figure:
        """Add a figure to the pool."""
        self._items.append(figure)
        return figure

    def add_many(self, figures: list[Figure]) -> list[Figure]:
        """Add multiple figures to the pool."""
        self._items.extend(figures)
        return figures


# ============================================================================
# Demo Pool
# ============================================================================

class DemoPool(BasePool[BaseDemo]):
    """Pool of demos (discriminated union: CodeDemo, LeanDemo, MarkdownDemo)."""

    _item_key = "demos"
    _item_cls = BaseDemo
    _demo_adapter = TypeAdapter(AnyDemo)

    def add(self, demo: BaseDemo) -> BaseDemo:
        """Add a demo to the pool."""
        self._items.append(demo)
        return demo

    @classmethod
    def load(cls, path: Path):
        """Load demos with discriminated union for correct subclass."""
        pool = cls()
        path = Path(path)
        if not path.exists():
            return pool
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pool._items = [
            cls._demo_adapter.validate_python(item)
            for item in data.get(cls._item_key, [])
        ]
        pool._load_extra(data)
        return pool
