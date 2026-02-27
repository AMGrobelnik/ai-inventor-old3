"""BasePool — generic base class for typed, serializable item pools.

Provides:
- BasePool[T]: In-memory collection with query, save/load, and prompt formatting
- parse_iteration(): Extract iteration number from pool item IDs
- get_type_abbrev(): Abbreviate artifact type names for IDs
- TYPE_ABBREVS: Mapping of full type names to abbreviations

Subclasses in aii_pipeline define concrete pools (StrategyPool, etc.)
with domain-specific schemas.
"""

import re
import json
from datetime import datetime
from pathlib import Path
from typing import TypeVar, Generic, ClassVar

from aii_lib.prompts import LLMPromptModel


# =============================================================================
# ID UTILITIES
# =============================================================================

# Type abbreviations for consistent ID naming across the pipeline
TYPE_ABBREVS = {
    "research": "research",
    "experiment": "exp",
    "dataset": "data",
    "evaluation": "eval",
    "proof": "proof",
}


def get_type_abbrev(type_name: str) -> str:
    """Get abbreviated type name for IDs (e.g., 'experiment' -> 'exp')."""
    return TYPE_ABBREVS.get(type_name, type_name[:4])


def parse_iteration(obj_id: str) -> int:
    """Parse iteration number from an object ID.

    All pool object IDs contain '_it{N}_' or '_it{N}__' pattern.
    Examples:
        'strat_v1_it3__gpt5_idx1' → 3
        'prop_exp_iter1_dir1_v1_it2__opus_idx1' → 2
        'exp_id1_it1__sonnet' → 1
        'narr_v2_it4__gpt5' → 4
    """
    match = re.search(r'_it(\d+)', obj_id)
    return int(match.group(1)) if match else 0


# =============================================================================
# BASE POOL
# =============================================================================

T = TypeVar("T")


class BasePool(Generic[T]):
    """Base class for all invention loop pools.

    Subclasses must set:
    - _item_key: JSON key for serialization (e.g., "strategies")
    - _item_cls: Pydantic model class for deserialization
    """

    _item_key: ClassVar[str]
    _item_cls: ClassVar[type]

    def __init__(self):
        self._items: list[T] = []

    def get_by_id(self, item_id: str) -> T | None:
        for item in self._items:
            if item.id == item_id:
                return item
        return None

    def get_by_iteration(self, iteration: int) -> list[T]:
        return [item for item in self._items if parse_iteration(item.id) == iteration]

    def get_all(self) -> list[T]:
        return self._items.copy()

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total": len(self._items),
            self._item_key: [item.model_dump() for item in self._items],
            "saved_at": datetime.now().isoformat(),
        }
        self._enrich_save_data(data)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _enrich_save_data(self, data: dict):
        """Override to add extra fields to save output."""

    @classmethod
    def load(cls, path: Path):
        pool = cls()
        path = Path(path)
        if not path.exists():
            return pool
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pool._items = [cls._item_cls.model_validate(item) for item in data.get(cls._item_key, [])]
        pool._load_extra(data)
        return pool

    def _load_extra(self, data: dict):
        """Override to load extra fields from saved data."""

    def get_all_prompt(self, *, include: set[str] | None = None, label: str = "Item") -> str:
        """Return all items formatted as YAML for LLM prompts.

        Args:
            include: Only include these fields per item. None = LLMPrompt-marked fields.
            label: Label for each YAML block.
        """
        items = self.get_all()
        if not items:
            return ""
        return LLMPromptModel.list_to_prompt_yaml(items, label=label, include=include, strip_nulls=True)
