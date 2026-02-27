"""Pool utilities for managing collections of items.

Modules:
    base_pool - Generic BasePool class with save/load and prompt formatting
"""

from .base_pool import (
    BasePool,
    TYPE_ABBREVS,
    get_type_abbrev,
    parse_iteration,
)

__all__ = [
    "BasePool",
    "TYPE_ABBREVS",
    "get_type_abbrev",
    "parse_iteration",
]
