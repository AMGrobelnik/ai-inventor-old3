"""Four Pools for invention loop - Strategy, Plan, Artifact, and Narrative.

Each pool manages an in-memory collection of items with:
- Query/filter methods
- save(path) / load(path) for single-JSON persistence

Pools are purely in-memory during the loop. Snapshots are saved to
iter_N/pools/ at each iteration checkpoint.

BasePool, parse_iteration, get_type_abbrev live in aii_lib.pools.
"""

import json
from pathlib import Path

from aii_lib.pools import BasePool, parse_iteration, get_type_abbrev  # noqa: F401
from aii_lib.agent_backend import ExpectedFile
from pydantic import TypeAdapter

from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.schema import Strategy
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.schema import (
    BasePlan, PlanType, AnyPlan,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.schema import BaseArtifact, ArtifactType
from aii_pipeline.prompts.steps._3_invention_loop._4_gen_narr.schema import Narrative


# ============================================================================
# Strategy Pool
# ============================================================================

class StrategyPool(BasePool[Strategy]):
    """Pool of strategies across iterations."""

    _item_key = "strategies"
    _item_cls = Strategy

    def add(self, strategy: Strategy) -> Strategy:
        """Add a strategy to the pool."""
        self._items.append(strategy)
        return strategy

    def add_many(self, strategies: list[Strategy]) -> list[Strategy]:
        """Add multiple strategies to the pool."""
        self._items.extend(strategies)
        return strategies

    def get_last_iter_idx(self) -> int:
        if not self._items:
            return 0
        return max(parse_iteration(s.id) for s in self._items)


# ============================================================================
# Plan Pool
# ============================================================================

class PlanPool(BasePool[BasePlan]):
    """Pool of plans (pending work)."""

    _item_key = "plans"
    _item_cls = BasePlan
    _plan_adapter = TypeAdapter(AnyPlan)

    def add(self, plan: BasePlan) -> BasePlan:
        """Add a plan to the pool. ID must already be set."""
        self._items.append(plan)
        return plan

    def get_by_type(self, ptype: PlanType) -> list[BasePlan]:
        return [p for p in self._items if p.type == ptype]

    @classmethod
    def load(cls, path: Path):
        """Load plans using discriminated union for correct subclass."""
        pool = cls()
        path = Path(path)
        if not path.exists():
            return pool
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pool._items = [
            cls._plan_adapter.validate_python(item)
            for item in data.get(cls._item_key, [])
        ]
        pool._load_extra(data)
        return pool


# ============================================================================
# Artifact Pool
# ============================================================================

class ArtifactPool(BasePool[BaseArtifact]):
    """Pool of completed artifacts (only successful ones stored)."""

    _item_key = "artifacts"
    _item_cls = BaseArtifact

    def add(
        self,
        id: str,
        plan: BasePlan,
        title: str = "",
        summary: str = "",
        workspace_path: str | None = None,
        out_expected_files: list[str] | None = None,
        out_demo_files: list[ExpectedFile] | None = None,
        out_dependency_files: dict[str, str | list[str] | None] | None = None,
    ) -> BaseArtifact:
        """Add an artifact to the pool. ID is the task_id from the executor."""
        artifact_type = ArtifactType(plan.type.value)

        artifact = BaseArtifact(
            id=id,
            type=artifact_type,
            title=title or plan.title,
            in_plan_id=plan.id,
            in_dependency_artifact_ids=list(plan.artifact_dependencies),
            summary=summary,
            workspace_path=workspace_path,
            out_expected_files=out_expected_files or [],
            out_demo_files=out_demo_files or [],
            out_dependency_files=out_dependency_files or {},
        )

        self._items.append(artifact)
        return artifact

    def get_by_type(self, atype: ArtifactType) -> list[BaseArtifact]:
        return [a for a in self._items if a.type == atype]

    def get_by_ids(self, ids: list[str]) -> list[BaseArtifact]:
        """Return artifacts matching the given IDs, in the order given."""
        lookup = {a.id: a for a in self._items}
        return [lookup[i] for i in ids if i in lookup]

    def get_prompt(
        self,
        *,
        ids: list[str] | None = None,
        include: set[str] | None = None,
        label: str = "Artifact",
    ) -> str:
        """Return filtered artifacts as prompt YAML.

        Args:
            ids: Only include these artifact IDs. None = all.
            include: Only include these fields per artifact. None = LLMPrompt-marked fields.
            label: Label for each YAML block.
        """
        from aii_lib.prompts import LLMPromptModel
        items = self.get_by_ids(ids) if ids is not None else self.get_all()
        if not items:
            return ""
        return LLMPromptModel.list_to_prompt_yaml(
            items,
            label=label,
            include=include,
            strip_nulls=True,
        )

    def _enrich_save_data(self, data: dict):
        data["by_type"] = {
            atype.value: len([a for a in self._items if a.type == atype])
            for atype in ArtifactType
        }


# ============================================================================
# Narrative Pool
# ============================================================================

class NarrativePool(BasePool[Narrative]):
    """Pool of research narratives across iterations."""

    _item_key = "narratives"
    _item_cls = Narrative

    def add(
        self,
        id: str,
        narrative: str,
        summary: str,
        artifacts_used: list[str],
        gaps: list[str],
        title: str = "",
    ) -> Narrative:
        """Add a narrative to the pool. ID is the task_id from the generator."""
        narr = Narrative(
            id=id,
            title=title,
            narrative=narrative,
            summary=summary,
            artifacts_used=artifacts_used,
            gaps=gaps,
        )
        self._items.append(narr)
        return narr

    def get_last_iter_idx(self) -> int:
        if not self._items:
            return 0
        return max(parse_iteration(n.id) for n in self._items)


# ============================================================================
# Helper: Save/Load all pools at once
# ============================================================================

def save_all_pools(
    pools_dir: Path,
    strategy_pool: StrategyPool,
    plan_pool: PlanPool,
    artifact_pool: ArtifactPool,
    narrative_pool: NarrativePool,
):
    """Save all four pools to a directory as individual JSON files."""
    pools_dir = Path(pools_dir)
    pools_dir.mkdir(parents=True, exist_ok=True)
    strategy_pool.save(pools_dir / "strategy_pool.json")
    plan_pool.save(pools_dir / "plan_pool.json")
    artifact_pool.save(pools_dir / "artifact_pool.json")
    narrative_pool.save(pools_dir / "narrative_pool.json")


def load_all_pools(pools_dir: Path) -> tuple[StrategyPool, PlanPool, ArtifactPool, NarrativePool]:
    """Load all four pools from a directory."""
    pools_dir = Path(pools_dir)
    return (
        StrategyPool.load(pools_dir / "strategy_pool.json"),
        PlanPool.load(pools_dir / "plan_pool.json"),
        ArtifactPool.load(pools_dir / "artifact_pool.json"),
        NarrativePool.load(pools_dir / "narrative_pool.json"),
    )
