"""AgentInitializer - Pre-agent setup and initialization utilities."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from aii_lib.telemetry import AIITelemetry


class AgentInitializer:
    """
    Handles pre-agent setup and initialization.

    Provides utilities for:
    - Workspace setup (create dirs, copy templates)
    - Dependency copying (copy dependency workspaces)
    - Dependency prompt generation (build markdown context)
    - Server health checks
    - AIITelemetry task registration
    """

    def __init__(
        self,
        telemetry: "AIITelemetry | None" = None,
        task_id: str | None = None,
        task_name: str | None = None,
    ):
        """
        Initialize the AgentInitializer.

        Args:
            telemetry: AIITelemetry instance for logging
            task_id: Task ID for telemetry sequencing
            task_name: Task name for display in logs
        """
        self.telemetry = telemetry
        self.task_id = task_id
        self.task_name = task_name

    def setup_workspace(
        self,
        workspace_dir: Path,
        template_dir: Path | None = None,
    ) -> Path:
        """
        Set up the agent workspace directory.

        If workspace exists, emits warning but doesn't delete.
        Creates directory if needed, copies template if provided.

        Args:
            workspace_dir: Target workspace directory
            template_dir: Optional template directory to copy from

        Returns:
            The workspace directory path
        """
        workspace_dir = Path(workspace_dir)

        if workspace_dir.exists():
            self._emit(f"Workspace already exists: {workspace_dir}", "WARN")
        else:
            workspace_dir.mkdir(parents=True, exist_ok=True)

        # Copy template if provided and exists
        if template_dir and Path(template_dir).exists():
            for item in Path(template_dir).iterdir():
                dest = workspace_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

        return workspace_dir

    def copy_dependencies(
        self,
        dependencies: list[Any],
        workspace_dir: Path,
        get_workspace_path: Callable[[Any], str | None] | None = None,
        get_id: Callable[[Any], str] | None = None,
        get_type: Callable[[Any], str] | None = None,
        get_title: Callable[[Any], str] | None = None,
        get_summary: Callable[[Any], str] | None = None,
    ) -> list[dict]:
        """
        Copy dependency artifact workspaces into current workspace.

        Args:
            dependencies: List of dependency artifacts (objects or dicts)
            workspace_dir: Target workspace directory
            get_workspace_path: Function to get workspace path from artifact
                               Default: artifact.result.get("workspace_path")
            get_id: Function to get artifact ID from artifact
                   Default: getattr(artifact, 'id', 'unknown')
            get_type: Function to get artifact type from artifact
                     Default: getattr(artifact, 'type', None)
            get_title: Function to get title from artifact
                      Default: reads from artifact_title.txt in workspace
            get_summary: Function to get summary from artifact
                        Default: reads from artifact_metadata.json in workspace

        Returns:
            List of copied dependency info dicts with keys:
            - id: Artifact ID
            - type: Artifact type
            - title: Artifact title
            - summary: Artifact summary (for research artifacts)
            - local_path: Relative path in workspace (e.g., "./dependencies/folder_name")
        """
        deps_dir = Path(workspace_dir) / "dependencies"
        deps_dir.mkdir(parents=True, exist_ok=True)

        copied_deps = []
        for artifact in dependencies:
            # Get artifact ID (for logging)
            if get_id:
                artifact_id = get_id(artifact)
            else:
                artifact_id = getattr(artifact, 'id', 'unknown')

            # Get workspace path
            if get_workspace_path:
                source_path = get_workspace_path(artifact)
            else:
                source_path = artifact.result.get("workspace_path") if hasattr(artifact, 'result') else None

            if not source_path:
                self._emit(f"Skipping dependency {artifact_id}: no workspace path", "WARN")
                continue

            source_workspace = Path(source_path)
            if not source_workspace.exists():
                self._emit(f"Skipping dependency: workspace not found at {source_path}", "WARN")
                continue

            # Get title (for metadata/display, NOT for folder name)
            if get_title:
                title = get_title(artifact)
            else:
                title = self._read_artifact_title(source_workspace) or artifact_id

            # Use artifact ID for folder name (not title) to avoid long sanitized names
            folder_name = artifact_id

            # Copy workspace
            target_dir = deps_dir / folder_name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_workspace, target_dir)

            # Get summary
            if get_summary:
                summary = get_summary(artifact)
            else:
                summary = self._read_artifact_summary(source_workspace)

            # Get artifact type
            if get_type:
                artifact_type = get_type(artifact)
            else:
                artifact_type = getattr(artifact, 'type', None)
                if hasattr(artifact_type, 'value'):
                    artifact_type = artifact_type.value

            # Build dependency info
            copied_deps.append({
                "id": artifact_id,
                "type": str(artifact_type) if artifact_type else "unknown",
                "title": title,
                "summary": summary,
                "local_path": f"./dependencies/{folder_name}",
                "is_research": str(artifact_type).lower() == "research" if artifact_type else False,
            })

        self._emit(f"Copied {len(copied_deps)} dependencies to {deps_dir}")
        return copied_deps

    def gen_dependency_prompt(self, copied_deps: list[dict]) -> str:
        """
        Generate markdown prompt section from copied dependencies.

        Args:
            copied_deps: List from copy_dependencies()

        Returns:
            Markdown string describing dependencies and their paths
        """
        if not copied_deps:
            return ""

        lines = [
            "## Dependencies\n",
            "Each path below is a **folder** containing the artifact's workspace.",
            "Use `ls <folder>/` to explore contents and find data files.\n",
        ]
        for dep in copied_deps:
            dep_type = dep.get('type', 'unknown').upper()
            title = dep.get('title', dep.get('id', 'unknown'))
            lines.append(f"### [{dep.get('id', 'unknown')}] {dep_type}: {title}")
            lines.append(f"**Folder:** `{dep.get('local_path', 'unknown')}/`")

            # Include full summary for research artifacts
            if dep.get('is_research') and dep.get('summary'):
                lines.append(f"\n{dep['summary']}")

            lines.append("")  # Blank line between deps

        return "\n".join(lines)

    def ensure_servers(
        self,
        servers: list[dict] | None = None,
    ) -> dict[str, bool]:
        """
        Ensure required servers are running.

        Args:
            servers: List of server configs, each with keys:
                    - name: Server name
                    - command: Command to run
                    - port: Port to check
                    - working_dir: Working directory
                    - log_file: Log file path
                    - venv_activate: Optional venv activate script
                    - kill_pattern: Optional pattern for pkill

        Returns:
            Dict of {server_name: is_running}
        """
        from aii_lib.utils import ensure_server_running

        if not servers:
            return {}

        results = {}
        for server in servers:
            name = server.get("name", "unknown")
            try:
                is_running = ensure_server_running(
                    name=name,
                    command=server.get("command", ""),
                    port=server.get("port", 8000),
                    working_dir=Path(server.get("working_dir", ".")),
                    log_file=Path(server.get("log_file", f"/tmp/{name}.log")),
                    venv_activate=server.get("venv_activate"),
                    kill_pattern=server.get("kill_pattern"),
                    log_func=lambda msg: self._emit(msg),
                )
                results[name] = is_running
            except Exception as e:
                self._emit(f"Failed to start server {name}: {e}", "ERROR")
                raise

        return results

    def start_task(self, sequence: int | None = None) -> None:
        """
        Register task start with telemetry.

        Must be called before any emit_message calls for proper sequencing.

        Args:
            sequence: Optional sequence number for ordering
        """
        if self.telemetry and self.task_id and self.task_name:
            self.telemetry.emit_task_start(self.task_id, self.task_name, sequence=sequence)

    # === Private helpers ===

    def _emit(self, message: str, level: str = "INFO") -> None:
        """Emit message to telemetry if available."""
        if self.telemetry and self.task_id and self.task_name:
            self.telemetry.emit_message(level, message, self.task_name, self.task_id)

    def _read_artifact_title(self, workspace_dir: Path) -> str:
        """Read title from artifact_title.txt."""
        title_path = workspace_dir / "artifact_title.txt"
        if title_path.exists():
            return title_path.read_text(encoding="utf-8").strip()
        return ""

    def _read_artifact_summary(self, workspace_dir: Path) -> str:
        """Read summary from artifact_metadata.json."""
        import json
        metadata_path = workspace_dir / "artifact_metadata.json"
        if metadata_path.exists():
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                return data.get("summary", "")
            except (json.JSONDecodeError, IOError) as e:
                self._emit(f"Failed to read artifact metadata from {metadata_path}: {e}", "ERROR")
                raise
        return ""


