"""AgentFinalizer - Post-agent verification and finalization utilities."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from aii_lib.telemetry import AIITelemetry

# Default file size limit (GitHub's 100MB limit)
MAX_FILE_SIZE_MB = 100


class AgentFinalizer:
    """
    Handles post-agent verification and finalization.

    Provides utilities for:
    - File size checking (with agent retry for oversized files)
    - Metadata reading (summary + title from artifact_metadata.json)
    - Requirements.txt generation from venv
    - AIITelemetry task end signaling
    """

    def __init__(
        self,
        telemetry: "AIITelemetry | None" = None,
        task_id: str | None = None,
        task_name: str | None = None,
    ):
        """
        Initialize the AgentFinalizer.

        Args:
            telemetry: AIITelemetry instance for logging
            task_id: Task ID for telemetry sequencing
            task_name: Task name for display in logs
        """
        self.telemetry = telemetry
        self.task_id = task_id
        self.task_name = task_name

    # === File Size Check ===

    def check_oversized_files(
        self,
        workspace_dir: Path,
        max_size_mb: float = MAX_FILE_SIZE_MB,
    ) -> list[dict]:
        """
        Check for files exceeding size limit in workspace.

        Args:
            workspace_dir: Agent workspace directory
            max_size_mb: Maximum allowed file size in MB (default: 100)

        Returns:
            List of dicts with oversized file info: [{"path": str, "size_mb": float}, ...]
        """
        max_size_bytes = max_size_mb * 1024 * 1024
        oversized = []

        workspace = Path(workspace_dir)
        if not workspace.exists():
            return []

        for filepath in workspace.rglob("*"):
            if filepath.is_file():
                size = filepath.stat().st_size
                if size > max_size_bytes:
                    rel_path = filepath.relative_to(workspace)
                    size_mb = size / (1024 * 1024)
                    oversized.append({
                        "path": str(rel_path),
                        "size_mb": round(size_mb, 2),
                    })

        oversized.sort(key=lambda x: x["size_mb"], reverse=True)
        return oversized

    def get_oversized_files_prompt(
        self,
        oversized_files: list[dict],
        max_size_mb: float = MAX_FILE_SIZE_MB,
    ) -> str:
        """Generate a prompt to tell the agent to reduce file sizes."""
        files_list = "\n".join(
            f"  - {f['path']} ({f['size_mb']:.1f} MB)"
            for f in oversized_files
        )

        return f"""<CRITICAL_ERROR>
Some files in your workspace exceed the {max_size_mb}MB size limit for GitHub deployment.

OVERSIZED FILES:
{files_list}

You MUST reduce these files to under {max_size_mb}MB each. Use ONE of these strategies:

=== STRATEGY 1: SPLIT FILES (PREFERRED) ===
Split large files into smaller parts and update code to read them sequentially.

For data files (JSON, JSONL, CSV, Parquet):
1. Split the file into parts under {max_size_mb}MB each:
   - data.jsonl -> data_part_001.jsonl, data_part_002.jsonl, ...
2. Update ALL code that reads this file to handle the split parts
3. Delete the original large file after splitting

=== STRATEGY 2: COMPRESSION (FALLBACK) ===
Only use if splitting is not feasible (e.g., binary files, model weights).

1. Compress the file with gzip
2. Update ALL code to decompress before use
3. Delete the original uncompressed file

=== REQUIRED: UPDATE AND TEST CODE ===
After applying your chosen strategy, you MUST:

1. Find ALL code files that reference the modified files (use grep/search)
2. Update each file to work with the new format (split parts or compressed)
3. Run the updated code to verify it still works correctly
4. Fix any errors that occur until the code runs successfully

Do NOT skip testing - the code must actually execute without errors.

Start by listing the oversized files with `ls -lh`, then apply the appropriate strategy.
</CRITICAL_ERROR>"""

    async def verify_file_sizes_and_retry(
        self,
        workspace_dir: Path,
        agent: Any,
        max_size_mb: float = MAX_FILE_SIZE_MB,
        get_retry_prompt: Callable[[list[dict]], str] | None = None,
        max_retries: int = 2,
    ) -> tuple[list[dict], float]:
        """
        Check for oversized files and retry agent if needed.

        Args:
            workspace_dir: Agent workspace directory
            agent: Agent instance with async run() method
            max_size_mb: Maximum allowed file size in MB (default: 100)
            get_retry_prompt: Function that takes oversized files list and returns retry prompt
                             (default: uses self.get_oversized_files_prompt)
            max_retries: Maximum number of retry attempts (default: 2)

        Returns:
            Tuple of (still_oversized_files, total_retry_cost)
        """
        workspace_dir = Path(workspace_dir)
        total_cost = 0.0
        get_prompt = get_retry_prompt or (lambda files: self.get_oversized_files_prompt(files, max_size_mb))

        for retry in range(max_retries):
            oversized = self.check_oversized_files(workspace_dir, max_size_mb)

            if not oversized:
                return [], total_cost

            self._emit(
                f"Oversized files ({len(oversized)}): {', '.join(f['path'] for f in oversized)}. "
                f"Retry {retry + 1}/{max_retries}...",
                "WARN"
            )

            retry_prompt = get_prompt(oversized)
            result = await agent.run([retry_prompt])
            total_cost += result.total_cost

        # Final check after all retries
        still_oversized = self.check_oversized_files(workspace_dir, max_size_mb)
        if still_oversized:
            self._emit(
                f"Still oversized after {max_retries} retries: {', '.join(f['path'] for f in still_oversized)}",
                "ERROR"
            )

        return still_oversized, total_cost

    # === Metadata Reading ===

    def read_metadata(self, workspace_dir: Path) -> dict[str, str]:
        """
        Read metadata from artifact_metadata.json.

        Args:
            workspace_dir: Agent workspace directory

        Returns:
            Dict with 'summary' and 'title' keys (empty strings if not found)
        """
        workspace_dir = Path(workspace_dir)
        metadata_path = workspace_dir / "artifact_metadata.json"

        result = {
            "summary": "",
            "title": "",
        }

        if metadata_path.exists():
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                result["summary"] = data.get("summary", "")
                result["title"] = data.get("title", "")
            except (json.JSONDecodeError, IOError) as e:
                self._emit(f"Failed to read metadata: {e}", "ERROR")
                raise RuntimeError(f"Failed to read artifact_metadata.json: {e}") from e

        return result

    # === Requirements Generation ===

    def generate_requirements(
        self,
        workspace_dir: Path,
        output_file: str = "requirements.txt",
        venv_path: Path | str | None = None,
    ) -> Path | None:
        """
        Generate requirements.txt from venv using uv pip freeze.

        Args:
            workspace_dir: Agent workspace directory
            output_file: Output filename (default: requirements.txt)
            venv_path: Path to venv directory (default: workspace_dir/.venv)

        Returns:
            Path to generated requirements.txt, or None if failed
        """
        workspace_dir = Path(workspace_dir)
        venv = Path(venv_path) if venv_path else workspace_dir / ".venv"

        if not venv.exists():
            self._emit(f"No venv found at {venv}", "WARN")
            return None

        output_path = workspace_dir / output_file

        try:
            # Use uv pip freeze
            result = subprocess.run(
                ["uv", "pip", "freeze"],
                capture_output=True,
                text=True,
                cwd=str(workspace_dir),
                env={
                    **subprocess.os.environ,
                    "VIRTUAL_ENV": str(venv),
                },
            )

            if result.returncode != 0:
                self._emit(f"uv pip freeze failed: {result.stderr}", "ERROR")
                return None

            # Write requirements
            output_path.write_text(result.stdout, encoding="utf-8")
            self._emit(f"Generated {output_file} with {len(result.stdout.splitlines())} packages")
            return output_path

        except FileNotFoundError:
            self._emit("uv not found, falling back to pip freeze", "WARN")
            # Fallback to pip freeze
            pip_path = venv / "bin" / "pip"
            if not pip_path.exists():
                self._emit(f"pip not found at {pip_path}", "ERROR")
                return None

            result = subprocess.run(
                [str(pip_path), "freeze"],
                capture_output=True,
                text=True,
                cwd=str(workspace_dir),
            )

            if result.returncode != 0:
                self._emit(f"pip freeze failed: {result.stderr}", "ERROR")
                return None

            output_path.write_text(result.stdout, encoding="utf-8")
            self._emit(f"Generated {output_file} with {len(result.stdout.splitlines())} packages")
            return output_path

        except Exception as e:
            self._emit(f"Failed to generate requirements: {e}", "ERROR")
            raise

    # === AIITelemetry Task End ===

    def end_task(
        self,
        status: str,
        cost: float | None = None,
        **metadata,
    ) -> None:
        """
        Signal task completion to telemetry.

        Args:
            status: Status string (e.g., "Success", "Failed", "Timeout")
            cost: Optional cost to include in status message
            **metadata: Additional metadata to pass to emit_task_end
        """
        if not (self.telemetry and self.task_id and self.task_name):
            return

        if cost is not None:
            status_msg = f"{status}: ${cost:.4f}"
        else:
            status_msg = status

        self.telemetry.emit_task_end(self.task_id, self.task_name, status_msg, **metadata)

    def end_task_success(self, cost: float | None = None, **metadata) -> None:
        """Convenience method for successful task completion."""
        self.end_task("Success", cost=cost, **metadata)

    def end_task_failure(self, error: str, cost: float | None = None, **metadata) -> None:
        """Convenience method for failed task completion."""
        status = f"Failed: {error[:50]}" if len(error) > 50 else f"Failed: {error}"
        self.end_task(status, cost=cost, **metadata)

    def end_task_timeout(self, timeout_seconds: int, **metadata) -> None:
        """Convenience method for timeout task completion."""
        self.end_task(f"Timeout ({timeout_seconds}s)", **metadata)

    def end_task_error(self, error: str, **metadata) -> None:
        """Convenience method for error task completion."""
        error_msg = error[:50] if len(error) > 50 else error
        self.end_task(f"Error: {error_msg}", **metadata)


    # === Private helpers ===

    def _emit(self, message: str, level: str = "INFO") -> None:
        """Emit message to telemetry if available."""
        if self.telemetry and self.task_id and self.task_name:
            self.telemetry.emit_message(level, message, self.task_name, self.task_id)
