"""B_CREATE_REPO Step - Create GitHub repository for the research artifacts.

Creates a new GitHub repository to host all research artifacts, code, and paper.
Uses gh CLI for repository creation.

Uses aii_lib for:
- AIITelemetry: Task tracking
"""

import json
import secrets
import subprocess
from datetime import datetime
from pathlib import Path
import re

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from aii_lib import AIITelemetry, MessageType, JSONSink

from aii_pipeline.utils import PipelineConfig, rel_path


class GitHubRepoCreationError(Exception):
    """Raised when GitHub repo creation fails (retriable)."""
    pass


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=32),
    retry=retry_if_exception_type(GitHubRepoCreationError),
    reraise=True,
)
def _create_repo_with_retry(repo_name: str, description: str) -> subprocess.CompletedProcess:
    """Create GitHub repo with tenacity retry on intermittent failures."""
    result = subprocess.run(
        [
            "gh", "repo", "create", repo_name,
            "--public",
            "--description", description,
            "--clone=false",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "Repository creation failed" in result.stderr:
        raise GitHubRepoCreationError(result.stderr)
    return result


def generate_short_id() -> str:
    """Generate a short unique ID (6 hex chars) for repo uniqueness."""
    return secrets.token_hex(3)  # 6 hex characters


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug.

    Truncates to 40 chars to leave room for "ai-invention-" prefix (13 chars),
    unique ID suffix (7 chars with hyphen), and stay under GitHub's 100 char limit.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')[:40]


async def run_create_repo_module(
    config: PipelineConfig,
    hypothesis: dict,
    telemetry: AIITelemetry,
    output_dir: Path | None = None,
) -> dict:
    """
    Run the B_CREATE_REPO step.

    Creates a GitHub repository for the research project.

    Args:
        config: Pipeline configuration
        hypothesis: Hypothesis dict with title
        telemetry: AIITelemetry instance
        output_dir: Output directory

    Returns:
        Dict with repo_url, repo_name, repo_owner
    """
    # Add module-specific JSON sink (removed after module completes)
    module_sinks = []
    if output_dir:
        s1 = JSONSink(output_dir / "create_repo_pipeline_messages.jsonl")
        s2 = JSONSink(output_dir / "create_repo_pipeline_messages_sequenced.jsonl", sequenced=True)
        telemetry.add_sink(s1)
        telemetry.add_sink(s2)
        module_sinks.extend([s1, s2])

    telemetry.start_module("B_CREATE_REPO")

    try:
        telemetry.emit(MessageType.INFO, "=" * 60)
        telemetry.emit(MessageType.INFO, "B_CREATE_REPO - Creating GitHub repository")
        telemetry.emit(MessageType.INFO, "=" * 60)

        # Generate repo name from hypothesis with unique ID
        title = hypothesis.get("title", "research-project")
        repo_slug = slugify(title)
        unique_id = generate_short_id()
        repo_name = f"ai-invention-{unique_id}-{repo_slug}"

        telemetry.emit(MessageType.INFO, f"   Repo name: {repo_name}")
        telemetry.emit(MessageType.INFO, f"   Unique ID: {unique_id}")

        task_id = "create_repo"
        task_name = task_id
        telemetry.emit_task_start(task_id, task_name)

        result = {
            "repo_name": repo_name,
            "repo_url": None,
            "repo_owner": None,
            "created": False,
            "error": None,
        }

        try:
            # Check if gh CLI is available
            check_gh = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
            )
            if check_gh.returncode != 0:
                telemetry.emit_message("WARNING", "gh CLI not available, skipping repo creation", task_name, task_id)
                result["error"] = "gh CLI not installed"
                telemetry.emit_task_end(task_id, task_name, "gh CLI not available")
                telemetry.emit_module_summary("B_CREATE_REPO")
                return result

            # Get current GitHub user
            user_result = subprocess.run(
                ["gh", "api", "user", "-q", ".login"],
                capture_output=True,
                text=True,
            )
            if user_result.returncode != 0:
                telemetry.emit_message("WARNING", "Not authenticated with gh CLI", task_name, task_id)
                result["error"] = "Not authenticated"
                telemetry.emit_task_end(task_id, task_name, "Not authenticated")
                telemetry.emit_module_summary("B_CREATE_REPO")
                return result

            owner = user_result.stdout.strip()
            result["repo_owner"] = owner

            # Check if repo already exists
            check_repo = subprocess.run(
                ["gh", "repo", "view", f"{owner}/{repo_name}"],
                capture_output=True,
                text=True,
            )
            if check_repo.returncode == 0:
                # Repo exists
                repo_url = f"https://github.com/{owner}/{repo_name}"
                telemetry.emit_message("INFO", f"Repo already exists: {repo_url}", task_name, task_id)
                result["repo_url"] = repo_url
                result["created"] = False
                telemetry.emit_task_end(task_id, task_name, f"Exists: {repo_url}")
                telemetry.emit_module_summary("B_CREATE_REPO")
                return result

            # Create the repository with tenacity retry (GitHub GraphQL can fail intermittently)
            description = hypothesis.get("hypothesis", "AI-generated research project")[:200]

            try:
                create_result = _create_repo_with_retry(repo_name, description)
                if create_result.returncode == 0:
                    repo_url = f"https://github.com/{owner}/{repo_name}"
                    telemetry.emit_message("SUCCESS", f"Created repo: {repo_url}", task_name, task_id)
                    result["repo_url"] = repo_url
                    result["created"] = True
                    telemetry.emit_task_end(task_id, task_name, f"Created: {repo_url}")
                else:
                    telemetry.emit_message("ERROR", f"Failed to create repo: {create_result.stderr}", task_name, task_id)
                    result["error"] = create_result.stderr
                    telemetry.emit_task_end(task_id, task_name, f"Error: {create_result.stderr[:50]}")
            except GitHubRepoCreationError as e:
                telemetry.emit_message("ERROR", f"Failed to create repo after retries: {e}", task_name, task_id)
                result["error"] = str(e)
                telemetry.emit_task_end(task_id, task_name, f"Error: {str(e)[:50]}")

        except Exception as e:
            telemetry.emit_message("ERROR", f"B_CREATE_REPO failed: {e}", task_name, task_id)
            result["error"] = str(e)
            telemetry.emit_task_end(task_id, task_name, f"Error: {e}")

        # Save output
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "repo_info.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    **result,
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "module": "create_repo",
                        "llm_provider": "gh_cli",
                        "output_dir": str(output_dir) if output_dir else None,
                    },
                }, f, indent=2, ensure_ascii=False)
            telemetry.emit(MessageType.INFO, f"   Saved to: {rel_path(output_file)}")

        telemetry.emit_module_summary("B_CREATE_REPO")

        return result

    finally:
        # Always remove module sinks to prevent leak into subsequent modules
        for sink in module_sinks:
            sink.flush()
            telemetry.remove_sink(sink)
