"""Fetch Claude Code usage statistics programmatically.

Uses tmux to interact with the claude CLI and capture /usage output.
Requires: tmux, claude CLI.
"""

import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class ClaudeUsage:
    """Claude Code usage statistics."""

    current_session: int | None = None
    current_week_all_models: int | None = None
    current_week_sonnet: int | None = None

    def __str__(self) -> str:
        return (
            f"current_session: {self.current_session}%\n"
            f"current_week_all_models: {self.current_week_all_models}%\n"
            f"current_week_sonnet: {self.current_week_sonnet}%"
        )


def _parse_usage(raw_output: str) -> ClaudeUsage:
    """Parse usage percentages from raw tmux capture output.

    Finds all N% occurrences in order and assigns them positionally:
    1st = current_session, 2nd = current_week_all_models, 3rd = current_week_sonnet.
    """
    # Match "N% used" with possible invisible/whitespace chars between them
    percentages = re.findall(r"(\d+)\s*%\s*u\s*s\s*e\s*d", raw_output)
    return ClaudeUsage(
        current_session=int(percentages[0]) if len(percentages) > 0 else None,
        current_week_all_models=int(percentages[1]) if len(percentages) > 1 else None,
        current_week_sonnet=int(percentages[2]) if len(percentages) > 2 else None,
    )


def _check_prerequisites() -> None:
    """Verify tmux and claude are available. Raises RuntimeError if not."""
    if not shutil.which("tmux"):
        raise RuntimeError("tmux not found on PATH")
    if not shutil.which("claude"):
        raise RuntimeError("claude CLI not found on PATH")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=3),
    reraise=True,
)
def get_claude_usage(timeout_seconds: int = 15) -> ClaudeUsage:
    """
    Fetch Claude Code usage statistics.

    Runs `claude "/usage"` in a tmux session, captures the output,
    and parses usage percentages.

    Args:
        timeout_seconds: How long to wait for usage data to load.

    Returns:
        ClaudeUsage dataclass with usage percentages.

    Raises:
        RuntimeError: If unable to fetch or parse usage data.
    """
    _check_prerequisites()

    uid = uuid.uuid4().hex[:8]
    session_name = f"claude_usage_{uid}"
    raw_output_file = Path(f"/tmp/claude_usage_raw_{uid}.txt")

    # Build env: unset CLAUDECODE to prevent "nested session" detection
    env = {**os.environ}
    env.pop("CLAUDECODE", None)

    try:
        # Start tmux session with claude
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, "claude"],
            check=True,
            capture_output=True,
            env=env,
        )

        # Wait for CLI to initialize, then send /usage command
        time.sleep(5)
        subprocess.run(
            ["tmux", "send-keys", "-t", session_name, "/usage"],
            check=True,
            capture_output=True,
        )
        time.sleep(1)
        subprocess.run(
            ["tmux", "send-keys", "-t", session_name, "Enter"],
            check=True,
            capture_output=True,
        )

        # Wait for usage data to load
        time.sleep(timeout_seconds)

        # Capture pane content
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session_name, "-p", "-S", "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        raw_output = result.stdout
        raw_output_file.write_text(raw_output)

        # Parse usage
        usage = _parse_usage(raw_output)

        # Validate we got data
        if usage.current_week_all_models is None:
            raise RuntimeError("Failed to parse usage data from output")

        return usage

    finally:
        # Gracefully exit: Esc to close /usage dialog, then /exit
        subprocess.run(
            ["tmux", "send-keys", "-t", session_name, "Escape"],
            capture_output=True,
        )
        time.sleep(0.5)
        subprocess.run(
            ["tmux", "send-keys", "-t", session_name, "/exit", "Enter"],
            capture_output=True,
        )
        time.sleep(1)
        # Force kill the tmux session and all processes in it
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            capture_output=True,
        )
        # Clean up temp file
        raw_output_file.unlink(missing_ok=True)


if __name__ == "__main__":
    # Test the function
    usage = get_claude_usage()
    print(usage)
