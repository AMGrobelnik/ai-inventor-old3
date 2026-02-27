"""Background usage monitor for Claude Code rate limiting."""

import asyncio
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

from aii_lib.agents.claude.utils.usage import ClaudeUsage, get_claude_usage
from aii_lib.telemetry import log

# Config file paths (priority order: pipeline config > local config > defaults)
PIPELINE_CONFIG_PATH = Path(__file__).parents[6] / "aii_pipeline" / "config.yaml"
LOCAL_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class UsageMonitor:
    """
    Background monitor for Claude Code usage.

    Checks usage periodically and blocks calls when usage exceeds threshold.
    """

    _instance: "UsageMonitor | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "UsageMonitor":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        self._config = self._load_config()
        self._latest_usage: ClaudeUsage | None = None
        self._is_rate_limited = threading.Event()
        self._rate_limit_start: float | None = None
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._telemetry_file: Path | None = None
        self._on_rate_limit_callbacks: list[Callable[[ClaudeUsage], None]] = []
        self._on_rate_limit_clear_callbacks: list[Callable[[ClaudeUsage], None]] = []

        if self._config["telemetry"]["enabled"]:
            self._telemetry_file = Path(self._config["telemetry"]["log_file"])

    def _load_config(self) -> dict:
        """Load configuration from yaml file.

        Priority: aii_pipeline/config.yaml > local config.yaml > defaults
        """
        # Try pipeline config first (claude_agent_global section)
        if PIPELINE_CONFIG_PATH.exists():
            with open(PIPELINE_CONFIG_PATH) as f:
                pipeline_config = yaml.safe_load(f)
                if pipeline_config and "claude_agent_global" in pipeline_config:
                    return pipeline_config["claude_agent_global"]

        # Fall back to local config
        if LOCAL_CONFIG_PATH.exists():
            with open(LOCAL_CONFIG_PATH) as f:
                return yaml.safe_load(f)

        # Default config
        return {
            "usage_tracking": {
                "enabled": True,
                "check_interval_seconds": 60,
                "thresholds": {
                    "current_session": 70,
                    "current_week_all_models": 90,
                    "current_week_sonnet": 90,
                },
            },
            "telemetry": {"enabled": True, "log_file": "/tmp/claude_usage_telemetry.jsonl"},
        }

    def _log_telemetry(self, usage: ClaudeUsage, is_rate_limited: bool) -> None:
        """Log usage data to telemetry file."""
        if not self._telemetry_file:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "usage": asdict(usage),
            "is_rate_limited": is_rate_limited,
            "thresholds": self._config["usage_tracking"]["thresholds"],
        }

        with open(self._telemetry_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _check_threshold(self, usage: ClaudeUsage) -> bool:
        """Check if any monitored metric exceeds its threshold (no logging here).

        Each metric has its own threshold. Set to null in config to disable.
        """
        thresholds = self._config["usage_tracking"]["thresholds"]

        session_threshold = thresholds.get("current_session")
        if session_threshold is not None and usage.current_session is not None:
            if usage.current_session >= session_threshold:
                return True

        all_models_threshold = thresholds.get("current_week_all_models")
        if all_models_threshold is not None and usage.current_week_all_models is not None:
            if usage.current_week_all_models >= all_models_threshold:
                return True

        sonnet_threshold = thresholds.get("current_week_sonnet")
        if sonnet_threshold is not None and usage.current_week_sonnet is not None:
            if usage.current_week_sonnet >= sonnet_threshold:
                return True

        return False

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        interval = self._config["usage_tracking"]["check_interval_seconds"]
        thresholds = self._config["usage_tracking"]["thresholds"]

        while not self._stop_event.is_set():
            try:
                usage = get_claude_usage()
                self._latest_usage = usage
                was_rate_limited = self._is_rate_limited.is_set()
                is_over_threshold = self._check_threshold(usage)

                self._log_telemetry(usage, is_over_threshold)

                if is_over_threshold:
                    if not was_rate_limited:
                        self._is_rate_limited.set()
                        self._rate_limit_start = time.time()
                        for cb in self._on_rate_limit_callbacks:
                            cb(usage)
                    # Log warning on each check while rate limited
                    elapsed_min = int((time.time() - self._rate_limit_start) / 60) if self._rate_limit_start else 0
                    log.warning(
                        f"⏳ Usage over threshold - waiting ({elapsed_min}min) | "
                        f"session: {usage.current_session if usage.current_session is not None else '?'}%/{thresholds.get('current_session', '-')}% | "
                        f"all_models: {usage.current_week_all_models if usage.current_week_all_models is not None else '?'}%/{thresholds.get('current_week_all_models', '-')}% | "
                        f"sonnet: {usage.current_week_sonnet if usage.current_week_sonnet is not None else '?'}%/{thresholds.get('current_week_sonnet', '-')}%"
                    )
                else:
                    if was_rate_limited:
                        elapsed_min = int((time.time() - self._rate_limit_start) / 60) if self._rate_limit_start else 0
                        log.success(f"Usage dropped below thresholds - resuming (waited {elapsed_min}min)")
                        self._is_rate_limited.clear()
                        self._rate_limit_start = None
                        for cb in self._on_rate_limit_clear_callbacks:
                            cb(usage)


            except Exception as e:
                log.exception(f"Usage monitor check failed: {e}")

            self._stop_event.wait(interval)


    def start(self) -> None:
        """Start the background monitor."""
        if not self._config["usage_tracking"]["enabled"]:
            return

        if self._monitor_thread and self._monitor_thread.is_alive():
            return  # Already running, no need to log

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self) -> None:
        """Stop the background monitor."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

    def wait_for_capacity(self, timeout: float | None = None) -> bool:
        """
        Block until usage is below threshold.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if capacity available, False if timed out.
        """
        # If no usage data yet, do an immediate check
        if self._latest_usage is None:
            try:
                usage = get_claude_usage()
                self._latest_usage = usage
                if self._check_threshold(usage):
                    self._is_rate_limited.set()
                    self._rate_limit_start = time.time()
                self._log_telemetry(usage, self._is_rate_limited.is_set())
            except Exception as e:
                log.warning(f"Usage check unavailable — continuing without monitoring: {e}")
                return True  # Don't block pipeline on monitoring failure

        if not self._is_rate_limited.is_set():
            return True

        # Wait for rate limit to clear (monitor loop logs status)
        start = time.time()
        while self._is_rate_limited.is_set():
            if timeout and (time.time() - start) > timeout:
                log.error(f"Timeout waiting for Claude capacity after {timeout}s")
                return False
            time.sleep(1)
        return True

    async def async_wait_for_capacity(self, timeout: float | None = None) -> bool:
        """
        Async version of wait_for_capacity. Uses asyncio.sleep to not block event loop.

        This allows asyncio.wait_for (agent_timeout) to cancel the wait,
        and lets other async tasks run during the wait.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if capacity available, False if timed out.
        """
        # If no usage data yet, do an immediate check
        if self._latest_usage is None:
            try:
                usage = get_claude_usage()
                self._latest_usage = usage
                if self._check_threshold(usage):
                    self._is_rate_limited.set()
                    self._rate_limit_start = time.time()
                self._log_telemetry(usage, self._is_rate_limited.is_set())
            except Exception as e:
                log.warning(f"Usage check unavailable — continuing without monitoring: {e}")
                return True  # Don't block pipeline on monitoring failure

        if not self._is_rate_limited.is_set():
            return True

        # Wait for rate limit to clear (monitor loop logs status)
        start = time.time()
        while self._is_rate_limited.is_set():
            if timeout and (time.time() - start) > timeout:
                log.error(f"Timeout waiting for Claude capacity after {timeout}s")
                return False
            await asyncio.sleep(1)
        return True

    def is_rate_limited(self) -> bool:
        """Check if currently rate limited."""
        return self._is_rate_limited.is_set()

    def get_latest_usage(self) -> ClaudeUsage | None:
        """Get the most recent usage data."""
        return self._latest_usage

    def on_rate_limit(self, callback: Callable[[ClaudeUsage], None]) -> None:
        """Register callback for when rate limit is hit."""
        self._on_rate_limit_callbacks.append(callback)

    def on_rate_limit_clear(self, callback: Callable[[ClaudeUsage], None]) -> None:
        """Register callback for when rate limit clears."""
        self._on_rate_limit_clear_callbacks.append(callback)


# Global monitor instance
_monitor: UsageMonitor | None = None


def get_monitor() -> UsageMonitor:
    """Get the global usage monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = UsageMonitor()
    return _monitor


def require_capacity(timeout: float | None = None) -> bool:
    """
    Sync helper - wait for capacity before proceeding.

    Use this at the start of any sync function that calls Claude.
    For async code, use async_require_capacity instead.

    Args:
        timeout: Maximum seconds to wait

    Returns:
        True if capacity available, False if timed out.
    """
    monitor = get_monitor()
    return monitor.wait_for_capacity(timeout=timeout)


async def async_require_capacity(timeout: float | None = None) -> bool:
    """
    Async helper - wait for capacity before proceeding.

    Use this at the start of any async function that calls Claude.
    Uses asyncio.sleep so it doesn't block the event loop.

    Args:
        timeout: Maximum seconds to wait

    Returns:
        True if capacity available, False if timed out.
    """
    monitor = get_monitor()
    return await monitor.async_wait_for_capacity(timeout=timeout)
