"""
Ability Service - FastAPI server with dedicated process per endpoint.

Each endpoint has ONE dedicated process that can handle multiple concurrent
requests via an internal ThreadPoolExecutor. This provides:
- Process isolation (crash in one endpoint doesn't affect others)
- Pre-warmed imports (init_func runs once per process)
- Concurrent request handling (ThreadPoolExecutor inside each process)
- Auto-cleanup of temp files older than 1 hour

Architecture:
    FastAPI (main process)
        ↓ sends request via queue
    Worker Process (one per endpoint)
        ↓ runs init_func once
        ↓ handles requests via internal ThreadPoolExecutor

Usage:
    from aii_lib.abilities.ability_server.ability_service import ability_service, app

    # Register endpoints (typically in endpoints.py)
    ability_service.register("aii_hf_search_datasets", handler, init_func, max_threads=30)

    # Run server
    uvicorn aii_lib.abilities.ability_server.endpoints:app --port 8100
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from typing import Any, Callable
import asyncio
import faulthandler
import os
import signal
import sys
import time
import traceback
import uuid

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# =============================================================================
# Loguru configuration with colors
# =============================================================================

# ANSI color codes for worker categories
_ANSI_COLORS = {
    "yellow": "\033[33m",      # HuggingFace
    "blue": "\033[34m",        # Web tools
    "green": "\033[32m",       # Lean/Mathlib
    "magenta": "\033[35m",     # OpenRouter
    "cyan": "\033[36m",        # OWID
    "white": "\033[37m",       # JSON
    "bold": "\033[1m",         # Server
    "reset": "\033[0m",
}

# Worker category colors (by prefix)
_WORKER_COLORS = {
    "aii_hf_": "yellow",           # HuggingFace - gold/yellow
    "aii_web_": "blue",            # Web tools - blue
    "aii_verify_": "blue",         # Verify quotes - blue
    "aii_lean": "green",           # Lean - green
    "aii_mathlib_": "green",       # Mathlib - green
    "aii_openrouter_": "magenta",  # OpenRouter - magenta/purple
    "aii_owid_": "cyan",           # OWID - cyan
    "aii_json_": "white",          # JSON - white
    "dblp_": "blue",               # DBLP - blue (bibliography)
    "tooluniverse_": "magenta",    # ToolUniverse MCP - magenta
    "server": "bold",              # Server - bold white
}


def _get_worker_color_code(source: str) -> str:
    """Get ANSI color code for a worker based on its name prefix."""
    for prefix, color_name in _WORKER_COLORS.items():
        if source.startswith(prefix):
            return _ANSI_COLORS.get(color_name, "")
    return _ANSI_COLORS["white"]


def _format_with_source(record):
    """Custom formatter that colors the source and message based on worker type."""
    source = record["extra"].get("source", "")
    color = _get_worker_color_code(source)
    reset = _ANSI_COLORS["reset"]
    # Pad source to 22 chars (longest worker name)
    source_padded = f"{source: <22}"
    record["extra"]["colored_source"] = f"{color}{source_padded}{reset}"
    record["extra"]["color"] = color
    record["extra"]["reset"] = reset
    return "<dim>{time:HH:mm:ss}</dim> | <level>{level: <7}</level> | {extra[colored_source]} | {extra[color]}{message}{extra[reset]}\n"


def _format_without_source(record):
    """Formatter for logs without source."""
    return "<dim>{time:HH:mm:ss}</dim> | <level>{level: <7}</level> | {message}\n"


# Configure loguru: TIME | LEVEL | SOURCE | MESSAGE
logger.remove()
logger.add(
    sys.stderr,
    format=_format_with_source,
    level="DEBUG",
    filter=lambda record: "source" in record["extra"],
    colorize=True,
)
logger.add(
    sys.stderr,
    format=_format_without_source,
    level="DEBUG",
    filter=lambda record: "source" not in record["extra"],
    colorize=True,
)

# Load config from server_config.yaml (cached at module load)
_CONFIG_FILE = Path(__file__).parent / "server_config.yaml"
_server_config: dict = {}
if _CONFIG_FILE.exists():
    with open(_CONFIG_FILE) as f:
        _server_config = yaml.safe_load(f) or {}

DEFAULT_TIMEOUT = float(_server_config.get("server", {}).get("timeout", 180.0))

# Retry configuration
_retry_config = _server_config.get("retry", {})
RETRY_MODE = _retry_config.get("mode", "exponential")  # Only "exponential" supported for now
RETRY_MAX_RETRIES = int(_retry_config.get("max_retries", 3))
RETRY_MIN_BACKOFF = float(_retry_config.get("min_backoff", 1.0))
RETRY_MAX_BACKOFF = float(_retry_config.get("max_backoff", 20.0))

if RETRY_MODE != "exponential":
    logger.warning(f"Invalid retry mode '{RETRY_MODE}', defaulting to 'exponential'")
    RETRY_MODE = "exponential"

# Transient error patterns (case-insensitive matching)
_TRANSIENT_ERRORS = (
    "connection aborted",
    "connection reset",
    "connection refused",
    "remotedisconnected",
    "remote end closed",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "too many requests",
    "rate limit",
    # DNS resolution failures
    "name resolution",
    "nameresolutionerror",
    "temporary failure in name resolution",
    "failed to resolve",
    "getaddrinfo failed",
    "nodename nor servname provided",
    # Network unreachable
    "network is unreachable",
    "no route to host",
    "max retries exceeded",
)


def with_retry(func: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """
    Decorator that adds retry logic with exponential backoff for transient errors.

    Retries on connection drops, timeouts, rate limits, and similar transient failures.
    Backoff: min_backoff * 2^attempt, capped at max_backoff.
    """
    import functools
    import random

    @functools.wraps(func)
    def wrapper(req: dict) -> dict:
        last_error = None

        for attempt in range(RETRY_MAX_RETRIES + 1):
            result = func(req)

            # Success or non-transient error
            if result.get("success", False):
                return result

            error_msg = str(result.get("error", "")).lower()

            # Check if error is transient
            is_transient = any(err in error_msg for err in _TRANSIENT_ERRORS)

            if not is_transient or attempt >= RETRY_MAX_RETRIES:
                return result

            # Calculate backoff with jitter
            backoff = min(RETRY_MIN_BACKOFF * (2 ** attempt), RETRY_MAX_BACKOFF)
            backoff = backoff * (0.8 + 0.4 * random.random())  # ±20% jitter
            last_error = result.get("error")

            logger.bind(source="retry").warning(
                f"Transient error (attempt {attempt + 1}/{RETRY_MAX_RETRIES + 1}), "
                f"retrying in {backoff:.1f}s: {last_error}"
            )
            time.sleep(backoff)

        return result

    return wrapper


# Crash log directory
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
_CRASH_LOG_DIR = _PROJECT_ROOT / "logs" / "ability_crashes"
_CRASH_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Enable faulthandler for segfaults in main process
faulthandler.enable()

# =============================================================================
# Temp file cleanup configuration
# =============================================================================

# Temp directories to clean (auto-discover all temp/ folders under .claude/skills/)
_SKILLS_DIR = _PROJECT_ROOT / ".claude" / "skills"


def _get_temp_cleanup_dirs() -> list[Path]:
    """Find all temp directories under .claude/skills/ (excluding archived)."""
    dirs = []
    if not _SKILLS_DIR.exists():
        return dirs
    for skill_dir in _SKILLS_DIR.iterdir():
        if not skill_dir.is_dir() or skill_dir.name == "archived":
            continue
        temp_dir = skill_dir / "temp"
        if temp_dir.exists() and temp_dir.is_dir():
            dirs.append(temp_dir)
    return dirs

# Cleanup configuration from server_config.yaml
_cleanup_config = _server_config.get("cleanup", {})
# Worker memory recycling configuration
_worker_memory_config = _server_config.get("worker_memory", {})
WORKER_MAX_MEMORY_MB = float(_worker_memory_config.get("max_memory_mb", 2048))
WORKER_MEMORY_CHECK_INTERVAL = int(_worker_memory_config.get("check_interval", 10))

CLEANUP_MAX_AGE_HOURS = float(_cleanup_config.get("max_age_hours", 1.0))
CLEANUP_INTERVAL_MINUTES = float(_cleanup_config.get("interval_minutes", 10.0))


def _cleanup_old_temp_files() -> dict:
    """
    Remove temp files older than CLEANUP_MAX_AGE_HOURS.

    Recursively cleans all temp/ directories under .claude/skills/.

    Returns:
        Dict with cleanup statistics
    """
    log = logger.bind(source="cleanup")
    stats = {"checked": 0, "deleted": 0, "freed_mb": 0.0, "errors": [], "dirs_scanned": []}
    max_age_seconds = CLEANUP_MAX_AGE_HOURS * 3600
    now = time.time()

    for temp_dir in _get_temp_cleanup_dirs():
        stats["dirs_scanned"].append(str(temp_dir))
        if not temp_dir.exists():
            continue

        try:
            # Recursively walk all files in temp directory
            for file_path in temp_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                stats["checked"] += 1
                try:
                    # Check file age based on modification time
                    mtime = file_path.stat().st_mtime
                    age_seconds = now - mtime

                    if age_seconds > max_age_seconds:
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        file_path.unlink()
                        stats["deleted"] += 1
                        stats["freed_mb"] += size_mb
                        log.debug(f"Deleted {file_path.name} ({size_mb:.1f}MB, {age_seconds/3600:.1f}h old)")
                except Exception as e:
                    stats["errors"].append(f"{file_path.name}: {e}")

            # Clean up empty directories after file deletion
            for dir_path in sorted(temp_dir.rglob("*"), reverse=True):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    try:
                        dir_path.rmdir()
                        log.debug(f"Removed empty dir: {dir_path.name}")
                    except Exception:
                        pass

        except Exception as e:
            stats["errors"].append(f"{temp_dir}: {e}")

    if stats["deleted"] > 0:
        log.info(f"Cleaned {stats['deleted']} files, freed {stats['freed_mb']:.1f}MB")

    return stats


async def _cleanup_task():
    """Background task that periodically cleans up old temp files."""
    log = logger.bind(source="cleanup")
    interval_seconds = CLEANUP_INTERVAL_MINUTES * 60

    log.info(f"Started (max_age={CLEANUP_MAX_AGE_HOURS}h, interval={CLEANUP_INTERVAL_MINUTES}min)")

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            _cleanup_old_temp_files()
        except asyncio.CancelledError:
            log.info("Cleanup task cancelled")
            break
        except Exception as e:
            log.error(f"Cleanup error: {e}")


def _get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # KB to MB on Linux
    except Exception:
        return -1.0


def _write_crash_log(name: str, error_type: str, message: str, tb: str = ""):
    """Write crash info to dedicated crash log file."""
    log = logger.bind(source=name)
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        crash_file = _CRASH_LOG_DIR / f"crash_{name}_{timestamp}.log"
        with open(crash_file, "w") as f:
            f.write(f"CRASH LOG - {name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"PID: {os.getpid()}\n")
            f.write(f"Memory MB: {_get_memory_mb():.1f}\n")
            f.write(f"Error Type: {error_type}\n")
            f.write(f"Message: {message}\n")
            f.write(f"{'='*60}\n")
            if tb:
                f.write(f"Traceback:\n{tb}\n")
        log.error(f"Crash log written: {crash_file}")
    except Exception as e:
        log.error(f"Failed to write crash log: {e}")


def _worker_process(
    name: str,
    handler: Callable[[dict], dict],
    init_func: Callable[[], None] | None,
    request_queue: Queue,
    response_queues: dict,
    max_threads: int,
):
    """
    Worker process that handles requests for one endpoint.

    Runs init_func once, then processes requests via ThreadPoolExecutor.
    Catches ALL exceptions to prevent silent crashes.
    """
    # Create bound logger for this worker
    log = logger.bind(source=name)

    # Enable faulthandler in worker process
    faulthandler.enable()

    # Write faulthandler output to crash file on segfault
    try:
        fault_file = _CRASH_LOG_DIR / f"segfault_{name}_{os.getpid()}.log"
        fault_fd = open(fault_file, "w")
        faulthandler.enable(file=fault_fd)
        faulthandler.register(signal.SIGUSR1, file=fault_fd, all_threads=True)
    except Exception as e:
        log.warning(f"Could not setup faulthandler file: {e}")

    # Signal handler for clean shutdown logging
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        log.warning(f"Received signal {sig_name} (pid={os.getpid()})")
        _write_crash_log(name, "SIGNAL", f"Received {sig_name}", traceback.format_stack(frame))
        sys.exit(128 + signum)

    # Register signal handlers
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        try:
            signal.signal(sig, signal_handler)
        except Exception:
            pass

    log.info(f"Worker started (pid={os.getpid()}, mem={_get_memory_mb():.1f}MB)")

    # Run init function (imports, warmup)
    if init_func:
        try:
            log.info("Running init_func...")
            start = time.perf_counter()
            init_func()
            elapsed = time.perf_counter() - start
            log.info(f"Init complete ({elapsed:.2f}s, mem={_get_memory_mb():.1f}MB)")
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Init FAILED: {e}\n{tb}")
            _write_crash_log(name, "INIT_ERROR", str(e), tb)
            raise RuntimeError(f"Init function failed for {name}: {e}") from e

    # Create thread pool for concurrent request handling
    executor = ThreadPoolExecutor(max_workers=max_threads)
    request_count = 0
    error_count = 0

    def handle_request(request_id: str, request_dict: dict):
        """Handle a single request in a thread."""
        nonlocal request_count, error_count
        request_count += 1
        req_start = time.perf_counter()

        try:
            log.debug(f"REQ#{request_count} started (id={request_id[:8]})")
            result = handler(request_dict)
            # Handle async handlers
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            elapsed = time.perf_counter() - req_start
            log.debug(f"REQ#{request_count} completed ({elapsed:.2f}s)")
            return request_id, result
        except Exception as e:
            error_count += 1
            elapsed = time.perf_counter() - req_start
            tb = traceback.format_exc()
            log.error(f"REQ#{request_count} FAILED ({elapsed:.2f}s): {e}\n{tb}")
            _write_crash_log(name, "HANDLER_ERROR", str(e), tb)
            return request_id, {"success": False, "error": str(e), "traceback": tb}

    def send_response(future):
        """Callback to send response back to main process."""
        try:
            request_id, result = future.result()
            if request_id in response_queues:
                response_queues[request_id].put(result)
            else:
                log.warning(f"Response queue missing for {request_id[:8]}")
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Failed to send response: {e}\n{tb}")
            _write_crash_log(name, "RESPONSE_ERROR", str(e), tb)

    log.info(f"Worker ready (max_threads={max_threads}, pid={os.getpid()})")

    # Main loop - process requests
    while True:
        try:
            # Wait for request with timeout to allow periodic health logging
            try:
                msg = request_queue.get(timeout=60)
            except Empty:
                # Periodic health log
                log.debug(f"Health: reqs={request_count}, errs={error_count}, mem={_get_memory_mb():.1f}MB")
                continue

            if msg is None:  # Shutdown signal
                log.info(f"Shutdown signal received, stopping... (reqs={request_count}, errs={error_count})")
                executor.shutdown(wait=True)
                break

            request_id, request_dict, response_queue = msg

            # Submit to thread pool
            future = executor.submit(handle_request, request_id, request_dict)
            # Store response queue for callback
            response_queues[request_id] = response_queue
            future.add_done_callback(send_response)

            # Memory-based worker recycling
            if WORKER_MEMORY_CHECK_INTERVAL > 0 and request_count % WORKER_MEMORY_CHECK_INTERVAL == 0:
                mem_mb = _get_memory_mb()
                if mem_mb > WORKER_MAX_MEMORY_MB:
                    log.warning(
                        f"Memory limit exceeded ({mem_mb:.0f}MB > {WORKER_MAX_MEMORY_MB:.0f}MB), "
                        f"recycling worker (reqs={request_count}, errs={error_count})"
                    )
                    # Wait for in-flight requests to complete
                    executor.shutdown(wait=True)
                    break

        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"MAIN LOOP ERROR: {e}\n{tb}")
            _write_crash_log(name, "MAIN_LOOP_ERROR", str(e), tb)
            # Continue running - don't crash the worker

    log.info(f"Worker stopped (pid={os.getpid()})")


class WorkerHandle:
    """Handle to a worker process for one endpoint."""

    def __init__(
        self,
        name: str,
        handler: Callable[[dict], dict],
        init_func: Callable[[], None] | None,
        max_threads: int = 10,
    ):
        self.name = name
        self.handler = handler
        self.init_func = init_func
        self.max_threads = max_threads
        self.request_queue: Queue = Queue()
        self.response_queues: dict = {}  # Shared dict for responses
        self.process: Process | None = None
        self._manager = None  # Reusable manager for response queues
        self._restart_count = 0
        self._last_restart = None
        self._log = logger.bind(source=name)

    def start(self):
        """Start the worker process."""
        if self.process is not None and self.process.is_alive():
            return

        # Clean up old manager if exists
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass

        # Create ONE manager for this worker (reused for all requests)
        from multiprocessing import Manager
        self._manager = Manager()
        self.response_queues = self._manager.dict()

        # Create new request queue (old one may be corrupted)
        self.request_queue = Queue()

        self.process = Process(
            target=_worker_process,
            args=(
                self.name,
                self.handler,
                self.init_func,
                self.request_queue,
                self.response_queues,
                self.max_threads,
            ),
            daemon=True,
        )
        self.process.start()
        self._last_restart = datetime.now()
        self._log.info(f"Worker process started (pid={self.process.pid}, restart_count={self._restart_count})")

    def stop(self):
        """Stop the worker process."""
        if self.process is None:
            return

        self._log.info(f"Stopping worker (pid={self.process.pid})...")

        # Send shutdown signal
        try:
            self.request_queue.put(None)
            self.process.join(timeout=5)
        except Exception as e:
            self._log.warning(f"Error during graceful shutdown: {e}")

        if self.process.is_alive():
            self._log.warning("Worker didn't stop gracefully, terminating...")
            self.process.terminate()
            self.process.join(timeout=2)

        if self.process.is_alive():
            self._log.error("Worker still alive after terminate, killing...")
            self.process.kill()
            self.process.join(timeout=1)

        self.process = None
        self._log.info("Worker stopped")

    def _handle_worker_death(self):
        """Handle worker death - log details and prepare for restart."""
        if self.process is None:
            return

        exitcode = self.process.exitcode
        pid = self.process.pid

        # Detailed crash logging
        if exitcode is not None and exitcode != 0:
            if exitcode < 0:
                # Killed by signal
                try:
                    sig_name = signal.Signals(-exitcode).name
                except ValueError:
                    sig_name = str(-exitcode)
                error_msg = f"Worker killed by signal {sig_name} (exitcode={exitcode})"
                self._log.error(f"{error_msg} (pid={pid})")
                _write_crash_log(self.name, "SIGNAL_DEATH", error_msg, f"Signal: {sig_name}\nPID: {pid}")
            else:
                error_msg = f"Worker exited with code {exitcode}"
                self._log.error(f"{error_msg} (pid={pid})")
                _write_crash_log(self.name, "EXIT_CODE", error_msg, f"Exit code: {exitcode}\nPID: {pid}")
        else:
            self._log.warning(f"Worker died unexpectedly (pid={pid}, exitcode={exitcode})")
            _write_crash_log(self.name, "UNEXPECTED_DEATH", "Worker died unexpectedly", f"PID: {pid}\nExitcode: {exitcode}")

        self._restart_count += 1

    async def call(self, request: dict, timeout: float = None) -> dict:
        """Send request to worker and wait for response."""
        if timeout is None:
            timeout = DEFAULT_TIMEOUT

        # Check if worker is alive, restart if needed
        if self.process is None or not self.process.is_alive():
            if self.process is not None:
                self._handle_worker_death()
            self._log.info("Restarting worker...")
            self.start()
            # Give worker time to initialize
            await asyncio.sleep(0.5)

        # Reuse the manager's Queue (no new Manager per request!)
        if self._manager is None:
            self._log.error("Manager not initialized!")
            return {"success": False, "error": "Worker manager not initialized"}

        try:
            response_queue = self._manager.Queue()
        except Exception as e:
            self._log.error(f"Failed to create response queue: {e}")
            # Manager might be dead, restart worker
            self._handle_worker_death()
            self.start()
            await asyncio.sleep(0.5)
            try:
                response_queue = self._manager.Queue()
            except Exception as e2:
                return {"success": False, "error": f"Failed to create response queue: {e2}"}

        request_id = str(uuid.uuid4())

        # Send request
        try:
            self.request_queue.put((request_id, request, response_queue))
        except Exception as e:
            self._log.error(f"Failed to send request: {e}")
            return {"success": False, "error": f"Failed to send request: {e}"}

        # Wait for response (async-friendly)
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, response_queue.get, True, timeout),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            self._log.warning(f"Request timed out after {timeout}s (id={request_id[:8]})")
            return {"success": False, "error": f"Request timed out after {timeout}s"}
        except Exception as e:
            tb = traceback.format_exc()
            self._log.error(f"Error waiting for response: {e}\n{tb}")
            return {"success": False, "error": str(e), "traceback": tb}


class AbilityService:
    """
    Single FastAPI server with one dedicated process per endpoint.

    Each endpoint's process runs init_func once for heavy imports,
    then handles concurrent requests via internal ThreadPoolExecutor.

    Worker crashes are isolated - they don't crash the server.
    Dead workers are automatically restarted on next request.

    Also supports:
    - Mounting MCP servers with proper lifespan management
    - Spawning ToolUniverse MCP as a separate subprocess (crash isolated)
    """

    def __init__(self):
        self.workers: dict[str, WorkerHandle] = {}
        self._mcp_apps: list[tuple[str, Any]] = []  # (path, mcp_app) pairs for lifespan
        self._mcp_subprocess: Process | None = None  # ToolUniverse MCP subprocess
        self._mcp_subprocess_port: int = 8101  # Port for MCP subprocess
        self.app = FastAPI(
            title="Ability Service",
            description="FastAPI server for AI Inventor abilities",
            version="1.0.0",
            lifespan=self._lifespan,
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Health check endpoint with detailed worker status
        @self.app.get("/health")
        async def health():
            worker_status = {}
            for name, w in self.workers.items():
                alive = w.process is not None and w.process.is_alive()
                worker_status[name] = {
                    "alive": alive,
                    "pid": w.process.pid if w.process else None,
                    "restart_count": w._restart_count,
                    "last_restart": w._last_restart.isoformat() if w._last_restart else None,
                }

            # ToolUniverse MCP subprocess status
            mcp_subprocess_status = None
            if self._mcp_subprocess is not None:
                mcp_subprocess_status = {
                    "alive": self._mcp_subprocess.is_alive(),
                    "pid": self._mcp_subprocess.pid,
                    "port": self._mcp_subprocess_port,
                    "url": f"http://localhost:{self._mcp_subprocess_port}/mcp",
                }

            return {
                "status": "ok",
                "endpoints": list(self.workers.keys()),
                "workers": worker_status,
                "mcp_servers": {path: {"status": "mounted"} for path, _ in self._mcp_apps},
                "tooluniverse_mcp": mcp_subprocess_status,
                "crash_log_dir": str(_CRASH_LOG_DIR),
            }

        # List endpoints
        @self.app.get("/endpoints")
        async def list_endpoints():
            return {"endpoints": list(self.workers.keys())}

        # Temp directory status
        @self.app.get("/temp_status")
        async def temp_status():
            """Get status of temp directories (size, file count, oldest file)."""
            status = {
                "cleanup_config": {
                    "max_age_hours": CLEANUP_MAX_AGE_HOURS,
                    "interval_minutes": CLEANUP_INTERVAL_MINUTES,
                },
                "directories": {},
            }
            now = time.time()

            for temp_dir in _get_temp_cleanup_dirs():
                dir_status = {
                    "exists": temp_dir.exists(),
                    "path": str(temp_dir),
                    "total_size_mb": 0.0,
                    "file_count": 0,
                    "oldest_file_hours": None,
                    "files": [],
                }

                if temp_dir.exists():
                    for f in temp_dir.rglob("*"):
                        if f.is_file():
                            try:
                                stat = f.stat()
                                size_mb = stat.st_size / (1024 * 1024)
                                age_hours = (now - stat.st_mtime) / 3600
                                dir_status["total_size_mb"] += size_mb
                                dir_status["file_count"] += 1
                                if dir_status["oldest_file_hours"] is None or age_hours > dir_status["oldest_file_hours"]:
                                    dir_status["oldest_file_hours"] = age_hours
                                # Show relative path from temp_dir
                                rel_path = str(f.relative_to(temp_dir))
                                dir_status["files"].append({
                                    "name": rel_path,
                                    "size_mb": round(size_mb, 2),
                                    "age_hours": round(age_hours, 2),
                                })
                            except Exception:
                                pass

                dir_status["total_size_mb"] = round(dir_status["total_size_mb"], 2)
                if dir_status["oldest_file_hours"]:
                    dir_status["oldest_file_hours"] = round(dir_status["oldest_file_hours"], 2)
                # Sort files by age (oldest first)
                dir_status["files"].sort(key=lambda x: x["age_hours"], reverse=True)
                status["directories"][temp_dir.name] = dir_status

            return status

        # Manual cleanup trigger
        @self.app.post("/cleanup")
        async def trigger_cleanup():
            """Manually trigger temp file cleanup."""
            stats = _cleanup_old_temp_files()
            return {
                "status": "ok",
                "checked": stats["checked"],
                "deleted": stats["deleted"],
                "freed_mb": round(stats["freed_mb"], 2),
                "errors": stats["errors"],
            }

    def _start_tooluniverse_mcp_subprocess(self) -> bool:
        """Start ToolUniverse MCP as a separate subprocess.

        Skipped if AII_SKIP_MCP_SUBPROCESS=1 (when running MCP in separate zellij session).
        """
        log = logger.bind(source="tooluniverse_mcp")

        # Skip if env var set (running in separate zellij session)
        if os.environ.get("AII_SKIP_MCP_SUBPROCESS") == "1":
            log.info("Skipping subprocess (AII_SKIP_MCP_SUBPROCESS=1, run separately)")
            return True

        if self._mcp_subprocess is not None and self._mcp_subprocess.is_alive():
            log.info("ToolUniverse MCP subprocess already running")
            return True

        try:
            # Get config
            max_workers = _server_config.get("endpoints", {}).get("tooluniverse_mcp", {}).get("max_threads", 30)

            # Import the run_server function from the dedicated module
            from aii_lib.abilities.ability_server.tooluniverse_mcp import run_server

            self._mcp_subprocess = Process(
                target=run_server,
                kwargs={"port": self._mcp_subprocess_port, "max_workers": max_workers},
                daemon=True,
            )
            self._mcp_subprocess.start()

            log.info(f"Started (pid={self._mcp_subprocess.pid}, port={self._mcp_subprocess_port}, max_workers={max_workers})")
            return True

        except Exception as e:
            log.error(f"Failed to start: {e}")
            import traceback
            log.error(traceback.format_exc())
            raise

    def _stop_tooluniverse_mcp_subprocess(self):
        """Stop ToolUniverse MCP subprocess."""
        log = logger.bind(source="tooluniverse_mcp")

        if self._mcp_subprocess is None:
            return

        log.info(f"Stopping (pid={self._mcp_subprocess.pid})...")

        try:
            self._mcp_subprocess.terminate()
            self._mcp_subprocess.join(timeout=5)
        except Exception as e:
            log.warning(f"Error during graceful shutdown: {e}")

        if self._mcp_subprocess.is_alive():
            log.warning("Subprocess didn't stop gracefully, killing...")
            self._mcp_subprocess.kill()
            self._mcp_subprocess.join(timeout=2)

        self._mcp_subprocess = None
        log.info("Stopped")

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """FastAPI lifespan handler - startup and shutdown."""
        log = logger.bind(source="server")
        log.info(f"Starting Ability Service with {len(self.workers)} endpoints")
        log.info(f"Crash logs: {_CRASH_LOG_DIR}")

        # Start ToolUniverse MCP subprocess (separate process for crash isolation)
        self._start_tooluniverse_mcp_subprocess()

        # Start all workers
        for name, worker in self.workers.items():
            try:
                worker.start()
            except Exception as e:
                logger.bind(source=name).error(f"Failed to start worker: {e}")

        # Start MCP app lifespans (for any mounted MCP apps)
        mcp_contexts = []
        for path, mcp_app in self._mcp_apps:
            try:
                if hasattr(mcp_app, 'lifespan') and mcp_app.lifespan:
                    ctx = mcp_app.lifespan(mcp_app)
                    await ctx.__aenter__()
                    mcp_contexts.append((path, ctx))
                    log.info(f"MCP server lifespan started: {path}")
            except Exception as e:
                log.error(f"Failed to start MCP lifespan for {path}: {e}")

        # Start temp file cleanup background task
        cleanup_task = asyncio.create_task(_cleanup_task())

        # Run initial cleanup on startup
        _cleanup_old_temp_files()

        yield

        # Cancel cleanup task
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

        # Shutdown MCP lifespans
        for path, ctx in reversed(mcp_contexts):
            try:
                await ctx.__aexit__(None, None, None)
                log.info(f"MCP server lifespan stopped: {path}")
            except Exception as e:
                log.error(f"Error stopping MCP lifespan for {path}: {e}")

        # Stop ToolUniverse MCP subprocess
        self._stop_tooluniverse_mcp_subprocess()

        # Shutdown all workers
        log.info("Shutting down Ability Service...")
        for name, worker in self.workers.items():
            try:
                worker.stop()
            except Exception as e:
                logger.bind(source=name).error(f"Error stopping worker: {e}")
        log.info("Ability Service stopped")

    def register(
        self,
        name: str,
        handler: Callable[[dict], dict],
        init_func: Callable[[], None] | None = None,
        max_threads: int = 10,
    ) -> None:
        """
        Register an endpoint with its own dedicated process.

        Args:
            name: Endpoint name (becomes POST /{name})
            handler: Function that takes dict request and returns dict response
            init_func: Function to run once in worker process (for imports/warmup)
            max_threads: Max concurrent requests per endpoint (default 10)
        """
        # Create worker handle
        worker = WorkerHandle(name, handler, init_func, max_threads)
        self.workers[name] = worker

        # Create endpoint factory to capture worker in closure
        def make_endpoint(w: WorkerHandle):
            async def endpoint(request: Request):
                """Handle request by dispatching to dedicated worker process."""
                try:
                    body = await request.json()
                    return await w.call(body)
                except Exception as e:
                    tb = traceback.format_exc()
                    w._log.error(f"Request handler error: {e}\n{tb}")
                    _write_crash_log(w.name, "REQUEST_HANDLER_ERROR", str(e), tb)
                    raise HTTPException(status_code=500, detail=str(e))
            return endpoint

        # Register endpoint with FastAPI
        self.app.add_api_route(
            f"/{name}",
            make_endpoint(worker),
            methods=["POST"],
            name=name,
            response_model=None,
        )

        logger.bind(source="server").info(f"Registered endpoint: POST /{name} (max_threads={max_threads})")

    def mount_mcp(
        self,
        path: str,
        mcp_app: Any,
        name: str = "MCP",
    ) -> None:
        """
        Mount an MCP server app with proper lifespan management.

        Args:
            path: URL path to mount at (e.g., "/tooluniverse_mcp")
            mcp_app: The MCP ASGI app (from smcp.streamable_http_app() etc.)
            name: Display name for logging
        """
        self.app.mount(path, mcp_app)
        self._mcp_apps.append((path, mcp_app))
        logger.bind(source="server").info(f"Mounted MCP server: {name} at {path}")


# Global instance
ability_service = AbilityService()
app = ability_service.app


# DEFAULT_PORT from config (loaded at top of file)
DEFAULT_PORT = _server_config.get("server", {}).get("port", 8100)


def run_server(port: int = DEFAULT_PORT, host: str = "0.0.0.0") -> None:
    """Run the ability service server."""
    import uvicorn

    log = logger.bind(source="server")
    log.info(f"Starting Ability Service on {host}:{port}")
    log.info(f"Crash logs directory: {_CRASH_LOG_DIR}")

    # Configure uvicorn logging to match loguru format
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)-7s | uvicorn                | %(message)s",
                "datefmt": "%H:%M:%S",
            },
            "access": {
                "format": "%(asctime)s | %(levelname)-7s | uvicorn.access         | %(client_addr)s - %(request_line)s %(status_code)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }

    uvicorn.run(app, host=host, port=port, log_config=log_config)


if __name__ == "__main__":
    run_server()
