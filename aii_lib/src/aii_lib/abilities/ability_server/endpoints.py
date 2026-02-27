"""
Ability Endpoints - Register skill core functions with the FastAPI service.

Skills are fully self-contained with hardcoded API keys. Core functions accept **kwargs
and are registered directly without wrapper handlers.

Usage:
    uvicorn aii_lib.abilities.ability_server.endpoints:app --port 8100
"""

import sys
from pathlib import Path

import yaml
from loguru import logger

from aii_lib.abilities.ability_server.ability_service import ability_service, app, DEFAULT_PORT, with_retry


# =============================================================================
# Config and paths
# =============================================================================

CONFIG_FILE = Path(__file__).parent / "server_config.yaml"
SKILLS_DIR = Path(__file__).parents[5] / ".claude" / "skills"

# Add skill script directories to sys.path
TOOLS_DIR = Path(__file__).parents[1] / "tools"

_skill_paths = [
    TOOLS_DIR / "_aii_web_tools",  # Web search/fetch (moved from aii_fast_web_research skill)
    SKILLS_DIR / "aii_hf_datasets" / "scripts",  # HuggingFace datasets (uses datasets library)
    SKILLS_DIR / "aii_lean" / "scripts",
    SKILLS_DIR / "aii_openrouter_llms" / "scripts",
    SKILLS_DIR / "aii_owid_datasets" / "scripts",
    SKILLS_DIR / "aii_json" / "scripts",
    SKILLS_DIR / "dblp_bib" / "scripts",
    SKILLS_DIR / "aii_image_gen_nano_banana" / "scripts",
]
for p in _skill_paths:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Load server config (for max_threads settings)
_server_config: dict = {}
if CONFIG_FILE.exists():
    with open(CONFIG_FILE) as f:
        _server_config = yaml.safe_load(f) or {}


def _threads(name: str) -> int:
    """Get max_threads for an endpoint from config."""
    return _server_config.get("endpoints", {}).get(name, {}).get("max_threads", 10)


# =============================================================================
# Import skill functions (fully self-contained with hardcoded API keys)
# =============================================================================

# HuggingFace (aii_hf_datasets skill - uses datasets library for better compatibility)
from aii_hf_search_datasets import init_search_datasets, core_search_datasets
from aii_hf_preview_datasets import init_preview_dataset, core_preview_dataset
from aii_hf_download_datasets import init_download_dataset, core_download_dataset

# Fast Web Research
from aii_fast_web_search import init_web_search, core_web_search
from aii_fast_web_fetch import init_web_fetch, core_web_fetch, core_web_grep
from aii_verify_quotes import init_verify_quotes, core_verify_quotes

# Lean
from aii_run_lean import init_run_lean, core_run_lean
from aii_lean_suggest import init_lean_suggest, core_lean_suggest
from aii_mathlib_semantic_search import init_mathlib_semantic_search, core_mathlib_semantic_search
from aii_mathlib_pattern_search import init_mathlib_pattern_search, core_mathlib_pattern_search

# OpenRouter
from aii_or_search_llms import init_openrouter_search, core_openrouter_search
from aii_or_call_llms import init_openrouter_call, core_openrouter_call
from aii_or_get_llm_params import init_openrouter_get_params, core_openrouter_get_params

# OWID
from aii_owid_search_datasets import init_owid_search, core_owid_search
from aii_owid_download_datasets import init_owid_download, core_owid_download

# JSON
from aii_json_validate_schema import init_json_validate, core_json_validate
from aii_json_format_mini_preview import init_json_format, core_json_format

# DBLP
from dblp_bib_search import init_dblp, core_dblp_search
from dblp_bib_fetch import core_dblp_fetch

# Image Generation (nano_banana)
from aii_image_gen_nano_banana import init_image_gen_nano_banana, core_image_gen_nano_banana




# =============================================================================
# Handlers (named functions for pickling compatibility with multiprocessing)
# All handlers wrapped with @with_retry for transient error recovery
# =============================================================================

# HuggingFace handlers (aii_hf_datasets skill)
@with_retry
def _hf_search(req): return core_search_datasets(**req)
@with_retry
def _hf_preview(req): return core_preview_dataset(**req)
@with_retry
def _hf_download(req): return core_download_dataset(**req)

@with_retry
def _web_search(req): return core_web_search(**req)
@with_retry
def _web_fetch(req): return core_web_fetch(**req)
@with_retry
def _web_grep(req): return core_web_grep(**req)
@with_retry
def _verify_quotes(req): return core_verify_quotes(**req)
@with_retry
def _lean(req): return core_run_lean(**req)
@with_retry
def _lean_suggest(req): return core_lean_suggest(**req)
@with_retry
def _mathlib_semantic_search(req): return core_mathlib_semantic_search(**req)
@with_retry
def _mathlib_pattern_search(req): return core_mathlib_pattern_search(**req)
@with_retry
def _openrouter_search(req): return core_openrouter_search(**req)
@with_retry
def _openrouter_call(req): return core_openrouter_call(**req)
@with_retry
def _openrouter_get_params(req): return core_openrouter_get_params(**req)
@with_retry
def _owid_search(req): return core_owid_search(**req)
@with_retry
def _owid_download(req): return core_owid_download(**req)
@with_retry
def _json_validate(req): return core_json_validate(**req)
@with_retry
def _json_format(req): return core_json_format(**req)
@with_retry
def _dblp_search(req): return core_dblp_search(**req)
@with_retry
def _dblp_fetch(req): return core_dblp_fetch(**req)
@with_retry
def _image_gen_nano_banana(req): return core_image_gen_nano_banana(**req)


# =============================================================================
# Register all endpoints
# =============================================================================

# HuggingFace (aii_hf_datasets skill)
ability_service.register("aii_hf_search_datasets", _hf_search, init_search_datasets, max_threads=_threads("aii_hf_search_datasets"))
ability_service.register("aii_hf_preview_datasets", _hf_preview, init_preview_dataset, max_threads=_threads("aii_hf_preview_datasets"))
ability_service.register("aii_hf_download_datasets", _hf_download, init_download_dataset, max_threads=_threads("aii_hf_download_datasets"))

# Web
ability_service.register("aii_web_search", _web_search, init_web_search, max_threads=_threads("aii_web_search"))
ability_service.register("aii_web_fetch", _web_fetch, init_web_fetch, max_threads=_threads("aii_web_fetch"))
ability_service.register("aii_web_fetch_grep", _web_grep, init_web_fetch, max_threads=_threads("aii_web_fetch_grep"))
ability_service.register("aii_verify_quotes", _verify_quotes, init_verify_quotes, max_threads=_threads("aii_verify_quotes"))

# Lean
ability_service.register("aii_lean", _lean, init_run_lean, max_threads=_threads("aii_lean"))
ability_service.register("aii_lean_suggest", _lean_suggest, init_lean_suggest, max_threads=_threads("aii_lean_suggest"))
ability_service.register("aii_mathlib_semantic_search", _mathlib_semantic_search, init_mathlib_semantic_search, max_threads=_threads("aii_mathlib_semantic_search"))
ability_service.register("aii_mathlib_pattern_search", _mathlib_pattern_search, init_mathlib_pattern_search, max_threads=_threads("aii_mathlib_pattern_search"))

# OpenRouter
ability_service.register("aii_openrouter_search", _openrouter_search, init_openrouter_search, max_threads=_threads("aii_openrouter_search"))
ability_service.register("aii_openrouter_call", _openrouter_call, init_openrouter_call, max_threads=_threads("aii_openrouter_call"))
ability_service.register("aii_openrouter_get_params", _openrouter_get_params, init_openrouter_get_params, max_threads=_threads("aii_openrouter_get_params"))

# OWID
ability_service.register("aii_owid_search_datasets", _owid_search, init_owid_search, max_threads=_threads("aii_owid_search_datasets"))
ability_service.register("aii_owid_download_datasets", _owid_download, init_owid_download, max_threads=_threads("aii_owid_download_datasets"))

# JSON
ability_service.register("aii_json_validate", _json_validate, init_json_validate, max_threads=_threads("aii_json_validate"))
ability_service.register("aii_json_format", _json_format, init_json_format, max_threads=_threads("aii_json_format"))

# DBLP (single-threaded — both share DBLP's per-IP rate limit)
ability_service.register("dblp_bib_search", _dblp_search, init_dblp, max_threads=_threads("dblp_bib_search"))
ability_service.register("dblp_bib_fetch", _dblp_fetch, init_dblp, max_threads=_threads("dblp_bib_fetch"))

# Image Generation (nano_banana)
ability_service.register("aii_image_gen_nano_banana", _image_gen_nano_banana, init_image_gen_nano_banana, max_threads=_threads("aii_image_gen_nano_banana"))


# =============================================================================
# Server entry point
# =============================================================================

def run_server(port: int = None, host: str = None, reload: bool = False) -> None:
    """Run the ability service server.

    Args:
        port: Port to run on (default: from config or 8100)
        host: Host to bind to (default: from config or 0.0.0.0)
        reload: Enable auto-reload on code changes (for development)
    """
    import uvicorn
    server_cfg = _server_config.get("server", {})
    port = port or server_cfg.get("port", DEFAULT_PORT)
    host = host or server_cfg.get("host", "0.0.0.0")
    logger.bind(source="server").info(f"Starting Ability Service on {host}:{port}" + (" (reload mode)" if reload else ""))
    uvicorn.run(
        "aii_lib.abilities.ability_server.endpoints:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(SKILLS_DIR), str(Path(__file__).parent)] if reload else None,
    )


if __name__ == "__main__":
    run_server()
