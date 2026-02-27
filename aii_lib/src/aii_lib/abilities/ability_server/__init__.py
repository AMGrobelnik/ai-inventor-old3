"""
Utilities for aii_lib abilities.

Provides HTTP client for the Ability Service, and convenience functions
for calling individual ability endpoints.

Note: Server components (ability_service, app, run_server) are lazy-loaded
to avoid importing FastAPI when only using the client (call_server).
"""

from typing import Any

from .ability_client import (
    call_server,
    server_available,
    get_ability_service_url,
)

# Lazy-load server components to avoid FastAPI import overhead (~600ms)
DEFAULT_PORT = 8100  # Duplicated here to avoid import

def __getattr__(name: str):
    """Lazy load server components only when accessed."""
    if name in ("ability_service", "app", "run_server"):
        from .ability_service import ability_service, app, run_server
        globals()["ability_service"] = ability_service
        globals()["app"] = app
        globals()["run_server"] = run_server
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# Convenience functions - call endpoints via HTTP
# =============================================================================

def call_ability(name: str, **kwargs) -> dict[str, Any]:
    """Call an ability by name with keyword arguments."""
    result = call_server(name, kwargs)
    if result is None:
        return {"success": False, "error": f"Ability service not available"}
    return result


def hf_search(query: str, limit: int = 5, tags: str = "", sort: str = "downloads") -> dict[str, Any]:
    """Search HuggingFace datasets."""
    return call_ability("aii_hf_search_datasets", query=query, limit=limit, tags=tags, sort=sort)


def hf_preview(dataset_id: str, config: str = None, split: str = "train", num_rows: int = 5) -> dict[str, Any]:
    """Preview a HuggingFace dataset."""
    return call_ability("aii_hf_preview_datasets", dataset_id=dataset_id, config=config, split=split, num_rows=num_rows)


def hf_download(dataset_id: str, config: str = None, split: str = None) -> dict[str, Any]:
    """Download a HuggingFace dataset."""
    return call_ability("aii_hf_download_datasets", dataset_id=dataset_id, config=config, split=split)


def aii_web_search(query: str, max_results: int = 10) -> dict[str, Any]:
    """Search the web using Serper.dev."""
    return call_ability("aii_web_search", query=query, max_results=max_results)


def aii_web_fetch(
    url: str,
    max_chars: int = 10000,
    char_offset: int = 0,
) -> dict[str, Any]:
    """Fetch a URL as markdown text with optional pagination."""
    return call_ability(
        "aii_web_fetch",
        url=url,
        max_chars=max_chars,
        char_offset=char_offset,
    )


def aii_web_fetch_grep(
    url: str,
    pattern: str,
    max_matches: int = 20,
    context_chars: int = 200,
    chars_before: int = None,
    chars_after: int = None,
    case_insensitive: bool = False,
) -> dict[str, Any]:
    """Grep through a URL (HTML or PDF) for a regex pattern."""
    kwargs = {
        "url": url,
        "pattern": pattern,
        "max_matches": max_matches,
        "context_chars": context_chars,
        "case_insensitive": case_insensitive,
    }
    if chars_before is not None:
        kwargs["chars_before"] = chars_before
    if chars_after is not None:
        kwargs["chars_after"] = chars_after
    return call_ability("aii_web_fetch_grep", **kwargs)


def dblp_bib_search(
    query: str,
    max_results: int = 5,
    year_from: int = None,
    year_to: int = None,
) -> dict[str, Any]:
    """Search DBLP bibliography (metadata only, no BibTeX)."""
    kwargs = {
        "query": query,
        "max_results": max_results,
    }
    if year_from is not None:
        kwargs["year_from"] = year_from
    if year_to is not None:
        kwargs["year_to"] = year_to
    return call_ability("dblp_bib_search", **kwargs)


def dblp_bib_fetch(
    dblp_keys: list[str] | str,
    years: list[int] | int = None,
) -> dict[str, Any]:
    """Fetch BibTeX entries from DBLP by key."""
    kwargs = {"dblp_keys": dblp_keys}
    if years is not None:
        kwargs["years"] = years
    return call_ability("dblp_bib_fetch", **kwargs)


def verify_quotes(text: str) -> dict[str, Any]:
    """Verify citations in text."""
    return call_ability("aii_verify_quotes", text=text)


def lean_run(code: str) -> dict[str, Any]:
    """Run Lean 4 code with Mathlib."""
    return call_ability("aii_lean", code=code)


def lean_suggest(code: str, tactics: str = "exact?,apply?,simp?,rw?,simp,aesop,omega,decide,ring,linarith,nlinarith,norm_num,field_simp,positivity") -> dict[str, Any]:
    """Try tactics at sorry positions in Lean 4 code."""
    return call_ability("aii_lean_suggest", code=code, tactics=tactics)


def owid_search(query: str, limit: int = 3) -> dict[str, Any]:
    """Search OWID tables (metadata only, no data download)."""
    return call_ability("aii_owid_search_datasets", query=query, limit=limit)


def owid_download(path: str, output_dir: str = None) -> dict[str, Any]:
    """Download an OWID table by path."""
    kwargs = {"path": path}
    if output_dir:
        kwargs["output_dir"] = output_dir
    return call_ability("aii_owid_download_datasets", **kwargs)


def json_validate(format_type: str, file_path: str, strict: bool = False) -> dict[str, Any]:
    """Validate a JSON file against a schema."""
    return call_ability("aii_json_validate", format_type=format_type, file_path=file_path, strict=strict)


def openrouter_search(query: str = "", limit: int = 10, series: str = "") -> dict[str, Any]:
    """Search OpenRouter models."""
    return call_ability("aii_openrouter_search", query=query, limit=limit, series=series)


def openrouter_call(model: str, input_text: str = None, **kwargs) -> dict[str, Any]:
    """Call an OpenRouter LLM model."""
    return call_ability("aii_openrouter_call", model=model, input_text=input_text, **kwargs)


def image_gen_nano_banana(
    prompt: str,
    output_path: str = "./generated_image.png",
    aspect_ratio: str = "16:9",
    image_size: str = "1K",
    style: str = None,
    negative_prompt: str = None,
    system_instruction: str = None,
) -> dict[str, Any]:
    """Generate an image using Gemini image models via OpenRouter."""
    kwargs = {
        "prompt": prompt,
        "output_path": output_path,
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
    }
    if style is not None:
        kwargs["style"] = style
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    if system_instruction is not None:
        kwargs["system_instruction"] = system_instruction
    return call_ability("aii_image_gen_nano_banana", **kwargs)


__all__ = [
    # Client functions
    "call_server",
    "server_available",
    "call_ability",
    "get_ability_service_url",
    # Service
    "ability_service",
    "app",
    "DEFAULT_PORT",
    "run_server",
    # Convenience functions
    "hf_search",
    "hf_preview",
    "hf_download",
    "aii_web_search",
    "aii_web_fetch",
    "aii_web_fetch_grep",
    "dblp_bib_search",
    "dblp_bib_fetch",
    "verify_quotes",
    "lean_run",
    "lean_suggest",
    "owid_search",
    "owid_download",
    "json_validate",
    "openrouter_search",
    "openrouter_call",
    "image_gen_nano_banana",
]
