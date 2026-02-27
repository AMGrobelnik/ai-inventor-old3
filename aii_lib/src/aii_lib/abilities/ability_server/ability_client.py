"""
Ability Client - HTTP client for calling the Ability Service.

Provides a simple interface to call ability endpoints via HTTP.

Usage:
    from aii_lib.abilities.ability_server import call_server, server_available

    # Check if service is available
    if server_available():
        result = call_server("aii_hf_search_datasets", {"query": "ML", "limit": 5})
"""

import os
from pathlib import Path
from typing import Any

import yaml

# Load config from server_config.yaml (cached at module load)
_CONFIG_FILE = Path(__file__).parent / "server_config.yaml"
_server_config: dict = {}
if _CONFIG_FILE.exists():
    with open(_CONFIG_FILE) as f:
        _server_config = yaml.safe_load(f) or {}

# Defaults from config
DEFAULT_PORT = _server_config.get("server", {}).get("port", 8100)
DEFAULT_TIMEOUT = float(_server_config.get("server", {}).get("timeout", 180.0))


def get_ability_service_url() -> str:
    """Get the ability service URL from environment or default."""
    host = os.environ.get("ABILITY_SERVICE_HOST", "localhost")
    port = os.environ.get("ABILITY_SERVICE_PORT", str(DEFAULT_PORT))
    return f"http://{host}:{port}"


def server_available(timeout: float = 2.0) -> bool:
    """
    Check if the ability service is available.

    Args:
        timeout: Timeout for health check request

    Returns:
        True if service is available, False otherwise
    """
    import httpx

    try:
        url = get_ability_service_url()
        response = httpx.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return False
    except Exception as e:
        raise RuntimeError(f"Unexpected error checking ability server health: {e}") from e


def call_server(
    endpoint: str,
    request: dict[str, Any],
    timeout: float = None,
) -> dict[str, Any] | None:
    """
    Call an ability endpoint via HTTP.

    Args:
        endpoint: Endpoint name (e.g., 'hf_search', 'web_fetch')
        request: Request data dict
        timeout: Request timeout in seconds

    Returns:
        Response dict from the endpoint, or None if unavailable
    """
    import httpx

    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    try:
        url = get_ability_service_url()
        response = httpx.post(
            f"{url}/{endpoint}",
            json=request,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise ConnectionError(f"Ability server unavailable for '{endpoint}': {e}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ability server error for '{endpoint}': {e.response.status_code} {e.response.text[:200]}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error calling ability server '{endpoint}': {e}") from e


# Aliases for convenience
call_ability = call_server
ability_available = server_available


__all__ = [
    "call_server",
    "server_available",
    "call_ability",
    "ability_available",
    "get_ability_service_url",
]
