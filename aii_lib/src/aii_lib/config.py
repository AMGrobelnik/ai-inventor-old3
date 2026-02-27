"""
Global configuration for aii_lib.

Singleton pattern - initialized once, accessible everywhere.

Config is serialized to /tmp/aii_config_cache.json so forked processes
can load API keys via load_from_cache().

Usage:
    # Initialize (called by aii_pipeline or consumer):
    from aii_lib.config import aii_config
    aii_config.init(api_keys={"serper": "...", "openrouter": "..."})

    # Or from PipelineConfig:
    aii_config.init_from_pipeline_config(pipeline_config)

    # Access from anywhere in aii_lib:
    from aii_lib.config import aii_config
    api_key = aii_config.api_keys.serper

    # In forked processes (init_func):
    from aii_lib.config import load_from_cache
    config = load_from_cache()  # Returns dict with api_keys
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
# Load project root .env (ai-inventor/.env)
_project_root = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_project_root / ".env")

# Cache file for passing config to forked processes
CONFIG_CACHE_PATH = Path("/tmp/aii_config_cache.json")



@dataclass
class APIKeysConfig:
    """API keys for external services. Reads from os.environ (populated by .env via load_dotenv)."""
    openai: str = ""
    openrouter: str = ""
    anthropic: str = ""
    gemini: str = ""
    serper: str = ""
    leanexplore: str = ""
    huggingface: str = ""

    @classmethod
    def from_env(cls) -> "APIKeysConfig":
        """Create from environment variables."""
        return cls(
            openai=os.environ.get("OPENAI_API_KEY", ""),
            openrouter=os.environ.get("OPENROUTER_API_KEY", ""),
            anthropic=os.environ.get("UNUSED_ANTHROPIC_API_KEY", ""),
            gemini=os.environ.get("GEMINI_API_KEY", ""),
            serper=os.environ.get("SERPER_API_KEY", ""),
            leanexplore=os.environ.get("LEANEXPLORE_API_KEY", ""),
            huggingface=os.environ.get("HF_TOKEN", ""),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "APIKeysConfig":
        """Create from dict, falling back to env vars for empty values."""
        return cls(
            openai=d.get("openai") or os.environ.get("OPENAI_API_KEY", ""),
            openrouter=d.get("openrouter") or os.environ.get("OPENROUTER_API_KEY", ""),
            anthropic=d.get("anthropic") or os.environ.get("UNUSED_ANTHROPIC_API_KEY", ""),
            gemini=d.get("gemini") or os.environ.get("GEMINI_API_KEY", ""),
            serper=d.get("serper") or os.environ.get("SERPER_API_KEY", ""),
            leanexplore=d.get("leanexplore") or os.environ.get("LEANEXPLORE_API_KEY", ""),
            huggingface=d.get("huggingface") or os.environ.get("HF_TOKEN", ""),
        )

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


class AiiLibConfig:
    """
    Global configuration singleton for aii_lib.

    Initialize once at startup, then access from anywhere.
    """
    _instance: Optional["AiiLibConfig"] = None
    _initialized: bool = False
    _servers_started: bool = False

    def __init__(self):
        self.api_keys: APIKeysConfig = APIKeysConfig.from_env()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict for serialization."""
        return {
            "api_keys": self.api_keys.to_dict(),
        }

    def serialize_to_cache(self) -> None:
        """Serialize config to cache file for forked processes to read."""
        CONFIG_CACHE_PATH.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def get(cls) -> "AiiLibConfig":
        """Get the singleton instance. Creates empty config if not initialized."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def start_ability_servers(cls) -> Dict[str, bool]:
        """
        Check ability service availability and start agent server.

        The ability service (FastAPI) should be started separately:
            uvicorn aii_lib.abilities.endpoints:app --port 8100

        Returns dict of {server_name: available}.
        """
        if cls._servers_started:
            return {}

        # Serialize config to cache file for worker processes
        instance = cls.get()
        instance.serialize_to_cache()

        results = {}

        # Check if ability service is available
        try:
            from aii_lib.utils import server_available
            if server_available():
                for name in ["hf_search", "hf_preview", "hf_download", "lean", "owid_query", "web_search", "web_fetch", "verify_quotes", "openrouter_search", "openrouter_call"]:
                    results[name] = True
        except Exception as e:
            import sys
            print(f"WARNING: Failed to check ability server availability: {e}", file=sys.stderr)

        cls._servers_started = True
        return results

    @classmethod
    def init(
        cls,
        api_keys: Optional[Dict[str, Any]] = None,
        start_servers: bool = True,
    ) -> "AiiLibConfig":
        """
        Initialize the global config.

        Args:
            api_keys: Dict with keys like {"serper": "...", "openrouter": "..."}
            start_servers: Whether to start ability servers (default: True)
        """
        instance = cls.get()

        if api_keys:
            instance.api_keys = APIKeysConfig.from_dict(api_keys)

        cls._initialized = True

        if start_servers:
            cls.start_ability_servers()

        return instance

    @classmethod
    def init_from_pipeline_config(cls, pipeline_config, start_servers: bool = True) -> "AiiLibConfig":
        """
        Initialize from a PipelineConfig object.

        Args:
            pipeline_config: PipelineConfig instance from aii_pipeline
            start_servers: Whether to start ability servers (default: True)
        """
        instance = cls.get()

        # Extract api_keys
        if hasattr(pipeline_config, 'api_keys'):
            ak = pipeline_config.api_keys
            instance.api_keys = APIKeysConfig(
                openai=getattr(ak, 'openai', ''),
                openrouter=getattr(ak, 'openrouter', ''),
                anthropic=getattr(ak, 'anthropic', ''),
                gemini=getattr(ak, 'gemini', ''),
                serper=getattr(ak, 'serper', ''),
                leanexplore=getattr(ak, 'leanexplore', ''),
                huggingface=getattr(ak, 'huggingface', ''),
            )

        cls._initialized = True

        if start_servers:
            cls.start_ability_servers()

        return instance

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if config has been explicitly initialized."""
        return cls._initialized

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (mainly for testing)."""
        cls._servers_started = False
        cls._instance = None
        cls._initialized = False


def load_from_cache() -> Dict[str, Any]:
    """
    Load config from cache file (for forked processes).

    Returns dict with api_keys, or empty dict if cache doesn't exist.
    This is called by init_func in forked server processes to get API keys.

    Usage in init_func:
        from aii_lib.config import load_from_cache
        config = load_from_cache()
        os.environ["SERPER_API_KEY"] = config.get("api_keys", {}).get("serper", "")
    """
    if CONFIG_CACHE_PATH.exists():
        try:
            return json.loads(CONFIG_CACHE_PATH.read_text())
        except Exception as e:
            import sys
            print(f"ERROR: Failed to load config cache from {CONFIG_CACHE_PATH}: {e}", file=sys.stderr)
            raise
    return {}


# Global singleton instance
aii_config = AiiLibConfig.get()
