#!/usr/bin/env python
"""
HuggingFace Dataset Preview Tool

Preview a dataset's metadata and sample rows.

Usage:
    python aii_hf_preview_datasets.py openai/gsm8k
    python aii_hf_preview_datasets.py glue --config mrpc --split validation
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

SERVER_NAME = "aii_hf_preview_datasets"
CONNECTION_TIMEOUT = 180  # seconds

# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Global HfApi instance for session reuse
_hf_api = None


def init_preview_dataset():
    """Initialize HuggingFace environment for preview."""
    global _hf_api
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(CONNECTION_TIMEOUT)

    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()

    import logging
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.repocard").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)

    # Pre-import to cache
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, DatasetCard
    from huggingface_hub.utils import RepositoryNotFoundError
    from datasets import load_dataset, get_dataset_config_names

    # Create global HfApi instance for session reuse
    _hf_api = HfApi()

    # Warmup API connection
    try:
        _hf_api.dataset_info("dair-ai/emotion")
        ds = load_dataset("dair-ai/emotion", split="train", streaming=True)
        next(iter(ds))
    except Exception:
        pass


def core_preview_dataset(**kwargs) -> dict:
    """
    Preview a HuggingFace dataset - metadata and sample rows.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "openai/gsm8k")
        config: Dataset configuration/subset name (optional)
        split: Split to preview (default: train)
        num_rows: Number of sample rows (default: 5, max: 20)

    Returns:
        Dict with metadata and sample rows
    """
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, DatasetCard
    from huggingface_hub.utils import RepositoryNotFoundError
    from datasets import load_dataset, get_dataset_config_names

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(CONNECTION_TIMEOUT)

    dataset_id = kwargs.get("dataset_id", "")
    config = kwargs.get("config") or None
    split = kwargs.get("split", "train")
    num_rows = min(kwargs.get("num_rows", 5), 20)

    def truncate(text, max_len=500):
        if not text:
            return ""
        text = str(text)
        return text[:max_len] + f"... (+{len(text) - max_len} chars)" if len(text) > max_len else text

    def truncate_value(value, max_array=3, max_str=200):
        # Handle None and primitives
        if value is None:
            return None
        if isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value[:max_str] + "..." if len(value) > max_str else value
        if isinstance(value, bytes):
            return f"<bytes: {len(value)} bytes>"
        if isinstance(value, list):
            return [truncate_value(v) for v in value[:max_array]]
        if isinstance(value, tuple):
            return [truncate_value(v) for v in value[:max_array]]
        if isinstance(value, dict):
            return {str(k): truncate_value(v) for k, v in list(value.items())[:max_array]}
        # Handle numpy/PIL/other objects
        type_name = type(value).__name__
        if hasattr(value, 'shape'):  # numpy array
            return f"<{type_name}: shape={value.shape}>"
        if hasattr(value, 'size') and hasattr(value, 'mode'):  # PIL Image
            return f"<{type_name}: size={value.size}, mode={value.mode}>"
        # Fallback: convert to string
        try:
            s = str(value)
            return s[:max_str] + "..." if len(s) > max_str else s
        except Exception:
            return f"<{type_name}>"

    result = {"success": True, "dataset_id": dataset_id, "config": config, "split": split}
    global _hf_api
    api = _hf_api  # Reuse global session

    # Fetch metadata in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_info = executor.submit(api.dataset_info, dataset_id)
        future_card = executor.submit(DatasetCard.load, dataset_id)
        future_configs = executor.submit(get_dataset_config_names, dataset_id)

        try:
            info = future_info.result()
            result["downloads"] = getattr(info, "downloads", 0)
            result["likes"] = getattr(info, "likes", 0)
            if getattr(info, "created_at", None):
                result["created_at"] = str(info.created_at)[:10]
            if getattr(info, "last_modified", None):
                result["last_modified"] = str(info.last_modified)[:10]
            if getattr(info, "tags", None):
                result["tags"] = info.tags
        except RepositoryNotFoundError:
            return {"success": False, "error": f"Dataset '{dataset_id}' not found"}
        except Exception:
            pass

        try:
            card = future_card.result()
            if card and card.text:
                result["description"] = truncate(card.text, 500)
        except Exception:
            pass

        try:
            config_names = future_configs.result()
            if config_names:
                result["configs"] = config_names
        except Exception:
            config_names = []

    # Determine config
    actual_config = config or (config_names[0] if config_names else None)
    result["config"] = actual_config

    # Load sample rows
    try:
        load_kwargs = {"path": dataset_id, "split": split, "streaming": True}
        if actual_config:
            load_kwargs["name"] = actual_config

        ds = load_dataset(**load_kwargs)
        rows, columns = [], []
        for i, row in enumerate(ds):
            if i >= num_rows:
                break
            rows.append(truncate_value(dict(row)))
            if i == 0:
                columns = list(row.keys())

        result["columns"] = columns
        result["sample_rows"] = rows
        result["num_sample_rows"] = len(rows)
    except Exception as e:
        result["sample_error"] = str(e)

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Preview a HuggingFace dataset")
    parser.add_argument("dataset_id", help="HuggingFace dataset ID")
    parser.add_argument("--config", default="", help="Dataset configuration")
    parser.add_argument("--split", default="train", help="Split to preview")
    parser.add_argument("--num-rows", type=int, default=5, help="Number of sample rows")
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server
    result = call_server(SERVER_NAME, {
        "dataset_id": args.dataset_id,
        "config": args.config,
        "split": args.split,
        "num_rows": args.num_rows,
    })

    if result is None:
        print("Error: Ability service not available. Start with: uvicorn aii_lib.abilities.endpoints:app --port 8100", file=sys.stderr)
        sys.exit(1)

    if result.get("success"):
        print(f"\n{'='*60}")
        print(f"Dataset: {result['dataset_id']}")
        print(f"{'='*60}")
        if result.get("downloads") is not None:
            print(f"Downloads: {result['downloads']:,} | Likes: {result.get('likes', 0)}")
        if result.get("description"):
            print(f"\nDescription: {result['description']}")
        if result.get("configs"):
            print(f"\nConfigs: {', '.join(result['configs'][:10])}")
        print(f"\n--- Sample Rows ({result['split']}) ---")
        if result.get("columns"):
            print(f"Columns: {', '.join(result['columns'][:15])}")
        for i, row in enumerate(result.get("sample_rows", []), 1):
            print(f"\nRow {i}:")
            for k, v in row.items():
                v_str = str(v)[:200] + "..." if len(str(v)) > 200 else str(v)
                print(f"  {k}: {v_str}")
    else:
        print(f"Error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
