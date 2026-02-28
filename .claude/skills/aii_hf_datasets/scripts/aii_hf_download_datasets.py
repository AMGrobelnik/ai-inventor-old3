#!/usr/bin/env python
"""
HuggingFace Dataset Download Tool

Download datasets from HuggingFace Hub.

Usage:
    python aii_hf_download_datasets.py openai/gsm8k --config main
    python aii_hf_download_datasets.py openai/gsm8k --config main --split train
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

SERVER_NAME = "aii_hf_download_datasets"
DATASETS_DIR = str(Path(__file__).parent.parent / "temp" / "datasets")
CONNECTION_TIMEOUT = 180  # seconds

# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def init_download_dataset():
    """Initialize HuggingFace environment for download."""
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(CONNECTION_TIMEOUT)

    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()

    import logging
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)

    # Pre-import to cache
    from datasets import load_dataset
    import pandas
    import json

    # Warmup with tiny dataset slice
    try:
        ds = load_dataset("dair-ai/emotion", split="train[:3]")
        ds.to_pandas()
    except Exception:
        pass


def core_download_dataset(**kwargs) -> dict:
    """
    Download a HuggingFace dataset.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "openai/gsm8k")
        config: Dataset configuration/subset name
        split: Specific split to load (optional, loads all if empty)
        output_dir: Directory to save files

    Returns:
        Dict with success status and file paths
    """
    import gc
    import json
    import traceback
    from datasets import load_dataset

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(CONNECTION_TIMEOUT)

    dataset_id = kwargs.get("dataset_id", "")
    config = kwargs.get("config") or None
    split = kwargs.get("split") or None
    output_dir = kwargs.get("output_dir") or DATASETS_DIR

    def truncate_value(value, max_array=3, max_str=200):
        if isinstance(value, list):
            return [truncate_value(v) for v in value[:max_array]]
        elif isinstance(value, str):
            return value[:max_str] + "..." if len(value) > max_str else value
        elif isinstance(value, dict):
            return {k: truncate_value(v) for k, v in value.items()}
        return value

    try:
        os.makedirs(output_dir, exist_ok=True)
        ds = load_dataset(dataset_id, config, split=split)
        safe_name = dataset_id.replace("/", "_")

        result = {
            "success": True,
            "dataset_id": dataset_id,
            "config": config,
            "splits": {},
            "output_files": [],
        }

        def save_split(split_ds, split_name):
            base_name = f"{safe_name}_{config}_{split_name}" if config else f"{safe_name}_{split_name}"

            # Mini (3 full rows) - extract first before streaming
            mini_data = [dict(split_ds[i]) for i in range(min(3, len(split_ds)))]
            mini_file = Path(output_dir) / f"mini_{base_name}.json"
            with open(mini_file, 'w') as f:
                json.dump(mini_data, f, indent=2, ensure_ascii=False, default=str)

            # Preview (3 truncated rows)
            preview_data = [truncate_value(row) for row in mini_data]
            preview_file = Path(output_dir) / f"preview_{base_name}.json"
            with open(preview_file, 'w') as f:
                json.dump(preview_data, f, indent=2, ensure_ascii=False, default=str)

            # Full dataset - stream to disk in chunks to avoid RAM explosion
            full_file = Path(output_dir) / f"full_{base_name}.json"
            chunk_size = 1000
            with open(full_file, 'w') as f:
                f.write('[\n')
                first = True
                for i in range(0, len(split_ds), chunk_size):
                    chunk = split_ds.select(range(i, min(i + chunk_size, len(split_ds))))
                    for row in chunk:
                        if not first:
                            f.write(',\n')
                        first = False
                        json.dump(dict(row), f, ensure_ascii=False, default=str)
                    del chunk
                    gc.collect()
                f.write('\n]')

            return {
                "num_rows": len(split_ds),
                "preview_file": str(preview_file),
                "mini_file": str(mini_file),
                "full_file": str(full_file),
            }

        if hasattr(ds, 'keys'):
            for split_name, split_ds in ds.items():
                try:
                    result["splits"][split_name] = save_split(split_ds, split_name)
                    result["output_files"].append(result["splits"][split_name]["full_file"])
                except Exception as e:
                    result["splits"][split_name] = {"error": str(e)}
                finally:
                    gc.collect()
        else:
            split_name = split or "train"
            try:
                result["splits"][split_name] = save_split(ds, split_name)
                result["output_files"].append(result["splits"][split_name]["full_file"])
            except Exception as e:
                result["splits"][split_name] = {"error": str(e)}
            finally:
                gc.collect()

        return result
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download a HuggingFace dataset")
    parser.add_argument("dataset_id", help="HuggingFace dataset ID")
    parser.add_argument("--config", default="", help="Dataset configuration")
    parser.add_argument("--split", default="", help="Specific split to load")
    parser.add_argument("--output-dir", default=DATASETS_DIR, help="Output directory")
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server
    result = call_server(SERVER_NAME, {
        "dataset_id": args.dataset_id,
        "config": args.config,
        "split": args.split,
        "output_dir": args.output_dir,
    }, timeout=180.0)

    if result is None:
        print("Error: Ability service not available. Start with: uvicorn aii_lib.abilities.endpoints:app --port 8100", file=sys.stderr)
        sys.exit(1)

    if result.get("success"):
        print(f"\n✓ Downloaded: {result['dataset_id']}")
        for split_name, info in result.get("splits", {}).items():
            print(f"\n  {split_name}:")
            if info.get("error"):
                print(f"    Error: {info['error']}")
            else:
                print(f"    Rows: {info.get('num_rows', '?')}")
                print(f"    Preview: {info.get('preview_file', '')}")
                print(f"    Mini: {info.get('mini_file', '')}")
                print(f"    Full: {info.get('full_file', '')}")
    else:
        print(f"Error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
