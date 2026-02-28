#!/usr/bin/env python
"""
HuggingFace Dataset Search Tool

Search and discover datasets on HuggingFace Hub with metadata.

Usage:
    python aii_hf_search_datasets.py --query "machine learning" --limit 5
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

SERVER_NAME = "aii_hf_search_datasets"
CONNECTION_TIMEOUT = 180  # seconds

# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Global HfApi instance for session reuse
_hf_api = None


def init_search_datasets():
    """Initialize HuggingFace environment for search."""
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

    # Pre-import to cache
    from huggingface_hub import HfApi, DatasetCard

    # Create global HfApi instance for session reuse
    _hf_api = HfApi()

    # Warmup API connection
    try:
        datasets = list(_hf_api.list_datasets(search="test", limit=1))
        if datasets:
            DatasetCard.load(datasets[0].id)
    except Exception:
        pass


def core_search_datasets(**kwargs) -> dict:
    """
    Search HuggingFace datasets.

    Args:
        query: Search query string
        limit: Maximum number of results (default: 5)
        tags: Comma-separated tags to filter by
        sort: Sort by 'downloads' or 'likes' (default: downloads)

    Returns:
        Dict with success, query, count, and results list
    """
    from huggingface_hub import DatasetCard

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(CONNECTION_TIMEOUT)

    query = kwargs.get("query", "")
    limit = kwargs.get("limit", 5)
    tags = kwargs.get("tags", "")
    sort = kwargs.get("sort", "downloads")

    global _hf_api
    api = _hf_api  # Reuse global session
    try:
        # tags are passed via filter param (tags= was deprecated in huggingface_hub)
        tag_filters = tags.split(",") if tags else None
        datasets = list(api.list_datasets(
            search=query,
            sort=sort,
            direction=-1,
            limit=limit,
            filter=tag_filters,
        ))

        results = []
        for ds in datasets:
            info = {
                "id": ds.id,
                "downloads": ds.downloads,
                "likes": ds.likes,
                "tags": ds.tags[:10] if ds.tags else [],
            }
            try:
                card = DatasetCard.load(ds.id)
                info["description"] = card.text[:500] if card.text else ""
            except Exception:
                info["description"] = ""
            results.append(info)

        return {"success": True, "query": query, "count": len(results), "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Search datasets on HuggingFace Hub")
    parser.add_argument("--query", default="", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Max results")
    parser.add_argument("--tags", default="", help="Filter by tags (comma-separated)")
    parser.add_argument("--sort", choices=["downloads", "likes"], default="downloads")
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server
    result = call_server(SERVER_NAME, {
        "query": args.query,
        "limit": args.limit,
        "tags": args.tags,
        "sort": args.sort,
    })

    if result is None:
        print("Error: Ability service not available. Start with: uvicorn aii_lib.abilities.endpoints:app --port 8100", file=sys.stderr)
        sys.exit(1)

    if result.get("success"):
        print(f"Found {result['count']} dataset(s) for query='{result['query']}'")
        for i, ds in enumerate(result.get("results", []), 1):
            print(f"\n{'='*60}")
            print(f"Dataset {i}: {ds['id']}")
            print(f"Downloads: {ds.get('downloads', 0):,} | Likes: {ds.get('likes', 0)}")
            if ds.get('description'):
                print(f"Description: {ds['description'][:200]}...")
            if ds.get('tags'):
                print(f"Tags: {', '.join(ds['tags'][:5])}")
    else:
        print(f"Error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
