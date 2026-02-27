#!/usr/bin/env python
"""
DBLP Bibliography Search

Search DBLP for papers by author/title/year. Returns metadata (title, authors,
venue, year, dblp_key). Use dblp_bib_fetch with the dblp_key to get BibTeX.

Usage:
    python dblp_bib_search.py --query "Vaswani attention 2017"
    python dblp_bib_search.py --query "Wei chain of thought" --max-results 3
"""

import argparse
import json
import sys
import time

import requests
from loguru import logger

SERVER_NAME = "dblp_bib_search"
DEFAULT_TIMEOUT = 180.0
SESSION_TIMEOUT = 30
POOL_CONNECTIONS = 10
POOL_MAXSIZE = 10

# DBLP API configuration (shared with dblp_bib_fetch.py)
DBLP_API_URL = "https://dblp.uni-trier.de/search/publ/api"
DBLP_BIB_URL = "https://dblp.uni-trier.de/rec/{key}.bib"
DBLP_MIN_INTERVAL = 2.0  # min seconds between any DBLP request to avoid 429
_dblp_last_request = 0.0  # timestamp of last DBLP request

# Session pooling for connection reuse
_session = None


def dblp_rate_limit() -> None:
    """Enforce minimum interval between all DBLP API requests."""
    global _dblp_last_request
    elapsed = time.time() - _dblp_last_request
    if elapsed < DBLP_MIN_INTERVAL:
        time.sleep(DBLP_MIN_INTERVAL - elapsed)
    _dblp_last_request = time.time()


def dblp_request_with_retry(url: str, params: dict = None, max_retries: int = 3) -> requests.Response:
    """Make a DBLP HTTP request with retry on 429."""
    global _session
    if _session is None:
        init_dblp()

    for attempt in range(max_retries + 1):
        response = _session.get(url, params=params, timeout=SESSION_TIMEOUT)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        wait = min(60, 2 ** attempt * 5)  # 5s, 10s, 20s, 60s
        logger.warning(f"DBLP 429 rate limit, retry {attempt + 1}/{max_retries} in {wait}s")
        time.sleep(wait)
    response.raise_for_status()  # raise on final failure
    return response


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

def init_dblp() -> None:
    """Initialize DBLP environment with session pooling and warmup request."""
    global _session
    import requests  # noqa: F811 - warm up import in worker process
    from requests.adapters import HTTPAdapter

    _session = requests.Session()
    adapter = HTTPAdapter(pool_maxsize=POOL_MAXSIZE, pool_connections=POOL_CONNECTIONS)
    _session.mount("https://", adapter)
    _session.mount("http://", adapter)
    _session.headers.update({
        "User-Agent": "aii-dblp-tool/1.0",
        "Accept": "application/json",
    })

    # Warmup
    try:
        _session.get(DBLP_API_URL, params={"q": "warmup", "format": "json", "h": 1}, timeout=10)
    except Exception:
        pass

    logger.info("DBLP tools initialized")


def core_dblp_search(**kwargs) -> dict:
    """
    Search DBLP for papers. Returns metadata (no BibTeX).

    Args:
        query: Search query string.
        max_results: Maximum papers to return (default: 5, max: 20).
        year_from: Only include papers from this year onward.
        year_to: Only include papers up to this year.

    Returns:
        Dict with success, total_found, returned, papers list.
    """
    query = kwargs.get("query", "")
    max_results = min(kwargs.get("max_results", 5), 20)
    year_from = kwargs.get("year_from")
    year_to = kwargs.get("year_to")

    try:
        params = {"q": query, "format": "json", "h": max_results * 2}
        dblp_rate_limit()
        response = dblp_request_with_retry(DBLP_API_URL, params=params)
        data = response.json()

        hits = data.get("result", {}).get("hits", {})
        total = int(hits.get("@total", "0"))
        publications = hits.get("hit", [])

        if not isinstance(publications, list):
            publications = [publications]

        papers = []
        for pub in publications:
            info = pub.get("info", {})

            # Parse authors
            authors_data = info.get("authors", {}).get("author", [])
            if not isinstance(authors_data, list):
                authors_data = [authors_data]
            authors = [
                a.get("text", "") if isinstance(a, dict) else str(a)
                for a in authors_data
            ]

            # Parse year
            year = None
            if info.get("year"):
                try:
                    year = int(info["year"])
                except (ValueError, TypeError):
                    pass

            # Year filtering
            if year_from and year and year < year_from:
                continue
            if year_to and year and year > year_to:
                continue

            # Extract DBLP key from URL
            dblp_url = info.get("url", "")
            dblp_key = dblp_url
            for prefix in ("https://dblp.org/rec/", "https://dblp.uni-trier.de/rec/"):
                dblp_key = dblp_key.replace(prefix, "")
            if dblp_key == dblp_url:
                dblp_key = ""

            papers.append({
                "title": info.get("title", ""),
                "authors": authors,
                "venue": info.get("venue", ""),
                "year": year,
                "dblp_key": dblp_key,
                "doi": info.get("doi", ""),
                "url": dblp_url,
            })

            if len(papers) >= max_results:
                break

        return {
            "success": True,
            "total_found": total,
            "returned": len(papers),
            "papers": papers,
        }

    except requests.exceptions.Timeout:
        logger.warning(f"DBLP search timed out for query: {query}")
        return {"success": False, "error": f"DBLP API timeout after {SESSION_TIMEOUT}s", "papers": []}
    except Exception as e:
        logger.error(f"DBLP search error: {e}")
        return {"success": False, "error": str(e), "papers": []}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Search DBLP for papers")
    parser.add_argument("--query", "-q", required=True, help="Search query (author + year works best)")
    parser.add_argument("--max-results", "-n", type=int, default=5, help="Max papers to return (default: 5, max: 20)")
    parser.add_argument("--year-from", type=int, default=None, help="Only papers from this year onward")
    parser.add_argument("--year-to", type=int, default=None, help="Only papers up to this year")
    parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server
    payload = {
        "query": args.query,
        "max_results": args.max_results,
    }
    if args.year_from is not None:
        payload["year_from"] = args.year_from
    if args.year_to is not None:
        payload["year_to"] = args.year_to

    result = call_server(SERVER_NAME, payload, timeout=DEFAULT_TIMEOUT)

    if result is None:
        print("Error: Ability service not available. Start with: bash aii_lib/src/aii_lib/utils/start_ability_server.sh", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if result.get("success"):
        print(f"Search: {args.query}")
        print(f"Found: {result.get('total_found', 0)} total, showing {result.get('returned', 0)}\n")
        for i, p in enumerate(result.get("papers", []), 1):
            authors = ", ".join(p.get("authors", []))
            print(f"{i}. {p.get('title', 'Untitled')}")
            print(f"   Authors: {authors}")
            print(f"   Venue: {p.get('venue', 'N/A')}  Year: {p.get('year', 'N/A')}")
            print(f"   DBLP key: {p.get('dblp_key', 'N/A')}")
            print()
    else:
        print(f"Error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
