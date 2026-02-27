#!/usr/bin/env python
"""
DBLP BibTeX Fetch

Fetch BibTeX entries from DBLP by key. Use dblp_bib_search.py first to find
papers and get their dblp_key, then pass those keys here.

Usage:
    python dblp_bib_fetch.py --keys "conf/nips/VaswaniSPUJGKP17" --years 2017
    python dblp_bib_fetch.py --keys "conf/nips/VaswaniSPUJGKP17" "conf/nips/YaoYZS00N23" --years 2017 2023
"""

import argparse
import json
import re
import sys

import requests
from loguru import logger

# Import shared DBLP infrastructure from search script
from dblp_bib_search import (
    DBLP_BIB_URL,
    dblp_rate_limit,
    dblp_request_with_retry,
    init_dblp,
)

SERVER_NAME = "dblp_bib_fetch"
DEFAULT_TIMEOUT = 180.0


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

def _fetch_bibtex(dblp_key: str, year: int | None = None) -> str:
    """Fetch BibTeX entry from DBLP by key.

    Args:
        dblp_key: DBLP key (e.g. "conf/nips/VaswaniSPUJGKP17").
        year: Publication year for reliable citation key generation.

    Returns:
        BibTeX string, or error comment string on failure.
    """
    if not dblp_key or dblp_key.isspace():
        return ""

    url = DBLP_BIB_URL.format(key=dblp_key)
    try:
        dblp_rate_limit()
        response = dblp_request_with_retry(url)
        if response.status_code == 200 and response.text.strip():
            bibtex = response.text

            # Generate a cleaner citation key: "AuthorYYYY"
            slug = dblp_key.split("/")[-1] if "/" in dblp_key else dblp_key
            author_match = re.match(r"([A-Z][a-z]+)", slug)

            # For arxiv keys (e.g. "abs-2305-14325"), extract author from BibTeX
            if not author_match:
                bib_author = re.search(r"author\s*=\s*\{([^}]+)\}", bibtex)
                if bib_author:
                    first_author = bib_author.group(1).split(" and ")[0].strip()
                    if "," in first_author:
                        last_name = first_author.split(",")[0].strip()
                    else:
                        last_name = first_author.split()[-1].strip()
                    author_match = re.match(r"([A-Z][a-z]+)", last_name)

            if author_match and year:
                new_key = f"{author_match.group(1)}{year}"
            elif author_match:
                yr_match = re.search(r"(\d{2})[a-z]?$", slug)
                if yr_match:
                    yr = yr_match.group(1)
                    yr = "20" + yr if int(yr) < 50 else "19" + yr
                    new_key = f"{author_match.group(1)}{yr}"
                else:
                    new_key = slug
            else:
                new_key = slug

            bibtex = re.sub(r"@(\w+)\{([^,]+),", rf"@\1{{{new_key},", bibtex, count=1)
            return bibtex

        return f"% BibTeX not found for key: {dblp_key} (HTTP {response.status_code})"
    except requests.exceptions.Timeout:
        return f"% Timeout fetching BibTeX for: {dblp_key}"
    except Exception as e:
        return f"% Error fetching BibTeX for {dblp_key}: {e}"


def core_dblp_fetch(**kwargs) -> dict:
    """
    Fetch BibTeX entries from DBLP by key.

    Args:
        dblp_keys: List of DBLP keys (e.g. ["conf/nips/VaswaniSPUJGKP17"]).
        years: Optional list of years (same length as dblp_keys) for citation keys.

    Returns:
        Dict with success, entries list (each has dblp_key + bibtex).
    """
    dblp_keys = kwargs.get("dblp_keys", [])
    years = kwargs.get("years", [])

    if isinstance(dblp_keys, str):
        dblp_keys = [dblp_keys]
    if isinstance(years, int):
        years = [years]

    entries = []
    for i, key in enumerate(dblp_keys):
        year = years[i] if i < len(years) else None
        bibtex = _fetch_bibtex(key, year=year)
        entries.append({"dblp_key": key, "bibtex": bibtex})

    return {
        "success": True,
        "returned": len(entries),
        "entries": entries,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch BibTeX entries from DBLP by key")
    parser.add_argument("--keys", "-k", nargs="+", required=True, help="DBLP key(s) from search results")
    parser.add_argument("--years", "-y", nargs="*", type=int, default=None, help="Publication year(s) for cleaner citation keys")
    parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON instead of BibTeX text")
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server
    payload = {"dblp_keys": args.keys}
    if args.years:
        payload["years"] = args.years

    result = call_server(SERVER_NAME, payload, timeout=DEFAULT_TIMEOUT)

    if result is None:
        print("Error: Ability service not available. Start with: bash aii_lib/src/aii_lib/utils/start_ability_server.sh", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if result.get("success"):
        entries = result.get("entries", [])
        print(f"Fetched {len(entries)} BibTeX entries\n")
        for entry in entries:
            bibtex = entry.get("bibtex", "")
            if bibtex:
                print(bibtex)
            else:
                print(f"% No BibTeX for key: {entry.get('dblp_key', 'unknown')}")
    else:
        print(f"Error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
