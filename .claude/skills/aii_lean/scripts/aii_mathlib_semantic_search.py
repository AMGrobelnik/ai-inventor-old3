#!/usr/bin/env python
"""
Mathlib Semantic Search Tool

Search Mathlib using natural language via LeanExplore API.

Usage:
    python aii_mathlib_semantic_search.py "fundamental theorem of calculus"
    python aii_mathlib_semantic_search.py "prime number" --limit 10 --deps
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

SERVER_NAME = "aii_mathlib_semantic_search"
DEFAULT_LIMIT = 5

API_KEY = os.environ.get("LEANEXPLORE_API_KEY", "")


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

def init_mathlib_semantic_search():
    """Initialize LeanExplore client and warmup connection."""
    import asyncio
    try:
        from lean_explore.api.client import ApiClient

        async def warmup():
            client = ApiClient(api_key=API_KEY or None)
            await client.search(query="test", limit=1)

        asyncio.run(warmup())
    except Exception:
        pass


def core_mathlib_semantic_search(**kwargs) -> dict:
    """
    Search Mathlib using natural language via LeanExplore API.

    Args:
        query: Natural language search query
        limit: Maximum number of results (default: 5)
        show_deps: Show dependencies of first result (default: False)
        packages: List of packages to filter (default: ["Mathlib"])

    Returns:
        Dict with success status and result string
    """
    import asyncio

    query = kwargs.get("query", "")
    limit = kwargs.get("limit", DEFAULT_LIMIT)
    show_deps = kwargs.get("show_deps", False)
    packages = kwargs.get("packages", ["Mathlib"])

    if not query:
        return {"success": False, "error": "Query is required"}

    try:
        from lean_explore.api.client import ApiClient
    except ImportError:
        return {"success": False, "error": "lean-explore package not installed. Run: pip install lean-explore"}

    async def do_search():
        client = ApiClient(api_key=API_KEY or None)
        response = await client.search(query=query, limit=limit, packages=packages)

        if not response.results:
            return f"No results found for: {query}"

        lines = [f"Found {response.count} results for: {query}\n"]

        for i, item in enumerate(response.results, 1):
            lines.append(f"[{i}] {item.name}")
            lines.append(f"    Module: {item.module}")
            lines.append(f"    ID: {item.id}")
            if item.source_link:
                lines.append(f"    Source: {item.source_link}")

            if item.source_text:
                stmt = item.source_text.strip()
                if len(stmt) > 200:
                    stmt = stmt[:200] + "..."
                lines.append(f"    Statement: {stmt}")

            if item.docstring:
                doc = item.docstring.strip().replace("\n", " ")[:150]
                lines.append(f"    Doc: {doc}")

            if item.informalization:
                info = item.informalization.strip().replace("\n", " ")[:150]
                lines.append(f"    Informal: {info}")

            lines.append("")

        # Optionally show dependencies of first result
        if show_deps and response.results:
            first = response.results[0]
            if first.dependencies:
                lines.append(f"Dependencies for {first.name}:")
                for dep_line in first.dependencies.strip().split("\n")[:10]:
                    lines.append(f"  - {dep_line.strip()}")

        return "\n".join(lines)

    try:
        return {"success": True, "result": asyncio.run(do_search())}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Search Mathlib with natural language (LeanExplore)")
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument("--limit", "-n", type=int, default=DEFAULT_LIMIT, help="Number of results (default: 5)")
    parser.add_argument("--deps", "-d", action="store_true", help="Show dependencies of first result")
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server
    result = call_server(SERVER_NAME, {
        "query": args.query,
        "limit": args.limit,
        "show_deps": args.deps,
    }, timeout=60.0)

    if result is None:
        print("Error: Ability service not available.", file=sys.stderr)
        sys.exit(1)

    if isinstance(result, dict):
        if result.get("success"):
            print(result.get("result", ""))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)
    else:
        print(result)


if __name__ == "__main__":
    main()
