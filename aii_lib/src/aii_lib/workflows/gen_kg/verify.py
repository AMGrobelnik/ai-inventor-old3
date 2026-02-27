"""Wikipedia URL verification for knowledge graph triples.

Verifies that Wikipedia URLs in triples actually exist using HTTP requests.
"""

import asyncio
from typing import Callable

from ...abilities.tools.aii_web_tools import aii_web_fetch_direct_async


async def _verify_single_url(url: str) -> dict:
    """Verify a single Wikipedia URL exists.

    Wikipedia returns HTTP 200 even for non-existent pages (shows "page does not exist" template).
    We must check the content for the non-existence message.

    Args:
        url: Wikipedia URL to verify

    Returns:
        Dict with keys: url, status ('valid' or 'invalid'), reason (if invalid)
    """
    # Check URL format first
    if not url.startswith("https://en.wikipedia.org/wiki/"):
        return {
            'url': url,
            'status': 'invalid',
            'reason': 'Invalid format: must start with https://en.wikipedia.org/wiki/'
        }

    try:
        # Fetch enough content to detect non-existent page message
        result = await aii_web_fetch_direct_async(url=url, max_chars=5000)

        if not result.get("success"):
            status_code = result.get("status_code", 0)
            return {
                'url': url,
                'status': 'invalid',
                'reason': result.get("error", f"HTTP {status_code}" if status_code else "Fetch failed")
            }

        # Check content for "page does not exist" indicators
        # Wikipedia uses redlink=1 parameter for non-existent pages
        content = result.get("content", "")
        if "action=edit&redlink=1" in content:
            return {
                'url': url,
                'status': 'invalid',
                'reason': 'Wikipedia article does not exist'
            }

        return {'url': url, 'status': 'valid'}

    except Exception as e:
        return {
            'url': url,
            'status': 'invalid',
            'reason': f'Error: {str(e)}'
        }


async def _verify_urls_async(urls: list[str]) -> list[dict]:
    """Verify multiple URLs in parallel."""
    tasks = [_verify_single_url(url) for url in urls]
    return await asyncio.gather(*tasks)


def verify_wikipedia_urls(
    triples: list[dict],
    callback: Callable[[str], None] | None = None,
) -> dict:
    """Verify all Wikipedia URLs in triples list.

    Args:
        triples: List of triple dicts, each with 'wikipedia_url' key
        callback: Optional callback for status messages

    Returns:
        Dict with keys:
            - valid: bool (True if all URLs valid)
            - total: int (total URLs checked)
            - verified: int (number of valid URLs)
            - failed: int (number of invalid URLs)
            - results: list[dict] (per-URL results with 'url', 'status', 'reason')
            - failed_triples: list[dict] (triples with invalid URLs)
    """
    if not triples:
        if callback:
            callback("[!] No triples to verify")
        return {
            'valid': True,
            'total': 0,
            'verified': 0,
            'failed': 0,
            'results': [],
            'failed_triples': [],
        }

    # Extract URLs from triples
    urls = [t.get('wikipedia_url', '') for t in triples if t.get('wikipedia_url')]

    if not urls:
        if callback:
            callback("[!] No Wikipedia URLs found in triples")
        return {
            'valid': False,
            'total': 0,
            'verified': 0,
            'failed': len(triples),
            'results': [],
            'failed_triples': triples,
        }

    # Run async verification
    def _run_in_new_loop(coro):
        """Run coroutine in a new event loop (for use when already in async context)."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    try:
        asyncio.get_running_loop()
        # Already in async context - run in thread pool with new loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(_run_in_new_loop, _verify_urls_async(urls))
            results = future.result()
    except RuntimeError:
        # No running loop - safe to use asyncio.run
        results = asyncio.run(_verify_urls_async(urls))

    # Build URL -> result mapping
    url_results = {r['url']: r for r in results}

    # Find failed triples
    failed_triples = []
    for triple in triples:
        url = triple.get('wikipedia_url', '')
        if url in url_results and url_results[url]['status'] == 'invalid':
            failed_triples.append({
                'triple': triple,
                'error': url_results[url].get('reason', 'Unknown error')
            })

    # Log results via callback
    for r in results:
        if callback:
            if r['status'] == 'valid':
                callback(f"[✓] Valid: {r['url']}")
            else:
                callback(f"[✗] Invalid: {r['url']}\n    Reason: {r.get('reason', 'Unknown')}")

    verified = sum(1 for r in results if r['status'] == 'valid')
    failed = len(results) - verified

    return {
        'valid': failed == 0 and len(results) > 0,
        'total': len(results),
        'verified': verified,
        'failed': failed,
        'results': results,
        'failed_triples': failed_triples,
    }


__all__ = ["verify_wikipedia_urls"]
