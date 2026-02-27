"""Cited argument classes for hypothesis auditing.

Classes that represent arguments backed by citations (quotes + URLs).
Used for novelty and feasibility auditing with citation verification.
"""

from typing import Callable

from pydantic import BaseModel, field_validator

from aii_lib import (
    extract_citations_with_url,
    extract_quotes_only,
    find_exact_match,
)
from aii_lib.abilities.tools.utils import content_cache, cache_content
from aii_lib.telemetry import logger


class NoveltyCitedArg(BaseModel):
    """A novelty argument backed by cited sources with direct quotes and URLs.

    Format: 'The argument text \"\"\"exact quote\"\"\" (https://url)...'
    Citations: \"\"\"triple quotes\"\"\" or ""double quotes"" always count,
               "single quotes" only if >=4 words.
    Used for novelty arguments where citations come from web search.
    """
    argument: str

    @field_validator('argument')
    @classmethod
    def has_citation(cls, v: str) -> str:
        """Verify at least one valid quote with a URL exists."""
        citations = extract_citations_with_url(v)
        if not citations:
            raise ValueError(
                'Argument must contain at least one citation with a URL. '
                'Use \"\"\"triple quotes\"\"\" (preferred) or ""double quotes"".'
            )
        return v

    def get_citations(self) -> list[tuple[str, str]]:
        """Extract all (quote, url) pairs from the argument."""
        return extract_citations_with_url(self.argument)

    async def _verify_single_citation(self, quote: str, url: str) -> dict:
        """Verify a single citation - uses cache if available, fetches if not."""
        from aii_lib.abilities.tools.aii_web_tools import aii_web_fetch_direct_async

        # 1. Check cache first
        if content_cache.has(url):
            content = content_cache.get(url)
            if content:
                if find_exact_match(quote, content):
                    return {'quote': quote, 'url': url, 'status': 'valid', 'source': 'cache'}
                else:
                    logger.warning(f"Citation verification failed: quote not found in cached content for URL: {url}")
                    return {'quote': quote, 'url': url, 'status': 'invalid', 'reason': 'quote_not_found', 'source': 'cache'}

        # 2. Not in cache - fetch fresh
        try:
            result = await aii_web_fetch_direct_async(url=url, max_chars=30000)
            if not result.get("success"):
                logger.warning(f"Citation verification failed: fetch unsuccessful for URL: {url}")
                return {'quote': quote, 'url': url, 'status': 'invalid', 'reason': 'fetch_failed', 'source': 'fetch'}

            content = result.get("content", "")
            if not content:
                logger.warning(f"Citation verification failed: empty content from URL: {url}")
                return {'quote': quote, 'url': url, 'status': 'invalid', 'reason': 'fetch_failed', 'source': 'fetch'}

            # 3. Save to cache for future use
            cache_content(url, content)

            # 4. Verify quote
            if find_exact_match(quote, content):
                return {'quote': quote, 'url': url, 'status': 'valid', 'source': 'fetch'}
            else:
                logger.warning(f"Citation verification failed: quote not found in fetched content for URL: {url}")
                return {'quote': quote, 'url': url, 'status': 'invalid', 'reason': 'quote_not_found', 'source': 'fetch'}

        except Exception as e:
            logger.warning(f"Citation verification failed: {e} for URL: {url}")
            return {'quote': quote, 'url': url, 'status': 'invalid', 'reason': f'fetch_error: {e}', 'source': 'fetch'}

    async def _verify_citations_async(self, citations: list[tuple[str, str]]) -> list[dict]:
        """Verify citations in parallel - cache first, fetch if needed."""
        import asyncio
        tasks = [self._verify_single_citation(quote, url) for quote, url in citations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    def verify_citations(self, callback: Callable[[str], None] | None = None) -> dict:
        """Verify all citations - uses cache if available, fetches if not."""
        import asyncio

        citations = self.get_citations()

        if not citations and callback:
            callback(f"[!] No citations found in argument")

        if not citations:
            return {
                'valid': False,
                'total': 0,
                'verified': 0,
                'failed': 0,
                'results': []
            }

        # Run async verification
        def _run_in_new_loop(coro):
            """Run coroutine in a new event loop (for use in thread pool)."""
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
                future = pool.submit(_run_in_new_loop, self._verify_citations_async(citations))
                results = future.result()
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            results = asyncio.run(self._verify_citations_async(citations))

        for r in results:
            if callback:
                status = r['status']
                reason = r.get('reason', '')
                source = r.get('source', 'cache')

                if status == 'valid':
                    icon = "✓"
                    msg = f"Quote verified ({source})"
                elif reason == 'fetch_failed':
                    icon = "✗"
                    msg = "URL fetch failed"
                elif 'fetch_error' in reason:
                    icon = "✗"
                    msg = f"Fetch error: {reason}"
                else:
                    icon = "✗"
                    msg = f"Quote not found on page ({source})"

                callback(f"[{icon}] {msg}\n    \"{r['quote']}\"\n    {r['url']}")

        verified = sum(1 for r in results if r['status'] == 'valid')
        failed = len(results) - verified

        return {
            'valid': failed == 0 and len(results) > 0,
            'total': len(results),
            'verified': verified,
            'failed': failed,
            'results': results
        }


class FeasibilityCitedArg(BaseModel):
    """A feasibility argument backed by quotes from provided resources.

    Format: 'The argument text \"\"\"exact quote from resources\"\"\"...'
    Citations: \"\"\"triple quotes\"\"\" or ""double quotes"" always count,
               "single quotes" only if >=4 words.
    Used for feasibility arguments where citations come from <available_resources>.
    """
    argument: str

    @field_validator('argument')
    @classmethod
    def has_citation(cls, v: str) -> str:
        """Verify at least one valid quote citation exists."""
        quotes = extract_quotes_only(v)
        if not quotes:
            raise ValueError(
                'Argument must contain at least one citation. '
                'Use \"\"\"triple quotes\"\"\" (preferred) or ""double quotes"".'
            )
        return v

    def get_quotes(self) -> list[str]:
        """Extract all quotes from the argument."""
        return extract_quotes_only(self.argument)

    def verify_against_resources(
        self,
        resources_text: str,
        callback: Callable[[str], None] | None = None
    ) -> dict:
        """Verify all quotes exist in the provided resources text."""
        quotes = self.get_quotes()
        results = []

        if not quotes and callback:
            callback(f"[!] No quotes found in argument")

        for quote in quotes:
            if find_exact_match(quote, resources_text):
                status = "valid"
                reason = "Quote found in resources"
                results.append({
                    'quote': quote,
                    'status': 'valid'
                })
            else:
                status = "invalid"
                reason = "Quote not found in resources"
                results.append({
                    'quote': quote,
                    'status': 'invalid',
                    'reason': reason
                })

            if callback:
                icon = "✓" if status == "valid" else "✗"
                callback(f"[{icon}] {reason}\n    \"{quote}\"")

        verified = sum(1 for r in results if r['status'] == 'valid')
        failed = len(results) - verified

        return {
            'valid': failed == 0 and len(results) > 0,
            'total': len(results),
            'verified': verified,
            'failed': failed,
            'results': results
        }


__all__ = [
    "NoveltyCitedArg",
    "FeasibilityCitedArg",
]
