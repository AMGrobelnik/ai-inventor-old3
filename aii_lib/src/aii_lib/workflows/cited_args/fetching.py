"""URL content fetching utilities - hybrid static + JS fallback.

Provides fast static fetching with aiohttp, falling back to
browser-based JS rendering when needed (e.g., for SPAs).
"""

import re
from dataclasses import dataclass

import aiohttp
from selectolax.parser import HTMLParser


# Max words to extract for quote matching
MAX_CONTENT_WORDS = 3000

# Minimum words to consider static fetch successful (avoid empty SPA shells)
MIN_CONTENT_WORDS = 50

# Indicators that JS rendering is needed
JS_REQUIRED_MARKERS = [
    'enable javascript',
    'requires javascript',
    'please enable javascript',
    'javascript is required',
    '<noscript>',
    'id="__next"',  # Next.js
    'id="root"',    # React
    'id="app"',     # Vue
]

# Headers to avoid bot detection
FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


@dataclass
class FetchResult:
    """Result of fetching a URL."""
    content: str | None
    used_js_fallback: bool


def _cap_words(content: str, max_words: int) -> str:
    """Cap content to max_words."""
    words = content.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return content


def _extract_text_from_html(html: str) -> str:
    """Extract clean text from HTML using selectolax."""
    tree = HTMLParser(html)
    # Remove non-content elements
    for tag in tree.css('script, style, nav, footer, header, aside, noscript'):
        tag.decompose()
    text = tree.text(separator=' ')
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _needs_js_rendering(html: str, text: str) -> bool:
    """Check if the page likely needs JS rendering."""
    html_lower = html.lower()
    # Check for JS-required markers
    for marker in JS_REQUIRED_MARKERS:
        if marker in html_lower:
            return True
    # Check if extracted text is too short (empty SPA shell)
    if len(text.split()) < MIN_CONTENT_WORDS:
        return True
    return False


async def _fetch_static(url: str, session: aiohttp.ClientSession) -> str | None:
    """Fast static fetch using aiohttp + selectolax. No JS rendering."""
    async with session.get(url, headers=FETCH_HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Static fetch failed for {url}: HTTP {resp.status}")
        html = await resp.text()
    return _extract_text_from_html(html)


def _lazy_import_crawl4ai():
    """Lazy import crawl4ai to avoid affecting global logging at module load time."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    return AsyncWebCrawler, BrowserConfig, CrawlerRunConfig


async def _fetch_with_js(url: str) -> str | None:
    """Fetch with full JS rendering using crawl4ai browser."""
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig = _lazy_import_crawl4ai()

    browser_config = BrowserConfig(verbose=False, headless=True)
    run_config = CrawlerRunConfig(verbose=False, log_console=False)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result_container = await crawler.arun(url=url, config=run_config)

    if hasattr(result_container, '_results') and result_container._results:
        result = result_container._results[0]
    else:
        result = result_container

    if not result.success:
        raise RuntimeError(f"JS fetch failed for {url}: crawl4ai reported failure")

    content = str(result.markdown) if hasattr(result, 'markdown') else None
    return content


async def fetch_url_content(
    url: str,
    session: aiohttp.ClientSession | None = None,
    skip_js_fallback: bool = False,
) -> FetchResult:
    """Fetch content from URL with hybrid static/JS approach.

    1. Try fast static fetch (aiohttp + selectolax)
    2. If content too short or fetch failed, fall back to browser (JS rendering)

    Returns:
        FetchResult with content and whether JS fallback was used
    """
    # Create session if not provided
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()

    try:
        # Try static fetch first (fast, no browser)
        content = await _fetch_static(url, session)

        if content and len(content.split()) >= MIN_CONTENT_WORDS:
            return FetchResult(
                content=_cap_words(content, MAX_CONTENT_WORDS),
                used_js_fallback=False
            )

        # Static fetch failed or content too short - try JS fallback
        if not skip_js_fallback:
            js_content = await _fetch_with_js(url)
            if js_content:
                return FetchResult(
                    content=_cap_words(js_content, MAX_CONTENT_WORDS),
                    used_js_fallback=True
                )

        return FetchResult(content=None, used_js_fallback=False)

    finally:
        if owns_session:
            await session.close()


__all__ = [
    "FetchResult",
    "fetch_url_content",
    "MAX_CONTENT_WORDS",
    "MIN_CONTENT_WORDS",
]
