'''Citation extraction utilities - quotes and URLs from text.

Handles various quote formats:
- Triple quotes \"\"\"quote\"\"\" (primary format)
- Double quotes ""quote"" (also accepted)
- Single quotes "quote" (only if >=4 words)

Also handles curly quote variants (" ").
'''

import re


# Quote character patterns (straight and curly)
QUOTE_CHARS = r'[""\u201C\u201D]'

# Triple quotes """quote""" - always a citation (primary format)
TRIPLE_QUOTE_PATTERN = QUOTE_CHARS + r'{3}(.+?)' + QUOTE_CHARS + r'{3}'

# Double quotes ""quote"" - also accepted for backwards compatibility
DOUBLE_QUOTE_PATTERN = QUOTE_CHARS + r'{2}(.+?)' + QUOTE_CHARS + r'{2}'

# Single quotes "quote" - only counts if >=4 words
SINGLE_QUOTE_PATTERN = QUOTE_CHARS + r'([^"""\u201C\u201D]+?)' + QUOTE_CHARS

# Min words for single quotes to count as citation (triple/double quotes always count)
MIN_WORDS_SINGLE_QUOTE = 4

# URL pattern
URL_PATTERN = r'https?://[^\s\)\]>]+'


def _count_words(text: str) -> int:
    """Count words in text (alphanumeric sequences)."""
    return len(re.findall(r'[a-zA-Z0-9]+', text))


def _is_url_or_parenthetical(text: str) -> bool:
    """Check if text is just a URL or parenthetical reference, not a real quote."""
    text = text.strip()
    # Contains URL
    if 'http://' in text or 'https://' in text or '://' in text:
        return True
    # Just parenthetical like "(1808.01033)" or "(url)"
    if text.startswith('(') and text.endswith(')'):
        return True
    return False


def _extract_all_quotes(text: str) -> list[tuple[str, int, bool]]:
    """Extract all quotes with position and whether they're triple/double-quoted.

    Priority: triple quotes > double quotes > single quotes (>=4 words)

    Returns:
        List of (quote_text, end_position, is_primary_quote) tuples
        is_primary_quote is True for triple/double quotes, False for single quotes
    """
    results = []
    priority_ranges = []  # Ranges used by triple/double quotes

    # Find triple quotes first (highest priority)
    for m in re.finditer(TRIPLE_QUOTE_PATTERN, text):
        results.append((m.group(1), m.end(), True))
        priority_ranges.append((m.start(), m.end()))

    # Find double quotes (but not if overlapping with triple quotes)
    for m in re.finditer(DOUBLE_QUOTE_PATTERN, text):
        # Check if this overlaps with any triple quote range
        overlaps = False
        for r_start, r_end in priority_ranges:
            if m.start() < r_end and m.end() > r_start:
                overlaps = True
                break
        if not overlaps:
            results.append((m.group(1), m.end(), True))
            priority_ranges.append((m.start(), m.end()))

    def is_inside_or_adjacent_to_priority(pos: int) -> bool:
        """Check if a position is inside or adjacent to any triple/double quote range."""
        for d_start, d_end in priority_ranges:
            # Inside the priority quote range
            if d_start <= pos < d_end:
                return True
            # Adjacent (within 1 char of boundaries)
            if d_start - 1 <= pos <= d_start or d_end - 1 <= pos <= d_end:
                return True
        return False

    # Find all quote character positions for single quote detection
    quote_char_pattern = re.compile(QUOTE_CHARS)
    quote_positions = [m.start() for m in quote_char_pattern.finditer(text)]

    # Try to find valid single-quote pairs
    # Start from positions that aren't inside/adjacent to double quotes
    i = 0
    while i < len(quote_positions):
        start_pos = quote_positions[i]

        # Skip if this position is inside or adjacent to a triple/double quote
        if is_inside_or_adjacent_to_priority(start_pos):
            i += 1
            continue

        # Look for a closing quote
        matched = False
        for j in range(i + 1, len(quote_positions)):
            end_pos = quote_positions[j]

            # Skip if closing position is inside/adjacent to triple/double quote
            if is_inside_or_adjacent_to_priority(end_pos):
                continue

            # Extract the content between quotes
            quote_text = text[start_pos + 1:end_pos]

            # Skip if empty or contains quote characters (nested quotes)
            if not quote_text or quote_char_pattern.search(quote_text):
                continue

            # Skip if it's just a URL or parenthetical
            if _is_url_or_parenthetical(quote_text):
                # Skip this candidate closing quote, try next one
                continue

            # Only include single quotes with >=3 words
            if _count_words(quote_text) >= MIN_WORDS_SINGLE_QUOTE:
                results.append((quote_text, end_pos + 1, False))
                # Move past this matched quote
                i = j + 1
                matched = True
                break
            else:
                # Less than 3 words - this is the first valid pair, but too short
                # Skip this start position entirely (greedy match)
                break

        if not matched:
            i += 1

    # Sort by position
    results.sort(key=lambda x: x[1])
    return results


def extract_citations_with_url(text: str) -> list[tuple[str, str]]:
    """Extract all quote citations and pair with their URLs.

    - Double quotes ""..."" always count as citations
    - Single quotes "..." only count if >2 words

    Quotes are paired with the next URL that appears after them.

    Returns:
        List of (quote, url) tuples
    """
    quote_matches = _extract_all_quotes(text)

    # Find all URLs with their positions
    url_matches = [(m.group(0), m.start()) for m in re.finditer(URL_PATTERN, text)]

    if not quote_matches or not url_matches:
        return []

    results = []
    for quote, quote_end_pos, _ in quote_matches:
        # Find the next URL that appears after this quote
        next_url = None
        for url, url_start_pos in url_matches:
            if url_start_pos >= quote_end_pos:
                next_url = url
                break

        if next_url:
            results.append((quote, next_url))

    return results


def extract_quotes_only(text: str) -> list[str]:
    """Extract all quote citations from text (no URL).

    - Double quotes ""..."" always count as citations
    - Single quotes "..." only count if >2 words

    Returns:
        List of quotes
    """
    return [quote for quote, _, _ in _extract_all_quotes(text)]


__all__ = [
    "QUOTE_CHARS",
    "TRIPLE_QUOTE_PATTERN",
    "DOUBLE_QUOTE_PATTERN",
    "SINGLE_QUOTE_PATTERN",
    "MIN_WORDS_SINGLE_QUOTE",
    "URL_PATTERN",
    "extract_citations_with_url",
    "extract_quotes_only",
]
