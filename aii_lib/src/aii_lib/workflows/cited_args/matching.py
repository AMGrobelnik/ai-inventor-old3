"""Quote matching utilities - fuzzy text matching for citation verification.

Provides robust matching that handles:
- Case differences
- [bracketed] editorial text
- Markdown links [text](url)
- PDF hyphenation (word-\n splits)
- Stopwords filtering
- Ellipsis gaps
"""

import re


# Common English stopwords/connectives to ignore in matching
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where',
    'why', 'how', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'shall', 'can', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'once', 'here', 'there', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
    'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we',
    'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his', 'hers', 'who', 'whom',
    'which', 'what', 'whose', 'while', 'although', 'because', 'since', 'unless',
    'until', 'whether', 'though', 'yet', 'however', 'therefore', 'thus', 'hence',
    'moreover', 'furthermore', 'additionally', 'consequently', 'nevertheless',
}

# Max gap between consecutive quote words (allows for minor differences)
MAX_WORD_GAP = 20

# Max gap when ellipsis (...) is implied (larger jump allowed)
MAX_ELLIPSIS_GAP = 500


def strip_brackets(text: str) -> str:
    """Handle [bracketed text] in quotes for matching.

    - [E] or [a] → keeps the letter (capitalization markers like "[E]nabling")
    - [DKW] or [ABC] → keeps acronyms (all caps, ≤5 chars)
    - [url] → keeps the whole thing if it's a URL
    - [emphasis added] or [...] → removes entirely (editorial notes)
    """
    def replace_bracket(match):
        content = match.group(1)
        # Keep if it looks like a URL
        if '://' in content or content.startswith('http'):
            return match.group(0)  # Keep the whole [url]
        # Keep single letters (capitalization markers like [E] in [E]nabling)
        if len(content) <= 2 and content.isalpha():
            return content  # Keep just the letter(s)
        # Keep short acronyms (all uppercase, ≤5 chars) like [DKW], [LLM], [API]
        if len(content) <= 5 and content.isupper() and content.isalpha():
            return content  # Keep the acronym
        return ''  # Remove editorial brackets like [...] or [emphasis added]

    return re.sub(r'\[([^\]]*)\]', replace_bracket, text)


def strip_markdown_links(text: str) -> str:
    """Convert markdown links [text](url) to just the text.

    This prevents URL fragments from polluting word extraction.
    """
    # [link text](url) -> link text
    # Also handles [link text](url "title") format
    return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)


def _fix_pdf_hyphenation(text: str) -> str:
    """Fix PDF hyphenation where words are split across lines.

    Pattern: 'methodolo-\\ngies' -> 'methodologies'
    Handles: word-\n, word-\r\n, word- \n (with optional space)
    """
    # Fix hyphenated line breaks: word- followed by newline and continuation
    # This joins 'methodolo-\ngies' into 'methodologies'
    text = re.sub(r'(\w)-\s*[\r\n]+\s*(\w)', r'\1\2', text)
    return text


def extract_words(text: str) -> list[str]:
    """Extract normalized words from text.

    - Fix PDF hyphenation (word-\\n splits)
    - Strip markdown links (keep link text, remove URLs)
    - Remove [bracketed editorial text]
    - Lowercase
    - Extract alphanumeric words only
    - Remove stopwords/connectives
    """
    # Fix PDF hyphenation first (before any other processing)
    text = _fix_pdf_hyphenation(text)
    # Strip markdown links first (before bracket handling)
    text = strip_markdown_links(text)
    # Remove bracketed editorial text
    text = strip_brackets(text)
    # Lowercase and extract words (alphanumeric sequences)
    words = re.findall(r'[a-z0-9]+', text.lower())
    # Filter out stopwords
    words = [w for w in words if w not in STOPWORDS]
    return words


def _try_match_from(
    quote_words: list[str],
    word_positions: dict[str, list[int]],
    start_pos: int
) -> bool:
    """Try to match all quote words starting from a position."""
    current_pos = start_pos

    for i, word in enumerate(quote_words[1:], 1):
        if word not in word_positions:
            return False

        # Find the next occurrence of this word after current_pos
        # but within a reasonable gap
        positions = word_positions[word]

        # Determine max gap - use larger gap if we might be crossing an ellipsis
        # (detected by larger jumps in successful matches)
        max_gap = MAX_WORD_GAP

        found_pos = None
        for pos in positions:
            if pos > current_pos:
                gap = pos - current_pos
                if gap <= max_gap:
                    found_pos = pos
                    break
                elif gap <= MAX_ELLIPSIS_GAP:
                    # Allow larger gap (ellipsis case) but keep looking for closer match
                    if found_pos is None:
                        found_pos = pos
                    break

        if found_pos is None:
            return False

        current_pos = found_pos

    return True


def find_exact_match(quote: str, content: str) -> bool:
    """Check if quote exists in content using sliding window matching.

    Uses a sliding window approach to find quote words in a contiguous
    region of content, with allowed gaps for ellipses (...).

    Handles:
    - Case differences (all lowercased)
    - [E]nabling style brackets (letter preserved, brackets removed)
    - ... or ellipsis wildcards (larger gaps allowed)
    - Different punctuation/unicode (all removed)
    - Stopwords filtered out
    """
    quote_words = extract_words(quote)
    content_words = extract_words(content)

    if not quote_words:
        return False

    # Build index of where each word appears in content
    word_positions: dict[str, list[int]] = {}
    for i, word in enumerate(content_words):
        if word not in word_positions:
            word_positions[word] = []
        word_positions[word].append(i)

    # Check if first quote word exists
    if quote_words[0] not in word_positions:
        return False

    # Try each occurrence of the first word as a potential starting point
    for start_pos in word_positions[quote_words[0]]:
        if _try_match_from(quote_words, word_positions, start_pos):
            return True

    return False


__all__ = [
    "STOPWORDS",
    "MAX_WORD_GAP",
    "MAX_ELLIPSIS_GAP",
    "strip_brackets",
    "strip_markdown_links",
    "extract_words",
    "find_exact_match",
]
