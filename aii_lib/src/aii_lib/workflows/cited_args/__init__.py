"""Cited argument generation workflow with citation utilities.

Structure:
├── workflow.py    - Main workflow (generate_cited_argument, configs)
├── citation.py    - Citation extraction (quotes, URLs)
├── fetching.py    - URL content fetching (static + JS fallback)
├── matching.py    - Quote matching utilities
└── cited_arg.py   - CitedArg classes with verification
"""

# Workflow - main API
from .workflow import (
    CitedArgsConfig,
    CitedArgsResult,
    ClaudeAgentConfig,
    generate_cited_argument,
    cap_results,
    collect_verified_arguments,
)


# Citation extraction
from .citation import (
    extract_citations_with_url,
    extract_quotes_only,
    QUOTE_CHARS,
    URL_PATTERN,
)

# URL fetching
from .fetching import (
    FetchResult,
    fetch_url_content,
)

# Quote matching
from .matching import (
    find_exact_match,
    extract_words,
    strip_brackets,
    strip_markdown_links,
    STOPWORDS,
)


__all__ = [
    # Workflow
    "CitedArgsConfig",
    "CitedArgsResult",
    "ClaudeAgentConfig",  # Config for Claude agent in workflow
    "generate_cited_argument",
    "cap_results",
    "collect_verified_arguments",
    # Citation extraction
    "extract_citations_with_url",
    "extract_quotes_only",
    "QUOTE_CHARS",
    "URL_PATTERN",
    # URL fetching
    "FetchResult",
    "fetch_url_content",
    # Quote matching
    "find_exact_match",
    "extract_words",
    "strip_brackets",
    "strip_markdown_links",
    "STOPWORDS",
]
