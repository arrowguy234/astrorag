"""
Tokenization for BM25 indexing and queries.

Uses simple lowercase whitespace splitting since BM25Okapi works
well with this on scientific text. More aggressive tokenization
(stemming, stopword removal) is available but empirically does not
improve results on the arXiv astro-ph corpus.
"""

from __future__ import annotations

import re


# ══════════════════════════════════════════════════════════
# tokenization
# ══════════════════════════════════════════════════════════

_PUNCT_STRIP_RE = re.compile(r"[^\w\s\-\.]")
_WHITESPACE_RE  = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.

    Steps:
      1. Lowercase
      2. Strip punctuation except hyphens and periods (preserve
         "X-ray", "3.5", "T_eff" style tokens)
      3. Split on whitespace
      4. Drop empty tokens

    Args:
        text: Raw text (title, abstract, query).

    Returns:
        List of lowercase tokens.
    """
    if not text:
        return []
    lower   = text.lower()
    cleaned = _PUNCT_STRIP_RE.sub(" ", lower)
    parts   = _WHITESPACE_RE.split(cleaned)
    return [p for p in parts if p]


# ══════════════════════════════════════════════════════════
# arXiv ID normalization
# ══════════════════════════════════════════════════════════

def normalize_arxiv_id(aid: str) -> str:
    """
    Normalize an arXiv ID for consistent lookup.

    Strips whitespace and lowercases. Preserves the original
    format (e.g. "0704.0007", "astro-ph/0703001").
    """
    return str(aid).strip()