"""
Find similar queries from the context library.

Uses simple TF-IDF-like word overlap scoring — no ML dependencies
required. Good enough for suggesting relevant library queries.
"""

from __future__ import annotations

import json
import re
from   collections import Counter
from   pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


# common stop words to ignore for matching
_STOPS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from",
    "with", "by", "and", "or", "how", "what", "why", "when", "where",
    "which", "who", "does", "do", "did", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "this", "that",
    "these", "those", "it", "its",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOPS]


def find_similar_queries(query: str, k: int = 5) -> list[dict]:
    """
    Return up to k similar library queries.

    Each result: {arxiv_id, query, subdomain, q_total, score}
    """

    query = query.strip()
    if not query:
        return []

    lib_path = DATA_DIR / "context_library.json"
    if not lib_path.exists():
        return []

    try:
        with open(lib_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    scored = []
    for entry in data.get("entries", {}).values():
        lib_query = entry.get("original_query", "")
        if not lib_query:
            continue

        lib_tokens = set(_tokenize(lib_query))
        if not lib_tokens:
            continue

        # Jaccard similarity
        overlap = len(query_tokens & lib_tokens)
        union   = len(query_tokens | lib_tokens)
        score   = overlap / union if union > 0 else 0.0

        if score > 0.10:  # threshold
            scored.append({
                "arxiv_id":  entry.get("arxiv_id", ""),
                "query":     lib_query,
                "subdomain": entry.get("subdomain", ""),
                "q_total":   entry.get("q_total", 0.0),
                "score":     score,
            })

    scored.sort(key=lambda x: -x["score"])
    return scored[:k]
