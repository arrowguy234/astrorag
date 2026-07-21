"""
arXiv PDF fetcher with URL construction and caching.

Handles the various arXiv ID formats:
    old style with slash:  astro-ph/0703001
    old style with hyphen: astro-ph-0703001  (dataset format)
    new style:             0704.0007, 2301.07688
    with version:          0704.0007v2

Caches PDFs to pdfs/{arxiv_id}.pdf so subsequent queries reuse them.
"""

from __future__ import annotations

import re
import time
from   pathlib import Path

import requests

from astrorag.logger import get_logger
from astrorag.paths  import get_paths

logger = get_logger(__name__)

USER_AGENT = "AstroRAG/1.0 (educational research)"


# ══════════════════════════════════════════════════════════
# arXiv ID handling
# ══════════════════════════════════════════════════════════

def normalize_arxiv_id_for_url(arxiv_id: str) -> str:
    """
    Normalise arxiv_id for URL construction.

    Rules:
    - strip whitespace
    - remove trailing version suffix (v1, v2, etc.)
    - convert old-style hyphen form (astro-ph-0308006) to slash form
      (astro-ph/0308006) — this is critical for arxiv.org URLs
    - preserve existing slash-form prefixes
    """
    aid = str(arxiv_id).strip()
    # remove version suffix
    aid = re.sub(r"v\d+$", "", aid)

    # convert astro-ph-0308006 → astro-ph/0308006
    # match: category (letters/hyphens optionally with .XX suffix)
    # followed by hyphen and 7+ digits
    m = re.match(
        r"^([a-z\-]+(?:\.[A-Z]{2})?)-(\d{7,})$",
        aid,
        flags=re.IGNORECASE,
    )
    if m and "/" not in aid:
        prefix = m.group(1)
        number = m.group(2)
        aid = f"{prefix}/{number}"

    return aid


def build_arxiv_urls(arxiv_id: str) -> list[str]:
    """
    Return a list of candidate URLs to try in order.

    Handles new-style, old-style with slash, and old-style with hyphen.
    """
    aid = normalize_arxiv_id_for_url(arxiv_id)
    urls = [
        f"https://arxiv.org/pdf/{aid}.pdf",
        f"https://arxiv.org/pdf/{aid}",
    ]

    # for old-style with slash, also try without prefix as last resort
    if "/" in aid:
        _, tail = aid.split("/", 1)
        urls.extend([
            f"https://arxiv.org/pdf/{tail}.pdf",
            f"https://arxiv.org/pdf/{tail}",
        ])

    # if original had hyphen form and normalisation created slash form,
    # also try the raw hyphen form as absolute last resort
    raw = str(arxiv_id).strip()
    raw = re.sub(r"v\d+$", "", raw)
    if raw != aid:
        urls.extend([
            f"https://arxiv.org/pdf/{raw}.pdf",
            f"https://arxiv.org/pdf/{raw}",
        ])

    return urls


# ══════════════════════════════════════════════════════════
# PDF fetching
# ══════════════════════════════════════════════════════════

def fetch_arxiv_pdf(
    arxiv_id:   str,
    cache_dir:  Path | None = None,
    timeout:    int = 60,
    force:      bool = False,
) -> tuple[Path | None, bool, str]:
    """
    Download an arXiv PDF, using the cache if available.

    Args:
        arxiv_id:  arXiv identifier (any format).
        cache_dir: Directory to cache PDFs (default: pdfs/).
        timeout:   HTTP timeout in seconds.
        force:     If True, re-download even if cached.

    Returns:
        Tuple of:
            path        Local Path if success, else None.
            from_cache  True if loaded from cache.
            error       Empty string on success, else error message.
    """
    cache_dir = cache_dir or get_paths().pdf_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[/\\]", "_", str(arxiv_id).strip())
    cache_path = cache_dir / f"{safe_name}.pdf"

    if cache_path.exists() and not force and cache_path.stat().st_size > 1000:
        logger.debug(f"PDF cached: {cache_path.name}")
        return cache_path, True, ""

    urls = build_arxiv_urls(arxiv_id)
    last_error = ""

    for url in urls:
        try:
            logger.debug(f"Fetching {url}")
            resp = requests.get(
                url,
                headers = {"User-Agent": USER_AGENT},
                timeout = timeout,
                allow_redirects = True,
            )
            if resp.status_code == 200 and len(resp.content) > 1000:
                cache_path.write_bytes(resp.content)
                size_kb = len(resp.content) / 1024
                logger.info(
                    f"Downloaded {arxiv_id} ({size_kb:.0f} KB) → "
                    f"{cache_path.name}"
                )
                return cache_path, False, ""
            last_error = f"HTTP {resp.status_code} at {url}"
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e} at {url}"
            continue

    logger.warning(f"Failed to fetch {arxiv_id}: {last_error}")
    return None, False, last_error