"""
JSON persistence for LibraryEntry objects.

One file per paper at results/library/{arxiv_id}.json.
Slashes in arxiv_ids (astro-ph/0703001) become underscores.
"""

from __future__ import annotations

import json
import re
from   pathlib import Path

from astrorag.library.models import LibraryEntry
from astrorag.logger         import get_logger
from astrorag.paths          import get_paths

logger = get_logger(__name__)


def _safe_filename(arxiv_id: str) -> str:
    """arxiv_id → safe filename (replace / and whitespace)."""
    return re.sub(r"[/\\\s]", "_", str(arxiv_id).strip()) + ".json"


def _library_dir() -> Path:
    d = get_paths().results_dir / "library"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ══════════════════════════════════════════════════════════
# read / write / delete
# ══════════════════════════════════════════════════════════

def save_entry(entry: LibraryEntry) -> Path:
    """Persist a LibraryEntry to disk. Returns the path written."""
    path = _library_dir() / _safe_filename(entry.arxiv_id)
    try:
        path.write_text(
            json.dumps(entry.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug(f"Library entry saved: {path.name}")
        return path
    except Exception as e:
        logger.warning(f"Failed to save library entry {path.name}: {e}")
        raise


def load_entry(arxiv_id: str) -> LibraryEntry | None:
    """Load one LibraryEntry by arxiv_id. Returns None if missing."""
    path = _library_dir() / _safe_filename(arxiv_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return LibraryEntry.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load library entry {path.name}: {e}")
        return None


def delete_entry(arxiv_id: str) -> bool:
    """Remove one LibraryEntry. Returns True if a file was deleted."""
    path = _library_dir() / _safe_filename(arxiv_id)
    if not path.exists():
        return False
    try:
        path.unlink()
        logger.debug(f"Library entry deleted: {path.name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete library entry {path.name}: {e}")
        return False


def list_entries() -> list[str]:
    """Return all arxiv_ids present in the library."""
    d = _library_dir()
    return sorted([
        p.stem.replace("_", "/") if p.stem.startswith("astro-ph")
        else p.stem
        for p in d.glob("*.json")
    ])