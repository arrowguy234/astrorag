"""
Persistent context library — main class.

Provides an in-memory view over persisted LibraryEntry files with
lazy loading, cache-and-write, and query-provenance tracking.
"""

from __future__ import annotations

from pathlib import Path

from astrorag.library.models      import LibraryEntry, LibraryStats, _now_iso
from astrorag.library.persistence import (
    delete_entry, list_entries, load_entry, save_entry, _library_dir,
)
from astrorag.llm.models          import StructuredSummary
from astrorag.logger              import get_logger

logger = get_logger(__name__)


class ContextLibrary:
    """
    Persistent per-paper knowledge base.

    Usage:
        lib = ContextLibrary()
        lib.load_all()                      # populate from disk

        entry = lib.get("1701.06747")        # returns LibraryEntry or None
        if entry is None:
            entry = lib.add_from_summary(
                arxiv_id="1701.06747",
                summary=stage5_result.summary,
                query="How does the ICM cool?",
                q_total=1.0,
            )
        else:
            lib.update_from_summary(
                arxiv_id="1701.06747",
                summary=new_summary,
                query="Different query",
                q_total=0.85,
            )
    """

    def __init__(self) -> None:
        self._cache: dict[str, LibraryEntry] = {}
        self._loaded_all = False

    # ══════════════════════════════════════════════════
    # bulk load
    # ══════════════════════════════════════════════════

    def load_all(self) -> int:
        """Load every entry from disk into the in-memory cache."""
        n = 0
        for aid in list_entries():
            if aid in self._cache:
                continue
            entry = load_entry(aid)
            if entry is not None:
                self._cache[entry.arxiv_id] = entry
                n += 1
        self._loaded_all = True
        logger.info(f"Library: loaded {n} entries from disk "
                    f"({len(self._cache)} total in memory)")
        return n

    # ══════════════════════════════════════════════════
    # single-entry access
    # ══════════════════════════════════════════════════

    def get(self, arxiv_id: str) -> LibraryEntry | None:
        """
        Return LibraryEntry for arxiv_id if present.
        Lazy loads from disk if not yet cached.
        """
        aid = str(arxiv_id).strip()
        if aid in self._cache:
            return self._cache[aid]
        entry = load_entry(aid)
        if entry is not None:
            self._cache[aid] = entry
        return entry

    def has(self, arxiv_id: str) -> bool:
        return self.get(arxiv_id) is not None

    # ══════════════════════════════════════════════════
    # add / update
    # ══════════════════════════════════════════════════

    def add_from_summary(
        self,
        arxiv_id:      str,
        summary:       StructuredSummary,
        query:         str,
        q_total:       float,
        pdf_path:      str = "",
        n_pages:       int = 0,
        section_names: list[str] | None = None,
    ) -> LibraryEntry:
        """
        Create a brand-new LibraryEntry.

        Fails if arxiv_id already present in cache. Use update_from_summary
        for merges.
        """
        aid = str(arxiv_id).strip()
        if aid in self._cache:
            raise ValueError(
                f"Entry for {aid} already exists — use update_from_summary"
            )

        now = _now_iso()
        entry = LibraryEntry(
            arxiv_id      = aid,
            summary       = summary,
            queries_used  = [query],
            first_seen    = now,
            last_updated  = now,
            n_analyses    = 1,
            pdf_path      = pdf_path,
            n_pages       = n_pages,
            section_names = section_names or [],
            best_q_total  = q_total,
            best_query    = query,
        )
        self._cache[aid] = entry
        save_entry(entry)
        logger.info(f"Library: added new entry for {aid} "
                    f"(Q={q_total:.3f})")
        return entry

    def update_from_summary(
        self,
        arxiv_id: str,
        summary:  StructuredSummary,
        query:    str,
        q_total:  float,
    ) -> LibraryEntry:
        """
        Merge a new summary into an existing entry.

        If no entry exists yet, delegates to add_from_summary with
        empty pipeline metadata.
        """
        aid = str(arxiv_id).strip()
        existing = self.get(aid)
        if existing is None:
            return self.add_from_summary(
                arxiv_id = aid,
                summary  = summary,
                query    = query,
                q_total  = q_total,
            )
        existing.merge_summary(summary, query, q_total)
        save_entry(existing)
        logger.info(
            f"Library: merged summary into {aid} "
            f"(now {existing.n_analyses} analyses, "
            f"best Q={existing.best_q_total:.3f})"
        )
        return existing

    # ══════════════════════════════════════════════════
    # deletion and iteration
    # ══════════════════════════════════════════════════

    def remove(self, arxiv_id: str) -> bool:
        aid = str(arxiv_id).strip()
        self._cache.pop(aid, None)
        return delete_entry(aid)

    def all_arxiv_ids(self) -> list[str]:
        if not self._loaded_all:
            self.load_all()
        return sorted(self._cache.keys())

    def size(self) -> int:
        return len(self._cache)

    # ══════════════════════════════════════════════════
    # stats
    # ══════════════════════════════════════════════════

    def stats(self) -> LibraryStats:
        if not self._loaded_all:
            self.load_all()

        queries = set()
        total_analyses = 0
        total_eq = 0
        total_nr = 0
        total_size = 0.0

        d = _library_dir()
        for aid, entry in self._cache.items():
            queries.update(entry.queries_used)
            total_analyses += entry.n_analyses
            total_eq       += len(entry.summary.key_equations)
            total_nr       += len(entry.summary.numerical_results)

        for p in d.glob("*.json"):
            try:
                total_size += p.stat().st_size / 1024.0
            except Exception:
                pass

        return LibraryStats(
            n_papers          = len(self._cache),
            n_queries         = len(queries),
            n_total_analyses  = total_analyses,
            n_equations_total = total_eq,
            n_numerical_total = total_nr,
            total_size_kb     = total_size,
        )


# ══════════════════════════════════════════════════════════
# module singleton
# ══════════════════════════════════════════════════════════

_library_instance: ContextLibrary | None = None


def get_library() -> ContextLibrary:
    global _library_instance
    if _library_instance is None:
        _library_instance = ContextLibrary()
    return _library_instance