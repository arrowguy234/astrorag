"""
Context Library — persistent storage of every paper AstroRAG has processed.

Each library entry contains the paper metadata, structured summary,
original query, and full chat history. Enables browsing past results
and continuing conversations.

Storage: JSON file at data/context_library.json, keyed by arxiv_id.
"""

from __future__ import annotations

import json
from   dataclasses import dataclass, field, asdict
from   datetime    import datetime
from   pathlib     import Path
from typing        import Any

from astrorag.chat.models import ChatSession
from astrorag.logger      import get_logger
from astrorag.paths       import get_paths

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# library entry
# ══════════════════════════════════════════════════════════

@dataclass
class LibraryEntry:
    """One paper in the context library."""

    arxiv_id:       str
    title:          str
    abstract:       str
    original_query: str
    subdomain:      str = ""

    # summary content
    paper_overview:      str = ""
    evidence_type:       str = ""
    instruments:         list[str] = field(default_factory=list)
    key_equations:       list[dict[str, str]] = field(default_factory=list)
    numerical_results:   list[dict[str, str]] = field(default_factory=list)
    sub_question_answers: dict[str, dict] = field(default_factory=dict)
    key_snippet:         str = ""
    key_findings:        list[str] = field(default_factory=list)
    methodology:         str = ""

    # quality
    q_total: float = 0.0
    q_f:     float = 0.0
    q_c:     float = 0.0
    q_i:     float = 0.0
    decision: str  = ""

    # timing
    total_seconds: float = 0.0
    pdf_pages:     int   = 0
    n_sections:    int   = 0

    # session history
    added_at:    str = ""
    updated_at:  str = ""
    view_count:  int = 0
    chat_sessions: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.added_at:
            self.added_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.added_at

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LibraryEntry":
        return cls(**d)


# ══════════════════════════════════════════════════════════
# library manager
# ══════════════════════════════════════════════════════════

class ContextLibrary:
    """
    Persistent JSON-backed store of processed papers.

    Usage:
        lib = get_library()
        lib.add_from_stage5(query, stage5_result)
        entries = lib.list_all()
        entry   = lib.get("1102.1481")
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (get_paths().data_dir / "context_library.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, LibraryEntry] = {}
        self._load()

    # ══════════════════════════════════════════════════
    # persistence
    # ══════════════════════════════════════════════════

    def _load(self) -> None:
        if not self.path.exists():
            logger.debug(f"No existing library at {self.path}")
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            for aid, entry_dict in data.get("entries", {}).items():
                self._entries[aid] = LibraryEntry.from_dict(entry_dict)
            logger.info(f"Loaded {len(self._entries)} entries from library")
        except Exception as e:
            logger.warning(f"Failed to load library: {e}")

    def _save(self) -> None:
        payload = {
            "version":     1,
            "updated_at":  datetime.now().isoformat(),
            "n_entries":   len(self._entries),
            "entries":     {aid: e.to_dict()
                            for aid, e in self._entries.items()},
        }
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            logger.debug(f"Saved library ({len(self._entries)} entries)")
        except Exception as e:
            logger.error(f"Failed to save library: {e}")

    # ══════════════════════════════════════════════════
    # CRUD operations
    # ══════════════════════════════════════════════════

    def add(self, entry: LibraryEntry) -> None:
        """Add or update a library entry. Preserves chat sessions."""
        aid = entry.arxiv_id
        if aid in self._entries:
            existing = self._entries[aid]
            entry.chat_sessions = existing.chat_sessions
            entry.view_count    = existing.view_count + 1
            entry.added_at      = existing.added_at
        entry.updated_at = datetime.now().isoformat()
        self._entries[aid] = entry
        self._save()
        logger.info(f"Library updated: {aid} ({entry.title[:60]})")

    def add_from_stage5(
        self,
        query:         str,
        stage5_result: Any,          # Stage5Result
        subdomain:     str = "",
    ) -> LibraryEntry:
        """
        Create and add a LibraryEntry from a Stage 5 result.

        Convenience method that pulls all relevant fields from the
        pipeline output.
        """
        s5 = stage5_result
        summary = s5.summary
        pdf     = s5.pdf_doc

        entry = LibraryEntry(
            arxiv_id       = s5.selected_arxiv_id,
            title          = getattr(pdf, "title", "") or summary.paper_overview[:80],
            abstract       = "",
            original_query = query,
            subdomain      = subdomain,
            paper_overview = summary.paper_overview,
            evidence_type  = summary.evidence_type,
            instruments    = list(summary.instruments),
            key_equations  = [
                {"equation": e.equation, "variables": e.variables}
                for e in summary.key_equations
            ],
            numerical_results = [
                {"quantity": n.quantity, "value": n.value,
                 "uncertainty": n.uncertainty, "unit": n.unit}
                for n in summary.numerical_results
            ],
            sub_question_answers = {
                qk: {
                    "answered":    a.answered,
                    "answer_text": a.answer_text,
                    "section":     a.section,
                }
                for qk, a in summary.sub_question_answers.items()
            },
            key_snippet   = summary.key_snippet,
            key_findings  = list(summary.key_findings),
            methodology   = summary.methodology,
            q_total       = s5.quality.scores.Q_total,
            q_f           = s5.quality.scores.Q_f,
            q_c           = s5.quality.scores.Q_c,
            q_i           = s5.quality.scores.Q_i,
            decision      = s5.quality.decision.value,
            total_seconds = s5.total_time_s,
            pdf_pages     = pdf.n_pages,
            n_sections    = len(pdf.sections),
        )
        self.add(entry)
        return entry

    def get(self, arxiv_id: str) -> LibraryEntry | None:
        entry = self._entries.get(arxiv_id)
        if entry is not None:
            entry.view_count += 1
            self._save()
        return entry

    def remove(self, arxiv_id: str) -> bool:
        if arxiv_id in self._entries:
            del self._entries[arxiv_id]
            self._save()
            return True
        return False

    def list_all(self) -> list[LibraryEntry]:
        return list(self._entries.values())

    def search(self, keyword: str) -> list[LibraryEntry]:
        """Case-insensitive substring search over title, query, and overview."""
        kw = keyword.lower().strip()
        if not kw:
            return self.list_all()
        results = []
        for e in self._entries.values():
            if (kw in e.title.lower()
                or kw in e.original_query.lower()
                or kw in e.paper_overview.lower()
                or kw in e.arxiv_id.lower()
                or kw in e.subdomain.lower()):
                results.append(e)
        return results

    def add_chat_session(
        self,
        arxiv_id: str,
        session:  ChatSession,
    ) -> None:
        """Append a chat session to a library entry."""
        if arxiv_id not in self._entries:
            logger.warning(f"Cannot add chat to unknown entry: {arxiv_id}")
            return
        self._entries[arxiv_id].chat_sessions.append(session.to_dict())
        self._entries[arxiv_id].updated_at = datetime.now().isoformat()
        self._save()

    def stats(self) -> dict[str, Any]:
        entries = list(self._entries.values())
        if not entries:
            return {"n_entries": 0}
        return {
            "n_entries":         len(entries),
            "n_with_equations":  sum(1 for e in entries if e.key_equations),
            "n_with_numerical":  sum(1 for e in entries if e.numerical_results),
            "subdomains":        sorted({e.subdomain for e in entries if e.subdomain}),
            "mean_q_total":      sum(e.q_total for e in entries) / len(entries),
            "total_chat_sessions": sum(len(e.chat_sessions) for e in entries),
        }


# ══════════════════════════════════════════════════════════
# module singleton
# ══════════════════════════════════════════════════════════

_library_instance: ContextLibrary | None = None

def get_library() -> ContextLibrary:
    global _library_instance
    if _library_instance is None:
        _library_instance = ContextLibrary()
    return _library_instance