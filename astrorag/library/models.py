"""
Data models for the persistent context library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime    import datetime, timezone
from typing      import Any

from astrorag.llm.models import (
    KeyEquation, NumericalResult, StructuredSummary, SubQuestionAnswer,
)


# ══════════════════════════════════════════════════════════
# entry
# ══════════════════════════════════════════════════════════

@dataclass
class LibraryEntry:
    """
    One paper's accumulated knowledge in the library.

    Contains the full StructuredSummary plus metadata about which
    queries have used this paper and when it was first/last analysed.
    """

    arxiv_id:      str
    summary:       StructuredSummary

    # ── provenance ──────────────────────────────────────
    queries_used:  list[str]  = field(default_factory=list)
    first_seen:    str        = ""       # ISO 8601 UTC
    last_updated:  str        = ""       # ISO 8601 UTC
    n_analyses:    int        = 1

    # ── pipeline metadata ───────────────────────────────
    pdf_path:      str        = ""
    n_pages:       int        = 0
    section_names: list[str]  = field(default_factory=list)

    # ── aggregate quality across all analyses ───────────
    best_q_total:  float      = 0.0
    best_query:    str        = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "arxiv_id":      self.arxiv_id,
            "queries_used":  self.queries_used,
            "first_seen":    self.first_seen,
            "last_updated":  self.last_updated,
            "n_analyses":    self.n_analyses,
            "pdf_path":      self.pdf_path,
            "n_pages":       self.n_pages,
            "section_names": self.section_names,
            "best_q_total":  self.best_q_total,
            "best_query":    self.best_query,
            "summary":       self.summary.model_dump(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LibraryEntry":
        return cls(
            arxiv_id      = d["arxiv_id"],
            summary       = StructuredSummary(**d["summary"]),
            queries_used  = d.get("queries_used", []),
            first_seen    = d.get("first_seen", ""),
            last_updated  = d.get("last_updated", ""),
            n_analyses    = d.get("n_analyses", 1),
            pdf_path      = d.get("pdf_path", ""),
            n_pages       = d.get("n_pages", 0),
            section_names = d.get("section_names", []),
            best_q_total  = d.get("best_q_total", 0.0),
            best_query    = d.get("best_query", ""),
        )

    def merge_summary(
        self,
        new_summary: StructuredSummary,
        query:       str,
        q_total:     float,
    ) -> None:
        """
        Merge a new summary into this entry.

        Union-based: new equations, values, and findings are appended
        (deduplicated). The paper_overview and methodology are kept
        from whichever analysis had the highest Q_total.
        """
        # update provenance
        if query not in self.queries_used:
            self.queries_used.append(query)
        self.last_updated = _now_iso()
        self.n_analyses  += 1

        # keep highest-quality version of narrative fields
        if q_total > self.best_q_total:
            self.summary.paper_overview = new_summary.paper_overview
            self.summary.methodology    = new_summary.methodology
            self.summary.key_snippet    = new_summary.key_snippet
            self.summary.evidence_type  = new_summary.evidence_type
            self.best_q_total           = q_total
            self.best_query             = query

        # union of instruments
        for inst in new_summary.instruments:
            if inst not in self.summary.instruments:
                self.summary.instruments.append(inst)

        # union of key equations by equation string
        existing_eqs = {eq.equation for eq in self.summary.key_equations}
        for eq in new_summary.key_equations:
            if eq.equation and eq.equation not in existing_eqs:
                self.summary.key_equations.append(eq)
                existing_eqs.add(eq.equation)

        # union of numerical results by (quantity, unit)
        existing_nr = {(nr.quantity, nr.unit)
                       for nr in self.summary.numerical_results}
        for nr in new_summary.numerical_results:
            key = (nr.quantity, nr.unit)
            if key not in existing_nr:
                self.summary.numerical_results.append(nr)
                existing_nr.add(key)

        # union of findings
        for finding in new_summary.key_findings:
            if finding not in self.summary.key_findings:
                self.summary.key_findings.append(finding)

        # union of limitations
        for lim in new_summary.limitations:
            if lim not in self.summary.limitations:
                self.summary.limitations.append(lim)

        # sub-question answers — add if this query's Qk not yet answered
        for qk, ans in new_summary.sub_question_answers.items():
            if qk not in self.summary.sub_question_answers:
                self.summary.sub_question_answers[qk] = ans
            else:
                existing = self.summary.sub_question_answers[qk]
                if not existing.answered and ans.answered:
                    self.summary.sub_question_answers[qk] = ans


# ══════════════════════════════════════════════════════════
# stats
# ══════════════════════════════════════════════════════════

@dataclass
class LibraryStats:
    """Aggregate statistics over the library."""

    n_papers:          int = 0
    n_queries:         int = 0
    n_total_analyses:  int = 0
    n_equations_total: int = 0
    n_numerical_total: int = 0
    total_size_kb:     float = 0.0

    def summary(self) -> str:
        return (
            f"Library Statistics\n"
            f"  Papers          : {self.n_papers:,}\n"
            f"  Unique queries  : {self.n_queries:,}\n"
            f"  Total analyses  : {self.n_total_analyses:,}\n"
            f"  Equations       : {self.n_equations_total:,}\n"
            f"  Numerical results: {self.n_numerical_total:,}\n"
            f"  Size on disk    : {self.total_size_kb:.1f} KB"
        )


# ══════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")