"""
Data models for retrieval results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing      import Any


# ══════════════════════════════════════════════════════════
# single result
# ══════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """
    A single retrieval result — one candidate paper.

    Contains the raw paper record plus retrieval metadata that
    downstream stages (Stage 2 graph, Stage 3 rerank) use.
    """

    # ── paper identifiers ───────────────────────────────
    arxiv_id:  str
    paper_idx: int

    # ── paper content ───────────────────────────────────
    title:    str = ""
    abstract: str = ""

    # ── retrieval scores ────────────────────────────────
    rank:            int   = 0
    bm25_score:      float = 0.0

    # ── augmented metadata ──────────────────────────────
    concepts:        list[str] = field(default_factory=list)
    concept_count:   int       = 0
    concept_overlap: int       = 0

    # ── downstream slots (filled by later stages) ───────
    ppr_score:  float = 0.0
    cluster:    int   = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "arxiv_id":         self.arxiv_id,
            "paper_idx":        self.paper_idx,
            "title":            self.title,
            "abstract":         self.abstract,
            "rank":             self.rank,
            "bm25_score":       self.bm25_score,
            "concepts":         self.concepts,
            "concept_count":    self.concept_count,
            "concept_overlap":  self.concept_overlap,
            "ppr_score":        self.ppr_score,
            "cluster":          self.cluster,
        }


# ══════════════════════════════════════════════════════════
# full run
# ══════════════════════════════════════════════════════════

@dataclass
class RetrievalRun:
    """
    Complete Stage 1 output for a single query.

    Contains the query, top-K results, timing information, and
    diagnostic metadata used for evaluation and logging.
    """

    query:          str
    results:        list[RetrievalResult]
    k:              int
    n_corpus:       int
    elapsed_s:      float
    top_score:      float = 0.0
    min_score_top:  float = 0.0
    mean_score_top: float = 0.0

    def top_arxiv_ids(self, n: int | None = None) -> list[str]:
        n = n or len(self.results)
        return [r.arxiv_id for r in self.results[:n]]

    def summary(self) -> str:
        return (
            f"Query    : {self.query[:70]}\n"
            f"Corpus   : {self.n_corpus:,} papers\n"
            f"Top-K    : {self.k}\n"
            f"Elapsed  : {self.elapsed_s:.3f}s\n"
            f"Top score: {self.top_score:.3f}\n"
            f"Mean top : {self.mean_score_top:.3f}"
        )