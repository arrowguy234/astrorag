"""
Data models for evaluation traces.

Every stage produces a Stage#Trace with the fields we need for
downstream analysis and paper metrics. A QueryTrace bundles all
stage traces for a single query. An EvaluationResult bundles all
QueryTraces for a full run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from   pathlib   import Path
from typing      import Any

# ══════════════════════════════════════════════════════════
# per-stage traces
# ══════════════════════════════════════════════════════════

@dataclass
class Stage0Trace:
    """Decomposition output."""
    q1:            str
    q2:            str
    q3:            str
    wavelength:    str
    catalogs:      list[str]
    query_type:    str
    fallback_used: bool
    latency_s:     float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage1Trace:
    """BM25 retrieval output."""
    n_corpus:       int
    top_k:          int
    top_arxiv_ids:  list[str]
    top_scores:     list[float]
    top_score:      float
    mean_top_score: float
    latency_s:      float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage2Trace:
    """Graph construction + PPR output."""
    n_nodes:            int
    n_edges:            int
    density:            float
    ppr_iterations:     int
    top_ppr_arxiv_ids:  list[str]
    top_ppr_scores:     list[float]
    n_clusters:         int
    cluster_sizes:      list[int]
    latency_s:          float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage3Trace:
    """Reranking output."""
    selected_arxiv_id: str
    selected_bm25_rank: int
    selected_bm25_score: float
    selected_ppr_score: float
    confidence:        float
    graph_adj_score:   float
    reason:            str
    fallback_pool:     list[int]
    fallback_used:     bool
    latency_s:         float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage4Trace:
    """PDF fetch/parse output."""
    arxiv_id:        str
    pdf_path:        str
    success:         bool
    n_pages:         int
    n_chars_total:   int
    n_sections:      int
    section_names:   list[str]
    extractor:       str
    fetch_seconds:   float
    parse_seconds:   float
    from_cache:      bool
    error:           str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage5Trace:
    """Deep summarisation + quality gate output."""
    final_arxiv_id:      str
    accepted:            bool
    decision:            str
    q_f:                 float
    q_c:                 float
    q_i:                 float
    q_total:             float
    n_claims_verified:   int
    n_claims_total:      int
    snippet_overlap:     float
    has_equations:       bool
    has_numerical_results: bool
    n_equations:         int
    n_numerical_results: int
    n_instruments:       int
    evidence_type:       str
    n_attempts:          int
    n_retries:           int
    n_reselections:      int
    fallback_used:       list[int]
    latency_s:           float
    paper_overview:      str = ""
    key_snippet:         str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════
# per-query bundle
# ══════════════════════════════════════════════════════════

@dataclass
class QueryTrace:
    """Full trace for a single query, all stages."""

    query_idx:      int
    query:          str
    subdomain:      str  = ""
    total_seconds:  float = 0.0
    success:        bool  = True
    error:          str   = ""

    stage0: Stage0Trace | None = None
    stage1: Stage1Trace | None = None
    stage2: Stage2Trace | None = None
    stage3: Stage3Trace | None = None
    stage4: Stage4Trace | None = None
    stage5: Stage5Trace | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_idx":     self.query_idx,
            "query":         self.query,
            "subdomain":     self.subdomain,
            "total_seconds": self.total_seconds,
            "success":       self.success,
            "error":         self.error,
            "stage0": self.stage0.to_dict() if self.stage0 else None,
            "stage1": self.stage1.to_dict() if self.stage1 else None,
            "stage2": self.stage2.to_dict() if self.stage2 else None,
            "stage3": self.stage3.to_dict() if self.stage3 else None,
            "stage4": self.stage4.to_dict() if self.stage4 else None,
            "stage5": self.stage5.to_dict() if self.stage5 else None,
        }


# ══════════════════════════════════════════════════════════
# full run
# ══════════════════════════════════════════════════════════

@dataclass
class EvaluationResult:
    """Bundle of all query traces plus run metadata."""

    run_id:           str
    query_set_name:   str
    n_queries:        int
    n_completed:      int
    n_succeeded:      int
    n_failed:         int
    total_wall_s:     float
    started_at:       str
    finished_at:      str
    traces:           list[QueryTrace] = field(default_factory=list)
    config_snapshot:  dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id":         self.run_id,
            "query_set_name": self.query_set_name,
            "n_queries":      self.n_queries,
            "n_completed":    self.n_completed,
            "n_succeeded":    self.n_succeeded,
            "n_failed":       self.n_failed,
            "total_wall_s":   self.total_wall_s,
            "started_at":     self.started_at,
            "finished_at":    self.finished_at,
            "config_snapshot": self.config_snapshot,
            "traces":         [t.to_dict() for t in self.traces],
        }

    def save(self, path: Path) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "EvaluationResult":
        import json
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        traces = [_dict_to_query_trace(t) for t in data.pop("traces", [])]
        return cls(**data, traces=traces)


# ══════════════════════════════════════════════════════════
# deserialisation helper
# ══════════════════════════════════════════════════════════

def _dict_to_query_trace(d: dict[str, Any]) -> QueryTrace:
    def _mk(cls, d_):
        return cls(**d_) if d_ else None

    return QueryTrace(
        query_idx     = d["query_idx"],
        query         = d["query"],
        subdomain     = d.get("subdomain", ""),
        total_seconds = d.get("total_seconds", 0.0),
        success       = d.get("success", True),
        error         = d.get("error", ""),
        stage0 = _mk(Stage0Trace, d.get("stage0")),
        stage1 = _mk(Stage1Trace, d.get("stage1")),
        stage2 = _mk(Stage2Trace, d.get("stage2")),
        stage3 = _mk(Stage3Trace, d.get("stage3")),
        stage4 = _mk(Stage4Trace, d.get("stage4")),
        stage5 = _mk(Stage5Trace, d.get("stage5")),
    )