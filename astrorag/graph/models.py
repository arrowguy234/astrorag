"""
Data models for graph construction outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ══════════════════════════════════════════════════════════
# raw signal matrices
# ══════════════════════════════════════════════════════════

@dataclass
class SignalMatrices:
    """
    The four signal matrices computed over the candidate papers.

    All are symmetric with zero diagonal, shape (K, K) where K is
    the number of BM25 candidates.
    """
    S1_concept:     np.ndarray            # concept embedding cosine similarity
    S2_biblio:      np.ndarray            # bibliographic coupling (Jaccard)
    S3_cocitation:  np.ndarray            # co-citation (geom mean norm)
    S4_domain:      np.ndarray            # domain hierarchy match

    W_combined:     np.ndarray            # weighted sum, thresholded

    n_edges_before: int = 0               # before threshold
    n_edges_after:  int = 0               # after threshold

    @property
    def n_nodes(self) -> int:
        return self.S1_concept.shape[0]

    @property
    def density(self) -> float:
        if self.n_nodes < 2:
            return 0.0
        max_edges = self.n_nodes * (self.n_nodes - 1) // 2
        return self.n_edges_after / max_edges if max_edges else 0.0


# ══════════════════════════════════════════════════════════
# cluster information
# ══════════════════════════════════════════════════════════

@dataclass
class ClusterInfo:
    """One cluster's summary information."""

    cluster_id:   int
    n_papers:     int
    member_idxs:  list[int]
    top_concepts: list[str]
    domains:      list[str]
    hub_idx:      int
    hub_arxiv_id: str
    hub_ppr:      float
    hub_title:    str


@dataclass
class ClusterSummary:
    """Full cluster summary passed to Stage 3 as LLM prompt context."""

    n_clusters:      int
    clusters:        list[ClusterInfo]
    cluster_labels:  np.ndarray            # (n_nodes,) integer cluster id per node
    prompt_text:     str = ""              # formatted text for LLM prompt

    def cluster_of(self, node_idx: int) -> int:
        return int(self.cluster_labels[node_idx])


# ══════════════════════════════════════════════════════════
# full graph context
# ══════════════════════════════════════════════════════════

@dataclass
class GraphContext:
    """
    Complete Stage 2 output.

    Passed to Stage 3 (LLM reranking) and Stage 5 (via metadata trail).
    """

    signals:          SignalMatrices
    ppr_scores:       np.ndarray           # normalised to [0, 1]
    ppr_raw:          np.ndarray           # raw PPR values before normalisation
    cluster_summary:  ClusterSummary

    n_nodes:          int = 0
    ppr_iterations:   int = 0
    elapsed_seconds:  float = 0.0

    def top_ppr_indices(self, n: int = 5) -> np.ndarray:
        return np.argsort(self.ppr_scores)[::-1][:n]

    def summary(self) -> str:
        s = self.signals
        return (
            f"Graph : {s.n_nodes} nodes  {s.n_edges_after} edges  "
            f"density={s.density:.1%}\n"
            f"PPR   : converged in {self.ppr_iterations} iters  "
            f"top={self.ppr_scores.max():.3f}\n"
            f"Time  : {self.elapsed_seconds:.3f}s"
        )