"""
Stage 2 — Context Graph and Personalised PageRank.

Given Stage 1's top-K candidates, constructs a weighted graph over
them using four signals, runs PPR seeded from BM25 scores, and
produces a K-means cluster summary for Stage 3's LLM prompt.
"""

from __future__ import annotations

import time
from   dataclasses import dataclass

import numpy as np

from astrorag.config             import Settings, get_settings
from astrorag.data               import CorpusData
from astrorag.graph              import (
    GraphContext,
    build_cluster_summary,
    compute_signal_matrices,
    personalized_pagerank,
)
from astrorag.logger             import get_logger
from astrorag.retrieval          import RetrievalRun

logger = get_logger(__name__)


class Stage2Graph:
    """
    Stage 2 — 4-signal Context Graph + PPR.

    Usage:
        stage2  = Stage2Graph(corpus=corpus_data)
        context = stage2.run(retrieval_run)
        print(context.summary())
        top_ppr = context.top_ppr_indices(5)
    """

    def __init__(
        self,
        corpus:   CorpusData,
        settings: Settings | None = None,
    ) -> None:
        self.corpus   = corpus
        self.settings = settings or get_settings()

    # ══════════════════════════════════════════════════
    # main run
    # ══════════════════════════════════════════════════

    def run(self, retrieval: RetrievalRun) -> GraphContext:
        """
        Build graph, run PPR, produce cluster summary.

        Args:
            retrieval: Output of Stage 1.

        Returns:
            GraphContext with signals, PPR scores, cluster summary.
        """
        results   = retrieval.results
        arxiv_ids = [r.arxiv_id for r in results]
        n         = len(results)

        if n < 2:
            raise ValueError(f"Stage 2 requires ≥2 papers, got {n}")

        logger.info(f"Stage 2 — building graph over {n} candidates")

        t0 = time.time()

        # ── compute signal matrices ─────────────────────
        signals = compute_signal_matrices(
            arxiv_ids = arxiv_ids,
            corpus    = self.corpus,
            settings  = self.settings,
        )

        # ── run PPR ─────────────────────────────────────
        bm25_scores = np.array(
            [r.bm25_score for r in results], dtype=np.float32
        )
        ppr_norm, ppr_raw, iterations = personalized_pagerank(
            W           = signals.W_combined,
            bm25_scores = bm25_scores,
            settings    = self.settings,
        )

        # ── mutate results with PPR scores ──────────────
        for i, r in enumerate(results):
            r.ppr_score = float(ppr_norm[i])

        # ── build cluster summary ───────────────────────
        cluster_summary = build_cluster_summary(
            results    = results,
            ppr_scores = ppr_norm,
            corpus     = self.corpus,
            settings   = self.settings,
        )

        # ── attach cluster labels to results ────────────
        for i, r in enumerate(results):
            r.cluster = cluster_summary.cluster_of(i)

        elapsed = time.time() - t0

        context = GraphContext(
            signals          = signals,
            ppr_scores       = ppr_norm,
            ppr_raw          = ppr_raw,
            cluster_summary  = cluster_summary,
            n_nodes          = n,
            ppr_iterations   = iterations,
            elapsed_seconds  = elapsed,
        )

        logger.info(
            f"Stage 2 done in {elapsed:.3f}s — "
            f"{signals.n_edges_after} edges, density={signals.density:.1%}, "
            f"top PPR={ppr_norm.max():.3f}"
        )

        return context