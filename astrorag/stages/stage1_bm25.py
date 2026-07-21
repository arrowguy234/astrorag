"""
Stage 1 — BM25 Retrieval.

Given a query, returns the top-K candidate papers ranked by BM25
score over the full corpus. Each result is enriched with concept
labels and query-concept overlap counts for use by downstream stages.
"""

from __future__ import annotations

import time
from   dataclasses import dataclass

import numpy as np

from astrorag.config              import Settings, get_settings
from astrorag.data                import CorpusData
from astrorag.logger              import get_logger
from astrorag.retrieval           import (
    BM25Index,
    RetrievalResult,
    RetrievalRun,
    build_bm25_index,
    tokenize,
    normalize_arxiv_id,
)

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# Stage 1 orchestrator
# ══════════════════════════════════════════════════════════

class Stage1BM25:
    """
    Stage 1 — BM25 Retrieval.

    Wraps the BM25 index with query enrichment logic. The index
    itself is built once and reused across queries — Stage1BM25
    should be constructed once per pipeline session.

    Usage:
        stage1 = Stage1BM25(corpus=corpus_data)
        run    = stage1.run("How do AGN jets suppress star formation?")
        print(run.summary())
        for r in run.results[:5]:
            print(r.arxiv_id, r.bm25_score, r.title[:60])
    """

    def __init__(
        self,
        corpus:   CorpusData,
        index:    BM25Index | None = None,
        settings: Settings  | None = None,
    ) -> None:
        self.corpus   = corpus
        self.settings = settings or get_settings()

        if index is None:
            logger.info("Stage 1 — building/loading BM25 index...")
            self.index = build_bm25_index(
                corpus        = corpus,
                settings      = self.settings,
                show_progress = True,
            )
        else:
            self.index = index

        # build fast paper_idx → position lookup
        self._paper_position: dict[str, int] = {
            aid: pos for pos, aid in enumerate(self.index.arxiv_ids)
        }

    # ══════════════════════════════════════════════════
    # main retrieval
    # ══════════════════════════════════════════════════

    def run(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalRun:
        """
        Retrieve top-K candidates for a query.

        Args:
            query: Natural language query.
            top_k: Override for number of results (default from settings).

        Returns:
            RetrievalRun with all results, timing, and diagnostics.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        top_k = top_k or self.settings.top_k
        logger.info(
            f"Stage 1 — BM25 retrieval for: {query[:60]} "
            f"(top-{top_k} of {self.index.n_docs:,})"
        )

        t0 = time.time()

        # ── score all documents ─────────────────────────
        query_tokens = tokenize(query)
        if not query_tokens:
            logger.warning("Query produced no tokens after tokenization")

        scores = self.index.bm25.get_scores(query_tokens)

        # ── top-K by score ──────────────────────────────
        top_idxs = np.argsort(scores)[::-1][:top_k]

        # ── build results ───────────────────────────────
        results = self._build_results(
            top_idxs = top_idxs,
            scores   = scores,
            query    = query,
        )

        elapsed = time.time() - t0

        top_scores = np.array([r.bm25_score for r in results])
        run = RetrievalRun(
            query          = query,
            results        = results,
            k              = top_k,
            n_corpus       = self.index.n_docs,
            elapsed_s      = elapsed,
            top_score      = float(top_scores.max())  if len(top_scores) else 0.0,
            min_score_top  = float(top_scores.min())  if len(top_scores) else 0.0,
            mean_score_top = float(top_scores.mean()) if len(top_scores) else 0.0,
        )

        logger.info(
            f"Stage 1 done in {elapsed:.3f}s — "
            f"top={run.top_score:.2f} mean={run.mean_score_top:.2f}"
        )
        return run

    # ── build result objects ────────────────────────────
    def _build_results(
        self,
        top_idxs: np.ndarray,
        scores:   np.ndarray,
        query:    str,
    ) -> list[RetrievalResult]:
        """
        Convert top-K indices into RetrievalResult objects
        with concept and overlap metadata.
        """
        query_words = set(tokenize(query))

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(top_idxs, start=1):
            idx        = int(idx)
            arxiv_id   = self.index.arxiv_ids[idx]
            paper_idx  = self.index.paper_idxs[idx]
            paper      = self.corpus.papers[idx]

            concepts = self.corpus.paper_to_concepts.get(arxiv_id, [])[:20]
            concept_text = " ".join(concepts).lower()
            concept_words = set(tokenize(concept_text))
            overlap = len(query_words & concept_words)

            result = RetrievalResult(
                arxiv_id        = arxiv_id,
                paper_idx       = paper_idx,
                title           = str(paper.get("title", "")),
                abstract        = str(paper.get("abstract", "")),
                rank            = rank,
                bm25_score      = float(scores[idx]),
                concepts        = concepts,
                concept_count   = len(self.corpus.paper_to_concepts.get(arxiv_id, [])),
                concept_overlap = overlap,
            )
            results.append(result)

        return results

    # ══════════════════════════════════════════════════
    # info accessors
    # ══════════════════════════════════════════════════

    def n_docs(self) -> int:
        return self.index.n_docs

    def index_stats(self) -> dict:
        return {
            "n_docs":     self.index.n_docs,
            "k1":         self.index.k1,
            "b":          self.index.b,
            "build_time": self.index.build_time_seconds,
        }