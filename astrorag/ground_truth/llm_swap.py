"""
LLM swap experiment.

Rerun Stage 3 (rerank) and Stage 5 (summarise) with a different LLM
model. Everything else (corpus, BM25, graph, PDF extraction) stays
identical, so we isolate the effect of the LLM choice.
"""

from __future__ import annotations

import copy
import time
from   dataclasses import dataclass, field, asdict
from   pathlib     import Path
from typing        import Any

from astrorag.config  import Settings, get_settings
from astrorag.data    import CorpusData
from astrorag.logger  import get_logger
from astrorag.stages  import (
    Stage0Decompose, Stage1BM25, Stage2Graph,
    Stage3Rerank, Stage4PDF, Stage5Summarise,
)

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# result
# ══════════════════════════════════════════════════════════

@dataclass
class SwapResult:
    """Result of running the pipeline with a specific LLM."""

    llm_model:            str
    query_idx:            int
    query:                str
    selected_arxiv_id:    str
    stage3_bm25_rank:     int
    stage3_confidence:    float
    q_total:              float
    q_f:                  float
    q_c:                  float
    q_i:                  float
    decision:             str
    n_equations:          int
    n_numerical_results:  int
    total_seconds:        float
    error:                str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════
# experiment runner
# ══════════════════════════════════════════════════════════

class LLMSwapExperiment:
    """
    Rerun Stages 3 and 5 with a specified LLM model.

    Stages 0-2 (corpus, BM25, graph) reuse cached state.
    Stage 3 selects paper using the swap LLM.
    Stage 4 fetches PDF for the (possibly different) selection.
    Stage 5 summarises using the swap LLM.
    """

    def __init__(
        self,
        corpus:   CorpusData,
        settings: Settings | None = None,
    ) -> None:
        self.corpus   = corpus
        self.settings = settings or get_settings()

        # Stages 0-2 with default (baseline) LLM
        self.stage0 = Stage0Decompose(settings=self.settings)
        self.stage1 = Stage1BM25(corpus=corpus, settings=self.settings)
        self.stage2 = Stage2Graph(corpus=corpus, settings=self.settings)

    def run_query(
        self,
        query_idx:  int,
        query:      str,
        swap_model: str,
        sleep_s:    float = 15.0,
    ) -> SwapResult:
        """Run one query with the given LLM at Stages 3 and 5."""
        t0 = time.time()

        # settings copy with swapped model
        swapped_settings = copy.copy(self.settings)
        swapped_settings.groq_model = swap_model

        # per-query Stage 3 / 4 / 5 with the swap model
        stage3 = Stage3Rerank(settings=swapped_settings)
        stage4 = Stage4PDF(settings=swapped_settings)
        stage5 = Stage5Summarise(settings=swapped_settings, stage4=stage4)

        try:
            s0 = self.stage0.run(query)
            s1 = self.stage1.run(query, top_k=50)
            s2 = self.stage2.run(s1)

            s3 = stage3.run(
                retrieval     = s1,
                graph_context = s2,
                decomposition = s0.decomposition,
            )

            s4 = stage4.run(s3)
            if not s4.success:
                # try fallback pool
                for pool_idx in list(s3.fallback_pool):
                    next_paper = s1.results[pool_idx]
                    s4 = stage4.run(next_paper)
                    if s4.success:
                        s3.selected_result = next_paper
                        break
                if not s4.success:
                    return SwapResult(
                        llm_model            = swap_model,
                        query_idx            = query_idx,
                        query                = query,
                        selected_arxiv_id    = s3.selected_result.arxiv_id,
                        stage3_bm25_rank     = s3.selected_result.rank,
                        stage3_confidence    = s3.confidence,
                        q_total              = 0.0,
                        q_f                  = 0.0,
                        q_c                  = 0.0,
                        q_i                  = 0.0,
                        decision             = "PDF_FAIL",
                        n_equations          = 0,
                        n_numerical_results  = 0,
                        total_seconds        = time.time() - t0,
                        error                = f"PDF: {s4.error}",
                    )

            s5 = stage5.run(
                decomposition = s0.decomposition,
                retrieval     = s1,
                stage3_result = s3,
                initial_pdf   = s4,
            )

            return SwapResult(
                llm_model            = swap_model,
                query_idx            = query_idx,
                query                = query,
                selected_arxiv_id    = s5.selected_arxiv_id,
                stage3_bm25_rank     = s3.selected_result.rank,
                stage3_confidence    = s3.confidence,
                q_total              = s5.quality.scores.Q_total,
                q_f                  = s5.quality.scores.Q_f,
                q_c                  = s5.quality.scores.Q_c,
                q_i                  = s5.quality.scores.Q_i,
                decision             = s5.quality.decision.value,
                n_equations          = len(s5.summary.key_equations),
                n_numerical_results  = len(s5.summary.numerical_results),
                total_seconds        = time.time() - t0,
            )

        except Exception as e:
            logger.error(f"Swap failed for {swap_model} on Q{query_idx}: {e}")
            return SwapResult(
                llm_model            = swap_model,
                query_idx            = query_idx,
                query                = query,
                selected_arxiv_id    = "",
                stage3_bm25_rank     = 0,
                stage3_confidence    = 0.0,
                q_total              = 0.0,
                q_f                  = 0.0,
                q_c                  = 0.0,
                q_i                  = 0.0,
                decision             = "ERROR",
                n_equations          = 0,
                n_numerical_results  = 0,
                total_seconds        = time.time() - t0,
                error                = f"{type(e).__name__}: {e}",
            )
