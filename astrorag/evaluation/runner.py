"""
Evaluation harness runner.

Runs the full pipeline on a query set, records structured traces
per query, and saves the aggregate result as JSON. Supports resuming
from a partial run.

Includes fallback pool handling: if Stage 4 fails on the primary
paper selection, iterates through Stage 3's fallback pool until a
PDF successfully fetches and parses.
"""

from __future__ import annotations

import json
import time
import traceback
from   datetime  import datetime
from   pathlib   import Path
from typing      import Iterable

from astrorag.config              import Settings, get_settings
from astrorag.data                import CorpusData, DataLoader
from astrorag.data.models         import LoadConfig
from astrorag.evaluation.models   import (
    EvaluationResult,
    QueryTrace,
    Stage0Trace,
    Stage1Trace,
    Stage2Trace,
    Stage3Trace,
    Stage4Trace,
    Stage5Trace,
)
from astrorag.evaluation.queries  import (
    EvaluationQuery,
    DEFAULT_QUERY_SET,
)
from astrorag.logger              import get_logger
from astrorag.stages              import (
    Stage0Decompose,
    Stage1BM25,
    Stage2Graph,
    Stage3Rerank,
    Stage4PDF,
    Stage5Summarise,
)

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# main runner
# ══════════════════════════════════════════════════════════

class EvaluationRunner:
    """
    Runs the full AstroRAG pipeline over a query set.

    Usage:
        runner = EvaluationRunner()
        result = runner.run(queries=DEFAULT_QUERY_SET, output_path=Path("out.json"))
    """

    def __init__(
        self,
        settings:      Settings | None = None,
        corpus:        CorpusData | None = None,
        top_k:         int  = 50,
        sleep_between_queries: float = 0.0,
    ) -> None:
        self.settings              = settings or get_settings()
        self.top_k                 = top_k
        self.sleep_between_queries = sleep_between_queries

        # ── load corpus ────────────────────────────────
        if corpus is None:
            logger.info("Loading corpus for evaluation...")
            config = LoadConfig(
                sample_size  = self.settings.sample_size,
                use_cache    = True,
                show_progress= True,
            )
            corpus = DataLoader(config=config).load()
        self.corpus = corpus

        # ── construct stages ───────────────────────────
        self.stage0 = Stage0Decompose(settings=self.settings)
        self.stage1 = Stage1BM25(corpus=self.corpus, settings=self.settings)
        self.stage2 = Stage2Graph(corpus=self.corpus, settings=self.settings)
        self.stage3 = Stage3Rerank(settings=self.settings)
        self.stage4 = Stage4PDF(settings=self.settings)
        self.stage5 = Stage5Summarise(settings=self.settings, stage4=self.stage4)

    # ══════════════════════════════════════════════════
    # main entry
    # ══════════════════════════════════════════════════

    def run(
        self,
        queries:      Iterable[EvaluationQuery],
        output_path:  Path,
        query_set_name: str = "default",
        resume:       bool = False,
    ) -> EvaluationResult:
        """
        Run the pipeline over a set of queries and save the result.

        Args:
            queries:        Iterable of EvaluationQuery objects.
            output_path:    Where to save the aggregate JSON result.
            query_set_name: Human-readable name for this run.
            resume:         If True and output_path exists, skip queries
                            already processed.

        Returns:
            EvaluationResult with all traces.
        """
        queries = list(queries)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        started_at = datetime.now().isoformat()
        t_start = time.time()

        # ── resume logic ────────────────────────────────
        traces: list[QueryTrace] = []
        completed_idxs:  set[int] = set()

        if resume and output_path.exists():
            try:
                prev = EvaluationResult.load(output_path)
                traces = prev.traces
                completed_idxs = {t.query_idx for t in traces if t.success}
                logger.info(
                    f"Resuming: {len(completed_idxs)} queries "
                    f"already completed"
                )
            except Exception as e:
                logger.warning(f"Resume failed, starting fresh: {e}")
                traces = []

        # ── run each query ──────────────────────────────
        n_succeeded = sum(1 for t in traces if t.success)
        n_failed    = sum(1 for t in traces if not t.success)

        for q in queries:
            if q.idx in completed_idxs:
                logger.info(f"Skipping query {q.idx} (already completed)")
                continue

            logger.info(f"═" * 70)
            logger.info(f"Query {q.idx}/{len(queries)}: {q.query[:80]}")
            logger.info(f"Subdomain: {q.subdomain}")

            trace = self._run_query(q)
            traces.append(trace)

            if trace.success:
                n_succeeded += 1
            else:
                n_failed += 1

            # incremental save after each query
            result = EvaluationResult(
                run_id          = run_id,
                query_set_name  = query_set_name,
                n_queries       = len(queries),
                n_completed     = len(traces),
                n_succeeded     = n_succeeded,
                n_failed        = n_failed,
                total_wall_s    = time.time() - t_start,
                started_at      = started_at,
                finished_at     = datetime.now().isoformat(),
                traces          = traces,
                config_snapshot = self._config_snapshot(),
            )
            result.save(output_path)
            logger.info(f"Saved incremental result → {output_path}")

            if self.sleep_between_queries > 0:
                logger.debug(
                    f"Sleeping {self.sleep_between_queries}s "
                    f"between queries"
                )
                time.sleep(self.sleep_between_queries)

        # final save
        result.finished_at  = datetime.now().isoformat()
        result.total_wall_s = time.time() - t_start
        result.save(output_path)

        logger.info("═" * 70)
        logger.info(
            f"Evaluation done: {n_succeeded} succeeded, "
            f"{n_failed} failed in {result.total_wall_s:.1f}s"
        )
        return result

    # ══════════════════════════════════════════════════
    # helpers
    # ══════════════════════════════════════════════════

    def _make_stage4_trace(self, s4) -> Stage4Trace:
        """Build a Stage4Trace from a Stage 4 result object."""
        return Stage4Trace(
            arxiv_id       = s4.arxiv_id,
            pdf_path       = str(s4.pdf_path) if s4.pdf_path else "",
            success        = s4.success,
            n_pages        = s4.n_pages,
            n_chars_total  = s4.n_chars_total,
            n_sections     = len(s4.sections),
            section_names  = list(s4.sections.keys()),
            extractor      = s4.extractor,
            fetch_seconds  = s4.fetch_seconds,
            parse_seconds  = s4.parse_seconds,
            from_cache     = s4.from_cache,
            error          = s4.error,
        )

    # ══════════════════════════════════════════════════
    # one query
    # ══════════════════════════════════════════════════

    def _run_query(self, q: EvaluationQuery) -> QueryTrace:
        """Run the full pipeline for one query and return the trace."""
        t0 = time.time()
        trace = QueryTrace(
            query_idx = q.idx,
            query     = q.query,
            subdomain = q.subdomain,
        )

        try:
            # ── stage 0 ────────────────────────────────
            s0 = self.stage0.run(q.query)
            trace.stage0 = Stage0Trace(
                q1            = s0.decomposition.sub_questions["Q1"],
                q2            = s0.decomposition.sub_questions["Q2"],
                q3            = s0.decomposition.sub_questions["Q3"],
                wavelength    = s0.decomposition.wavelength,
                catalogs      = list(s0.decomposition.catalogs),
                query_type    = s0.decomposition.query_type,
                fallback_used = s0.fallback_used,
                latency_s     = s0.total_time_s,
            )

            # ── stage 1 ────────────────────────────────
            s1 = self.stage1.run(q.query, top_k=self.top_k)
            trace.stage1 = Stage1Trace(
                n_corpus       = s1.n_corpus,
                top_k          = s1.k,
                top_arxiv_ids  = [r.arxiv_id  for r in s1.results[:20]],
                top_scores     = [r.bm25_score for r in s1.results[:20]],
                top_score      = s1.top_score,
                mean_top_score = s1.mean_score_top,
                latency_s      = s1.elapsed_s,
            )

            # ── stage 2 ────────────────────────────────
            s2 = self.stage2.run(s1)
            top_ppr_idxs   = s2.top_ppr_indices(20).tolist()
            trace.stage2 = Stage2Trace(
                n_nodes           = s2.n_nodes,
                n_edges           = s2.signals.n_edges_after,
                density           = s2.signals.density,
                ppr_iterations    = s2.ppr_iterations,
                top_ppr_arxiv_ids = [s1.results[i].arxiv_id for i in top_ppr_idxs],
                top_ppr_scores    = [float(s2.ppr_scores[i]) for i in top_ppr_idxs],
                n_clusters        = s2.cluster_summary.n_clusters,
                cluster_sizes     = [c.n_papers for c in s2.cluster_summary.clusters],
                latency_s         = s2.elapsed_seconds,
            )

            # ── stage 3 ────────────────────────────────
            s3 = self.stage3.run(
                retrieval     = s1,
                graph_context = s2,
                decomposition = s0.decomposition,
            )
            trace.stage3 = Stage3Trace(
                selected_arxiv_id   = s3.selected_result.arxiv_id,
                selected_bm25_rank  = s3.selected_result.rank,
                selected_bm25_score = s3.selected_result.bm25_score,
                selected_ppr_score  = s3.selected_result.ppr_score,
                confidence          = s3.confidence,
                graph_adj_score     = s3.graph_adj_score,
                reason              = s3.reason,
                fallback_pool       = list(s3.fallback_pool),
                fallback_used       = s3.fallback_used,
                latency_s           = s3.total_time_s,
            )

            # ── stage 4 ────────────────────────────────
            s4 = self.stage4.run(s3)
            trace.stage4 = self._make_stage4_trace(s4)

            # ── stage 4 fallback: try next candidate on PDF fail ─
            if not s4.success:
                logger.warning(
                    f"Primary PDF fetch failed for "
                    f"{s3.selected_result.arxiv_id}, trying fallback pool"
                )
                fallback_success = False
                for pool_idx in list(s3.fallback_pool):
                    next_paper = s1.results[pool_idx]
                    logger.info(
                        f"Trying fallback #{pool_idx}: {next_paper.arxiv_id}"
                    )
                    s4 = self.stage4.run(next_paper)
                    if s4.success:
                        logger.info(
                            f"Fallback succeeded with {next_paper.arxiv_id}"
                        )
                        # update stage 3 result to reflect actual selection
                        s3.selected_result = next_paper
                        s3.selected_idx    = pool_idx
                        # remove used fallback from pool
                        s3.fallback_pool.remove(pool_idx)
                        # refresh stage 4 trace
                        trace.stage4 = self._make_stage4_trace(s4)
                        # update stage 3 trace to reflect new selection
                        trace.stage3.selected_arxiv_id   = next_paper.arxiv_id
                        trace.stage3.selected_bm25_rank  = next_paper.rank
                        trace.stage3.selected_bm25_score = next_paper.bm25_score
                        trace.stage3.selected_ppr_score  = next_paper.ppr_score
                        trace.stage3.fallback_pool       = list(s3.fallback_pool)
                        trace.stage3.fallback_used       = True
                        fallback_success = True
                        break

                if not fallback_success:
                    trace.success = False
                    trace.error   = (
                        f"All PDF fetches failed (primary "
                        f"+ {len(s3.fallback_pool)} fallback candidates)"
                    )
                    trace.total_seconds = time.time() - t0
                    return trace

            # ── stage 5 ────────────────────────────────
            s5 = self.stage5.run(
                decomposition = s0.decomposition,
                retrieval     = s1,
                stage3_result = s3,
                initial_pdf   = s4,
            )
            trace.stage5 = Stage5Trace(
                final_arxiv_id        = s5.selected_arxiv_id,
                accepted              = s5.accepted,
                decision              = s5.quality.decision.value,
                q_f                   = s5.quality.scores.Q_f,
                q_c                   = s5.quality.scores.Q_c,
                q_i                   = s5.quality.scores.Q_i,
                q_total               = s5.quality.scores.Q_total,
                n_claims_verified     = s5.quality.scores.n_claims_verified,
                n_claims_total        = s5.quality.scores.n_claims_total,
                snippet_overlap       = s5.quality.scores.snippet_overlap,
                has_equations         = s5.quality.scores.has_equations,
                has_numerical_results = s5.quality.scores.has_numerical_results,
                n_equations           = len(s5.summary.key_equations),
                n_numerical_results   = len(s5.summary.numerical_results),
                n_instruments         = len(s5.summary.instruments),
                evidence_type         = s5.summary.evidence_type,
                n_attempts            = s5.n_attempts,
                n_retries             = s5.n_retries,
                n_reselections        = s5.n_reselections,
                fallback_used         = list(s5.fallback_pool_used),
                latency_s             = s5.total_time_s,
                paper_overview        = s5.summary.paper_overview[:300],
                key_snippet           = s5.summary.key_snippet[:300],
            )

        except Exception as e:
            trace.success = False
            trace.error   = f"{type(e).__name__}: {e}"
            logger.error(
                f"Query {q.idx} failed with {type(e).__name__}: {e}"
            )
            logger.debug(traceback.format_exc())

        trace.total_seconds = time.time() - t0
        return trace

    # ══════════════════════════════════════════════════
    # config snapshot
    # ══════════════════════════════════════════════════

    def _config_snapshot(self) -> dict:
        """Serialisable snapshot of the settings used for this run."""
        s = self.settings
        return {
            "top_k":         self.top_k,
            "bm25_k1":       s.bm25_k1,
            "bm25_b":        s.bm25_b,
            "edge_threshold": s.edge_threshold,
            "n_clusters":    s.n_clusters,
            "w_s1_concept":  s.w_s1_concept,
            "w_s2_biblio":   s.w_s2_biblio,
            "w_s3_cocitation": s.w_s3_cocitation,
            "w_s4_domain":   s.w_s4_domain,
            "ppr_alpha":     s.ppr_alpha,
            "q_accept":      s.q_accept_threshold,
            "q_retry":       s.q_retry_threshold,
            "q_weight_f":    s.q_weight_faithfulness,
            "q_weight_c":    s.q_weight_coverage,
            "q_weight_i":    s.q_weight_consistency,
            "groq_model":    s.groq_model,
            "max_reselect":  s.max_reselect_attempts,
        }