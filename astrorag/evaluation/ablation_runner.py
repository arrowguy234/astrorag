"""
Ablation runner — runs each variant on the query set without
modifying the original pipeline stages.

Design principle: the six pipeline stages are called normally, but
their outputs are conditionally IGNORED or REPLACED based on the
active AblationVariant. This preserves the original code exactly
while enabling clean comparisons.
"""

from __future__ import annotations

import time
import traceback
from   copy      import deepcopy
from   dataclasses import dataclass, field
from   datetime  import datetime
from   pathlib   import Path
from typing      import Iterable

import numpy as np

from astrorag.config              import Settings, get_settings
from astrorag.data                import CorpusData, DataLoader
from astrorag.data.models         import LoadConfig
from astrorag.evaluation.ablation import AblationVariant, ABLATION_VARIANTS
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
from astrorag.evaluation.queries  import EvaluationQuery
from astrorag.evaluation.runner   import EvaluationRunner
from astrorag.extraction          import assess_quality
from astrorag.llm.models          import (
    NumericalResult,
    StructuredSummary,
    SubQuestionAnswer,
)
from astrorag.logger              import get_logger
from astrorag.pdf                 import PDFDocument
from astrorag.retrieval           import RetrievalResult
from astrorag.stages              import (
    Stage0Decompose,
    Stage1BM25,
    Stage2Graph,
    Stage3Rerank,
    Stage3Result,
    Stage4PDF,
    Stage5Summarise,
    Stage5Result,
)
from astrorag.extraction.quality  import QualityAssessment, QualityDecision, QualityScores

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# ablation runner
# ══════════════════════════════════════════════════════════

class AblationRunner:
    """
    Run a specific ablation variant over a query set.

    Reuses existing Stage classes; conditionally bypasses their
    outputs according to the variant flags.
    """

    def __init__(
        self,
        variant:  AblationVariant,
        settings: Settings   | None = None,
        corpus:   CorpusData | None = None,
        top_k:    int  = 50,
        sleep_between_queries: float = 0.0,
    ) -> None:
        self.variant  = variant
        self.settings = settings or get_settings()
        self.top_k    = top_k
        self.sleep_between_queries = sleep_between_queries

        # load corpus once
        if corpus is None:
            logger.info(f"[{variant.name}] Loading corpus...")
            config = LoadConfig(
                sample_size  = self.settings.sample_size,
                use_cache    = True,
                show_progress= True,
            )
            corpus = DataLoader(config=config).load()
        self.corpus = corpus

        # construct all stages — unchanged from original
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
        queries:     Iterable[EvaluationQuery],
        output_path: Path,
    ) -> EvaluationResult:
        """Run this variant across the query set and save results."""
        queries = list(queries)
        run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
        started_at = datetime.now().isoformat()
        t_start    = time.time()

        traces:      list[QueryTrace] = []
        n_succeeded = 0
        n_failed    = 0

        for q in queries:
            logger.info(f"═" * 70)
            logger.info(
                f"[{self.variant.name}] Query {q.idx}/{len(queries)}: "
                f"{q.query[:70]}"
            )

            trace = self._run_query(q)
            traces.append(trace)

            if trace.success:
                n_succeeded += 1
            else:
                n_failed += 1

            # incremental save
            result = EvaluationResult(
                run_id          = f"{run_id}_{self.variant.name}",
                query_set_name  = f"ablation_{self.variant.name}",
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

            if self.sleep_between_queries > 0:
                time.sleep(self.sleep_between_queries)

        result.finished_at  = datetime.now().isoformat()
        result.total_wall_s = time.time() - t_start
        result.save(output_path)

        logger.info(
            f"[{self.variant.name}] Done: {n_succeeded}/{len(queries)} "
            f"in {result.total_wall_s:.1f}s"
        )
        return result

    # ══════════════════════════════════════════════════
    # one query — applies variant switches
    # ══════════════════════════════════════════════════

    def _run_query(self, q: EvaluationQuery) -> QueryTrace:
        t0 = time.time()
        trace = QueryTrace(
            query_idx = q.idx,
            query     = q.query,
            subdomain = q.subdomain,
        )

        try:
            # ── Stage 0 (always run) ────────────────────
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

            # ── Stage 1 (always run) ────────────────────
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

            # ── Stage 2 (always run for timing; conditionally used) ─
            s2 = self.stage2.run(s1)
            trace.stage2 = self._make_stage2_trace(s2, s1)

            # if graph disabled, wipe PPR scores so Stage 3 sees BM25 only
            if not self.variant.use_graph:
                logger.debug(f"[{self.variant.name}] disabling graph reranking")
                for r in s1.results:
                    r.ppr_score = 0.0
                s2.ppr_scores = np.zeros_like(s2.ppr_scores)

            # ── Stage 3 (LLM rerank OR top-BM25 direct) ─
            if self.variant.use_llm_rerank:
                s3 = self.stage3.run(
                    retrieval     = s1,
                    graph_context = s2,
                    decomposition = s0.decomposition,
                )
            else:
                # bypass: pick top-BM25 directly
                s3 = self._make_bm25_direct_result(s1, s2)
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

            # ── Stage 4 (fetch PDF unless disabled) ─────
            if self.variant.use_pdf:
                s4 = self.stage4.run(s3)
                trace.stage4 = self._make_stage4_trace(s4)

                # PDF-fallback if primary fetch fails
                if not s4.success:
                    logger.warning(
                        f"[{self.variant.name}] primary PDF fetch failed, "
                        f"trying fallback pool"
                    )
                    fallback_success = False
                    for pool_idx in list(s3.fallback_pool):
                        next_paper = s1.results[pool_idx]
                        s4 = self.stage4.run(next_paper)
                        if s4.success:
                            s3.selected_result = next_paper
                            s3.fallback_pool.remove(pool_idx)
                            trace.stage4 = self._make_stage4_trace(s4)
                            fallback_success = True
                            break
                    if not fallback_success:
                        trace.success = False
                        trace.error = "All PDF fetches failed"
                        trace.total_seconds = time.time() - t0
                        return trace
            else:
                # bypass: construct a fake PDFDocument from abstract only
                s4 = self._make_abstract_only_pdf(s3.selected_result)
                trace.stage4 = self._make_stage4_trace(s4)

            # ── Stage 5 (full summarisation OR abstract passthrough) ─
            if self.variant.use_full_summary:
                if self.variant.use_quality_gate:
                    # full pipeline behaviour
                    s5 = self.stage5.run(
                        decomposition = s0.decomposition,
                        retrieval     = s1,
                        stage3_result = s3,
                        initial_pdf   = s4,
                    )
                else:
                    # quality gate off: one attempt, force accept
                    s5 = self._run_stage5_no_gate(
                        s0.decomposition, s1, s3, s4
                    )
            else:
                # bypass entirely: return abstract as answer
                s5 = self._make_abstract_only_summary(
                    s0.decomposition, s3.selected_result, s4
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
                f"[{self.variant.name}] Query {q.idx} failed: "
                f"{type(e).__name__}: {e}"
            )
            logger.debug(traceback.format_exc())

        trace.total_seconds = time.time() - t0
        return trace

    # ══════════════════════════════════════════════════
    # bypass helpers
    # ══════════════════════════════════════════════════

    def _make_bm25_direct_result(
        self,
        s1,
        s2,
    ) -> Stage3Result:
        """
        Bypass Stage 3: pick top-BM25 paper directly.
        Fallback pool = next 5 by BM25 score.
        """
        top = s1.results[0]
        pool = list(range(1, min(6, len(s1.results))))
        return Stage3Result(
            selected_result = top,
            selected_idx    = 0,
            fallback_pool   = pool,
            confidence      = 1.0,       # no LLM confidence available
            reason          = "BM25 rank #1 (LLM rerank disabled)",
            graph_adj_score = top.bm25_score,
            llm_response    = None,
            fallback_used   = False,
            total_time_s    = 0.0,
        )

    def _make_abstract_only_pdf(
        self,
        paper: RetrievalResult,
    ) -> PDFDocument:
        """
        Bypass Stage 4: build a PDFDocument from abstract text only.
        Fake a single "Abstract" section.
        """
        from astrorag.pdf import Section
        text = f"{paper.title}\n\n{paper.abstract}"
        sections = {
            "Abstract": Section(
                name="Abstract",
                text=paper.abstract,
                char_start=0,
                char_end=len(paper.abstract),
            ),
        }
        return PDFDocument(
            arxiv_id      = paper.arxiv_id,
            pdf_path      = None,
            full_text     = text,
            sections      = sections,
            n_pages       = 0,
            n_chars_total = len(text),
            extractor     = "abstract_only",
            fetch_seconds = 0.0,
            parse_seconds = 0.0,
            from_cache    = False,
            success       = True,
            error         = "",
        )

    def _make_abstract_only_summary(
        self,
        decomposition,
        paper: RetrievalResult,
        pdf:   PDFDocument,
    ) -> Stage5Result:
        """
        Bypass Stage 5 entirely: return abstract as answer for each Q.
        """
        answers = {
            qk: SubQuestionAnswer(
                answered    = bool(paper.abstract),
                answer_text = paper.abstract[:500],
                section     = "Abstract",
                equations   = [],
                values      = [],
            )
            for qk in ("Q1", "Q2", "Q3")
        }
        summary = StructuredSummary(
            paper_overview       = paper.abstract[:300] if paper.abstract else "",
            sub_question_answers = answers,
            evidence_type        = "unknown",
            instruments          = [],
            key_equations        = [],
            numerical_results    = [],
            key_findings         = [],
            methodology          = "",
            limitations          = [],
            key_snippet          = paper.abstract[:200],
        )

        # score anyway for comparison
        quality = assess_quality(
            summary    = summary,
            paper_text = pdf.full_text,
            settings   = self.settings,
        )

        return Stage5Result(
            selected_arxiv_id  = paper.arxiv_id,
            summary            = summary,
            quality            = quality,
            pdf_doc            = pdf,
            llm_response       = None,
            n_attempts         = 1,
            n_reselections     = 0,
            n_retries          = 0,
            fallback_pool_used = [],
            total_time_s       = 0.0,
        )

    def _run_stage5_no_gate(
        self,
        decomposition,
        s1,
        s3,
        s4,
    ) -> Stage5Result:
        """
        Run Stage 5 once, accept whatever comes back.
        Bypass the retry / re-selection loop.
        """
        t0 = time.time()

        summary, llm_resp = self.stage5._summarise(
            decomposition  = decomposition,
            paper_title    = s3.selected_result.title,
            paper_abstract = s3.selected_result.abstract,
            pdf_doc        = s4,
            query          = decomposition.original_query,
        )

        # score for reporting, but force ACCEPT
        quality = assess_quality(
            summary    = summary,
            paper_text = s4.full_text,
            settings   = self.settings,
        )
        # override decision to ACCEPT since gate is disabled
        forced_quality = QualityAssessment(
            scores   = quality.scores,
            decision = QualityDecision.ACCEPT,
            reason   = "Quality gate disabled (ablation variant)",
        )

        return Stage5Result(
            selected_arxiv_id  = s3.selected_result.arxiv_id,
            summary            = summary,
            quality            = forced_quality,
            pdf_doc            = s4,
            llm_response       = llm_resp,
            n_attempts         = 1,
            n_reselections     = 0,
            n_retries          = 0,
            fallback_pool_used = [],
            total_time_s       = time.time() - t0,
        )

    # ══════════════════════════════════════════════════
    # trace builders
    # ══════════════════════════════════════════════════

    def _make_stage2_trace(self, s2, s1) -> Stage2Trace:
        top_ppr_idxs = s2.top_ppr_indices(20).tolist()
        return Stage2Trace(
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

    def _make_stage4_trace(self, s4) -> Stage4Trace:
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

    def _config_snapshot(self) -> dict:
        s = self.settings
        return {
            "variant":       self.variant.name,
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
            "groq_model":    s.groq_model,
            "max_reselect":  s.max_reselect_attempts,
            "flags": {
                "use_graph":        self.variant.use_graph,
                "use_llm_rerank":   self.variant.use_llm_rerank,
                "use_pdf":          self.variant.use_pdf,
                "use_quality_gate": self.variant.use_quality_gate,
                "use_full_summary": self.variant.use_full_summary,
            },
        }