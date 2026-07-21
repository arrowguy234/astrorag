"""Step 10 tests — ablation runner."""

from __future__ import annotations

import pytest


@pytest.mark.step2
class TestVariantRegistry:
    def test_all_variants_have_names(self):
        from astrorag.evaluation import ABLATION_VARIANTS
        for v in ABLATION_VARIANTS:
            assert v.name
            assert v.description

    def test_full_is_baseline(self):
        from astrorag.evaluation import get_variant
        v = get_variant("full")
        assert v.is_baseline is True
        assert v.use_graph
        assert v.use_llm_rerank
        assert v.use_pdf
        assert v.use_quality_gate

    def test_bm25_only_disables_all(self):
        from astrorag.evaluation import get_variant
        v = get_variant("bm25_only")
        assert v.use_graph          is False
        assert v.use_llm_rerank     is False
        assert v.use_pdf            is False
        assert v.use_quality_gate   is False
        assert v.use_full_summary   is False

    def test_no_graph_only_disables_graph(self):
        from astrorag.evaluation import get_variant
        v = get_variant("no_graph")
        assert v.use_graph          is False
        assert v.use_llm_rerank     is True
        assert v.use_pdf            is True

    def test_unknown_variant_raises(self):
        from astrorag.evaluation import get_variant
        with pytest.raises(ValueError):
            get_variant("nonexistent")

    def test_get_all_variant_names(self):
        from astrorag.evaluation import get_all_variant_names
        names = get_all_variant_names()
        assert "full" in names
        assert "no_graph" in names
        assert "bm25_only" in names
        assert len(names) == 6


@pytest.mark.step2
class TestComparisonMetrics:
    def _make_result(self, name="test", q_totals=None):
        from astrorag.evaluation.models import (
            EvaluationResult, QueryTrace, Stage5Trace,
        )
        q_totals = q_totals or [1.0, 1.0, 1.0]
        traces = []
        for i, q in enumerate(q_totals):
            traces.append(QueryTrace(
                query_idx=i, query=f"q{i}", subdomain="test",
                success=True, total_seconds=1.0,
                stage5=Stage5Trace(
                    final_arxiv_id=f"p{i}", accepted=True,
                    decision="ACCEPT" if q >= 0.75 else "RE-SELECT",
                    q_f=q, q_c=1.0, q_i=q, q_total=q,
                    n_claims_verified=5, n_claims_total=5,
                    snippet_overlap=0.8,
                    has_equations=True, has_numerical_results=True,
                    n_equations=2, n_numerical_results=3, n_instruments=1,
                    evidence_type="observational",
                    n_attempts=1, n_retries=0, n_reselections=0,
                    fallback_used=[], latency_s=5.0,
                ),
            ))
        return EvaluationResult(
            run_id="test", query_set_name=f"ablation_{name}",
            n_queries=len(traces), n_completed=len(traces),
            n_succeeded=len(traces), n_failed=0,
            total_wall_s=len(traces)*1.0,
            started_at="", finished_at="",
            traces=traces,
        )

    def test_baseline_delta_is_zero(self):
        from astrorag.evaluation import compute_variant_comparison
        baseline = self._make_result("full", [1.0, 1.0, 1.0])
        cmp = compute_variant_comparison(baseline, baseline)
        assert cmp.delta_q_total == pytest.approx(0.0)
        assert cmp.retrieval_overlap == pytest.approx(1.0)

    def test_degraded_variant_negative_delta(self):
        from astrorag.evaluation import compute_variant_comparison
        baseline = self._make_result("full", [1.0, 1.0, 1.0])
        variant  = self._make_result("no_graph", [0.8, 0.7, 0.9])
        cmp = compute_variant_comparison(variant, baseline)
        assert cmp.delta_q_total < 0

    def test_format_ablation_table_contains_headers(self):
        from astrorag.evaluation import (
            compute_variant_comparison, format_ablation_table,
        )
        baseline = self._make_result("full",     [1.0, 1.0, 1.0])
        variant  = self._make_result("no_graph", [0.8, 0.7, 0.9])
        cmps = [
            compute_variant_comparison(baseline, baseline),
            compute_variant_comparison(variant,  baseline),
        ]
        text = format_ablation_table(cmps)
        assert "Ablation Study" in text
        assert "Q_total"        in text
        assert "no_graph"       in text