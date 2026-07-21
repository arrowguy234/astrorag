"""Step 9 tests — evaluation harness."""

from __future__ import annotations

import pytest


@pytest.mark.step2
class TestQuerySet:
    def test_default_query_set_size(self):
        from astrorag.evaluation import DEFAULT_QUERY_SET
        assert len(DEFAULT_QUERY_SET) >= 15

    def test_all_queries_have_subdomains(self):
        from astrorag.evaluation import DEFAULT_QUERY_SET
        for q in DEFAULT_QUERY_SET:
            assert q.subdomain != ""

    def test_get_query_set_limit(self):
        from astrorag.evaluation import get_query_set
        subset = get_query_set(n=5)
        assert len(subset) == 5

    def test_get_query_set_filter_subdomain(self):
        from astrorag.evaluation import get_query_set
        cosmo = get_query_set(subdomains=["cosmology"])
        assert all(q.subdomain == "cosmology" for q in cosmo)


@pytest.mark.step2
class TestTraceModels:
    def test_stage0_trace_serializable(self):
        from astrorag.evaluation.models import Stage0Trace
        t = Stage0Trace(
            q1="a", q2="b", q3="c",
            wavelength="X-ray", catalogs=["Chandra"],
            query_type="observational", fallback_used=False,
            latency_s=0.5,
        )
        d = t.to_dict()
        assert d["q1"] == "a"
        assert d["wavelength"] == "X-ray"

    def test_query_trace_to_dict_with_none_stages(self):
        from astrorag.evaluation.models import QueryTrace
        t = QueryTrace(query_idx=1, query="test")
        d = t.to_dict()
        assert d["stage0"] is None
        assert d["query"]  == "test"

    def test_evaluation_result_roundtrip(self, tmp_path):
        from astrorag.evaluation.models import (
            EvaluationResult, QueryTrace, Stage0Trace,
        )
        er = EvaluationResult(
            run_id="test_run",
            query_set_name="test",
            n_queries=1, n_completed=1, n_succeeded=1, n_failed=0,
            total_wall_s=1.5,
            started_at="2026-01-01T00:00:00",
            finished_at="2026-01-01T00:00:01",
            traces=[QueryTrace(
                query_idx=1, query="test", subdomain="test",
                stage0=Stage0Trace(
                    q1="a", q2="b", q3="c", wavelength="X-ray",
                    catalogs=[], query_type="general", fallback_used=False,
                    latency_s=0.1,
                ),
            )],
        )
        p = tmp_path / "result.json"
        er.save(p)
        loaded = EvaluationResult.load(p)
        assert loaded.n_queries    == 1
        assert loaded.traces[0].query == "test"
        assert loaded.traces[0].stage0.q1 == "a"


@pytest.mark.step2
class TestMetricsComputation:
    def _make_result(self, n=3, all_accept=True):
        from astrorag.evaluation.models import (
            EvaluationResult, QueryTrace, Stage5Trace,
        )
        traces = []
        for i in range(n):
            decision = "ACCEPT" if all_accept else ("RETRY" if i == 0 else "ACCEPT")
            traces.append(QueryTrace(
                query_idx=i, query=f"q{i}", subdomain="test",
                success=True, total_seconds=1.0,
                stage5=Stage5Trace(
                    final_arxiv_id=f"p{i}", accepted=True,
                    decision=decision,
                    q_f=0.9, q_c=1.0, q_i=0.85, q_total=0.92,
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
            run_id="test", query_set_name="test",
            n_queries=n, n_completed=n, n_succeeded=n, n_failed=0,
            total_wall_s=n*1.0,
            started_at="", finished_at="",
            traces=traces,
        )

    def test_accept_rate_all_accept(self):
        from astrorag.evaluation import compute_metrics
        r = self._make_result(n=5, all_accept=True)
        m = compute_metrics(r)
        assert m.accept_rate == 1.0

    def test_mean_q_total(self):
        from astrorag.evaluation import compute_metrics
        r = self._make_result(n=5, all_accept=True)
        m = compute_metrics(r)
        assert m.mean_q_total == pytest.approx(0.92)

    def test_frac_with_equations(self):
        from astrorag.evaluation import compute_metrics
        r = self._make_result(n=3, all_accept=True)
        m = compute_metrics(r)
        assert m.frac_with_equations == 1.0

    def test_format_metrics_table(self):
        from astrorag.evaluation import compute_metrics, format_metrics_table
        r = self._make_result(n=3, all_accept=True)
        m = compute_metrics(r)
        text = format_metrics_table(m)
        assert "Evaluation Metrics" in text
        assert "ACCEPT rate"        in text