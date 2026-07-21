"""
Step 5 tests — Stage 2 graph construction and PPR.
"""

from __future__ import annotations

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════
# model tests (no data needed)
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestGraphModels:
    def test_signal_matrices_density(self):
        from astrorag.graph.models import SignalMatrices
        W = np.zeros((5, 5), dtype=np.float32)
        W[0, 1] = W[1, 0] = 0.5
        W[2, 3] = W[3, 2] = 0.7
        sm = SignalMatrices(
            S1_concept    = W,
            S2_biblio     = W,
            S3_cocitation = W,
            S4_domain     = W,
            W_combined    = W,
            n_edges_after = 2,
        )
        assert sm.n_nodes == 5
        assert sm.density == pytest.approx(2 / 10)


# ══════════════════════════════════════════════════════════
# PPR unit tests with synthetic graph
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestPPR:
    def test_ppr_returns_normalised_array(self):
        from astrorag.graph.ppr import personalized_pagerank
        W = np.array([
            [0.0, 0.5, 0.3, 0.0],
            [0.5, 0.0, 0.4, 0.2],
            [0.3, 0.4, 0.0, 0.6],
            [0.0, 0.2, 0.6, 0.0],
        ], dtype=np.float32)
        bm25 = np.array([10.0, 8.0, 6.0, 4.0])
        norm, raw, iters = personalized_pagerank(W, bm25)
        assert norm.shape  == (4,)
        assert norm.min() == pytest.approx(0.0)
        assert norm.max() == pytest.approx(1.0)
        assert iters > 0

    def test_ppr_favors_high_bm25_when_isolated(self):
        # disconnected graph — PPR should just return personalisation
        from astrorag.graph.ppr import personalized_pagerank
        W = np.zeros((4, 4), dtype=np.float32)
        bm25 = np.array([10.0, 1.0, 1.0, 1.0])
        norm, raw, iters = personalized_pagerank(W, bm25)
        # highest BM25 paper should have highest PPR
        assert np.argmax(norm) == 0


# ══════════════════════════════════════════════════════════
# integration tests over small corpus
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
@pytest.mark.requires_data
class TestStage2Small:
    N = 2000

    @pytest.fixture(scope="class")
    def corpus(cls):
        from astrorag.data import DataLoader, LoadConfig
        loader = DataLoader(config=LoadConfig(
            sample_size=cls.N, use_cache=True, show_progress=False,
        ))
        return loader.load()

    @pytest.fixture(scope="class")
    def stage1(cls, corpus):
        from astrorag.stages import Stage1BM25
        return Stage1BM25(corpus=corpus)

    @pytest.fixture(scope="class")
    def stage2(cls, corpus):
        from astrorag.stages import Stage2Graph
        return Stage2Graph(corpus=corpus)

    def test_stage2_runs_end_to_end(self, stage1, stage2):
        run     = stage1.run("black hole mass galaxy", top_k=30)
        context = stage2.run(run)
        assert context.n_nodes == 30
        assert context.ppr_scores.shape == (30,)

    def test_ppr_scores_in_range(self, stage1, stage2):
        run     = stage1.run("dark matter halos", top_k=30)
        context = stage2.run(run)
        assert 0.0 <= context.ppr_scores.min()
        assert context.ppr_scores.max() <= 1.0

    def test_cluster_labels_assigned(self, stage1, stage2):
        run     = stage1.run("cosmology cluster counts", top_k=30)
        context = stage2.run(run)
        labels = context.cluster_summary.cluster_labels
        assert len(labels) == 30
        assert labels.min() >= 0

    def test_results_get_ppr_and_cluster(self, stage1, stage2):
        run     = stage1.run("galaxy formation", top_k=30)
        context = stage2.run(run)
        for r in run.results:
            assert 0.0 <= r.ppr_score <= 1.0
            assert r.cluster >= 0

    def test_cluster_summary_prompt_populated(self, stage1, stage2):
        run     = stage1.run("gravitational waves neutron stars", top_k=30)
        context = stage2.run(run)
        text = context.cluster_summary.prompt_text
        assert "CONTEXT GRAPH CLUSTER SUMMARY" in text
        assert "Cluster 0"                     in text

    def test_signal_matrices_symmetric(self, stage1, stage2):
        run     = stage1.run("supernova remnants", top_k=30)
        context = stage2.run(run)
        s = context.signals
        for M in (s.S1_concept, s.S2_biblio, s.S3_cocitation,
                  s.S4_domain, s.W_combined):
            assert np.allclose(M, M.T, atol=1e-5)
            assert np.all(np.diag(M) == 0)

    def test_top_ppr_indices(self, stage1, stage2):
        run     = stage1.run("AGN cavities cluster", top_k=30)
        context = stage2.run(run)
        top5 = context.top_ppr_indices(5)
        assert len(top5) == 5
        # PPR scores should be non-increasing
        vals = context.ppr_scores[top5]
        assert list(vals) == sorted(vals, reverse=True)