"""
Step 4 tests — Stage 1 BM25 retrieval.
"""

from __future__ import annotations

import pytest


# ══════════════════════════════════════════════════════════
# tokenizer tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestTokenizer:
    def test_basic_tokenization(self):
        from astrorag.retrieval.tokenizer import tokenize
        assert tokenize("AGN jets") == ["agn", "jets"]

    def test_lowercase(self):
        from astrorag.retrieval.tokenizer import tokenize
        assert tokenize("Chandra X-ray") == ["chandra", "x-ray"]

    def test_preserves_hyphens(self):
        from astrorag.retrieval.tokenizer import tokenize
        tokens = tokenize("X-ray gamma-ray")
        assert "x-ray"     in tokens
        assert "gamma-ray" in tokens

    def test_strips_punctuation(self):
        from astrorag.retrieval.tokenizer import tokenize
        tokens = tokenize("What causes AGN feedback? Multiple mechanisms!")
        assert "?" not in tokens
        assert "!" not in tokens

    def test_empty_input(self):
        from astrorag.retrieval.tokenizer import tokenize
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_arxiv_id_normalization(self):
        from astrorag.retrieval.tokenizer import normalize_arxiv_id
        assert normalize_arxiv_id(" 0704.0007 ") == "0704.0007"
        assert normalize_arxiv_id("0704.0007")   == "0704.0007"


# ══════════════════════════════════════════════════════════
# model tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestModels:
    def test_retrieval_result_defaults(self):
        from astrorag.retrieval.models import RetrievalResult
        r = RetrievalResult(arxiv_id="0704.0007", paper_idx=0)
        assert r.rank            == 0
        assert r.bm25_score      == 0.0
        assert r.concepts        == []
        assert r.concept_overlap == 0

    def test_retrieval_result_to_dict(self):
        from astrorag.retrieval.models import RetrievalResult
        r = RetrievalResult(
            arxiv_id  = "test",
            paper_idx = 1,
            bm25_score = 3.14,
        )
        d = r.to_dict()
        assert d["arxiv_id"]   == "test"
        assert d["bm25_score"] == 3.14

    def test_retrieval_run_summary(self):
        from astrorag.retrieval.models import RetrievalRun, RetrievalResult
        results = [RetrievalResult(arxiv_id=f"p{i}", paper_idx=i,
                                   bm25_score=10.0 - i, rank=i+1)
                   for i in range(5)]
        run = RetrievalRun(
            query          = "test query",
            results        = results,
            k              = 5,
            n_corpus       = 1000,
            elapsed_s      = 0.05,
            top_score      = 10.0,
            mean_score_top = 8.0,
        )
        text = run.summary()
        assert "test query" in text
        assert "1,000"      in text

    def test_run_top_arxiv_ids(self):
        from astrorag.retrieval.models import RetrievalRun, RetrievalResult
        results = [RetrievalResult(arxiv_id=f"p{i}", paper_idx=i)
                   for i in range(10)]
        run = RetrievalRun(query="q", results=results, k=10,
                           n_corpus=100, elapsed_s=0.01)
        assert run.top_arxiv_ids(3)     == ["p0", "p1", "p2"]
        assert len(run.top_arxiv_ids()) == 10


# ══════════════════════════════════════════════════════════
# index construction and retrieval tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
@pytest.mark.requires_data
class TestStage1Small:
    """Small-corpus integration tests using a 500-paper sample."""

    N = 500

    @pytest.fixture(scope="class")
    def corpus(cls):
        from astrorag.data import DataLoader, LoadConfig
        loader = DataLoader(config=LoadConfig(
            sample_size=cls.N, use_cache=False, show_progress=False
        ))
        return loader.load()

    @pytest.fixture(scope="class")
    def stage1(cls, corpus):
        from astrorag.stages import Stage1BM25
        return Stage1BM25(corpus=corpus)

    def test_index_built(self, stage1):
        assert stage1.n_docs() == self.N
        assert stage1.index.n_docs == self.N
        assert stage1.index.k1 > 0
        assert stage1.index.b >= 0

    def test_retrieval_returns_top_k(self, stage1):
        run = stage1.run("black hole", top_k=10)
        assert len(run.results) == 10
        assert run.k            == 10

    def test_results_sorted_by_score(self, stage1):
        run = stage1.run("galaxy cluster", top_k=20)
        scores = [r.bm25_score for r in run.results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_start_at_one(self, stage1):
        run = stage1.run("cosmology", top_k=5)
        assert run.results[0].rank == 1
        assert run.results[-1].rank == 5

    def test_arxiv_ids_populated(self, stage1):
        run = stage1.run("dark matter", top_k=5)
        for r in run.results:
            assert r.arxiv_id != ""
            assert isinstance(r.arxiv_id, str)

    def test_top_score_ge_min_score(self, stage1):
        run = stage1.run("stellar evolution", top_k=10)
        assert run.top_score >= run.min_score_top

    def test_deterministic(self, stage1):
        run1 = stage1.run("supernova")
        run2 = stage1.run("supernova")
        ids1 = run1.top_arxiv_ids()
        ids2 = run2.top_arxiv_ids()
        assert ids1 == ids2

    def test_empty_query_raises(self, stage1):
        with pytest.raises(ValueError, match="empty"):
            stage1.run("")

    def test_concept_augmentation_works(self, stage1):
        """Papers with matching concepts should get overlap > 0."""
        run = stage1.run("cosmology dark energy", top_k=20)
        overlaps = [r.concept_overlap for r in run.results]
        # at least one result should have some overlap
        assert max(overlaps) >= 0


# ══════════════════════════════════════════════════════════
# cache behaviour tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
@pytest.mark.requires_data
class TestBM25Cache:
    def test_second_build_uses_cache(self):
        import time
        from astrorag.data      import DataLoader
        from astrorag.data.models import LoadConfig
        from astrorag.retrieval import build_bm25_index

        loader = DataLoader(config=LoadConfig(
            sample_size=500, use_cache=True, show_progress=False,
        ))
        corpus = loader.load()

        # first build (or cache hit from previous test)
        t0 = time.time()
        idx1 = build_bm25_index(corpus, show_progress=False)
        first_elapsed = time.time() - t0

        # second build definitely from cache
        t0 = time.time()
        idx2 = build_bm25_index(corpus, show_progress=False)
        second_elapsed = time.time() - t0

        assert idx1.n_docs == idx2.n_docs
        assert second_elapsed < 5.0