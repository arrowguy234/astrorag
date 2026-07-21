"""
Step 6 tests — Stage 3 graph-primed LLM reranking.
"""

from __future__ import annotations

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════
# prompt construction tests (no LLM needed)
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestPromptConstruction:
    def test_format_annotated_abstracts(self):
        from astrorag.retrieval             import RetrievalResult
        from astrorag.stages.stage3_rerank  import _format_annotated_abstracts

        results = [
            RetrievalResult(
                arxiv_id="1234", paper_idx=0,
                title="AGN feedback",
                abstract="This paper describes AGN feedback mechanisms",
                cluster=1,
            ),
            RetrievalResult(
                arxiv_id="5678", paper_idx=1,
                title="Star formation",
                abstract="This describes star formation in dwarf galaxies",
                cluster=0,
            ),
        ]
        ppr = np.array([0.95, 0.42])
        text = _format_annotated_abstracts(results, ppr)
        assert "PAPER #0" in text
        assert "PAPER #1" in text
        assert "PPR=0.95" in text
        assert "cluster=1" in text
        assert "AGN feedback" in text

    def test_build_user_prompt_structure(self):
        from astrorag.llm.models              import QueryDecomposition
        from astrorag.stages.stage3_rerank    import _build_user_prompt

        d = QueryDecomposition(
            original_query = "test query",
            sub_questions  = {"Q1": "mech?", "Q2": "evid?", "Q3": "quant?"},
            wavelength     = "X-ray",
        )
        prompt = _build_user_prompt(
            query           = "test query",
            decomposition   = d,
            cluster_summary = "CLUSTER SUMMARY:\nCluster 0 stuff",
            annotated       = "[PAPER #0]\n...",
            n_candidates    = 10,
        )
        # cluster summary must come first
        assert prompt.index("CLUSTER SUMMARY") < prompt.index("RESEARCH QUERY")
        assert prompt.index("SUB-QUESTIONS")   < prompt.index("CANDIDATE PAPERS")
        assert "0-9" in prompt   # index range mentioned


# ══════════════════════════════════════════════════════════
# fallback path (no LLM)
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestFallbackPath:
    def test_fallback_selects_highest_ppr(self):
        from astrorag.retrieval  import RetrievalResult
        from astrorag.stages     import Stage3Rerank

        results = [
            RetrievalResult(arxiv_id=f"p{i}", paper_idx=i)
            for i in range(5)
        ]
        ppr = np.array([0.1, 0.9, 0.3, 0.5, 0.2])

        stage3 = Stage3Rerank()
        result = stage3._fallback_by_ppr(
            results=results, ppr_scores=ppr, t_start=0.0,
        )
        assert result.selected_idx  == 1
        assert result.fallback_used is True
        assert len(result.fallback_pool) == 4

    def test_fallback_confidence_is_ppr(self):
        from astrorag.retrieval  import RetrievalResult
        from astrorag.stages     import Stage3Rerank

        results = [RetrievalResult(arxiv_id="x", paper_idx=0)]
        ppr = np.array([0.75])
        stage3 = Stage3Rerank()
        result = stage3._fallback_by_ppr(results, ppr, 0.0)
        assert result.confidence == pytest.approx(0.75)


# ══════════════════════════════════════════════════════════
# integration tests over small corpus
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
@pytest.mark.requires_data
class TestStage3Integration:
    N = 2000

    @pytest.fixture(scope="class")
    def pipeline(cls):
        from astrorag.data      import DataLoader
        from astrorag.data.models import LoadConfig
        from astrorag.stages    import (
            Stage0Decompose, Stage1BM25, Stage2Graph, Stage3Rerank,
        )
        corpus = DataLoader(config=LoadConfig(
            sample_size=cls.N, use_cache=True, show_progress=False,
        )).load()
        return {
            "corpus": corpus,
            "stage0": Stage0Decompose(),
            "stage1": Stage1BM25(corpus=corpus),
            "stage2": Stage2Graph(corpus=corpus),
            "stage3": Stage3Rerank(),
        }

    def _run_end_to_end(self, pipeline, query: str, top_k: int = 30,
                        use_llm: bool = True):
        s0 = pipeline["stage0"].run(query, rule_based_only=not use_llm)
        s1 = pipeline["stage1"].run(query, top_k=top_k)
        s2 = pipeline["stage2"].run(s1)
        s3 = pipeline["stage3"].run(
            retrieval     = s1,
            graph_context = s2,
            decomposition = s0.decomposition,
            use_llm       = use_llm,
        )
        return s0, s1, s2, s3

    def test_pipeline_fallback_mode(self, pipeline):
        _, _, _, s3 = self._run_end_to_end(
            pipeline, "AGN feedback X-ray cavities", use_llm=False,
        )
        assert s3.selected_result is not None
        assert s3.fallback_used   is True

    def test_pipeline_produces_valid_selection(self, pipeline):
        _, s1, _, s3 = self._run_end_to_end(
            pipeline, "black hole mass galaxy scaling", use_llm=False,
        )
        assert 0 <= s3.selected_idx < len(s1.results)
        assert s3.selected_result.arxiv_id in [r.arxiv_id for r in s1.results]

    def test_fallback_pool_excludes_selected(self, pipeline):
        _, _, _, s3 = self._run_end_to_end(
            pipeline, "cosmological simulations", use_llm=False,
        )
        assert s3.selected_idx not in s3.fallback_pool

    def test_graph_adj_score_in_range(self, pipeline):
        _, _, _, s3 = self._run_end_to_end(
            pipeline, "gravitational lensing", use_llm=False,
        )
        assert 0.0 <= s3.graph_adj_score <= 1.0


@pytest.mark.step2
@pytest.mark.requires_api
@pytest.mark.requires_data
class TestStage3WithLLM:
    """LLM tests — small sample to keep API cost low."""
    N = 2000

    def test_llm_selects_valid_index(self):
        from astrorag.data      import DataLoader
        from astrorag.data.models import LoadConfig
        from astrorag.stages    import (
            Stage0Decompose, Stage1BM25, Stage2Graph, Stage3Rerank,
        )
        corpus = DataLoader(config=LoadConfig(
            sample_size=self.N, use_cache=True, show_progress=False,
        )).load()

        s0 = Stage0Decompose().run("AGN feedback in galaxy clusters")
        s1 = Stage1BM25(corpus=corpus).run(s0.decomposition.original_query, top_k=30)
        s2 = Stage2Graph(corpus=corpus).run(s1)
        s3 = Stage3Rerank().run(
            retrieval=s1, graph_context=s2, decomposition=s0.decomposition,
            use_llm=True,
        )

        assert 0 <= s3.selected_idx < 30
        assert s3.confidence > 0.0
        assert len(s3.reason) > 0