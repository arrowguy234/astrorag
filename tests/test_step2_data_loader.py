"""
Step 2 tests — verify data loader produces correct output.

Uses a small sample_size for speed. Full-corpus load tested
separately by benchmark script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════
# models tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestModels:
    def test_paper_model_accepts_valid_record(self):
        from astrorag.data.models import Paper
        p = Paper(paper_idx=1, arxiv_id="0709.2152",
                  title="X-ray cavities", abstract="Some abstract")
        assert p.arxiv_id == "0709.2152"
        assert p.title    == "X-ray cavities"

    def test_paper_arxiv_id_strips_whitespace(self):
        from astrorag.data.models import Paper
        p = Paper(paper_idx=1, arxiv_id="  0709.2152  ")
        assert p.arxiv_id == "0709.2152"

    def test_paper_searchable_text_with_concepts(self):
        from astrorag.data.models import Paper
        p = Paper(paper_idx=1, arxiv_id="test",
                  title="AGN feedback", abstract="Jets suppress SF")
        text = p.searchable_text(concept_labels=["AGN Feedback"])
        assert "AGN feedback"  in text
        assert "Jets suppress" in text
        assert "AGN Feedback"  in text

    def test_corpus_stats_summary(self):
        from astrorag.data.models import CorpusStats
        s = CorpusStats(n_papers=1000, n_concepts=500,
                        concept_emb_dim=100)
        text = s.summary()
        assert "1,000" in text
        assert "500"   in text

    def test_load_config_defaults(self):
        from astrorag.data.models import LoadConfig
        c = LoadConfig()
        assert c.sample_size    == 408_590
        assert c.use_cache      is True
        assert c.force_reload   is False


# ══════════════════════════════════════════════════════════
# streaming tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
@pytest.mark.requires_data
class TestStreaming:
    def test_iter_abstracts_yields_dicts(self):
        from astrorag.config          import get_settings
        from astrorag.data.streaming  import iter_abstracts

        path = get_settings().dataset_files["abstracts"]
        records = list(iter_abstracts(path, limit=10, show_progress=False))
        assert len(records) == 10
        for r in records:
            assert isinstance(r, dict)

    def test_iter_abstracts_respects_limit(self):
        from astrorag.config          import get_settings
        from astrorag.data.streaming  import iter_abstracts

        path = get_settings().dataset_files["abstracts"]
        records = list(iter_abstracts(path, limit=5, show_progress=False))
        assert len(records) == 5

    def test_iter_citations_filters_by_keep_ids(self):
        from astrorag.config          import get_settings
        from astrorag.data.streaming  import iter_citations

        path = get_settings().dataset_files["citations"]
        keep = {"0", "1", "2"}
        records = list(iter_citations(path, keep_ids=keep,
                                       show_progress=False))
        for r in records[:100]:
            aid = str(r.get("paper_idx", r.get("id", "")))
            assert aid in keep


# ══════════════════════════════════════════════════════════
# cache tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestCache:
    def test_cache_manager_creates_directory(self, tmp_path):
        from astrorag.data.cache import CacheManager
        cache_dir = tmp_path / "test_cache"
        cm = CacheManager(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_source_hash_stable_across_calls(self, tmp_path):
        from astrorag.data.cache import CacheManager

        f1 = tmp_path / "file1.txt"
        f1.write_text("hello")
        f2 = tmp_path / "file2.txt"
        f2.write_text("world")

        cm = CacheManager(cache_dir=tmp_path / "cache")
        h1 = cm.compute_source_hash({"a": f1, "b": f2})
        h2 = cm.compute_source_hash({"a": f1, "b": f2})
        assert h1 == h2

    def test_source_hash_changes_on_file_modification(self, tmp_path):
        import time
        from astrorag.data.cache import CacheManager

        f = tmp_path / "file.txt"
        f.write_text("original")
        cm  = CacheManager(cache_dir=tmp_path / "cache")
        h1  = cm.compute_source_hash({"f": f})
        time.sleep(0.05)
        f.write_text("modified")
        h2  = cm.compute_source_hash({"f": f})
        assert h1 != h2

    def test_cache_save_and_load_roundtrip(self, tmp_path):
        from astrorag.data.cache import CacheManager, CacheKey

        cm  = CacheManager(cache_dir=tmp_path / "cache")
        key = CacheKey(sample_size=100, source_hash="abc123")
        payload = {"data": [1, 2, 3], "name": "test"}
        cm.save(key, payload)
        assert cm.exists(key)

        loaded = cm.load(key)
        assert loaded == payload

    def test_cache_load_returns_none_when_missing(self, tmp_path):
        from astrorag.data.cache import CacheManager, CacheKey
        cm  = CacheManager(cache_dir=tmp_path / "cache")
        key = CacheKey(sample_size=999, source_hash="nonexistent")
        assert cm.load(key) is None

    def test_cache_clear_removes_files(self, tmp_path):
        from astrorag.data.cache import CacheManager, CacheKey
        cm = CacheManager(cache_dir=tmp_path / "cache")
        for i in range(3):
            key = CacheKey(sample_size=i, source_hash=f"h{i}")
            cm.save(key, {"i": i})
        assert len(cm.list_cached()) == 3
        n = cm.clear()
        assert n == 3
        assert len(cm.list_cached()) == 0


# ══════════════════════════════════════════════════════════
# loader tests — full pipeline
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
@pytest.mark.requires_data
class TestDataLoader:
    """Small-sample integration tests. Use N=500 for speed."""

    N = 500

    @pytest.fixture(scope="class")
    def data(self):
        from astrorag.data import DataLoader, LoadConfig
        loader = DataLoader(config=LoadConfig(
            sample_size=self.N, use_cache=False, show_progress=False
        ))
        return loader.load()

    def test_papers_loaded(self, data):
        assert data.n_papers() == self.N
        assert isinstance(data.papers, list)
        assert isinstance(data.papers[0], dict)

    def test_concept_embeddings_loaded(self, data):
        assert isinstance(data.concept_emb, np.ndarray)
        assert data.concept_emb.dtype == np.float32
        assert data.concept_emb.ndim  == 2
        assert data.concept_emb.shape[1] > 0

    def test_vocabulary_loaded(self, data):
        assert isinstance(data.concepts_df, pd.DataFrame)
        assert len(data.concepts_df) > 0
        assert data.lbl_col in data.concepts_df.columns

    def test_pc_mapping_loaded(self, data):
        assert isinstance(data.pc_mapping, pd.DataFrame)
        assert data.pid_col in data.pc_mapping.columns
        assert data.cid_col in data.pc_mapping.columns

    def test_concept_lookups_populated(self, data):
        assert isinstance(data.arxiv_to_cidx, dict)
        assert isinstance(data.paper_to_concepts, dict)
        # some papers must have concepts assigned
        assert len(data.arxiv_to_cidx) > 0

    def test_citation_lookups_populated(self, data):
        assert isinstance(data.paper_refs, dict)
        assert isinstance(data.paper_cited_by, dict)
        for v in list(data.paper_refs.values())[:10]:
            assert isinstance(v, set)

    def test_get_paper_vector_returns_correct_shape(self, data):
        # find a paper that has concepts
        aids_with_concepts = list(data.arxiv_to_cidx.keys())
        if not aids_with_concepts:
            pytest.skip("no papers with concepts in sample")
        aid = aids_with_concepts[0]
        vec = data.get_paper_vector(aid)
        assert vec.shape  == (data.concept_emb.shape[1],)
        assert vec.dtype  == np.float32
        assert np.any(vec != 0)

    def test_get_paper_vector_zero_for_missing(self, data):
        vec = data.get_paper_vector("nonexistent_paper_id")
        assert vec.shape  == (data.concept_emb.shape[1],)
        assert np.all(vec == 0)

    def test_stats_populated(self, data):
        stats = data.stats
        assert stats.n_papers        == self.N
        assert stats.n_concepts      > 0
        assert stats.concept_emb_dim > 0
        assert stats.load_time_seconds > 0

    def test_load_corpus_convenience_function(self):
        from astrorag.data import load_corpus
        data = load_corpus(sample_size=100, use_cache=False)
        assert data.n_papers() == 100


# ══════════════════════════════════════════════════════════
# cache integration tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
@pytest.mark.requires_data
class TestLoaderCache:
    def test_second_load_uses_cache_and_is_faster(self):
        import time
        from astrorag.data      import DataLoader
        from astrorag.data.models import LoadConfig

        # first load builds cache
        cfg1 = LoadConfig(sample_size=500, use_cache=True,
                          force_reload=True, show_progress=False)
        t0   = time.time()
        d1   = DataLoader(config=cfg1).load()
        elapsed_1 = time.time() - t0

        # second load hits cache
        cfg2 = LoadConfig(sample_size=500, use_cache=True,
                          force_reload=False, show_progress=False)
        t0   = time.time()
        d2   = DataLoader(config=cfg2).load()
        elapsed_2 = time.time() - t0

        assert d1.n_papers() == d2.n_papers()
        # cache should be at least 3x faster
        assert elapsed_2 < elapsed_1 / 2

    def test_force_reload_bypasses_cache(self):
        from astrorag.data      import DataLoader
        from astrorag.data.models import LoadConfig

        loader = DataLoader(config=LoadConfig(
            sample_size=100, use_cache=True,
            force_reload=True, show_progress=False,
        ))
        data = loader.load()
        assert data.n_papers() == 100