"""
BM25 index construction, caching, and loading.

Uses rank_bm25.BM25Okapi with configurable k1 and b parameters.
The index is built once over the corpus and pickled for reuse.
For the full 408,590-paper corpus:
  - Build: 3-5 minutes on CPU
  - Load : 20-40 seconds from pickle
  - Query: ~50 ms
"""

from __future__ import annotations

import hashlib
import pickle
import time
from   dataclasses import dataclass, field
from   pathlib     import Path

import numpy as np
from   rank_bm25 import BM25Okapi
from   tqdm.auto import tqdm

from astrorag.config              import Settings, get_settings
from astrorag.data                import CorpusData
from astrorag.logger              import get_logger
from astrorag.paths               import get_paths
from astrorag.retrieval.tokenizer import tokenize

logger = get_logger(__name__)

# increment when internal cache structure changes
INDEX_SCHEMA_VERSION = 1


# ══════════════════════════════════════════════════════════
# index container
# ══════════════════════════════════════════════════════════

@dataclass
class BM25Index:
    """
    Wrapped BM25Okapi index with metadata.

    The bm25 object holds the actual index; arxiv_ids is the
    ordered list of paper IDs matching the bm25 internal order.
    """
    bm25:        BM25Okapi
    arxiv_ids:   list[str]
    paper_idxs:  list[int]
    k1:          float
    b:           float
    n_docs:      int
    schema:      int   = INDEX_SCHEMA_VERSION
    build_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        if len(self.arxiv_ids) != self.n_docs:
            raise ValueError(
                f"arxiv_ids length ({len(self.arxiv_ids)}) "
                f"!= n_docs ({self.n_docs})"
            )


# ══════════════════════════════════════════════════════════
# cache management
# ══════════════════════════════════════════════════════════

def _cache_key(
    n_docs: int,
    k1:     float,
    b:      float,
    source_hash: str,
) -> str:
    """Deterministic key for a BM25 cache file."""
    raw = f"n{n_docs}_k1_{k1}_b_{b}_s{INDEX_SCHEMA_VERSION}_{source_hash[:12]}"
    return raw


def _cache_path(key: str) -> Path:
    paths = get_paths()
    d     = paths.data_dir / "bm25_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"bm25_{key}.pkl"


def _source_hash(corpus: CorpusData) -> str:
    """Hash of the corpus so cache invalidates when data changes."""
    parts = [
        str(len(corpus.papers)),
        str(corpus.stats.n_papers_with_concepts),
        str(corpus.stats.total_paper_concept_edges),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


# ══════════════════════════════════════════════════════════
# index construction
# ══════════════════════════════════════════════════════════

def build_bm25_index(
    corpus:       CorpusData,
    settings:     Settings | None = None,
    show_progress: bool = True,
    save_cache:   bool = True,
    force_rebuild: bool = False,
) -> BM25Index:
    """
    Build the BM25 index over the full corpus.

    Args:
        corpus:        Loaded CorpusData object.
        settings:      Configuration (uses defaults if None).
        show_progress: Show tqdm progress bar during tokenization.
        save_cache:    If True, save the index to disk after building.
        force_rebuild: If True, ignore any existing cache.

    Returns:
        BM25Index ready for queries.
    """
    settings = settings or get_settings()
    n_docs   = len(corpus.papers)
    if n_docs == 0:
        raise ValueError("Corpus contains no papers")

    src_hash   = _source_hash(corpus)
    cache_key  = _cache_key(
        n_docs      = n_docs,
        k1          = settings.bm25_k1,
        b           = settings.bm25_b,
        source_hash = src_hash,
    )
    cache_path = _cache_path(cache_key)

    # ── try loading cached index ────────────────────────
    if not force_rebuild and cache_path.exists():
        try:
            t0 = time.time()
            with open(cache_path, "rb") as fh:
                index = pickle.load(fh)
            elapsed = time.time() - t0
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Loaded BM25 index from cache in {elapsed:.1f}s "
                f"({size_mb:.0f} MB, {index.n_docs:,} docs)"
            )
            return index
        except Exception as e:
            logger.warning(f"Cache load failed, rebuilding: {e}")

    # ── build from scratch ──────────────────────────────
    logger.info(
        f"Building BM25 index — {n_docs:,} docs, "
        f"k1={settings.bm25_k1}, b={settings.bm25_b}"
    )
    t0 = time.time()

    # ── tokenise all documents ──────────────────────────
    tokenized_corpus: list[list[str]] = []
    arxiv_ids:  list[str] = []
    paper_idxs: list[int] = []

    iterator = tqdm(
        corpus.papers,
        desc    = "Tokenising",
        unit    = "docs",
        disable = not show_progress,
    )
    for paper in iterator:
        aid = str(paper.get("arxiv_id", "")).strip()

        # get concept labels for this paper (up to 15)
        concepts = corpus.paper_to_concepts.get(aid, [])[:15]

        # build searchable text: title + abstract + concept labels
        parts = [
            paper.get("title", ""),
            paper.get("abstract", ""),
        ]
        parts.extend(concepts)
        combined = " ".join(str(p) for p in parts if p)

        tokens = tokenize(combined)
        tokenized_corpus.append(tokens)
        arxiv_ids.append(aid)
        paper_idxs.append(int(paper.get("paper_idx", len(paper_idxs))))

    logger.info(f"  Tokenized {len(tokenized_corpus):,} documents")

    # ── build BM25 ──────────────────────────────────────
    logger.info(f"  Building BM25Okapi (may take 2-3 minutes for full corpus)...")
    t_bm25 = time.time()
    bm25 = BM25Okapi(
        tokenized_corpus,
        k1 = settings.bm25_k1,
        b  = settings.bm25_b,
    )
    bm25_build = time.time() - t_bm25
    logger.info(f"  BM25 built in {bm25_build:.1f}s")

    elapsed = time.time() - t0

    index = BM25Index(
        bm25        = bm25,
        arxiv_ids   = arxiv_ids,
        paper_idxs  = paper_idxs,
        k1          = settings.bm25_k1,
        b           = settings.bm25_b,
        n_docs      = n_docs,
        build_time_seconds = elapsed,
    )

    logger.info(
        f"BM25 index built: {n_docs:,} docs in {elapsed:.1f}s"
    )

    # ── save cache ──────────────────────────────────────
    if save_cache:
        try:
            t_save = time.time()
            with open(cache_path, "wb") as fh:
                pickle.dump(index, fh, protocol=4)
            elapsed_save = time.time() - t_save
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Cached BM25 index in {elapsed_save:.1f}s "
                f"({size_mb:.0f} MB) → {cache_path.name}"
            )
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    return index


# ══════════════════════════════════════════════════════════
# convenience loader
# ══════════════════════════════════════════════════════════

def load_bm25_index(
    corpus:   CorpusData,
    settings: Settings | None = None,
) -> BM25Index:
    """
    Load or build the BM25 index for a corpus.

    Convenience wrapper — uses cache if available, else builds.
    """
    return build_bm25_index(
        corpus        = corpus,
        settings      = settings,
        show_progress = False,
        save_cache    = True,
        force_rebuild = False,
    )