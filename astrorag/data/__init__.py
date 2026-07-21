"""
AstroRAG data loading subpackage.

Provides efficient loading and access to the 408,590-paper corpus:
- abstracts (JSONL, gzipped)
- concept embeddings (NPZ)
- paper-concept mappings (CSV, gzipped)
- concept vocabulary (CSV, gzipped)
- citation network (JSONL, gzipped)
- paper index mappings (CSV, gzipped)
- publication years (NPY)

All loaded data is exposed through the CorpusData object with typed,
validated accessors used by every downstream pipeline stage.
"""

from astrorag.data.cache     import CacheManager, get_cache_manager
from astrorag.data.loader    import CorpusData, DataLoader, load_corpus
from astrorag.data.models    import (
    Paper,
    Concept,
    CitationRecord,
    CorpusStats,
    LoadConfig,
)
from astrorag.data.streaming import (
    iter_abstracts,
    iter_citations,
    count_lines_gz,
)

__all__ = [
    # main entry
    "load_corpus",
    "DataLoader",
    "CorpusData",
    # models
    "Paper",
    "Concept",
    "CitationRecord",
    "CorpusStats",
    "LoadConfig",
    # streaming
    "iter_abstracts",
    "iter_citations",
    "count_lines_gz",
    # cache
    "CacheManager",
    "get_cache_manager",
]
