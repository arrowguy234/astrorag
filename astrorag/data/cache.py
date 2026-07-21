"""
Pickle-based cache for parsed corpus lookups.

Loading and parsing the raw dataset files takes 5-10 minutes for the
full 408,590-paper corpus. This module caches the parsed lookups so
subsequent runs load in under 30 seconds.

Cache invalidation is based on:
- Configured sample size
- Modification times of source dataset files
- Cache schema version
"""

from __future__ import annotations

import hashlib
import pickle
import time
from   dataclasses import dataclass
from   pathlib     import Path
from   typing      import Any

from astrorag.logger import get_logger
from astrorag.paths  import get_paths

logger = get_logger(__name__)

# increment when internal cache structure changes
CACHE_SCHEMA_VERSION = 1


@dataclass
class CacheKey:
    """
    Cache key derived from source file mtimes and sample size.

    Two loads with the same key are guaranteed to produce identical
    data, so cached lookups are safe to reuse.
    """
    sample_size:  int
    source_hash:  str
    schema:       int = CACHE_SCHEMA_VERSION

    def filename(self) -> str:
        """Compact filename for this cache key."""
        return (
            f"corpus_cache_"
            f"n{self.sample_size}_"
            f"s{self.schema}_"
            f"{self.source_hash[:12]}.pkl"
        )


class CacheManager:
    """
    Manages read/write of parsed corpus lookups to disk.

    Uses pickle protocol 4 for large object support.
    Cache files are stored in the data/ directory.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or (get_paths().data_dir / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache directory: {self.cache_dir}")

    # ── key generation ──────────────────────────────────
    def compute_source_hash(self, source_paths: dict[str, Path]) -> str:
        """
        Hash source file mtimes and sizes to detect changes.

        If any source file changes, the hash changes and the cache is
        invalidated automatically.
        """
        parts = []
        for name in sorted(source_paths):
            p = source_paths[name]
            if p.exists():
                stat = p.stat()
                parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
            else:
                parts.append(f"{name}:missing")

        h = hashlib.sha256("|".join(parts).encode()).hexdigest()
        return h

    def make_key(
        self,
        sample_size:  int,
        source_paths: dict[str, Path],
    ) -> CacheKey:
        return CacheKey(
            sample_size = sample_size,
            source_hash = self.compute_source_hash(source_paths),
        )

    # ── read / write ────────────────────────────────────
    def path_for(self, key: CacheKey) -> Path:
        return self.cache_dir / key.filename()

    def exists(self, key: CacheKey) -> bool:
        return self.path_for(key).exists()

    def load(self, key: CacheKey) -> dict[str, Any] | None:
        """
        Load cached data if present, else return None.

        Returns the full data dict as it was when saved.
        Returns None on any read error, prompting a fresh load.
        """
        cache_path = self.path_for(key)
        if not cache_path.exists():
            return None

        try:
            t0 = time.time()
            with open(cache_path, "rb") as fh:
                data = pickle.load(fh)
            elapsed = time.time() - t0
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Loaded cache in {elapsed:.1f}s "
                f"({size_mb:.1f} MB): {cache_path.name}"
            )
            return data
        except Exception as e:
            logger.warning(f"Cache load failed for {cache_path.name}: {e}")
            return None

    def save(self, key: CacheKey, data: dict[str, Any]) -> None:
        """Save parsed corpus data to disk under the given key."""
        cache_path = self.path_for(key)
        try:
            t0 = time.time()
            with open(cache_path, "wb") as fh:
                pickle.dump(data, fh, protocol=4)
            elapsed = time.time() - t0
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Saved cache in {elapsed:.1f}s "
                f"({size_mb:.1f} MB): {cache_path.name}"
            )
        except Exception as e:
            logger.warning(f"Cache save failed for {cache_path.name}: {e}")

    def clear(self) -> int:
        """Remove all cache files. Returns count of files deleted."""
        n = 0
        for p in self.cache_dir.glob("corpus_cache_*.pkl"):
            p.unlink()
            n += 1
        logger.info(f"Cleared {n} cache file(s) from {self.cache_dir}")
        return n

    def list_cached(self) -> list[Path]:
        """List all cache files with their sizes."""
        return sorted(self.cache_dir.glob("corpus_cache_*.pkl"))


# ── module singleton ────────────────────────────────────
_cache_instance: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance