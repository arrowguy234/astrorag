"""
Memory-efficient streaming iterators for large corpus files.

Rather than loading multi-gigabyte JSONL files into memory,
these iterators yield one record at a time so downstream code
can process incrementally.
"""

from __future__ import annotations

import gzip
import json
from   collections.abc import Iterator
from   pathlib         import Path
from   typing          import Any

from tqdm.auto import tqdm

from astrorag.logger import get_logger

logger = get_logger(__name__)


def count_lines_gz(path: Path) -> int:
    """
    Count total lines in a gzipped file.

    Slower than a plain wc -l but necessary for progress bar
    display when total line count is not known in advance.
    """
    logger.debug(f"Counting lines in {path.name}...")
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            count += 1
    return count


def iter_abstracts(
    path:          Path,
    limit:         int  = None,
    show_progress: bool = True,
) -> Iterator[dict[str, Any]]:
    """
    Yield paper records from abstracts_all.jsonl.gz one at a time.

    Args:
        path:          Path to abstracts_all.jsonl.gz.
        limit:         Maximum records to yield (None = all).
        show_progress: Show tqdm progress bar.

    Yields:
        Dict for each paper record; malformed lines are skipped.
    """
    if not path.exists():
        raise FileNotFoundError(f"Abstracts file not found: {path}")

    iterator: Iterator[str]
    total = limit if limit else None

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        iterator = tqdm(
            fh,
            desc     = "Loading abstracts",
            total    = total,
            unit     = "papers",
            disable  = not show_progress,
        )
        for i, line in enumerate(iterator):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {i}: {e}")
                continue


def iter_citations(
    path:          Path,
    keep_ids:      set[str] = None,
    show_progress: bool     = True,
) -> Iterator[dict[str, Any]]:
    """
    Yield citation records from citations_indexed.jsonl.gz.

    Args:
        path:          Path to citations_indexed.jsonl.gz.
        keep_ids:      If provided, only yield records whose paper_idx
                       is in this set (used to filter to loaded sample).
        show_progress: Show tqdm progress bar.

    Yields:
        Dict for each citation record.
    """
    if not path.exists():
        raise FileNotFoundError(f"Citations file not found: {path}")

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        iterator = tqdm(
            fh,
            desc     = "Loading citations",
            unit     = "records",
            disable  = not show_progress,
        )
        for i, line in enumerate(iterator):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed citation line {i}: {e}")
                continue
            if keep_ids is not None:
                aid = str(rec.get("paper_idx", rec.get("id", "")))
                if aid not in keep_ids:
                    continue
            yield rec