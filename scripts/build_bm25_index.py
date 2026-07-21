#!/usr/bin/env python
"""
Pre-build the BM25 index over the full 408,590-paper corpus.

Runs once. Takes 3-5 minutes on CPU. After running, subsequent
retrievals load from cache in about 20-40 seconds.

Usage:
    python scripts/build_bm25_index.py
    python scripts/build_bm25_index.py --size 50000    # smaller
    python scripts/build_bm25_index.py --force         # rebuild
"""

from __future__ import annotations

import argparse
import sys
from   pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel   import Panel

from astrorag.data      import DataLoader
from astrorag.data.models import LoadConfig
from astrorag.logger    import setup_logging
from astrorag.retrieval import build_bm25_index

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",  type=int, default=None,
                        help="Sample size (default from config)")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if cache exists")
    args = parser.parse_args()

    setup_logging(level="INFO")

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Building BM25 Index[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    # ── load corpus ─────────────────────────────────────
    config = LoadConfig(
        sample_size  = args.size or 408_590,
        use_cache    = True,
        show_progress= True,
    )
    loader = DataLoader(config=config)
    corpus = loader.load()

    console.print()
    console.print(f"Corpus loaded: {corpus.n_papers():,} papers")
    console.print()

    # ── build index ─────────────────────────────────────
    index = build_bm25_index(
        corpus        = corpus,
        show_progress = True,
        save_cache    = True,
        force_rebuild = args.force,
    )

    console.print()
    console.print(Panel.fit(
        f"[bold green]✓ Index ready — {index.n_docs:,} docs "
        f"in {index.build_time_seconds:.1f}s[/bold green]",
        border_style="green",
    ))
    console.print()


if __name__ == "__main__":
    main()