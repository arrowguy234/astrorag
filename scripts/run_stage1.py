#!/usr/bin/env python
"""
CLI to test Stage 1 BM25 retrieval.

Usage:
    python scripts/run_stage1.py "AGN feedback"
    python scripts/run_stage1.py "cooling flows" --k 20
    python scripts/run_stage1.py --all-defaults
"""

from __future__ import annotations

import argparse
import sys
from   pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table

from astrorag.data      import DataLoader
from astrorag.data.models import LoadConfig
from astrorag.logger    import setup_logging
from astrorag.stages    import Stage1BM25

console = Console()


DEFAULT_QUERIES = [
    "How do AGN jets suppress star formation in massive elliptical galaxies through X-ray cavity observations?",
    "What is the relationship between galaxy stellar mass and central supermassive black hole mass?",
    "How does the intracluster medium cool in galaxy clusters?",
    "How are photometric redshifts calibrated using SDSS and DESI?",
    "What mechanisms quench star formation in massive early-type galaxies?",
]


def print_run(query: str, run, show_n: int = 10):
    console.rule(f"[cyan]{query[:80]}[/cyan]")

    table = Table(show_header=True, header_style="bold cyan",
                  title=f"Top {show_n} of {run.n_corpus:,}")
    table.add_column("Rank", justify="right")
    table.add_column("arXiv ID",  style="green")
    table.add_column("BM25",      justify="right")
    table.add_column("Overlap",   justify="right")
    table.add_column("Title",     max_width=50)

    for r in run.results[:show_n]:
        title = r.title or (r.abstract[:50] + "...")
        table.add_row(
            str(r.rank),
            r.arxiv_id,
            f"{r.bm25_score:.2f}",
            str(r.concept_overlap),
            title[:50],
        )
    console.print(table)

    console.print(
        f"\n  Elapsed  : {run.elapsed_s*1000:.1f} ms   "
        f"Top score: {run.top_score:.2f}   "
        f"Mean top: {run.mean_score_top:.2f}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--k", type=int, default=10,
                        help="Number of results to show")
    parser.add_argument("--size", type=int, default=None,
                        help="Corpus sample size (default from config)")
    parser.add_argument("--all-defaults", action="store_true")
    args = parser.parse_args()

    setup_logging(level="INFO")

    # ── load corpus ─────────────────────────────────────
    console.print(Panel.fit(
        "[bold cyan]Stage 1 — BM25 Retrieval[/bold cyan]",
        border_style="cyan",
    ))
    config = LoadConfig(
        sample_size  = args.size or 408_590,
        use_cache    = True,
        show_progress= True,
    )
    loader = DataLoader(config=config)
    corpus = loader.load()

    # ── build/load index ────────────────────────────────
    stage1 = Stage1BM25(corpus=corpus)

    # ── run queries ─────────────────────────────────────
    if args.all_defaults or args.query is None:
        for q in DEFAULT_QUERIES:
            run = stage1.run(q, top_k=args.k)
            print_run(q, run, show_n=args.k)
    else:
        run = stage1.run(args.query, top_k=args.k)
        print_run(args.query, run, show_n=args.k)


if __name__ == "__main__":
    main()