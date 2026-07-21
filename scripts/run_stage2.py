#!/usr/bin/env python
"""
End-to-end Stage 0 → 1 → 2 test script.

Runs query decomposition, BM25 retrieval, then graph construction
with PPR and cluster summary for one or more queries.

Usage:
    python scripts/run_stage2.py "AGN feedback star formation"
    python scripts/run_stage2.py --all-defaults
    python scripts/run_stage2.py --size 50000 "cooling flows"
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
from astrorag.stages    import Stage0Decompose, Stage1BM25, Stage2Graph

console = Console()

DEFAULT_QUERIES = [
    "How do AGN jets suppress star formation in massive elliptical galaxies through X-ray cavity observations?",
    "What is the relationship between galaxy stellar mass and central supermassive black hole mass?",
    "How does the intracluster medium cool in galaxy clusters?",
]


def run_one_query(
    query:  str,
    stage0: Stage0Decompose,
    stage1: Stage1BM25,
    stage2: Stage2Graph,
    top_k:  int,
) -> None:
    console.rule(f"[cyan]{query[:80]}[/cyan]")

    # ── stage 0 ────────────────────────────────────────
    s0 = stage0.run(query)
    d = s0.decomposition
    console.print(Panel(
        f"[bold]Q1[/bold]: {d.sub_questions['Q1']}\n"
        f"[bold]Q2[/bold]: {d.sub_questions['Q2']}\n"
        f"[bold]Q3[/bold]: {d.sub_questions['Q3']}\n"
        f"Wavelength: {d.wavelength}   Catalogs: {', '.join(d.catalogs) or '(none)'}",
        title="Stage 0", border_style="blue",
    ))

    # ── stage 1 ────────────────────────────────────────
    run = stage1.run(query, top_k=top_k)
    console.print(
        f"[bold]Stage 1[/bold]: {top_k} candidates from "
        f"{run.n_corpus:,} in {run.elapsed_s*1000:.0f} ms  "
        f"top BM25={run.top_score:.2f}"
    )

    # ── stage 2 ────────────────────────────────────────
    context = stage2.run(run)
    console.print(
        f"[bold]Stage 2[/bold]: {context.n_nodes} nodes, "
        f"{context.signals.n_edges_after} edges, "
        f"density={context.signals.density:.1%}, "
        f"PPR converged in {context.ppr_iterations} iters "
        f"({context.elapsed_seconds*1000:.0f} ms)"
    )

    # ── top 10 by PPR ──────────────────────────────────
    top10 = context.top_ppr_indices(10)
    table = Table(title="Top 10 by PPR", show_header=True,
                  header_style="bold cyan")
    table.add_column("PPR Rank", justify="right")
    table.add_column("BM25 Rank", justify="right")
    table.add_column("Cluster", justify="right")
    table.add_column("arXiv ID", style="green")
    table.add_column("PPR", justify="right")
    table.add_column("BM25", justify="right")
    table.add_column("Title", max_width=40)

    for ppr_rank, idx in enumerate(top10, 1):
        r = run.results[idx]
        table.add_row(
            str(ppr_rank),
            str(r.rank),
            str(r.cluster),
            r.arxiv_id,
            f"{r.ppr_score:.3f}",
            f"{r.bm25_score:.2f}",
            (r.title or r.abstract[:40])[:40],
        )
    console.print(table)

    # ── cluster summary ────────────────────────────────
    console.print(Panel(
        context.cluster_summary.prompt_text,
        title="Cluster Summary (LLM prompt context)",
        border_style="yellow",
    ))
    console.print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--k",    type=int, default=50)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--all-defaults", action="store_true")
    args = parser.parse_args()

    setup_logging(level="INFO")

    # ── load corpus ────────────────────────────────────
    config = LoadConfig(
        sample_size  = args.size or 408_590,
        use_cache    = True,
        show_progress= True,
    )
    corpus = DataLoader(config=config).load()

    # ── init stages ────────────────────────────────────
    stage0 = Stage0Decompose()
    stage1 = Stage1BM25(corpus=corpus)
    stage2 = Stage2Graph(corpus=corpus)

    console.print()
    console.print(Panel.fit(
        "[bold cyan]AstroRAG Stages 0 → 1 → 2[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    if args.all_defaults or args.query is None:
        for q in DEFAULT_QUERIES:
            run_one_query(q, stage0, stage1, stage2, args.k)
    else:
        run_one_query(args.query, stage0, stage1, stage2, args.k)


if __name__ == "__main__":
    main()