#!/usr/bin/env python
"""
End-to-end Stages 0 → 1 → 2 → 3 test script.

Runs the full retrieval + graph + rerank pipeline for one or more
queries and displays the LLM's selection with graph-adjusted score.

Usage:
    python scripts/run_stage3.py "AGN feedback"
    python scripts/run_stage3.py --all-defaults
    python scripts/run_stage3.py --size 50000 "cooling flows"
"""

from __future__ import annotations

import argparse
import sys
from   pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table

from astrorag.data       import DataLoader
from astrorag.data.models import LoadConfig
from astrorag.logger     import setup_logging
from astrorag.stages     import (
    Stage0Decompose, Stage1BM25, Stage2Graph, Stage3Rerank,
)

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
    stage3: Stage3Rerank,
    top_k:  int,
) -> None:
    console.rule(f"[cyan]{query[:80]}[/cyan]")

    # ── stage 0 ────────────────────────────────────────
    s0 = stage0.run(query)
    d = s0.decomposition
    console.print(Panel(
        f"[bold]Q1[/bold]: {d.sub_questions['Q1'][:80]}\n"
        f"[bold]Q2[/bold]: {d.sub_questions['Q2'][:80]}\n"
        f"[bold]Q3[/bold]: {d.sub_questions['Q3'][:80]}",
        title="Stage 0 — decomposition", border_style="blue",
    ))

    # ── stage 1 ────────────────────────────────────────
    s1 = stage1.run(query, top_k=top_k)
    console.print(
        f"[bold]Stage 1[/bold]: {top_k} candidates from {s1.n_corpus:,} "
        f"in {s1.elapsed_s*1000:.0f} ms, top BM25={s1.top_score:.2f}"
    )

    # ── stage 2 ────────────────────────────────────────
    s2 = stage2.run(s1)
    console.print(
        f"[bold]Stage 2[/bold]: {s2.n_nodes} nodes, "
        f"{s2.signals.n_edges_after} edges, density={s2.signals.density:.1%}, "
        f"top PPR={s2.ppr_scores.max():.3f} in {s2.elapsed_seconds*1000:.0f} ms"
    )

    # ── stage 3 ────────────────────────────────────────
    s3 = stage3.run(retrieval=s1, graph_context=s2, decomposition=d)

    selected = s3.selected_result
    mode = "LLM" if not s3.fallback_used else "FALLBACK"

    console.print(Panel(
        f"[bold]Selected[/bold] : #{s3.selected_idx}  "
        f"[green]{selected.arxiv_id}[/green]\n"
        f"[bold]Title[/bold]    : {(selected.title or selected.abstract[:80])[:80]}\n"
        f"\n"
        f"BM25 rank    : #{selected.rank}   BM25 score: {selected.bm25_score:.2f}\n"
        f"PPR score    : {selected.ppr_score:.3f}   Cluster: {selected.cluster}\n"
        f"Confidence   : {s3.confidence:.3f}\n"
        f"Adjusted     : {s3.graph_adj_score:.3f}\n"
        f"Mode         : {mode}   Time: {s3.total_time_s*1000:.0f} ms\n"
        f"\n"
        f"[bold]Reason[/bold]: {s3.reason}",
        title="Stage 3 — LLM Rerank Decision", border_style="green",
    ))

    # ── fallback pool ──────────────────────────────────
    if s3.fallback_pool:
        table = Table(title="Fallback Pool (top-5 for Stage 5 re-selection)",
                      show_header=True, header_style="bold cyan")
        table.add_column("Pool Rank", justify="right")
        table.add_column("Cand Idx",  justify="right")
        table.add_column("arXiv",     style="green")
        table.add_column("BM25",      justify="right")
        table.add_column("PPR",       justify="right")
        table.add_column("Title",     max_width=45)
        for rank, idx in enumerate(s3.fallback_pool[:5], 1):
            r = s1.results[idx]
            table.add_row(
                str(rank), str(idx), r.arxiv_id,
                f"{r.bm25_score:.2f}",
                f"{r.ppr_score:.3f}",
                (r.title or r.abstract[:45])[:45],
            )
        console.print(table)

    console.print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--k",    type=int, default=50)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--all-defaults", action="store_true")
    args = parser.parse_args()

    setup_logging(level="INFO")

    config = LoadConfig(
        sample_size  = args.size or 408_590,
        use_cache    = True,
        show_progress= True,
    )
    corpus = DataLoader(config=config).load()

    stage0 = Stage0Decompose()
    stage1 = Stage1BM25(corpus=corpus)
    stage2 = Stage2Graph(corpus=corpus)
    stage3 = Stage3Rerank()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]AstroRAG Stages 0 → 1 → 2 → 3[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    if args.all_defaults or args.query is None:
        for q in DEFAULT_QUERIES:
            run_one_query(q, stage0, stage1, stage2, stage3, args.k)
    else:
        run_one_query(args.query, stage0, stage1, stage2, stage3, args.k)


if __name__ == "__main__":
    main()