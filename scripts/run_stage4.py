#!/usr/bin/env python
"""
End-to-end Stages 0 → 1 → 2 → 3 → 4.

Runs the full pipeline through PDF fetch and section parsing.

Usage:
    python scripts/run_stage4.py "AGN feedback"
    python scripts/run_stage4.py --all-defaults
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
    Stage0Decompose, Stage1BM25, Stage2Graph, Stage3Rerank, Stage4PDF,
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
    stage4: Stage4PDF,
    top_k:  int,
) -> None:
    console.rule(f"[cyan]{query[:80]}[/cyan]")

    s0 = stage0.run(query)
    d  = s0.decomposition
    console.print(
        f"[bold]Stage 0[/bold]: Q1='{d.sub_questions['Q1'][:60]}...'"
    )

    s1 = stage1.run(query, top_k=top_k)
    console.print(
        f"[bold]Stage 1[/bold]: top-{top_k} in {s1.elapsed_s*1000:.0f} ms"
    )

    s2 = stage2.run(s1)
    console.print(
        f"[bold]Stage 2[/bold]: {s2.n_nodes} nodes, "
        f"density={s2.signals.density:.1%}"
    )

    s3 = stage3.run(retrieval=s1, graph_context=s2, decomposition=d)
    selected = s3.selected_result
    console.print(
        f"[bold]Stage 3[/bold]: selected {selected.arxiv_id} "
        f"(BM25 rank #{selected.rank}, PPR={selected.ppr_score:.3f}, "
        f"conf={s3.confidence:.3f})"
    )

    # ── stage 4 ────────────────────────────────────────
    s4 = stage4.run(s3)

    if not s4.success:
        console.print(Panel(
            f"[bold red]Stage 4 FAILED[/bold red]: {s4.error}",
            border_style="red",
        ))
        console.print()
        return

    console.print(Panel(
        f"[bold]arXiv[/bold]     : {s4.arxiv_id}\n"
        f"[bold]Path[/bold]      : {s4.pdf_path.name if s4.pdf_path else 'n/a'}\n"
        f"[bold]Pages[/bold]     : {s4.n_pages}\n"
        f"[bold]Total chars[/bold]: {s4.n_chars_total:,}\n"
        f"[bold]Extractor[/bold] : {s4.extractor}\n"
        f"[bold]From cache[/bold]: {s4.from_cache}\n"
        f"[bold]Fetch[/bold]     : {s4.fetch_seconds:.2f}s   "
        f"[bold]Parse[/bold]: {s4.parse_seconds:.2f}s",
        title="Stage 4 — PDF Extraction",
        border_style="green",
    ))

    # ── sections table ─────────────────────────────────
    table = Table(title="Detected Sections", show_header=True,
                  header_style="bold cyan")
    table.add_column("Section", style="green")
    table.add_column("Chars", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Preview", max_width=40)

    for name, sec in s4.sections.items():
        preview = sec.text[:80].replace("\n", " ").strip()
        table.add_row(
            name,
            f"{sec.n_chars:,}",
            f"{sec.n_words:,}",
            preview + ("..." if len(sec.text) > 80 else ""),
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
    stage4 = Stage4PDF()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]AstroRAG Stages 0 → 1 → 2 → 3 → 4[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    if args.all_defaults or args.query is None:
        for q in DEFAULT_QUERIES:
            run_one_query(q, stage0, stage1, stage2, stage3, stage4, args.k)
    else:
        run_one_query(args.query, stage0, stage1, stage2, stage3, stage4, args.k)


if __name__ == "__main__":
    main()