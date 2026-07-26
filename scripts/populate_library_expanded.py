#!/usr/bin/env python
"""
Populate the context library from the expanded 45-query set.

Runs the full pipeline (Stages 0-5) on all new queries and writes
complete LibraryEntry records with real equations, numerical
results, and sub-question answers.

Uses --resume to skip queries whose selected paper is already in
the library.

Usage:
    # populate only the 25 NEW queries (idx 21-45)
    python scripts/populate_library_expanded.py --new-only --sleep 30

    # populate all 45, resuming from existing library
    python scripts/populate_library_expanded.py --sleep 30 --resume

    # test with just 3 new queries first
    python scripts/populate_library_expanded.py --new-only --n 3 --sleep 15
"""

from __future__ import annotations

import argparse
import sys
import time
from   pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel   import Panel

from astrorag.chat        import get_library
from astrorag.data        import DataLoader
from astrorag.data.models import LoadConfig
from astrorag.evaluation.expanded_queries import get_expanded_set
from astrorag.logger      import setup_logging
from astrorag.stages      import (
    Stage0Decompose, Stage1BM25, Stage2Graph,
    Stage3Rerank, Stage4PDF, Stage5Summarise,
)

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-only", action="store_true",
                        help="Only run the 25 new queries (idx 21-45)")
    parser.add_argument("--n", type=int, default=None,
                        help="Run first N queries only")
    parser.add_argument("--sleep", type=float, default=30.0,
                        help="Sleep between queries (Groq rate limit)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip queries whose arxiv ID already in library")
    args = parser.parse_args()

    setup_logging(level="INFO")

    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Expanded Library Population[/bold cyan]\n"
        f"Runs full pipeline over expanded query set",
        border_style="cyan",
    ))
    console.print()

    # load corpus
    console.print("[cyan]Loading corpus...[/cyan]")
    corpus = DataLoader(config=LoadConfig(
        sample_size   = 408_590,
        use_cache     = True,
        show_progress = False,
    )).load()

    # stages
    stage0 = Stage0Decompose()
    stage1 = Stage1BM25(corpus=corpus)
    stage2 = Stage2Graph(corpus=corpus)
    stage3 = Stage3Rerank()
    stage4 = Stage4PDF()
    stage5 = Stage5Summarise(stage4=stage4)

    lib = get_library()
    console.print(f"Library currently has {len(lib.list_all())} entries")

    # select queries
    queries = get_expanded_set(n=args.n, only_new=args.new_only)
    console.print(f"Will process {len(queries)} queries")

    n_added   = 0
    n_skipped = 0
    n_failed  = 0

    for q in queries:
        console.print(f"\n[bold]Query {q.idx}:[/bold] [{q.subdomain}] "
                      f"{q.query[:70]}")

        try:
            s0 = stage0.run(q.query)
            s1 = stage1.run(q.query, top_k=50)
            s2 = stage2.run(s1)
            s3 = stage3.run(retrieval=s1, graph_context=s2,
                           decomposition=s0.decomposition)
            console.print(f"  Selected: {s3.selected_result.arxiv_id}")

            # resume check
            if args.resume and lib.get(s3.selected_result.arxiv_id):
                console.print(f"  [yellow]Already in library — skipping[/yellow]")
                n_skipped += 1
                continue

            s4 = stage4.run(s3)
            if not s4.success:
                console.print(f"  [yellow]PDF failed, trying fallback[/yellow]")
                for pool_idx in list(s3.fallback_pool):
                    next_paper = s1.results[pool_idx]
                    s4 = stage4.run(next_paper)
                    if s4.success:
                        s3.selected_result = next_paper
                        break
                if not s4.success:
                    console.print(f"  [red]All PDF attempts failed[/red]")
                    n_failed += 1
                    continue

            s5 = stage5.run(
                decomposition = s0.decomposition,
                retrieval     = s1,
                stage3_result = s3,
                initial_pdf   = s4,
            )
            console.print(
                f"  Q_total={s5.quality.scores.Q_total:.3f} "
                f"→ {s5.quality.decision.value}"
            )

            entry = lib.add_from_stage5(q.query, s5, subdomain=q.subdomain)
            console.print(f"  [green]✓ Saved[/green]")
            n_added += 1

        except Exception as e:
            console.print(f"  [red]✗ {type(e).__name__}: {e}[/red]")
            n_failed += 1

        if args.sleep > 0:
            time.sleep(args.sleep)

    console.print()
    console.print(Panel.fit(
        f"[bold]Done[/bold]\n"
        f"Added: {n_added}\n"
        f"Skipped: {n_skipped}\n"
        f"Failed: {n_failed}\n"
        f"Library now has {len(lib.list_all())} entries",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
