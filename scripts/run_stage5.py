#!/usr/bin/env python
"""
Full end-to-end AstroRAG pipeline through Stage 5.

Usage:
    python scripts/run_stage5.py "AGN feedback"
    python scripts/run_stage5.py --all-defaults
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
    Stage0Decompose, Stage1BM25, Stage2Graph,
    Stage3Rerank, Stage4PDF, Stage5Summarise,
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
    stage5: Stage5Summarise,
    top_k:  int,
) -> None:
    console.rule(f"[cyan]{query[:80]}[/cyan]")

    s0 = stage0.run(query)
    s1 = stage1.run(query, top_k=top_k)
    s2 = stage2.run(s1)
    s3 = stage3.run(retrieval=s1, graph_context=s2, decomposition=s0.decomposition)
    s4 = stage4.run(s3)

    if not s4.success:
        console.print(f"[red]Stage 4 failed: {s4.error}[/red]\n")
        return

    console.print(
        f"[bold]Pipeline progress[/bold]: selected {s3.selected_result.arxiv_id} "
        f"({s4.n_pages} pages, {s4.n_chars_total:,} chars, "
        f"{len(s4.sections)} sections)\n"
    )

    s5 = stage5.run(
        decomposition = s0.decomposition,
        retrieval     = s1,
        stage3_result = s3,
        initial_pdf   = s4,
    )

    q  = s5.quality
    ss = s5.summary

    # ── overview panel ─────────────────────────────────
    console.print(Panel(
        f"[bold]Paper[/bold]     : [green]{s5.selected_arxiv_id}[/green]\n"
        f"[bold]Overview[/bold]  : {ss.paper_overview[:200]}\n"
        f"[bold]Evidence[/bold]  : {ss.evidence_type}\n"
        f"[bold]Instruments[/bold]: {', '.join(ss.instruments) or '(none)'}\n"
        f"[bold]Methodology[/bold]: {ss.methodology[:200] or '(none extracted)'}",
        title="Stage 5 — Structured Summary",
        border_style="green",
    ))

    # ── sub-question answers ───────────────────────────
    for qk, ans in ss.sub_question_answers.items():
        status = "[green]✓[/green]" if ans.answered else "[red]✗[/red]"
        details = ans.answer_text[:250]
        eq_note = (f"\n   Equations: {', '.join(ans.equations[:3])}"
                   if ans.equations else "")
        val_note = (f"\n   Values: {', '.join(ans.values[:3])}"
                    if ans.values else "")
        console.print(Panel(
            f"{details}{eq_note}{val_note}",
            title=f"{status} {qk} — Section: {ans.section or 'unknown'}",
            border_style="blue",
        ))

    # ── key equations ──────────────────────────────────
    if ss.key_equations:
        table = Table(title="Key Equations", show_header=True)
        table.add_column("Equation", style="cyan")
        table.add_column("Variables", max_width=40)
        table.add_column("Section", style="green")
        for eq in ss.key_equations[:5]:
            table.add_row(eq.equation[:40], eq.variables[:40], eq.section)
        console.print(table)

    # ── numerical results ──────────────────────────────
    if ss.numerical_results:
        table = Table(title="Numerical Results", show_header=True)
        table.add_column("Quantity", style="cyan")
        table.add_column("Value", justify="right", style="yellow")
        table.add_column("Uncertainty", justify="right")
        table.add_column("Unit", style="green")
        for nr in ss.numerical_results[:8]:
            table.add_row(nr.quantity[:30], nr.value, nr.uncertainty, nr.unit)
        console.print(table)

    # ── quality gate ───────────────────────────────────
    color = "green" if q.decision.value == "ACCEPT" else \
            "yellow" if q.decision.value == "RETRY" else "red"
    console.print(Panel(
        f"[bold]Q_f (faithfulness)[/bold] : {q.scores.Q_f:.3f}   "
        f"({q.scores.n_claims_verified}/{q.scores.n_claims_total} claims verified)\n"
        f"[bold]Q_c (coverage)[/bold]     : {q.scores.Q_c:.3f}\n"
        f"[bold]Q_i (consistency)[/bold]  : {q.scores.Q_i:.3f}   "
        f"snippet overlap={q.scores.snippet_overlap:.2f}\n"
        f"\n"
        f"[bold]Q_total[/bold] = 0.40 · Q_f + 0.35 · Q_c + 0.25 · Q_i "
        f"= [bold]{q.scores.Q_total:.3f}[/bold]\n"
        f"\n"
        f"[bold]Decision[/bold]     : [{color}]{q.decision.value}[/{color}]\n"
        f"[bold]Attempts[/bold]     : {s5.n_attempts}   "
        f"(retries={s5.n_retries}, reselects={s5.n_reselections})\n"
        f"[bold]Total time[/bold]   : {s5.total_time_s:.2f}s",
        title="Quality Gate",
        border_style=color,
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
    stage5 = Stage5Summarise()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]AstroRAG Full Pipeline (Stages 0–5)[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    if args.all_defaults or args.query is None:
        for q in DEFAULT_QUERIES:
            run_one_query(q, stage0, stage1, stage2, stage3,
                          stage4, stage5, args.k)
    else:
        run_one_query(args.query, stage0, stage1, stage2, stage3,
                      stage4, stage5, args.k)


if __name__ == "__main__":
    main()