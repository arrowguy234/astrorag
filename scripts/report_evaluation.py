#!/usr/bin/env python
"""
Read an existing evaluation JSON and print aggregate metrics.

Usage:
    python scripts/report_evaluation.py results/evaluation.json
    python scripts/report_evaluation.py results/evaluation.json --traces
"""

from __future__ import annotations

import argparse
import sys
from   pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table

from astrorag.evaluation import compute_metrics, format_metrics_table
from astrorag.evaluation.models import EvaluationResult

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path",       type=Path)
    parser.add_argument("--traces",   action="store_true",
                        help="Also print per-query summary table")
    args = parser.parse_args()

    if not args.path.exists():
        console.print(f"[red]File not found: {args.path}[/red]")
        sys.exit(1)

    result = EvaluationResult.load(args.path)
    m      = compute_metrics(result)

    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Evaluation Report — {args.path.name}[/bold cyan]",
        border_style="cyan",
    ))
    console.print()
    console.print(format_metrics_table(m))

    if args.traces:
        console.print()
        table = Table(title="Per-query summary", show_header=True,
                      header_style="bold cyan")
        table.add_column("Idx", justify="right")
        table.add_column("Subdomain")
        table.add_column("Selected arXiv", style="green")
        table.add_column("Decision")
        table.add_column("Q_total", justify="right")
        table.add_column("Time (s)", justify="right")

        for t in result.traces:
            if not t.success:
                table.add_row(
                    str(t.query_idx), t.subdomain, "-",
                    f"[red]FAIL[/red]", "-",
                    f"{t.total_seconds:.1f}",
                )
                continue
            aid       = t.stage5.final_arxiv_id if t.stage5 else "-"
            decision  = t.stage5.decision       if t.stage5 else "-"
            q_total   = f"{t.stage5.q_total:.3f}" if t.stage5 else "-"
            color = (
                "green"  if decision == "ACCEPT" else
                "yellow" if decision == "RETRY"  else "red"
            )
            table.add_row(
                str(t.query_idx),
                t.subdomain,
                aid,
                f"[{color}]{decision}[/{color}]",
                q_total,
                f"{t.total_seconds:.1f}",
            )
        console.print(table)


if __name__ == "__main__":
    main()