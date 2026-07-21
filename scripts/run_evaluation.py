#!/usr/bin/env python
"""
Run full AstroRAG pipeline on evaluation query set.

Usage:
    # full 20-query evaluation
    python scripts/run_evaluation.py

    # smaller subset for quick testing
    python scripts/run_evaluation.py --n 5

    # specific subdomain(s)
    python scripts/run_evaluation.py --subdomains cosmology "galaxy formation"

    # resume from partial run
    python scripts/run_evaluation.py --resume

    # rate-limit friendly (sleep between queries)
    python scripts/run_evaluation.py --sleep 30
"""

from __future__ import annotations

import argparse
import sys
from   pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel   import Panel

from astrorag.evaluation import (
    EvaluationRunner,
    get_query_set,
    compute_metrics,
    format_metrics_table,
)
from astrorag.logger import setup_logging

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Run only first N queries")
    parser.add_argument("--subdomains", nargs="+", default=None,
                        help="Filter to these subdomains")
    parser.add_argument("--k", type=int, default=50,
                        help="Top-K for BM25 (default 50)")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Sleep between queries (rate limits)")
    parser.add_argument("--output", type=str,
                        default="results/evaluation.json",
                        help="Output JSON path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--name", type=str, default="default",
                        help="Query set name for identification")
    args = parser.parse_args()

    setup_logging(level="INFO")

    console.print()
    console.print(Panel.fit(
        "[bold cyan]AstroRAG Evaluation Harness[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    # ── select queries ─────────────────────────────────
    queries = get_query_set(
        n          = args.n,
        subdomains = args.subdomains,
    )
    console.print(f"Running {len(queries)} queries")
    for q in queries:
        console.print(f"  [{q.idx:2d}] [{q.subdomain}] {q.query[:70]}...")
    console.print()

    # ── run ────────────────────────────────────────────
    runner = EvaluationRunner(
        top_k                 = args.k,
        sleep_between_queries = args.sleep,
    )
    result = runner.run(
        queries        = queries,
        output_path    = Path(args.output),
        query_set_name = args.name,
        resume         = args.resume,
    )

    # ── metrics ────────────────────────────────────────
    metrics = compute_metrics(result)
    console.print()
    console.print(format_metrics_table(metrics))
    console.print(f"\nFull traces saved → [green]{args.output}[/green]")


if __name__ == "__main__":
    main()