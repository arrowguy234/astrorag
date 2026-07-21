#!/usr/bin/env python
"""
Run ablation study — one or more variants over the query set.

Usage:
    # run all 6 variants (long — ~5 hours on Groq free tier)
    python scripts/run_ablation.py --all

    # run specific variant
    python scripts/run_ablation.py --variant no_graph

    # smoke test with 3 queries per variant
    python scripts/run_ablation.py --all --n 3
"""

from __future__ import annotations

import argparse
import sys
from   pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel   import Panel

from astrorag.data       import DataLoader
from astrorag.data.models import LoadConfig
from astrorag.evaluation import (
    AblationRunner,
    ABLATION_VARIANTS,
    get_variant,
    get_query_set,
)
from astrorag.logger     import setup_logging

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",     action="store_true",
                        help="Run all 6 ablation variants")
    parser.add_argument("--variant", type=str, default=None,
                        help="Run only this variant")
    parser.add_argument("--n",       type=int, default=None,
                        help="Run first N queries only")
    parser.add_argument("--sleep",   type=float, default=30.0,
                        help="Sleep between queries (Groq rate limit)")
    parser.add_argument("--k",       type=int, default=50,
                        help="Stage 1 top-k")
    parser.add_argument("--output-dir", type=str,
                        default="results/ablation",
                        help="Directory for per-variant JSON outputs")
    args = parser.parse_args()

    setup_logging(level="INFO")

    # ── select variants ──────────────────────────────
    if args.variant:
        variants = [get_variant(args.variant)]
    elif args.all:
        variants = list(ABLATION_VARIANTS)
    else:
        parser.error("Must specify --all or --variant")

    # ── select queries ───────────────────────────────
    queries = get_query_set(n=args.n)

    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Ablation Study[/bold cyan]\n"
        f"Variants: {[v.name for v in variants]}\n"
        f"Queries : {len(queries)}\n"
        f"Output  : {args.output_dir}",
        border_style="cyan",
    ))
    console.print()

    # ── load corpus once, reuse across variants ──────
    console.print("[cyan]Loading corpus (once, shared across variants)...[/cyan]")
    config = LoadConfig(
        sample_size  = 408_590,
        use_cache    = True,
        show_progress= True,
    )
    corpus = DataLoader(config=config).load()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        console.print()
        console.rule(f"[bold green]Variant: {variant.name}[/bold green]")
        console.print(f"  {variant.description}\n")

        runner = AblationRunner(
            variant               = variant,
            corpus                = corpus,
            top_k                 = args.k,
            sleep_between_queries = args.sleep,
        )
        out_path = out_dir / f"ablation_{variant.name}.json"
        runner.run(queries=queries, output_path=out_path)
        console.print(f"[green]✓ {variant.name} → {out_path}[/green]")


if __name__ == "__main__":
    main()