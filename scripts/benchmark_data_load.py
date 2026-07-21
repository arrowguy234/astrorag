#!/usr/bin/env python
"""
Benchmark the data loader on various sample sizes.

Run this to verify the full 408,590-paper load works end-to-end
and measure exact timing and memory usage on your specific server.

Usage:
    python scripts/benchmark_data_load.py                    # default
    python scripts/benchmark_data_load.py --full             # full 408K
    python scripts/benchmark_data_load.py --size 50000       # 50K
    python scripts/benchmark_data_load.py --no-cache         # bypass
"""

from __future__ import annotations

import argparse
import sys
import time
from   pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table   import Table

from astrorag.data   import DataLoader
from astrorag.data.models import LoadConfig
from astrorag.logger import setup_logging

console = Console()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full",     action="store_true",
                   help="load the full 408,590 papers")
    p.add_argument("--size",     type=int, default=5000,
                   help="sample size to test (default 5000)")
    p.add_argument("--no-cache", action="store_true",
                   help="bypass and rebuild cache")
    p.add_argument("--multi",    action="store_true",
                   help="benchmark multiple sizes 1K/10K/50K/100K")
    return p.parse_args()


def run_benchmark(sample_size: int, use_cache: bool = True) -> dict:
    setup_logging(level="INFO")

    console.rule(f"Benchmarking sample_size = {sample_size:,}")

    cfg = LoadConfig(
        sample_size  = sample_size,
        use_cache    = use_cache,
        force_reload = not use_cache,
        show_progress= True,
    )

    t0     = time.time()
    loader = DataLoader(config=cfg)
    data   = loader.load()
    elapsed = time.time() - t0

    console.print()
    console.print(data.stats.summary())

    return {
        "sample_size":  sample_size,
        "n_papers":     data.n_papers(),
        "load_time_s":  elapsed,
        "memory_mb":    data.stats.memory_usage_mb,
        "n_concepts":   data.stats.n_papers_with_concepts,
        "n_citations":  data.stats.n_papers_with_citations,
        "cache":        use_cache,
    }


def main():
    args = parse_args()

    results: list[dict] = []

    if args.multi:
        for n in (1_000, 10_000, 50_000, 100_000):
            results.append(run_benchmark(n, use_cache=not args.no_cache))
    elif args.full:
        results.append(run_benchmark(408_590, use_cache=not args.no_cache))
    else:
        results.append(run_benchmark(args.size, use_cache=not args.no_cache))

    # summary table
    console.print()
    console.rule("Benchmark Summary")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Sample size", justify="right")
    table.add_column("Papers loaded", justify="right")
    table.add_column("Load time (s)", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Cache", justify="center")
    for r in results:
        table.add_row(
            f"{r['sample_size']:,}",
            f"{r['n_papers']:,}",
            f"{r['load_time_s']:.1f}",
            f"{r['memory_mb']:.0f}",
            "Y" if r["cache"] else "N",
        )
    console.print(table)


if __name__ == "__main__":
    main()