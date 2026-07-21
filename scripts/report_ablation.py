#!/usr/bin/env python
"""
Compare ablation variant results against the baseline (full) run.

Usage:
    python scripts/report_ablation.py
    python scripts/report_ablation.py --dir results/ablation
"""

from __future__ import annotations

import argparse
import sys
from   pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from astrorag.evaluation import (
    compute_variant_comparison,
    format_ablation_table,
    load_all_variants,
)

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="results/ablation",
                        help="Directory containing ablation_*.json files")
    parser.add_argument("--baseline", type=str, default="full",
                        help="Which variant to treat as baseline")
    args = parser.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.exists():
        console.print(f"[red]Directory not found: {results_dir}[/red]")
        sys.exit(1)

    baseline, variants = load_all_variants(results_dir, args.baseline)

    baseline_cmp = compute_variant_comparison(baseline, baseline)
    variant_cmps = [
        compute_variant_comparison(v, baseline)
        for v in variants.values()
    ]

    # baseline first, then variants
    all_cmps = [baseline_cmp] + variant_cmps

    console.print()
    console.print(format_ablation_table(all_cmps))
    console.print()


if __name__ == "__main__":
    main()