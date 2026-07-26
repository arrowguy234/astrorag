#!/usr/bin/env python
"""Generate readable report from ground truth study results."""

from __future__ import annotations

import argparse
import json
import sys
from   pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel

console = Console()


def report_multi_judge(path: Path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    console.rule("[bold cyan]Multi-Judge Panel[/bold cyan]")

    # per-judge table
    table = Table(title="Per-Judge Metrics")
    table.add_column("Judge", style="cyan")
    table.add_column("Papers", justify="right")
    table.add_column("Faith", justify="right")
    table.add_column("Cover", justify="right")
    table.add_column("Tech", justify="right")
    table.add_column("Overall", justify="right", style="bold")
    table.add_column("Accept%", justify="right")

    for j in data.get("per_judge", []):
        n = j["n_papers_judged"]
        dist = j["verdict_distribution"]
        accept_pct = dist["accept"] / n * 100 if n else 0
        table.add_row(
            j["judge"],
            str(n),
            f"{j['mean_faithfulness']:.2f}",
            f"{j['mean_coverage']:.2f}",
            f"{j['mean_technical_accuracy']:.2f}",
            f"{j['mean_overall']:.2f}",
            f"{accept_pct:.0f}%",
        )
    console.print(table)

    # aggregate
    agg = data.get("aggregate", {})
    pw = agg.get("pairwise_agreement", {})
    console.print(Panel.fit(
        f"Total verdicts: {agg.get('n_verdicts_total', 0)}\n"
        f"Errors: {agg.get('n_verdicts_err', 0)}\n"
        f"Mean overall (all judges): {agg.get('mean_overall_all_judges', 0):.2f}\n"
        f"Pairwise verdict agreement: {pw.get('agreement', 0):.1%} "
        f"({pw.get('n_agree', 0)}/{pw.get('n_pairs', 0)})",
        title="Aggregate",
        border_style="cyan",
    ))


def report_llm_swap(path: Path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    console.rule("[bold cyan]LLM Swap Experiment[/bold cyan]")

    # group results by model
    by_model: dict[str, list] = {}
    for r in data.get("results", []):
        by_model.setdefault(r["llm_model"], []).append(r)

    table = Table(title="Per-Model Metrics")
    table.add_column("Model", style="cyan")
    table.add_column("N ok", justify="right")
    table.add_column("N err", justify="right")
    table.add_column("Mean Q_total", justify="right", style="bold")
    table.add_column("Mean Q_f", justify="right")
    table.add_column("Mean Q_c", justify="right")
    table.add_column("Mean Q_i", justify="right")
    table.add_column("Accept%", justify="right")
    table.add_column("Mean #eq", justify="right")

    for model, rs in by_model.items():
        ok = [r for r in rs if not r.get("error")]
        n_ok, n_err = len(ok), len(rs) - len(ok)
        if not ok:
            continue
        mean = lambda k: sum(r[k] for r in ok) / n_ok
        accept_pct = sum(1 for r in ok if r["decision"] == "ACCEPT") / n_ok * 100

        table.add_row(
            model,
            str(n_ok),
            str(n_err),
            f"{mean('q_total'):.3f}",
            f"{mean('q_f'):.3f}",
            f"{mean('q_c'):.3f}",
            f"{mean('q_i'):.3f}",
            f"{accept_pct:.0f}%",
            f"{mean('n_equations'):.1f}",
        )
    console.print(table)


def report_retrieval(path: Path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    console.rule("[bold cyan]Retrieval Ground Truth[/bold cyan]")

    console.print(Panel.fit(
        f"Judge model: {data['judge_model']}\n"
        f"Queries evaluated: {data['n_valid']}\n"
        f"[bold]Precision@1: {data['precision_at_1']:.1%}[/bold]\n"
        f"[bold]Precision@3: {data['precision_at_3']:.1%}[/bold]",
        title="Summary",
        border_style="green",
    ))

    # per-query detail
    table = Table(title="Per-Query Agreement")
    table.add_column("Q", justify="right")
    table.add_column("AstroRAG", style="cyan")
    table.add_column("Judge Top-3")
    table.add_column("Match", justify="center")

    for r in data.get("results", []):
        marker = "🎯" if r["astrorag_in_judge_top1"] else \
                 "✅" if r["astrorag_in_judge_top3"] else "❌"
        table.add_row(
            str(r["query_idx"]),
            r["astrorag_selected"],
            ", ".join(r["judge_top3"]),
            marker,
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/ground_truth")
    args = parser.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.exists():
        console.print(f"[red]{results_dir} does not exist[/red]")
        sys.exit(1)

    mj = results_dir / "multi_judge.json"
    if mj.exists():
        report_multi_judge(mj)
        console.print()

    swap = results_dir / "llm_swap.json"
    if swap.exists():
        report_llm_swap(swap)
        console.print()

    rgt = results_dir / "retrieval_gt.json"
    if rgt.exists():
        report_retrieval(rgt)


if __name__ == "__main__":
    main()
