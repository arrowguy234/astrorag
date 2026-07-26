#!/usr/bin/env python
"""
Ground truth evaluation CLI — runs all three studies.

Usage:
    # run all three studies
    python scripts/run_ground_truth.py --all

    # just multi-judge
    python scripts/run_ground_truth.py --study judges

    # just LLM swap
    python scripts/run_ground_truth.py --study swap

    # just retrieval ground truth
    python scripts/run_ground_truth.py --study retrieval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from   datetime import datetime
from   pathlib  import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel   import Panel

from astrorag.chat import get_library
from astrorag.data import DataLoader
from astrorag.data.models import LoadConfig
from astrorag.evaluation import get_query_set
from astrorag.ground_truth import (
    MultiJudgePanel,
    LLMSwapExperiment,
    RetrievalGroundTruth,
    GroundTruthConfig,
)
from astrorag.logger import setup_logging

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Run all three studies")
    parser.add_argument("--study", choices=["judges", "swap", "retrieval"])
    parser.add_argument("--n", type=int, default=None,
                        help="Run first N queries only")
    parser.add_argument("--sleep", type=float, default=15.0)
    parser.add_argument("--output-dir", default="results/ground_truth")
    args = parser.parse_args()

    setup_logging(level="INFO")
    config = GroundTruthConfig(sleep_between=args.sleep)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    console.print(Panel.fit(
        "[bold cyan]AstroRAG Ground Truth Evaluation[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    # ── Study 1: Multi-Judge ──────────────────────────
    if args.all or args.study == "judges":
        console.rule("[bold green]Study 1: Multi-Judge Panel[/bold green]")
        lib = get_library()
        entries = lib.list_all()
        if args.n:
            entries = entries[:args.n]
        console.print(f"Judging {len(entries)} library entries with "
                      f"{len(config.judge_models)} judges")

        panel = MultiJudgePanel(
            judge_models  = config.judge_models,
            sleep_between = config.sleep_between,
        )
        report = panel.evaluate(entries)
        report.finished_at = datetime.now().isoformat()

        out_path = output_dir / "multi_judge.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
        console.print(f"[green]✓ Saved to {out_path}[/green]")

    # ── Study 2: LLM Swap ─────────────────────────────
    if args.all or args.study == "swap":
        console.rule("[bold green]Study 2: LLM Swap[/bold green]")

        # load corpus once
        console.print("Loading corpus for swap experiment...")
        corpus = DataLoader(config=LoadConfig(
            sample_size = 408_590,
            use_cache   = True,
            show_progress = False,
        )).load()

        experiment = LLMSwapExperiment(corpus=corpus)

        queries = get_query_set(n=args.n)
        all_results = []

        for swap_model in config.swap_models:
            console.print(f"\n[cyan]Running swap with: {swap_model}[/cyan]")
            for q in queries:
                console.print(f"  Q{q.idx}: {q.query[:70]}")
                result = experiment.run_query(
                    query_idx  = q.idx,
                    query      = q.query,
                    swap_model = swap_model,
                    sleep_s    = config.sleep_between,
                )
                all_results.append(result.to_dict())
                if result.error:
                    console.print(f"    [red]✗ {result.error}[/red]")
                else:
                    console.print(
                        f"    [green]✓ {result.selected_arxiv_id} "
                        f"Q_total={result.q_total:.3f}[/green]"
                    )
                time.sleep(config.sleep_between)

        out_path = output_dir / "llm_swap.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({
                "study_name":  "llm_swap",
                "swap_models": config.swap_models,
                "n_queries":   len(queries),
                "results":     all_results,
                "finished_at": datetime.now().isoformat(),
            }, fh, indent=2, ensure_ascii=False)
        console.print(f"[green]✓ Saved to {out_path}[/green]")

    # ── Study 3: Retrieval Ground Truth ───────────────
    if args.all or args.study == "retrieval":
        console.rule("[bold green]Study 3: Retrieval Ground Truth[/bold green]")

        # ensure corpus loaded
        console.print("Loading corpus for retrieval GT...")
        corpus = DataLoader(config=LoadConfig(
            sample_size = 408_590,
            use_cache   = True,
            show_progress = False,
        )).load()

        rgt = RetrievalGroundTruth(
            corpus      = corpus,
            judge_model = config.retrieval_gt_model,
        )

        # get library entries to know AstroRAG's actual selection
        lib = get_library()
        entries_by_query = {e.original_query: e for e in lib.list_all()}

        queries = get_query_set(n=args.n)
        all_results = []

        for q in queries:
            entry = entries_by_query.get(q.query)
            if not entry:
                console.print(f"  [yellow]Q{q.idx} not in library — skipping[/yellow]")
                continue

            console.print(f"\n  Q{q.idx}: {q.query[:70]}")
            console.print(f"  AstroRAG selected: {entry.arxiv_id}")

            result = rgt.evaluate_query(
                query_idx         = q.idx,
                query             = q.query,
                astrorag_selected = entry.arxiv_id,
            )
            all_results.append(result.to_dict())

            if result.error:
                console.print(f"    [red]✗ {result.error}[/red]")
            else:
                match_marker = "🎯 top-1" if result.astrorag_in_judge_top1 else \
                               "✅ top-3" if result.astrorag_in_judge_top3 else \
                               "❌ not in top-3"
                console.print(
                    f"    Judge top-3: {result.judge_top3}  {match_marker}"
                )
            time.sleep(config.sleep_between)

        # aggregate metrics
        valid = [r for r in all_results if not r.get("error")]
        p_at_1 = sum(1 for r in valid if r["astrorag_in_judge_top1"]) / len(valid) if valid else 0
        p_at_3 = sum(1 for r in valid if r["astrorag_in_judge_top3"]) / len(valid) if valid else 0

        out_path = output_dir / "retrieval_gt.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({
                "study_name":  "retrieval_ground_truth",
                "judge_model": config.retrieval_gt_model,
                "n_queries":   len(all_results),
                "n_valid":     len(valid),
                "precision_at_1": p_at_1,
                "precision_at_3": p_at_3,
                "results":     all_results,
                "finished_at": datetime.now().isoformat(),
            }, fh, indent=2, ensure_ascii=False)

        console.print()
        console.print(Panel.fit(
            f"[bold]Retrieval Ground Truth Summary[/bold]\n"
            f"Judge model: {config.retrieval_gt_model}\n"
            f"Precision@1: {p_at_1:.1%}\n"
            f"Precision@3: {p_at_3:.1%}\n"
            f"Saved: {out_path}",
            border_style="green",
        ))


if __name__ == "__main__":
    main()
