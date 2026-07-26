#!/usr/bin/env python
"""
Compare citation counts of AstroRAG selections vs judge selections.

For each query in the retrieval GT results, look up how many times
each selected paper is cited within the 408K astro-ph corpus. Compare
AstroRAG's Stage 3 pick against the 70B judge's top-3 picks.

Hypothesis: AstroRAG's four-signal graph reranker (which explicitly
uses bibliographic coupling and co-citation) selects more highly-cited
papers than an abstract-only semantic LLM judge.

Note: paper_cited_by is keyed directly by arxiv_id string (not by
integer paper_idx). Direct dict lookup is O(1).
"""

from __future__ import annotations

import json
import statistics
import sys
from   pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table   import Table

from astrorag.data        import DataLoader
from astrorag.data.models import LoadConfig
from astrorag.logger      import setup_logging

console = Console()


def main():
    setup_logging(level="WARNING")

    # ── load corpus ──────────────────────────────────
    console.print("[cyan]Loading corpus for citation lookup...[/cyan]")
    corpus = DataLoader(config=LoadConfig(
        sample_size   = 408_590,
        use_cache     = True,
        show_progress = False,
    )).load()

    # ── build citation count map ─────────────────────
    # paper_cited_by is dict {arxiv_id: [list of citer arxiv_ids]}
    console.print("Building citation count index...")

    citation_counts: dict[str, int] = {
        arxiv_id: len(citers)
        for arxiv_id, citers in corpus.paper_cited_by.items()
    }

    total_indexed = len(citation_counts)
    n_with_cits = sum(1 for v in citation_counts.values() if v > 0)
    console.print(
        f"Indexed {total_indexed} papers, {n_with_cits} with non-zero citations"
    )

    # sanity check — show top 5 most-cited papers in the corpus
    top5 = sorted(citation_counts.items(), key=lambda x: -x[1])[:5]
    console.print(f"\nTop 5 most-cited papers in corpus:")
    for aid, cnt in top5:
        console.print(f"  {aid}: {cnt} citations")
    console.print()

    # ── load retrieval GT results ────────────────────
    gt_path = Path("results/ground_truth/retrieval_gt.json")
    if not gt_path.exists():
        console.print(
            f"[red]{gt_path} not found. Run "
            f"scripts/run_ground_truth.py --study retrieval first.[/red]"
        )
        sys.exit(1)

    with open(gt_path, encoding="utf-8") as fh:
        gt = json.load(fh)

    console.print(f"Loaded {len(gt['results'])} queries from retrieval GT\n")

    # ── per-query comparison ──────────────────────────
    astrorag_citations   = []
    judge_top1_citations = []
    judge_top3_citations = []

    table = Table(title="Citation Count Comparison")
    table.add_column("Q",              justify="right")
    table.add_column("Query",          max_width=35)
    table.add_column("AstroRAG pick",  style="cyan")
    table.add_column("Cit",            justify="right")
    table.add_column("Judge #1",       style="green")
    table.add_column("Cit",            justify="right")
    table.add_column("Δ",              justify="right")

    for r in gt.get("results", []):
        if r.get("error"):
            continue

        astro_id   = r["astrorag_selected"]
        judge_top3 = r["judge_top3"]

        astro_cit = citation_counts.get(astro_id, 0)
        astrorag_citations.append(astro_cit)

        if judge_top3:
            j1     = judge_top3[0]
            j1_cit = citation_counts.get(j1, 0)
            judge_top1_citations.append(j1_cit)
            for jid in judge_top3:
                judge_top3_citations.append(citation_counts.get(jid, 0))

            delta = astro_cit - j1_cit
            delta_style = ("[green]" if delta > 0 else
                          "[red]" if delta < 0 else "[yellow]")
            delta_str = f"{delta_style}{delta:+d}[/]"

            table.add_row(
                str(r["query_idx"]),
                r["query"][:35] + ("..." if len(r["query"]) > 35 else ""),
                astro_id,
                str(astro_cit),
                j1,
                str(j1_cit),
                delta_str,
            )

    console.print(table)
    console.print()

    # ── aggregate stats ───────────────────────────────
    def stat_line(name: str, xs: list[int]) -> str:
        if not xs:
            return f"  {name}: no data"
        return (
            f"  {name:26s} n={len(xs):3d}  "
            f"median={statistics.median(xs):7.1f}  "
            f"mean={statistics.mean(xs):8.1f}  "
            f"max={max(xs):5d}"
        )

    console.print("[bold]Citation Count Summary[/bold]")
    console.print(stat_line("AstroRAG selections",  astrorag_citations))
    console.print(stat_line("Judge top-1 picks",    judge_top1_citations))
    console.print(stat_line("Judge top-3 (all)",    judge_top3_citations))
    console.print()

    # ── overall verdict ───────────────────────────────
    if astrorag_citations and judge_top1_citations:
        astro_med = statistics.median(astrorag_citations)
        judge_med = statistics.median(judge_top1_citations)

        if astro_med > judge_med:
            ratio = astro_med / max(judge_med, 1)
            console.print(
                f"[green]▶ AstroRAG picks have {ratio:.1f}× higher "
                f"median citations than judge top-1 picks "
                f"({astro_med:.0f} vs {judge_med:.0f})[/green]"
            )
        elif judge_med > astro_med:
            ratio = judge_med / max(astro_med, 1)
            console.print(
                f"[yellow]▶ Judge top-1 picks have {ratio:.1f}× higher "
                f"median citations than AstroRAG picks "
                f"({judge_med:.0f} vs {astro_med:.0f})[/yellow]"
            )
        else:
            console.print(
                f"[yellow]▶ Median citation counts equal "
                f"at {astro_med:.0f}[/yellow]"
            )

    # ── per-query win/loss tally ──────────────────────
    console.print()
    console.print("[bold]Per-Query Comparison Summary[/bold]")

    deltas = []
    for r in gt.get("results", []):
        if r.get("error") or not r["judge_top3"]:
            continue
        astro_cit = citation_counts.get(r["astrorag_selected"], 0)
        j1_cit    = citation_counts.get(r["judge_top3"][0], 0)
        deltas.append(astro_cit - j1_cit)

    if deltas:
        n_astro_higher = sum(1 for d in deltas if d > 0)
        n_judge_higher = sum(1 for d in deltas if d < 0)
        n_tied         = sum(1 for d in deltas if d == 0)

        console.print(
            f"  AstroRAG picks more-cited paper: "
            f"[green]{n_astro_higher}/{len(deltas)} "
            f"({n_astro_higher/len(deltas)*100:.0f}%)[/green]"
        )
        console.print(
            f"  Judge top-1 more-cited: "
            f"[red]{n_judge_higher}/{len(deltas)} "
            f"({n_judge_higher/len(deltas)*100:.0f}%)[/red]"
        )
        console.print(
            f"  Tied: [yellow]{n_tied}/{len(deltas)}[/yellow]"
        )

    # ── save results ──────────────────────────────────
    out_path = Path("results/ground_truth/citation_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "study_name":            "citation_comparison",
        "n_queries":             len(astrorag_citations),
        "astrorag_citations":    astrorag_citations,
        "judge_top1_citations":  judge_top1_citations,
        "judge_top3_citations":  judge_top3_citations,
        "median_astrorag":       statistics.median(astrorag_citations) if astrorag_citations else 0,
        "median_judge_top1":     statistics.median(judge_top1_citations) if judge_top1_citations else 0,
        "median_judge_top3":     statistics.median(judge_top3_citations) if judge_top3_citations else 0,
        "mean_astrorag":         statistics.mean(astrorag_citations) if astrorag_citations else 0,
        "mean_judge_top1":       statistics.mean(judge_top1_citations) if judge_top1_citations else 0,
        "max_astrorag":          max(astrorag_citations) if astrorag_citations else 0,
        "max_judge_top1":        max(judge_top1_citations) if judge_top1_citations else 0,
        "deltas":                deltas,
        "n_astrorag_higher":     sum(1 for d in deltas if d > 0),
        "n_judge_higher":        sum(1 for d in deltas if d < 0),
        "n_tied":                sum(1 for d in deltas if d == 0),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    console.print(f"\n[green]Saved to {out_path}[/green]")


if __name__ == "__main__":
    main()
