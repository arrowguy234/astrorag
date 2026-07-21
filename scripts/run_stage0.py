#!/usr/bin/env python
"""
CLI to test Stage 0 query decomposition.

Usage:
    python scripts/run_stage0.py "How do AGN jets suppress star formation?"
    python scripts/run_stage0.py --rules "What causes cooling flows?"
    python scripts/run_stage0.py --file tests/test_queries.toml
"""

from __future__ import annotations

import argparse
import sys
from   pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel   import Panel

from astrorag.logger        import setup_logging
from astrorag.stages        import Stage0Decompose

console = Console()


DEFAULT_QUERIES = [
    "How do AGN jets suppress star formation in massive elliptical galaxies through X-ray cavity observations?",
    "What is the relationship between galaxy stellar mass and central supermassive black hole mass?",
    "How does the intracluster medium cool in galaxy clusters and what regulates the cooling flow?",
    "How are photometric redshifts calibrated using spectroscopic survey data from SDSS and DESI?",
    "What mechanisms quench star formation in massive early-type galaxies at high redshift?",
]


def run_query(stage: Stage0Decompose, query: str, rules_only: bool) -> None:
    console.rule(f"[cyan]{query[:80]}[/cyan]")
    result = stage.run(query=query, rule_based_only=rules_only)
    d = result.decomposition

    body = (
        f"[bold]Q1[/bold] (mechanism)     : {d.sub_questions['Q1']}\n"
        f"[bold]Q2[/bold] (evidence)      : {d.sub_questions['Q2']}\n"
        f"[bold]Q3[/bold] (quantitative)  : {d.sub_questions['Q3']}\n\n"
        f"Wavelength : [yellow]{d.wavelength}[/yellow]\n"
        f"Catalogs   : {', '.join(d.catalogs) or '(none)'}\n"
        f"Type       : {d.query_type}\n\n"
        f"Method     : {'RULES' if result.fallback_used else 'LLM'}\n"
        f"Time       : {result.total_time_s:.2f}s"
    )
    if result.llm_response and not result.fallback_used:
        r = result.llm_response
        body += (
            f"\nTokens     : in={r.input_tokens} out={r.output_tokens}\n"
            f"Retries    : {r.retries}"
        )

    console.print(Panel(body, border_style="green"))
    console.print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None,
                        help="Query text (omit to run default set)")
    parser.add_argument("--rules", action="store_true",
                        help="Skip LLM, use rule-based only")
    parser.add_argument("--all-defaults", action="store_true",
                        help="Run all default test queries")
    args = parser.parse_args()

    setup_logging(level="INFO")
    stage = Stage0Decompose()

    if args.all_defaults or args.query is None:
        console.print(Panel.fit(
            "[bold cyan]Running default query set[/bold cyan]",
            border_style="cyan",
        ))
        for q in DEFAULT_QUERIES:
            run_query(stage, q, rules_only=args.rules)
    else:
        run_query(stage, args.query, rules_only=args.rules)


if __name__ == "__main__":
    main()