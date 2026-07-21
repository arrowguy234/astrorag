"""
Cross-variant comparison metrics for the ablation study.

Compares each variant's EvaluationResult against the baseline
("full") variant and reports delta metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from astrorag.evaluation.metrics import (
    EvaluationMetrics,
    compute_metrics,
)
from astrorag.evaluation.models  import EvaluationResult


@dataclass
class VariantComparison:
    """One variant's metrics with deltas from baseline."""
    variant_name:   str
    n_queries:      int
    accept_rate:    float
    mean_q_total:   float
    mean_q_f:       float
    mean_q_c:       float
    mean_q_i:       float
    mean_n_equations: float
    mean_n_numerical: float
    mean_total_latency: float

    # deltas from baseline
    delta_accept_rate: float = 0.0
    delta_q_total:     float = 0.0
    delta_q_f:         float = 0.0
    delta_q_c:         float = 0.0
    delta_q_i:         float = 0.0
    delta_latency:     float = 0.0

    # retrieval overlap with baseline
    retrieval_overlap: float = 0.0   # fraction of queries selecting same arxiv_id


# ══════════════════════════════════════════════════════════
# comparison logic
# ══════════════════════════════════════════════════════════

def compute_variant_comparison(
    variant_result:  EvaluationResult,
    baseline_result: EvaluationResult,
) -> VariantComparison:
    """
    Compute a single variant's metrics with deltas from baseline.

    Uses paired comparison (same query indices) for the deltas.
    """
    v_metrics = compute_metrics(variant_result)
    b_metrics = compute_metrics(baseline_result)

    # retrieval overlap — how often did this variant pick the same paper?
    baseline_by_idx = {
        t.query_idx: (t.stage5.final_arxiv_id if t.stage5 else None)
        for t in baseline_result.traces if t.success
    }
    variant_by_idx = {
        t.query_idx: (t.stage5.final_arxiv_id if t.stage5 else None)
        for t in variant_result.traces if t.success
    }
    common_idxs = set(baseline_by_idx) & set(variant_by_idx)
    if common_idxs:
        overlap = sum(
            1 for idx in common_idxs
            if baseline_by_idx[idx] == variant_by_idx[idx]
        ) / len(common_idxs)
    else:
        overlap = 0.0

    return VariantComparison(
        variant_name       = variant_result.query_set_name.replace("ablation_", ""),
        n_queries          = v_metrics.n_queries,
        accept_rate        = v_metrics.accept_rate,
        mean_q_total       = v_metrics.mean_q_total,
        mean_q_f           = v_metrics.mean_q_f,
        mean_q_c           = v_metrics.mean_q_c,
        mean_q_i           = v_metrics.mean_q_i,
        mean_n_equations   = v_metrics.mean_n_equations,
        mean_n_numerical   = v_metrics.mean_n_numerical_results,
        mean_total_latency = v_metrics.mean_total_latency,
        delta_accept_rate  = v_metrics.accept_rate  - b_metrics.accept_rate,
        delta_q_total      = v_metrics.mean_q_total - b_metrics.mean_q_total,
        delta_q_f          = v_metrics.mean_q_f     - b_metrics.mean_q_f,
        delta_q_c          = v_metrics.mean_q_c     - b_metrics.mean_q_c,
        delta_q_i          = v_metrics.mean_q_i     - b_metrics.mean_q_i,
        delta_latency      = v_metrics.mean_total_latency - b_metrics.mean_total_latency,
        retrieval_overlap  = overlap,
    )


# ══════════════════════════════════════════════════════════
# formatting
# ══════════════════════════════════════════════════════════

def format_ablation_table(comparisons: list[VariantComparison]) -> str:
    """
    Render the cross-variant comparison as a plain-text table.

    Baseline row appears first with zero deltas.
    """
    header = "═" * 100
    lines = [
        header,
        "  Ablation Study — Cross-Variant Comparison",
        header,
        "",
        f"  {'Variant':<20} {'Accept%':>8} {'Q_total':>8} {'Q_f':>7} "
        f"{'Q_c':>7} {'Q_i':>7} {'#Eq':>5} {'#Num':>5} {'Overlap':>8} {'Time(s)':>8}",
        "  " + "-" * 96,
    ]

    for c in comparisons:
        lines.append(
            f"  {c.variant_name:<20} "
            f"{c.accept_rate*100:>7.1f}% "
            f"{c.mean_q_total:>8.3f} "
            f"{c.mean_q_f:>7.3f} "
            f"{c.mean_q_c:>7.3f} "
            f"{c.mean_q_i:>7.3f} "
            f"{c.mean_n_equations:>5.1f} "
            f"{c.mean_n_numerical:>5.1f} "
            f"{c.retrieval_overlap*100:>7.1f}% "
            f"{c.mean_total_latency:>7.1f}"
        )

    lines.extend([
        "",
        "  Delta from baseline (positive = variant improves; negative = variant degrades):",
        "  " + "-" * 96,
        f"  {'Variant':<20} {'ΔAccept':>8} {'ΔQ_tot':>8} "
        f"{'ΔQ_f':>7} {'ΔQ_c':>7} {'ΔQ_i':>7} {'':>5} {'':>5} {'':>8} {'ΔTime':>8}",
        "  " + "-" * 96,
    ])

    for c in comparisons:
        sign = lambda x: f"+{x:.3f}" if x > 0 else f"{x:.3f}"
        pct  = lambda x: f"+{x*100:.1f}%" if x > 0 else f"{x*100:.1f}%"
        lines.append(
            f"  {c.variant_name:<20} "
            f"{pct(c.delta_accept_rate):>8} "
            f"{sign(c.delta_q_total):>8} "
            f"{sign(c.delta_q_f):>7} "
            f"{sign(c.delta_q_c):>7} "
            f"{sign(c.delta_q_i):>7} "
            f"{'':>5} {'':>5} {'':>8} "
            f"{sign(c.delta_latency):>8}"
        )

    lines.extend(["", header])
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# convenience loader
# ══════════════════════════════════════════════════════════

def load_all_variants(
    results_dir: Path,
    baseline_name: str = "full",
) -> tuple[EvaluationResult, dict[str, EvaluationResult]]:
    """
    Load all ablation variant result files from a directory.

    Expected naming: ablation_{variant_name}.json
    """
    variants: dict[str, EvaluationResult] = {}
    for p in sorted(results_dir.glob("ablation_*.json")):
        variant_name = p.stem.replace("ablation_", "")
        try:
            variants[variant_name] = EvaluationResult.load(p)
        except Exception as e:
            print(f"Failed to load {p}: {e}")

    if baseline_name not in variants:
        raise ValueError(
            f"Baseline '{baseline_name}' not found. "
            f"Have: {list(variants)}"
        )

    baseline = variants.pop(baseline_name)
    return baseline, variants