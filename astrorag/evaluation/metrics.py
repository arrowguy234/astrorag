"""
Aggregate metrics computation over EvaluationResult.

Produces the numbers you cite in the paper:
- Accept rate  (fraction of queries with Stage 5 ACCEPT)
- Mean Q_f / Q_c / Q_i / Q_total
- Mean latency per stage
- Re-selection rate
- Median top-BM25 score (retrieval strength)
- Density of graph (context clustering)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing      import Any

from astrorag.evaluation.models import EvaluationResult, QueryTrace


# ══════════════════════════════════════════════════════════
# aggregate metrics
# ══════════════════════════════════════════════════════════

@dataclass
class EvaluationMetrics:
    """Aggregate metrics across all queries in an EvaluationResult."""

    n_queries:     int = 0
    n_succeeded:   int = 0
    n_failed:      int = 0
    success_rate:  float = 0.0

    # quality
    accept_rate:   float = 0.0
    retry_rate:    float = 0.0
    reselect_rate: float = 0.0
    mean_q_f:      float = 0.0
    mean_q_c:      float = 0.0
    mean_q_i:      float = 0.0
    mean_q_total:  float = 0.0

    # summary content
    frac_with_equations:          float = 0.0
    frac_with_numerical_results:  float = 0.0
    mean_n_equations:             float = 0.0
    mean_n_numerical_results:     float = 0.0
    mean_n_instruments:           float = 0.0

    # retrieval
    mean_bm25_top_score:  float = 0.0
    mean_ppr_iterations:  float = 0.0
    mean_graph_density:   float = 0.0

    # cost
    mean_n_attempts:      float = 0.0
    mean_n_reselections:  float = 0.0
    mean_stage5_attempts_when_failed: float = 0.0

    # latency (seconds)
    mean_stage0_latency:  float = 0.0
    mean_stage1_latency:  float = 0.0
    mean_stage2_latency:  float = 0.0
    mean_stage3_latency:  float = 0.0
    mean_stage4_latency:  float = 0.0
    mean_stage5_latency:  float = 0.0
    mean_total_latency:   float = 0.0
    median_total_latency: float = 0.0

    # per-subdomain accept rate
    accept_rate_by_subdomain: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        return d


# ══════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════

def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _frac(xs: list[bool]) -> float:
    return sum(1 for x in xs if x) / len(xs) if xs else 0.0


# ══════════════════════════════════════════════════════════
# main entry
# ══════════════════════════════════════════════════════════

def compute_metrics(result: EvaluationResult) -> EvaluationMetrics:
    """
    Compute aggregate metrics over an EvaluationResult.
    """
    traces = result.traces
    n = len(traces)
    succeeded = [t for t in traces if t.success]
    failed    = [t for t in traces if not t.success]

    m = EvaluationMetrics(
        n_queries    = n,
        n_succeeded  = len(succeeded),
        n_failed     = len(failed),
        success_rate = len(succeeded) / n if n else 0.0,
    )

    # traces with stage 5 output
    with_s5 = [t for t in succeeded if t.stage5 is not None]

    if with_s5:
        # ── quality metrics ────────────────────────────
        decisions = [t.stage5.decision for t in with_s5]
        m.accept_rate   = decisions.count("ACCEPT")    / len(decisions)
        m.retry_rate    = decisions.count("RETRY")     / len(decisions)
        m.reselect_rate = decisions.count("RE-SELECT") / len(decisions)

        m.mean_q_f     = _mean([t.stage5.q_f     for t in with_s5])
        m.mean_q_c     = _mean([t.stage5.q_c     for t in with_s5])
        m.mean_q_i     = _mean([t.stage5.q_i     for t in with_s5])
        m.mean_q_total = _mean([t.stage5.q_total for t in with_s5])

        # ── summary content ────────────────────────────
        m.frac_with_equations = _frac(
            [t.stage5.has_equations for t in with_s5]
        )
        m.frac_with_numerical_results = _frac(
            [t.stage5.has_numerical_results for t in with_s5]
        )
        m.mean_n_equations = _mean(
            [float(t.stage5.n_equations) for t in with_s5]
        )
        m.mean_n_numerical_results = _mean(
            [float(t.stage5.n_numerical_results) for t in with_s5]
        )
        m.mean_n_instruments = _mean(
            [float(t.stage5.n_instruments) for t in with_s5]
        )

        # ── cost ───────────────────────────────────────
        m.mean_n_attempts     = _mean(
            [float(t.stage5.n_attempts) for t in with_s5]
        )
        m.mean_n_reselections = _mean(
            [float(t.stage5.n_reselections) for t in with_s5]
        )

        m.mean_stage5_latency = _mean(
            [t.stage5.latency_s for t in with_s5]
        )

    # ── retrieval ──────────────────────────────────────
    with_s1 = [t for t in succeeded if t.stage1 is not None]
    if with_s1:
        m.mean_bm25_top_score = _mean(
            [t.stage1.top_score for t in with_s1]
        )
        m.mean_stage1_latency = _mean(
            [t.stage1.latency_s for t in with_s1]
        )

    # ── graph ──────────────────────────────────────────
    with_s2 = [t for t in succeeded if t.stage2 is not None]
    if with_s2:
        m.mean_ppr_iterations = _mean(
            [float(t.stage2.ppr_iterations) for t in with_s2]
        )
        m.mean_graph_density = _mean(
            [t.stage2.density for t in with_s2]
        )
        m.mean_stage2_latency = _mean(
            [t.stage2.latency_s for t in with_s2]
        )

    # ── per-stage latency ──────────────────────────────
    for stage_key, attr in [
        ("stage0", "mean_stage0_latency"),
        ("stage3", "mean_stage3_latency"),
        ("stage4", "mean_stage4_latency"),
    ]:
        vals = [
            getattr(t, stage_key).latency_s if stage_key != "stage4"
            else getattr(t, stage_key).fetch_seconds
                 + getattr(t, stage_key).parse_seconds
            for t in succeeded
            if getattr(t, stage_key) is not None
        ]
        setattr(m, attr, _mean(vals))

    # ── total latency ──────────────────────────────────
    totals = [t.total_seconds for t in succeeded]
    m.mean_total_latency   = _mean(totals)
    m.median_total_latency = statistics.median(totals) if totals else 0.0

    # ── per-subdomain accept rate ──────────────────────
    subdomains: dict[str, list[bool]] = {}
    for t in with_s5:
        subdomains.setdefault(t.subdomain, []).append(
            t.stage5.decision == "ACCEPT"
        )
    m.accept_rate_by_subdomain = {
        k: _frac(v) for k, v in subdomains.items()
    }

    return m


# ══════════════════════════════════════════════════════════
# rendering
# ══════════════════════════════════════════════════════════

def format_metrics_table(m: EvaluationMetrics) -> str:
    """Render metrics as a plain-text table for CLI display."""
    lines = [
        "═" * 60,
        "  AstroRAG Evaluation Metrics",
        "═" * 60,
        "",
        f"  Queries               : {m.n_queries}",
        f"  Succeeded             : {m.n_succeeded}",
        f"  Failed                : {m.n_failed}",
        f"  Success rate          : {m.success_rate:.1%}",
        "",
        "  Quality gate:",
        f"    ACCEPT rate         : {m.accept_rate:.1%}",
        f"    RETRY rate          : {m.retry_rate:.1%}",
        f"    RE-SELECT rate      : {m.reselect_rate:.1%}",
        f"    Mean Q_f (faithful) : {m.mean_q_f:.3f}",
        f"    Mean Q_c (coverage) : {m.mean_q_c:.3f}",
        f"    Mean Q_i (consistent): {m.mean_q_i:.3f}",
        f"    Mean Q_total        : {m.mean_q_total:.3f}",
        "",
        "  Summary content:",
        f"    With equations      : {m.frac_with_equations:.1%}",
        f"    With numerical      : {m.frac_with_numerical_results:.1%}",
        f"    Mean # equations    : {m.mean_n_equations:.1f}",
        f"    Mean # numerical    : {m.mean_n_numerical_results:.1f}",
        f"    Mean # instruments  : {m.mean_n_instruments:.1f}",
        "",
        "  Retrieval:",
        f"    Mean top BM25       : {m.mean_bm25_top_score:.2f}",
        f"    Mean graph density  : {m.mean_graph_density:.1%}",
        f"    Mean PPR iterations : {m.mean_ppr_iterations:.1f}",
        "",
        "  Cost:",
        f"    Mean # attempts     : {m.mean_n_attempts:.2f}",
        f"    Mean # reselections : {m.mean_n_reselections:.2f}",
        "",
        "  Latency (s):",
        f"    Mean Stage 0        : {m.mean_stage0_latency:.2f}",
        f"    Mean Stage 1        : {m.mean_stage1_latency:.2f}",
        f"    Mean Stage 2        : {m.mean_stage2_latency:.2f}",
        f"    Mean Stage 3        : {m.mean_stage3_latency:.2f}",
        f"    Mean Stage 4        : {m.mean_stage4_latency:.2f}",
        f"    Mean Stage 5        : {m.mean_stage5_latency:.2f}",
        f"    Mean total          : {m.mean_total_latency:.2f}",
        f"    Median total        : {m.median_total_latency:.2f}",
        "",
    ]

    if m.accept_rate_by_subdomain:
        lines.append("  ACCEPT rate by subdomain:")
        for sd, rate in sorted(m.accept_rate_by_subdomain.items()):
            lines.append(f"    {sd:<20} : {rate:.1%}")
        lines.append("")

    lines.append("═" * 60)
    return "\n".join(lines)