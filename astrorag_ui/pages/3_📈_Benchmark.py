"""
Benchmark page — display 20-query evaluation and ablation results.
"""

from pathlib import Path
import json

import pandas as pd
import streamlit as st


st.set_page_config(page_title="AstroRAG — Benchmark", page_icon="📈", layout="wide")


# ══════════════════════════════════════════════════════════
# data
# ══════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data
def load_benchmark():
    path = DATA_DIR / "eval_full.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_ablation():
    ablation_dir = DATA_DIR / "ablation"
    if not ablation_dir.exists():
        return {}
    results = {}
    for p in sorted(ablation_dir.glob("ablation_*.json")):
        variant = p.stem.replace("ablation_", "")
        with open(p, encoding="utf-8") as fh:
            results[variant] = json.load(fh)
    return results


benchmark = load_benchmark()
ablation  = load_ablation()


# ══════════════════════════════════════════════════════════
# header
# ══════════════════════════════════════════════════════════

st.markdown("# 📈 Benchmark Results")

if not benchmark and not ablation:
    st.error("No benchmark data found in `data/eval_full.json` or `data/ablation/`.")
    st.stop()


# ══════════════════════════════════════════════════════════
# tabs
# ══════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊 20-Query Evaluation",
    "🔬 Ablation Study",
    "⏱ Latency Breakdown",
])


# ── Tab 1: 20-query eval ────────────────────────
with tab1:
    if not benchmark:
        st.warning("No 20-query benchmark data available.")
    else:
        traces = benchmark.get("traces", [])
        succeeded = [t for t in traces if t.get("success")]
        with_s5 = [t for t in succeeded if t.get("stage5")]

        st.markdown("### Aggregate Metrics")

        # summary
        n_total = len(traces)
        n_ok = len(succeeded)
        n_accept = sum(1 for t in with_s5
                       if t.get("stage5", {}).get("decision") == "ACCEPT")

        mean_q = sum(t.get("stage5", {}).get("q_total", 0)
                     for t in with_s5) / len(with_s5) if with_s5 else 0
        mean_qf = sum(t.get("stage5", {}).get("q_f", 0)
                      for t in with_s5) / len(with_s5) if with_s5 else 0
        mean_qc = sum(t.get("stage5", {}).get("q_c", 0)
                      for t in with_s5) / len(with_s5) if with_s5 else 0
        mean_qi = sum(t.get("stage5", {}).get("q_i", 0)
                      for t in with_s5) / len(with_s5) if with_s5 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Queries", f"{n_ok}/{n_total}")
        c2.metric("ACCEPT rate", f"{n_accept/n_total*100:.1f}%")
        c3.metric("Mean Q_total", f"{mean_q:.3f}")
        c4.metric("Success rate", f"{n_ok/n_total*100:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Mean Q_f", f"{mean_qf:.3f}")
        c6.metric("Mean Q_c", f"{mean_qc:.3f}")
        c7.metric("Mean Q_i", f"{mean_qi:.3f}")
        total_wall = benchmark.get("total_wall_s", 0)
        c8.metric("Total wall time", f"{total_wall/60:.1f} min")

        st.divider()

        # per-query table
        st.markdown("### Per-Query Results")

        table_data = []
        for t in traces:
            s5 = t.get("stage5") or {}
            table_data.append({
                "Idx":       t.get("query_idx", ""),
                "Subdomain": t.get("subdomain", ""),
                "Query":     t.get("query", "")[:80]
                             + ("..." if len(t.get("query", "")) > 80 else ""),
                "Selected":  s5.get("final_arxiv_id", "n/a"),
                "Q_total":   round(s5.get("q_total", 0), 3),
                "Decision":  s5.get("decision", "FAIL"),
                "Time (s)":  round(t.get("total_seconds", 0), 1),
            })

        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Q_total": st.column_config.ProgressColumn(
                    "Q_total", min_value=0.0, max_value=1.0, format="%.3f",
                ),
            },
            height=500,
        )


# ── Tab 2: Ablation ─────────────────────────────
with tab2:
    if not ablation:
        st.warning("No ablation data available.")
    else:
        st.markdown("### Six-Variant Ablation Study")
        st.markdown(
            "Each variant runs the same 20-query benchmark with one pipeline "
            "component disabled, isolating its contribution."
        )

        # aggregate metrics per variant
        variant_metrics = {}
        for name, result in ablation.items():
            traces = result.get("traces", [])
            with_s5 = [t for t in traces
                       if t.get("success") and t.get("stage5")]
            if not with_s5:
                continue
            variant_metrics[name] = {
                "n_queries":    len(traces),
                "accept_rate":  sum(1 for t in with_s5
                                    if t.get("stage5", {}).get("decision") == "ACCEPT")
                                / len(traces),
                "mean_q_total": sum(t.get("stage5", {}).get("q_total", 0)
                                    for t in with_s5) / len(with_s5),
                "mean_q_f":     sum(t.get("stage5", {}).get("q_f", 0)
                                    for t in with_s5) / len(with_s5),
                "mean_q_c":     sum(t.get("stage5", {}).get("q_c", 0)
                                    for t in with_s5) / len(with_s5),
                "mean_q_i":     sum(t.get("stage5", {}).get("q_i", 0)
                                    for t in with_s5) / len(with_s5),
                "mean_time":    sum(t.get("total_seconds", 0)
                                    for t in traces) / len(traces),
            }

        # compute overlap with 'full' baseline
        if "full" in variant_metrics:
            baseline_traces = ablation.get("full", {}).get("traces", [])
            baseline_papers = {
                t.get("query_idx"): t.get("stage5", {}).get("final_arxiv_id")
                for t in baseline_traces if t.get("success") and t.get("stage5")
            }

            for name in variant_metrics:
                if name == "full":
                    variant_metrics[name]["overlap"] = 1.0
                    continue
                var_traces = ablation[name].get("traces", [])
                var_papers = {
                    t.get("query_idx"): t.get("stage5", {}).get("final_arxiv_id")
                    for t in var_traces if t.get("success") and t.get("stage5")
                }
                common = set(baseline_papers) & set(var_papers)
                if common:
                    matches = sum(1 for i in common
                                  if baseline_papers[i] == var_papers[i])
                    variant_metrics[name]["overlap"] = matches / len(common)
                else:
                    variant_metrics[name]["overlap"] = 0.0

        # ablation table
        variant_order = ["full", "no_graph", "no_llm_rerank",
                         "no_quality_gate", "no_pdf", "bm25_only"]
        table_rows = []
        for name in variant_order:
            if name not in variant_metrics:
                continue
            m = variant_metrics[name]
            table_rows.append({
                "Variant":       name,
                "Accept %":      round(m["accept_rate"] * 100, 1),
                "Q_total":       round(m["mean_q_total"], 3),
                "Q_f":           round(m["mean_q_f"], 3),
                "Q_c":           round(m["mean_q_c"], 3),
                "Q_i":           round(m["mean_q_i"], 3),
                "Overlap %":     round(m.get("overlap", 0) * 100, 1),
                "Time (s)":      round(m["mean_time"], 1),
            })

        df = pd.DataFrame(table_rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Q_total": st.column_config.ProgressColumn(
                    "Q_total", min_value=0.0, max_value=1.0, format="%.3f",
                ),
                "Accept %": st.column_config.ProgressColumn(
                    "Accept %", min_value=0.0, max_value=100.0, format="%.1f",
                ),
                "Overlap %": st.column_config.ProgressColumn(
                    "Overlap %", min_value=0.0, max_value=100.0, format="%.1f",
                ),
            },
        )

        st.divider()

        # deltas from baseline
        if "full" in variant_metrics:
            st.markdown("### Deltas from Baseline (Full Pipeline)")
            baseline_m = variant_metrics["full"]
            delta_rows = []
            for name in variant_order:
                if name == "full" or name not in variant_metrics:
                    continue
                m = variant_metrics[name]
                delta_rows.append({
                    "Variant":     name,
                    "ΔAccept":     round((m["accept_rate"] - baseline_m["accept_rate"]) * 100, 1),
                    "ΔQ_total":    round(m["mean_q_total"] - baseline_m["mean_q_total"], 3),
                    "ΔQ_f":        round(m["mean_q_f"] - baseline_m["mean_q_f"], 3),
                    "ΔQ_c":        round(m["mean_q_c"] - baseline_m["mean_q_c"], 3),
                    "ΔQ_i":        round(m["mean_q_i"] - baseline_m["mean_q_i"], 3),
                    "ΔTime (s)":   round(m["mean_time"] - baseline_m["mean_time"], 1),
                })

            st.dataframe(pd.DataFrame(delta_rows),
                         use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("""
### Key Findings

- **PDF extraction essential** — removing Stage 4 causes Q_total to drop by 0.657
- **BM25-only baseline fails** — Q_total collapses to 0.303 (0% acceptance)
- **LLM reranking materially changes retrieval** — 70% of queries pick a different paper
- **Quality gate provides reliability** — marginal Q_total impact but prevents low-quality output
- **Graph provides small marginal contribution** on this benchmark
""")


# ── Tab 3: Latency ──────────────────────────────
with tab3:
    if not benchmark:
        st.warning("No benchmark data available.")
    else:
        traces = benchmark.get("traces", [])
        succeeded = [t for t in traces if t.get("success")]

        st.markdown("### Per-Stage Latency (mean over 20 queries)")

        stage_names = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5"]
        stage_labels = {
            "stage0": "Stage 0 — decomposition",
            "stage1": "Stage 1 — BM25",
            "stage2": "Stage 2 — graph + PPR",
            "stage3": "Stage 3 — LLM rerank",
            "stage4": "Stage 4 — PDF fetch",
            "stage5": "Stage 5 — summarisation",
        }

        latencies = []
        for stage in stage_names:
            vals = []
            for t in succeeded:
                s = t.get(stage)
                if s:
                    if stage == "stage4":
                        vals.append(s.get("fetch_seconds", 0) + s.get("parse_seconds", 0))
                    else:
                        vals.append(s.get("latency_s", 0))
            if vals:
                latencies.append({
                    "Stage":    stage_labels[stage],
                    "Mean (s)": round(sum(vals) / len(vals), 2),
                    "Min (s)":  round(min(vals), 2),
                    "Max (s)":  round(max(vals), 2),
                })

        # total
        totals = [t.get("total_seconds", 0) for t in succeeded]
        if totals:
            latencies.append({
                "Stage":    "**Total (end-to-end)**",
                "Mean (s)": round(sum(totals) / len(totals), 2),
                "Min (s)":  round(min(totals), 2),
                "Max (s)":  round(max(totals), 2),
            })

        st.dataframe(pd.DataFrame(latencies),
                     use_container_width=True, hide_index=True)

        st.info("""
**Note:** Stage 5 dominates latency due to Groq free-tier rate limits
(6,000 tokens/minute). On a paid tier or self-hosted model, Stage 5 would
drop to ~5-10 seconds per query, bringing total latency under 30 seconds.
""")
