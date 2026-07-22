"""
Papers page — browse the context library.
"""

from pathlib import Path
import json

import pandas as pd
import streamlit as st


st.set_page_config(page_title="AstroRAG — Papers", page_icon="📊", layout="wide")


# ══════════════════════════════════════════════════════════
# data
# ══════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data
def load_library():
    path = DATA_DIR / "context_library.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("entries", {})


library = load_library()

if not library:
    st.error("Context library is empty. Populate `data/context_library.json` first.")
    st.stop()


# ══════════════════════════════════════════════════════════
# header
# ══════════════════════════════════════════════════════════

st.markdown("# 📊 Context Library")
st.caption("All papers AstroRAG has processed, with structured summaries and extracted content.")


# ══════════════════════════════════════════════════════════
# filters
# ══════════════════════════════════════════════════════════

entries = list(library.values())

col1, col2 = st.columns([2, 1])

with col1:
    search = st.text_input(
        "🔍 Search",
        placeholder="Filter by arxiv ID, query, subdomain, or title...",
    )

with col2:
    all_subdomains = ["(all)"] + sorted(
        {e.get("subdomain", "") for e in entries if e.get("subdomain")}
    )
    subdomain_filter = st.selectbox("Subdomain", all_subdomains)


# apply filters
def matches(e, kw):
    if not kw:
        return True
    kw = kw.lower()
    return any(kw in str(e.get(f, "")).lower()
               for f in ["arxiv_id", "title", "original_query", "paper_overview", "subdomain"])

filtered = [e for e in entries if matches(e, search)]
if subdomain_filter != "(all)":
    filtered = [e for e in filtered if e.get("subdomain") == subdomain_filter]


st.markdown(f"### {len(filtered)} papers")


# ══════════════════════════════════════════════════════════
# summary table
# ══════════════════════════════════════════════════════════

if filtered:
    table_data = []
    for e in sorted(filtered, key=lambda x: x.get("q_total", 0), reverse=True):
        table_data.append({
            "arXiv":     e.get("arxiv_id", ""),
            "Subdomain": e.get("subdomain", "n/a"),
            "Q_total":   round(e.get("q_total", 0), 3),
            "Decision":  e.get("decision", ""),
            "📐":         len(e.get("key_equations", [])),
            "🔢":         len(e.get("numerical_results", [])),
            "💬":         len(e.get("chat_sessions", [])),
            "Query":     (e.get("original_query", "")[:80]
                          + ("..." if len(e.get("original_query", "")) > 80 else "")),
        })

    df = pd.DataFrame(table_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Q_total": st.column_config.ProgressColumn(
                "Q_total",
                help="Composite quality score",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
        },
        height=300,
    )


st.divider()


# ══════════════════════════════════════════════════════════
# paper detail viewer
# ══════════════════════════════════════════════════════════

st.markdown("## 📖 Paper Details")

paper_options = {
    f"arXiv:{e.get('arxiv_id')} — {(e.get('title') or e.get('original_query', ''))[:80]}":
        e.get("arxiv_id")
    for e in sorted(filtered, key=lambda x: x.get("updated_at", ""), reverse=True)
}

if not paper_options:
    st.info("No papers match your filters.")
    st.stop()

selected_label = st.selectbox(
    "Select a paper",
    list(paper_options.keys()),
)
selected_arxiv = paper_options[selected_label]
entry = library[selected_arxiv]


# ── header ─────────────────────────────────────
decision = entry.get("decision", "")
decision_emoji = {"ACCEPT": "🟢", "RETRY": "🟡", "RE-SELECT": "🔴"}.get(decision, "⚪")

st.markdown(f"""
<div style="background:#f8f9fa; padding:20px; border-radius:10px;
            border-left:5px solid #1e3a5f; margin-bottom:20px;">
  <h2 style="margin:0;">arXiv:{entry.get('arxiv_id')}</h2>
  <p style="margin:8px 0 0 0; color:#495057;">
    <em>{entry.get('original_query', '')}</em>
  </p>
  <p style="margin:5px 0 0 0; font-size:0.9em;">
    <strong>{decision_emoji} {decision}</strong> &nbsp;•&nbsp;
    Q_total = <strong>{entry.get('q_total', 0):.3f}</strong> &nbsp;•&nbsp;
    Subdomain: <em>{entry.get('subdomain', 'n/a')}</em>
  </p>
</div>
""", unsafe_allow_html=True)


# ── metrics row ────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("PDF Pages", entry.get("pdf_pages", 0))
m2.metric("Sections", entry.get("n_sections", 0))
m3.metric("Equations", len(entry.get("key_equations", [])))
m4.metric("Numerical", len(entry.get("numerical_results", [])))
m5.metric("Time", f"{entry.get('total_seconds', 0):.1f}s")


# ── action row ─────────────────────────────────
if st.button("💬 Chat with this paper", type="primary", use_container_width=True):
    st.session_state.current_arxiv_id = selected_arxiv
    st.session_state.chat_messages = []
    st.switch_page("pages/2_💬_Chat.py")


st.divider()


# ══════════════════════════════════════════════════════════
# tabbed detail view
# ══════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Summary",
    "📐 Equations",
    "🔢 Numerical",
    "✅ Quality Gate",
    "🔍 Raw JSON",
])


# ── Tab 1: Summary ──────────────────────────────
with tab1:
    st.markdown("### Paper Overview")
    if entry.get("paper_overview"):
        st.info(entry["paper_overview"])
    else:
        st.warning("No overview available.")

    if entry.get("instruments"):
        st.markdown(f"**🔭 Instruments:** {', '.join(entry['instruments'])}")

    if entry.get("evidence_type"):
        st.markdown(f"**Evidence type:** `{entry['evidence_type']}`")

    st.divider()

    st.markdown("### Sub-Question Answers")

    for qk in ["Q1", "Q2", "Q3"]:
        sqa = entry.get("sub_question_answers", {}).get(qk)
        if not sqa:
            continue
        check = "✅" if sqa.get("answered") else "❌"
        label = {"Q1": "Mechanism", "Q2": "Evidence", "Q3": "Quantitative"}[qk]
        section = sqa.get("section", "unknown")

        with st.expander(f"{check} {qk} — {label}   _(Section: {section})_", expanded=True):
            st.write(sqa.get("answer_text", "No answer."))

    if entry.get("methodology"):
        st.divider()
        st.markdown("### Methodology")
        st.write(entry["methodology"])

    if entry.get("key_findings"):
        st.divider()
        st.markdown("### Key Findings")
        for f in entry["key_findings"]:
            st.markdown(f"- {f}")

    if entry.get("key_snippet"):
        st.divider()
        st.markdown("### Verbatim Key Snippet")
        st.info(f"> _{entry['key_snippet']}_")


# ── Tab 2: Equations ────────────────────────────
with tab2:
    st.markdown("### 📐 Extracted Equations")
    equations = entry.get("key_equations", [])
    if equations:
        st.write(f"Extracted **{len(equations)}** equations from the paper.")
        for i, eq in enumerate(equations, 1):
            with st.container(border=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**Equation {i}**")
                    st.code(eq.get("equation", ""), language="latex")
                with col2:
                    st.markdown(f"**Variables:**")
                    st.write(eq.get("variables", "n/a"))
    else:
        st.warning("No equations extracted for this paper.")


# ── Tab 3: Numerical ────────────────────────────
with tab3:
    st.markdown("### 🔢 Numerical Results")
    numerical = entry.get("numerical_results", [])
    if numerical:
        df = pd.DataFrame(numerical)
        cols = ["quantity", "value", "uncertainty", "unit"]
        df = df[[c for c in cols if c in df.columns]]
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Extracted {len(numerical)} numerical measurements from the paper.")
    else:
        st.warning("No numerical results extracted for this paper.")


# ── Tab 4: Quality Gate ─────────────────────────
with tab4:
    q_total = entry.get("q_total", 0)
    q_f     = entry.get("q_f", 0)
    q_c     = entry.get("q_c", 0)
    q_i     = entry.get("q_i", 0)

    decision_color = {"ACCEPT": "green", "RETRY": "orange", "RE-SELECT": "red"}
    color = decision_color.get(decision, "gray")

    st.markdown(f"### Quality Gate: :{color}[{decision}] — Q_total = {q_total:.3f}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Q_f — Faithfulness (w=0.40)**")
        st.progress(q_f, text=f"{q_f:.3f}")
        st.caption("Fraction of claim sentences with paper-word-overlap > 0.40")

    with col2:
        st.markdown("**Q_c — Coverage (w=0.35)**")
        st.progress(q_c, text=f"{q_c:.3f}")
        st.caption("Fraction of sub-questions answered")

    with col3:
        st.markdown("**Q_i — Consistency (w=0.25)**")
        st.progress(q_i, text=f"{q_i:.3f}")
        st.caption("Snippet + evidence + technical-density penalties")

    st.divider()
    st.markdown("**Composite Q_total**")
    st.progress(q_total, text=f"{q_total:.3f}")

    st.info("""
**Decision thresholds:**
- 🟢 ACCEPT: `Q_total ≥ 0.75`
- 🟡 RETRY: `0.50 ≤ Q_total < 0.75`
- 🔴 RE-SELECT: `Q_total < 0.50`
""")


# ── Tab 5: Raw ──────────────────────────────────
with tab5:
    st.markdown("### 🔍 Raw JSON")
    st.json(entry, expanded=False)
