"""
AstroRAG — Streamlit UI (read-only).

This app displays results from AstroRAG runs that were executed
offline on the SDSU JupyterHub server. The full pipeline (corpus
loading, BM25, graph reranking, PDF extraction) does not run here.

Users can:
- Browse the context library (all processed papers)
- View structured summaries with equations and numerical results
- Chat with any paper (LLM calls only, no retrieval)
- View benchmark and ablation results
"""

from   pathlib import Path
import json

import streamlit as st


# ══════════════════════════════════════════════════════════
# page config
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "AstroRAG",
    page_icon  = "🌌",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════
# data loading (cached)
# ══════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_library():
    """Load the context library JSON."""
    path = DATA_DIR / "context_library.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("entries", {})


@st.cache_data
def load_benchmark():
    """Load the 20-query benchmark results."""
    path = DATA_DIR / "eval_full.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_ablation():
    """Load all ablation variant results."""
    ablation_dir = DATA_DIR / "ablation"
    if not ablation_dir.exists():
        return {}
    results = {}
    for p in sorted(ablation_dir.glob("ablation_*.json")):
        variant = p.stem.replace("ablation_", "")
        with open(p, encoding="utf-8") as fh:
            results[variant] = json.load(fh)
    return results


# ══════════════════════════════════════════════════════════
# session state
# ══════════════════════════════════════════════════════════

if "current_arxiv_id" not in st.session_state:
    st.session_state.current_arxiv_id = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ══════════════════════════════════════════════════════════
# home page content
# ══════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(90deg, #1e3a5f, #2c5282);
            color:white; padding:30px; border-radius:12px; margin-bottom:25px;">
  <h1 style="margin:0; font-size:2.8em;">🌌 AstroRAG</h1>
  <p style="margin:8px 0 0 0; font-size:1.3em; opacity:0.9;">
    Evidence-Aware Graph-Augmented Retrieval over 408,590 arXiv Astrophysics Papers
  </p>
  <p style="margin:15px 0 0 0; opacity:0.85; font-size:0.95em;">
    <strong>Author:</strong> Surinder Singh Chhabra
    &nbsp;•&nbsp; <strong>Institution:</strong> SDSU CSRC
    &nbsp;•&nbsp; <strong>Contact:</strong> schhabra@sdsu.edu
  </p>
</div>
""", unsafe_allow_html=True)


# quick stats row
library = load_library()
benchmark = load_benchmark()

if library:
    n_papers = len(library)
    n_with_eq = sum(1 for e in library.values() if e.get("key_equations"))
    n_with_num = sum(1 for e in library.values() if e.get("numerical_results"))
    mean_q = sum(e.get("q_total", 0) for e in library.values()) / n_papers if n_papers else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Papers Processed", n_papers)
    c2.metric("📐 With Equations", f"{n_with_eq}/{n_papers}")
    c3.metric("🔢 With Numerical", f"{n_with_num}/{n_papers}")
    c4.metric("✅ Mean Q_total", f"{mean_q:.3f}")


st.divider()


# ══════════════════════════════════════════════════════════
# about section
# ══════════════════════════════════════════════════════════

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### About AstroRAG")
    st.markdown("""
AstroRAG is a **six-stage retrieval-augmented generation pipeline** built for
astrophysics research. Given a natural-language research question, it:

1. **Decomposes** the query into three sub-questions (mechanism / evidence / quantitative)
2. **Retrieves** top-50 candidate papers via BM25 over the 408K-paper astro-ph corpus
3. **Reranks** using a four-signal citation graph and Personalised PageRank
4. **Selects** the best paper via graph-primed LLM reranking
5. **Extracts** full text with column-aware PDF parsing
6. **Summarises** with structured technical output (equations, values, instruments)

A composite quality gate ($Q_f$, $Q_c$, $Q_i$) validates the output and can trigger
re-selection from an LLM-provided fallback pool when quality is insufficient.

**This web UI is read-only.** Pipeline execution happens on the SDSU JupyterHub server
where the full 408K-paper corpus and BM25 index reside. Results are exported to a JSON
context library and displayed here.
""")

with col_right:
    st.markdown("### 📊 Benchmark Highlights")
    st.markdown("""
On a curated **20-query benchmark** spanning 10 astrophysics subdomains:

| Metric | Value |
|--------|-------|
| Success rate | **100%** (20/20) |
| Q_total mean | **0.981** |
| Papers with equations | **100%** |
| Papers with numerical | **100%** |
| Median latency | 78.9s |

**Six-variant ablation study** confirms:
- PDF extraction essential (ΔQ = −0.657 without)
- LLM rerank changes retrieval 70% of queries
- BM25-only baseline fails (Q_total = 0.303)

See the **📈 Benchmark** page for full details.
""")


st.divider()


# ══════════════════════════════════════════════════════════
# navigation cards
# ══════════════════════════════════════════════════════════

st.markdown("### Navigate")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
<div style="border:2px solid #1e3a5f; border-radius:10px; padding:20px;
            background:#f8f9fa; height:200px;">
  <h3 style="margin-top:0;">📊 Papers</h3>
  <p>Browse the context library. Every paper AstroRAG has processed with
  its structured summary, extracted equations, numerical results, and quality
  scores.</p>
</div>
""", unsafe_allow_html=True)
    if st.button("Open Papers →", use_container_width=True, key="nav_papers"):
        st.switch_page("pages/1_📊_Papers.py")

with nav_col2:
    st.markdown("""
<div style="border:2px solid #28a745; border-radius:10px; padding:20px;
            background:#f8f9fa; height:200px;">
  <h3 style="margin-top:0;">💬 Chat</h3>
  <p>Ask follow-up questions about any paper. The LLM has access to the paper's
  full summary, equations, numerical results, and methodology.</p>
</div>
""", unsafe_allow_html=True)
    if st.button("Open Chat →", use_container_width=True, key="nav_chat"):
        st.switch_page("pages/2_💬_Chat.py")

with nav_col3:
    st.markdown("""
<div style="border:2px solid #dc3545; border-radius:10px; padding:20px;
            background:#f8f9fa; height:200px;">
  <h3 style="margin-top:0;">📈 Benchmark</h3>
  <p>View the 20-query evaluation results, per-stage latency breakdown, and the
  six-variant ablation study establishing component contributions.</p>
</div>
""", unsafe_allow_html=True)
    if st.button("Open Benchmark →", use_container_width=True, key="nav_bench"):
        st.switch_page("pages/3_📈_Benchmark.py")


st.divider()


# ══════════════════════════════════════════════════════════
# sidebar
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### AstroRAG UI")
    st.caption("Read-only display of pipeline results")

    st.divider()

    if library:
        st.markdown(f"**📚 Library:** {len(library)} papers")
        # show subdomains
        subdomains = sorted(
            {e.get("subdomain", "") for e in library.values() if e.get("subdomain")}
        )
        if subdomains:
            st.markdown("**Subdomains:**")
            for sd in subdomains:
                count = sum(1 for e in library.values() if e.get("subdomain") == sd)
                st.markdown(f"- {sd} ({count})")
    else:
        st.warning("No library data loaded")

    st.divider()

    st.markdown("### Links")
    st.markdown("""
- [Paper (PDF)](#)
- [GitHub Repo](https://github.com/arrowguy234)
- [SDSU CSRC](https://www.csrc.sdsu.edu/)
""")
