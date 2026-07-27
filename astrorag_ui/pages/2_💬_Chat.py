"""
💬 Research Chat — deep Q&A on a selected paper.

Uses the chat_assist module for:
- Research-oriented system prompt (methodology, derivation, critique focus)
- Question template library grouped by category
- Section-aware paper context retrieval
- Dual summary display (query-focused + generalized) via Llama-3.3-70B
- On-demand PDF table extraction
- Larger max_tokens for detailed answers
- Automatic 70B model for depth, with 8B fallback
"""

import sys
from   pathlib import Path

# make sibling chat_assist package importable
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
import streamlit as st

from chat_assist.question_templates import get_templates_by_category
from chat_assist.qa_engine          import ResearchPaperQA


st.set_page_config(
    page_title = "AstroRAG — Research Chat",
    page_icon  = "💬",
    layout     = "wide",
)


# ══════════════════════════════════════════════════════════
# data loading
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
    st.error("Context library is empty.")
    st.stop()


# ══════════════════════════════════════════════════════════
# session state
# ══════════════════════════════════════════════════════════

if "current_arxiv_id" not in st.session_state:
    st.session_state.current_arxiv_id = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_model" not in st.session_state:
    st.session_state.chat_model = "llama-3.3-70b-versatile"


# ══════════════════════════════════════════════════════════
# header
# ══════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(90deg, #1e3a5f, #2c5282);
            color:white; padding:20px; border-radius:12px; margin-bottom:20px;">
  <h1 style="margin:0;">💬 Research Chat</h1>
  <p style="margin:8px 0 0 0; opacity:0.9;">
    Ask deep methodological, quantitative, and comparative questions about any paper.
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# paper selector
# ══════════════════════════════════════════════════════════

sorted_entries = sorted(library.values(), key=lambda x: x.get("q_total", 0), reverse=True)
paper_options = {
    f"arXiv:{e['arxiv_id']} — {(e.get('title') or e.get('original_query', ''))[:70]}":
        e["arxiv_id"]
    for e in sorted_entries
}

default_idx = 0
if st.session_state.current_arxiv_id:
    for i, aid in enumerate(paper_options.values()):
        if aid == st.session_state.current_arxiv_id:
            default_idx = i
            break

selected_label = st.selectbox(
    "Select a paper to analyze",
    list(paper_options.keys()),
    index = default_idx,
)
selected_arxiv = paper_options[selected_label]

if selected_arxiv != st.session_state.current_arxiv_id:
    st.session_state.current_arxiv_id = selected_arxiv
    st.session_state.chat_messages = []

entry = library[selected_arxiv]


# ══════════════════════════════════════════════════════════
# paper summary card + metrics
# ══════════════════════════════════════════════════════════

col1, col2 = st.columns([4, 1])

with col1:
    st.markdown(f"""
**Analyzing arXiv:{selected_arxiv}**

_Original query: {entry.get('original_query', '')}_
""")

with col2:
    if st.button("🔄 Reset chat", width="stretch"):
        st.session_state.chat_messages = []
        st.rerun()


mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Q_total",   f"{entry.get('q_total', 0):.3f}")
mc2.metric("Decision",  entry.get("decision", ""))
mc3.metric("Equations", len(entry.get("key_equations", [])))
mc4.metric("Numerical", len(entry.get("numerical_results", [])))

st.divider()


# ══════════════════════════════════════════════════════════
# dual summary display (query-focused + generalized)
# ══════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _cached_dual_summary(arxiv_id: str, entry_json: str, model: str):
    """Cache summaries per (arxiv_id, model) so we only generate once."""
    from chat_assist.dual_summary import generate_dual_summary
    entry_local = json.loads(entry_json)
    result = generate_dual_summary(entry_local, model=model)
    return {
        "query_focused": result.query_focused,
        "generalized":   result.generalized,
        "error":         result.error,
        "model_used":    result.model_used,
    }


with st.spinner("Generating detailed summaries with Llama-3.3-70B..."):
    dual = _cached_dual_summary(
        arxiv_id   = selected_arxiv,
        entry_json = json.dumps(entry),
        model      = st.session_state.chat_model,
    )

if dual["error"]:
    st.warning(f"Could not generate dual summary: {dual['error']}")
else:
    col_q, col_g = st.columns(2, gap="large")

    with col_q:
        st.markdown(
            "<div style='background:#e8f2ff; padding:12px 14px; "
            "border-left:4px solid #2c5282; border-radius:6px;'>"
            "<h4 style='margin:0 0 6px 0; color:#1e3a5f;'>"
            "🎯 Answer to Your Query</h4>"
            f"<div style='font-size:0.85em; color:#4a6688;'>"
            f"{entry.get('original_query', '')[:120]}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("")   # spacer
        st.markdown(dual["query_focused"] or "_(empty)_")

    with col_g:
        st.markdown(
            "<div style='background:#f0ede8; padding:12px 14px; "
            "border-left:4px solid #8a6d3b; border-radius:6px;'>"
            "<h4 style='margin:0 0 6px 0; color:#5a4523;'>"
            "📖 Paper Overview</h4>"
            f"<div style='font-size:0.85em; color:#6b5535;'>"
            f"Generalized summary, independent of query</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("")   # spacer
        st.markdown(dual["generalized"] or "_(empty)_")

    st.caption(
        f"_Both summaries generated by **{dual['model_used']}** in ~15–30s "
        f"and cached per paper._"
    )

st.divider()


# ══════════════════════════════════════════════════════════
# question templates panel
# ══════════════════════════════════════════════════════════

with st.expander(
    "📚 Research question templates (click to insert)",
    expanded = len(st.session_state.chat_messages) == 0,
):
    st.caption(
        "These are the questions researchers actually ask. Click any to "
        "load into your input box — you can edit before sending."
    )

    templates_by_cat = get_templates_by_category()

    for category, templates in templates_by_cat.items():
        st.markdown(f"**{category}**")
        cols = st.columns(2)
        for i, tmpl in enumerate(templates):
            with cols[i % 2]:
                depth_badge = {
                    "basic":    "🟢",
                    "medium":   "🟡",
                    "advanced": "🔴",
                }.get(tmpl.depth, "")
                if st.button(
                    f"{depth_badge} {tmpl.text}",
                    key   = f"tmpl_{category}_{i}",
                    width = "stretch",
                ):
                    st.session_state["chat_prefilled"] = tmpl.text
                    st.rerun()


st.divider()


# ══════════════════════════════════════════════════════════
# chat display
# ══════════════════════════════════════════════════════════

if not st.session_state.chat_messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"""
Ready to analyze **arXiv:{selected_arxiv}** in depth.

**I can help you with:**
- **Methodology:** sample selection, calibration, systematic errors, assumptions
- **Results:** exact numerical values, uncertainties, scaling relations
- **Derivations:** walk through equations step by step, explain physical reasoning
- **Comparisons:** relate this paper's approach to others (when cited)
- **Critical analysis:** limitations, failure modes, alternative interpretations
- **Extensions:** what would resolve open questions, follow-up observations
- **Tables:** extract tables from the PDF on demand

**I will:**
- ✓ Cite section names when I make claims
- ✓ Distinguish what the paper explicitly states from what I'm inferring
- ✓ Clearly say "the paper does not address this" when it doesn't
- ✓ Show reasoning and derivations when asked
- ✓ Fetch tables from the PDF when you ask about them

Use the **question templates above** for depth, or type your own question.
""")


# display messages
for msg in st.session_state.chat_messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        # replay any tables
        if msg.get("tables"):
            st.divider()
            st.markdown(f"### 📋 Tables ({len(msg['tables'])})")
            for t in msg["tables"]:
                caption = t.get("caption") or "_(no caption detected)_"
                st.markdown(
                    f"**Table {t['table_num']} on page {t['page_num']}** — "
                    f"{caption}"
                )
                st.markdown(t["markdown"])
                st.caption(f"{t['n_rows']} rows × {t['n_cols']} columns")
                st.divider()

        # show metadata if present
        if msg.get("metadata"):
            m = msg["metadata"]
            st.caption(
                f"_Intent: **{m['intent']}** &nbsp;•&nbsp; "
                f"Context: {m['context_chars']} chars &nbsp;•&nbsp; "
                f"Model: {m['model_used']}"
                + (f" &nbsp;•&nbsp; Tables: {m.get('n_tables', 0)}" if m.get('n_tables') else "")
                + "_"
            )


# ══════════════════════════════════════════════════════════
# input
# ══════════════════════════════════════════════════════════

prefilled = st.session_state.pop("chat_prefilled", "")
question = st.chat_input(
    f"Ask a research question about arXiv:{selected_arxiv}...",
)

if prefilled and not question:
    question = prefilled

if question:
    st.session_state.chat_messages.append({
        "role":    "user",
        "content": question,
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing paper..."):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_messages[:-1]
            ]
            qa = ResearchPaperQA(
                library_entry = entry,
                history       = history,
                model         = st.session_state.chat_model,
            )
            result = qa.ask(question)

        if result.error:
            st.error(f"⚠ {result.error}")
            answer = f"Error: {result.error}"
            metadata = None
        else:
            st.markdown(result.answer)

            if result.tables:
                st.divider()
                st.markdown(
                    f"### 📋 Tables from the PDF ({len(result.tables)} found)"
                )
                for t in result.tables:
                    caption = t.get("caption") or "_(no caption detected)_"
                    st.markdown(
                        f"**Table {t['table_num']} on page {t['page_num']}** — "
                        f"{caption}"
                    )
                    st.markdown(t["markdown"])
                    st.caption(
                        f"{t['n_rows']} rows × {t['n_cols']} columns "
                        f"&nbsp;•&nbsp; relevance {t['relevance']}"
                    )
                    st.divider()

            st.caption(
                f"_Intent: **{result.intent}** &nbsp;•&nbsp; "
                f"Context: {result.context_chars} chars &nbsp;•&nbsp; "
                f"Model: {result.model_used}"
                + (f" &nbsp;•&nbsp; Tables: {len(result.tables)}" if result.tables else "")
                + "_"
            )
            answer = result.answer
            metadata = {
                "intent":        result.intent,
                "context_chars": result.context_chars,
                "model_used":    result.model_used,
                "n_tables":      len(result.tables),
            }

    st.session_state.chat_messages.append({
        "role":     "assistant",
        "content":  answer,
        "metadata": metadata,
        "tables":   result.tables if hasattr(result, "tables") else [],
    })


# ══════════════════════════════════════════════════════════
# sidebar
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 💬 Research Chat")
    st.caption(
        "This chat is optimized for deep, technical questions about "
        "the selected paper — not casual summaries. Each answer is "
        "grounded in the paper's summary, equations, numerical values, "
        "and methodology. Ask about tables and the PDF is fetched on "
        "demand."
    )

    st.divider()

    st.markdown("### Model")
    model_choice = st.radio(
        "LLM for answers",
        options = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
        index = 0,
        help = (
            "70B produces deeper, more detailed answers but is slower. "
            "8B is fast but shallower. Recommendation: 70B for methodology "
            "and derivations, 8B for quick lookups."
        ),
    )
    st.session_state.chat_model = model_choice

    st.divider()

    st.markdown("### Current Paper")
    st.success(f"**arXiv:{selected_arxiv}**")
    st.caption(entry.get("original_query", "")[:100])

    st.divider()

    st.markdown("### Session")
    st.metric("Messages", len(st.session_state.chat_messages))

    if st.button("💾 Export chat", width="stretch"):
        chat_json = json.dumps({
            "arxiv_id": selected_arxiv,
            "messages": st.session_state.chat_messages,
        }, indent=2)
        st.download_button(
            "Download chat.json",
            chat_json,
            file_name = f"chat_{selected_arxiv}.json",
            mime      = "application/json",
            width     = "stretch",
        )