"""
Chat page — follow-up Q&A on a selected paper.

The LLM (Groq LLaMA-3.1-8B) is called with the paper's summary,
equations, numerical results, and methodology as context.
No retrieval happens here — only follow-up conversation on
already-processed papers.
"""

from pathlib import Path
import json
import os

import streamlit as st

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


st.set_page_config(page_title="AstroRAG — Chat", page_icon="💬", layout="wide")


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
    st.error("Context library is empty.")
    st.stop()


# ══════════════════════════════════════════════════════════
# session
# ══════════════════════════════════════════════════════════

if "current_arxiv_id" not in st.session_state:
    st.session_state.current_arxiv_id = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ══════════════════════════════════════════════════════════
# paper selection
# ══════════════════════════════════════════════════════════

st.markdown("# 💬 Chat with a Paper")

# paper selector
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
    "Select a paper to chat with",
    list(paper_options.keys()),
    index=default_idx,
)
selected_arxiv = paper_options[selected_label]

# if selection changed, reset chat
if selected_arxiv != st.session_state.current_arxiv_id:
    st.session_state.current_arxiv_id = selected_arxiv
    st.session_state.chat_messages = []

entry = library[selected_arxiv]


# ══════════════════════════════════════════════════════════
# header
# ══════════════════════════════════════════════════════════

col1, col2 = st.columns([4, 1])
with col1:
    st.markdown(f"""
Chatting with **arXiv:{selected_arxiv}**

_Query: {entry.get('original_query', '')}_
""")

with col2:
    if st.button("🔄 Reset chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()


# quick metadata
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Q_total", f"{entry.get('q_total', 0):.3f}")
mc2.metric("Decision", entry.get("decision", ""))
mc3.metric("Equations", len(entry.get("key_equations", [])))
mc4.metric("Numerical", len(entry.get("numerical_results", [])))


st.divider()


# ══════════════════════════════════════════════════════════
# LLM setup — Groq
# ══════════════════════════════════════════════════════════

def get_groq_client():
    """Get Groq client, checking env / secrets."""
    # try Streamlit secrets first
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    # fall back to env
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        return None
    return Groq(api_key=api_key)


def build_paper_context(entry: dict) -> str:
    """Assemble paper context block for LLM prompt."""
    parts = [
        f"PAPER: arXiv:{entry['arxiv_id']}",
        f"TITLE: {entry.get('title', '')}",
        f"",
        f"OVERVIEW: {entry.get('paper_overview', '')}",
        f"",
        f"EVIDENCE TYPE: {entry.get('evidence_type', '')}",
    ]
    if entry.get("instruments"):
        parts.append(f"INSTRUMENTS: {', '.join(entry['instruments'])}")
    parts.append("")

    if entry.get("key_equations"):
        parts.append("KEY EQUATIONS:")
        for eq in entry["key_equations"][:10]:
            parts.append(f"  - {eq.get('equation', '')} (variables: {eq.get('variables', '')})")
        parts.append("")

    if entry.get("numerical_results"):
        parts.append("NUMERICAL RESULTS:")
        for nr in entry["numerical_results"][:15]:
            parts.append(f"  - {nr.get('quantity', '')} = {nr.get('value', '')} "
                         f"± {nr.get('uncertainty', '')} {nr.get('unit', '')}")
        parts.append("")

    if entry.get("sub_question_answers"):
        parts.append("SUB-QUESTION ANSWERS:")
        for qk, sqa in entry["sub_question_answers"].items():
            if sqa.get("answered"):
                parts.append(f"  {qk} ({sqa.get('section', '')}): "
                             f"{sqa.get('answer_text', '')[:400]}")
        parts.append("")

    if entry.get("methodology"):
        parts.append(f"METHODOLOGY: {entry['methodology'][:600]}")
        parts.append("")

    if entry.get("key_snippet"):
        parts.append(f"KEY SNIPPET: \"{entry['key_snippet']}\"")

    return "\n".join(parts)


SYSTEM_PROMPT = (
    "You are an expert astrophysicist answering follow-up questions about a "
    "specific paper the user has retrieved. Use the paper's summary, equations, "
    "and numerical results provided in the context to answer precisely and "
    "quantitatively. If the answer is not in the context, say so clearly. "
    "Do not invent data. Cite section names when relevant. "
    "Keep answers focused and technical."
)


def ask_groq(question: str, entry: dict, history: list) -> str:
    """Send a follow-up question to Groq."""
    client = get_groq_client()
    if client is None:
        return ("⚠ GROQ_API_KEY not configured. Add it to `.streamlit/secrets.toml` "
                "or set as environment variable.")

    context = build_paper_context(entry)

    # build conversation
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role":    "user",
        "content": f"CONTEXT ABOUT THE PAPER:\n\n{context}\n\n"
                   f"I will now ask you follow-up questions about this paper.",
    })
    messages.append({
        "role":    "assistant",
        "content": f"Understood. I have the paper's summary, equations, "
                   f"numerical results, and methodology. Ask your question.",
    })

    for msg in history[:-1]:  # exclude the just-added user question
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model       = "llama-3.1-8b-instant",
            messages    = messages,
            temperature = 0.0,
            max_tokens  = 800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠ LLM error: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════
# chat display
# ══════════════════════════════════════════════════════════

if not HAS_GROQ:
    st.warning("`groq` package not installed. Install with `pip install groq`.")
    st.stop()


if not st.session_state.chat_messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"""
Hello! I'm ready to answer follow-up questions about **arXiv:{selected_arxiv}**.

I have access to:
- The paper's overview and methodology
- **{len(entry.get('key_equations', []))}** extracted equations
- **{len(entry.get('numerical_results', []))}** numerical measurements
- Sub-question answers with section attribution
- Key verbatim snippets

**Try asking:**
""")
        # subject-appropriate suggestions
        suggestions = [
            "What are the key equations in this paper?",
            "What numerical values did the paper measure?",
            "What instruments were used?",
            "Explain the methodology in simple terms.",
            "What are the main limitations?",
        ]
        for s in suggestions:
            st.markdown(f"- _{s}_")


# display messages
for msg in st.session_state.chat_messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# input
question = st.chat_input(f"Ask about arXiv:{selected_arxiv}...")

if question:
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            answer = ask_groq(question, entry, st.session_state.chat_messages)
        st.markdown(answer)

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})


# ══════════════════════════════════════════════════════════
# sidebar
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Current Paper")
    st.success(f"**arXiv:{selected_arxiv}**")
    st.caption(entry.get("original_query", "")[:100])

    st.divider()

    st.markdown("### Session")
    st.metric("Messages", len(st.session_state.chat_messages))

    st.divider()

    st.markdown("### About")
    st.caption(
        "Chat uses Groq's LLaMA-3.1-8B model with the paper's structured "
        "summary as context. No retrieval happens here — only follow-up "
        "Q&A on the already-processed paper."
    )
