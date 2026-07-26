"""
🧠 Query Builder — help users construct well-formed queries for AstroRAG.

Combines four modes:
1. Templates — click a pattern, fill in the blanks
2. Live decomposition preview — see how Stage 0 will interpret the query
3. Related queries — find similar successful queries from the library
4. Refinement loop — LLM asks clarifying questions

All UI. Doesn't touch the production pipeline.
"""

import sys
from   pathlib import Path

# make the sibling query_assist package importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from query_assist import (
    QUERY_TEMPLATES,
    get_template,
    preview_decomposition,
    find_similar_queries,
    assess_query_quality,
    suggest_refinements,
)
from query_assist.templates import fill_template
from query_assist.refiner   import build_refined_query


st.set_page_config(
    page_title = "AstroRAG — Query Builder",
    page_icon  = "🧠",
    layout     = "wide",
)


# ══════════════════════════════════════════════════════════
# session state
# ══════════════════════════════════════════════════════════

if "qb_query"          not in st.session_state:
    st.session_state.qb_query = ""
if "qb_template_id"    not in st.session_state:
    st.session_state.qb_template_id = None
if "qb_field_values"   not in st.session_state:
    st.session_state.qb_field_values = {}
if "qb_refinements"    not in st.session_state:
    st.session_state.qb_refinements = None
if "qb_refinement_answers" not in st.session_state:
    st.session_state.qb_refinement_answers = {}


# ══════════════════════════════════════════════════════════
# header
# ══════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(90deg, #1e3a5f, #2c5282);
            color:white; padding:20px; border-radius:12px; margin-bottom:20px;">
  <h1 style="margin:0;">🧠 Query Builder</h1>
  <p style="margin:8px 0 0 0; opacity:0.9;">
    Craft well-structured queries that match how AstroRAG thinks.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Astrophysics questions get much better results when they follow "
    "a structured pattern. Use the templates, refinement flow, or "
    "similarity search below to build a query, then send it to the "
    "chat or run it through the pipeline."
)


# ══════════════════════════════════════════════════════════
# tabs
# ══════════════════════════════════════════════════════════

tab_free, tab_templates, tab_refine = st.tabs([
    "✍️ Free-form",
    "📋 Templates",
    "🔀 Guided refinement",
])


# ══════════════════════════════════════════════════════════
# tab 1: free-form query with live analysis
# ══════════════════════════════════════════════════════════

with tab_free:
    st.markdown("### Type or paste your query")
    st.caption("As you type, we'll analyse quality, preview decomposition, "
               "and suggest similar library queries.")

    query = st.text_area(
        "Your query",
        value       = st.session_state.qb_query,
        height      = 80,
        placeholder = "e.g. How do AGN jets suppress star formation in massive elliptical galaxies?",
        key         = "qb_free_query",
        label_visibility = "collapsed",
    )

    if query != st.session_state.qb_query:
        st.session_state.qb_query = query

    if query.strip():
        col_left, col_right = st.columns([1, 1])

        # ── left: quality assessment ────────────
        with col_left:
            st.markdown("#### Quality Assessment")
            report = assess_query_quality(query)

            grade_color = {
                "A": "green", "B": "green", "C": "orange",
                "D": "red",   "F": "red",
            }[report.grade]

            st.markdown(
                f"<div style='font-size:1.4em;'>"
                f"Score: <strong>{report.score}/100</strong> — "
                f"Grade: <strong style='color:{grade_color};'>{report.grade}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if report.strengths:
                st.success("**Strengths**\n" +
                           "\n".join(f"- {s}" for s in report.strengths))

            if report.issues:
                st.warning("**Issues**\n" +
                           "\n".join(f"- {s}" for s in report.issues))

            if report.suggestions:
                st.info("**Suggestions**\n" +
                        "\n".join(f"- {s}" for s in report.suggestions))

        # ── right: stage 0 decomposition preview ─
        with col_right:
            st.markdown("#### Stage 0 Preview")
            st.caption("How AstroRAG will decompose your query into sub-questions:")

            if st.button("🔍 Preview decomposition", key="preview_decomp_btn"):
                with st.spinner("Calling Stage 0 LLM..."):
                    decomp = preview_decomposition(query)

                if decomp.error:
                    st.error(f"⚠ {decomp.error}")
                else:
                    st.markdown(
                        f"**Q1 (mechanism):** {decomp.q1_mechanism or '_none_'}"
                    )
                    st.markdown(
                        f"**Q2 (evidence):** {decomp.q2_evidence or '_none_'}"
                    )
                    st.markdown(
                        f"**Q3 (quantitative):** {decomp.q3_quantitative or '_none_'}"
                    )
                    st.divider()
                    st.markdown(f"**Wavelength:** `{decomp.wavelength}`")
                    if decomp.catalogs:
                        st.markdown(
                            f"**Catalogs:** {', '.join(decomp.catalogs)}"
                        )

        st.divider()

        # ── similar library queries ──────────────
        st.markdown("#### 💡 Similar successful queries from the library")

        similar = find_similar_queries(query, k=5)
        if not similar:
            st.caption("No similar queries found in the library yet.")
        else:
            for s in similar:
                sim_pct = int(s["score"] * 100)
                col_a, col_b = st.columns([5, 1])
                with col_a:
                    st.markdown(
                        f"**{s['query']}**  \n"
                        f"_{s['subdomain'] or 'n/a'}_ &nbsp;•&nbsp; "
                        f"arXiv:{s['arxiv_id']} &nbsp;•&nbsp; "
                        f"Q={s['q_total']:.2f} &nbsp;•&nbsp; "
                        f"{sim_pct}% similar"
                    )
                with col_b:
                    if st.button("Use", key=f"use_similar_{s['arxiv_id']}",
                                use_container_width=True):
                        st.session_state.qb_query = s["query"]
                        st.rerun()


# ══════════════════════════════════════════════════════════
# tab 2: template-based query construction
# ══════════════════════════════════════════════════════════

with tab_templates:
    st.markdown("### Pick a template")
    st.caption("Choose the pattern that matches your research question.")

    # template picker as cards
    template_cols = st.columns(len(QUERY_TEMPLATES))
    for i, tpl in enumerate(QUERY_TEMPLATES):
        with template_cols[i]:
            if st.button(f"{tpl.icon} {tpl.label}", key=f"tpl_pick_{tpl.id}",
                         use_container_width=True,
                         type="primary" if st.session_state.qb_template_id == tpl.id else "secondary"):
                st.session_state.qb_template_id = tpl.id
                st.session_state.qb_field_values = {}
                st.rerun()

    if st.session_state.qb_template_id:
        tpl = get_template(st.session_state.qb_template_id)

        st.divider()
        st.markdown(f"### {tpl.icon} {tpl.label} Query")
        st.caption(tpl.description)

        with st.expander("Example", expanded=False):
            st.info(tpl.example_filled)

        st.markdown(f"**Template:** `{tpl.template}`")
        st.divider()

        # field inputs
        st.markdown("### Fill in the blanks")

        for field in tpl.fields:
            fname = field["name"]

            col1, col2 = st.columns([3, 2])

            with col1:
                val = st.text_input(
                    field["label"],
                    value       = st.session_state.qb_field_values.get(fname, ""),
                    placeholder = field["placeholder"],
                    key         = f"tpl_field_{tpl.id}_{fname}",
                )
                st.session_state.qb_field_values[fname] = val

            with col2:
                if field.get("examples"):
                    st.caption("Examples (click to use):")
                    for ex in field["examples"][:4]:
                        if st.button(ex, key=f"tpl_ex_{tpl.id}_{fname}_{ex}",
                                     use_container_width=True):
                            st.session_state.qb_field_values[fname] = ex
                            st.rerun()

        # build preview
        st.divider()
        filled = fill_template(tpl, st.session_state.qb_field_values)
        st.markdown("### Preview")
        st.info(filled)

        if all(f["name"] in st.session_state.qb_field_values
               and st.session_state.qb_field_values[f["name"]].strip()
               for f in tpl.fields):
            if st.button("✓ Use this query", type="primary",
                         use_container_width=True, key="use_template_btn"):
                st.session_state.qb_query = filled
                st.success("Query set. Switch to Free-form tab or go to Chat.")


# ══════════════════════════════════════════════════════════
# tab 3: guided refinement loop
# ══════════════════════════════════════════════════════════

with tab_refine:
    st.markdown("### Guided refinement")
    st.caption(
        "Type a rough query — the LLM will ask a few clarifying questions "
        "to help you refine it before searching."
    )

    rough = st.text_input(
        "Rough query",
        value = st.session_state.qb_query,
        placeholder = "e.g. AGN feedback",
        key   = "qb_refine_input",
    )

    if st.button("🤔 Get clarifying questions",
                 key="get_refinements_btn",
                 disabled=not rough.strip()):
        with st.spinner("Asking LLM for clarifying questions..."):
            refinements = suggest_refinements(rough)
        st.session_state.qb_refinements = refinements
        st.session_state.qb_refinement_answers = {}

    if st.session_state.qb_refinements:
        refs = st.session_state.qb_refinements
        if refs.error:
            st.error(f"⚠ {refs.error}")
        elif not refs.questions:
            st.warning("No clarifying questions generated.")
        else:
            st.divider()
            st.markdown("### Answer these to refine:")

            for i, rq in enumerate(refs.questions):
                if not rq.options:
                    continue
                answer = st.radio(
                    rq.question,
                    options = rq.options,
                    key     = f"refine_q_{i}",
                    horizontal = True,
                )
                st.session_state.qb_refinement_answers[str(i)] = answer

            if st.button("✨ Build refined query",
                         type="primary",
                         key="build_refined_btn"):
                refined = build_refined_query(
                    refs.original_query,
                    st.session_state.qb_refinement_answers,
                )
                st.session_state.qb_query = refined
                st.success(f"Refined query set.")
                st.info(refined)


# ══════════════════════════════════════════════════════════
# footer: unified query display + actions
# ══════════════════════════════════════════════════════════

st.divider()

if st.session_state.qb_query.strip():
    st.markdown("### 🎯 Your current query")
    st.info(st.session_state.qb_query)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💬 Send to Chat", type="primary",
                     use_container_width=True):
            st.session_state["chat_prefilled"] = st.session_state.qb_query
            st.switch_page("pages/2_💬_Chat.py")

    with col2:
        if st.button("📋 Copy",  use_container_width=True):
            st.write("Copy from the box above.")

    with col3:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.qb_query = ""
            st.session_state.qb_template_id = None
            st.session_state.qb_field_values = {}
            st.session_state.qb_refinements = None
            st.session_state.qb_refinement_answers = {}
            st.rerun()


# ══════════════════════════════════════════════════════════
# sidebar
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🧠 About the Query Builder")
    st.caption(
        "Astrophysics queries perform best when they contain three components: "
        "a **mechanism** (what causes what), an **evidence type** (what "
        "observation supports it), and a **quantitative** focus (what's measured). "
        "This tool helps you construct such queries."
    )

    st.divider()

    st.markdown("### Why it matters")
    st.caption(
        "Vague queries like 'AGN feedback' pull hundreds of loosely related "
        "papers. Structured queries like 'How do AGN jets suppress star "
        "formation in massive elliptical galaxies through X-ray cavity "
        "observations?' let AstroRAG's Stage 0 decompose cleanly, so Stages "
        "1-5 can find the foundational paper."
    )
