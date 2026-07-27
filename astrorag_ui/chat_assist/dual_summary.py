"""
Generate two detailed summaries per paper for the Research Chat display:

  1. Query-focused summary  — how the paper answers the user's specific
     query, drawing on sub-question answers, equations, and numerical
     results.
  2. Generalized paper summary — what the paper is about on its own,
     independent of any query.

Runs once per paper on the first Chat page load (Streamlit-cached).
Uses Llama-3.3-70B-versatile for higher-quality prose than the
production 8B summariser.
"""

from __future__ import annotations

import json
import os
import re
from   dataclasses import dataclass


try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


@dataclass
class DualSummary:
    """The two-summary payload for one paper."""

    arxiv_id:            str
    query_focused:       str = ""
    generalized:         str = ""
    error:               str = ""
    model_used:          str = ""


# ══════════════════════════════════════════════════════════
# prompts
# ══════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are an expert astrophysics research assistant. You will be given "
    "the structured output of a retrieval pipeline for a single paper: the "
    "original user query, sub-question answers with section attribution, "
    "extracted equations, numerical results, methodology summary, limitations, "
    "and a verbatim key snippet. You will produce TWO distinct written "
    "summaries of this paper in careful, technical prose. Respond ONLY with "
    "valid JSON containing the two fields specified."
)


_USER_TEMPLATE = """Produce two summaries of the following paper.

═══ USER'S ORIGINAL QUERY ═══
{query}

═══ PIPELINE OUTPUT FOR THIS PAPER ═══
arXiv ID: {arxiv_id}

Paper overview (from Stage 5):
{paper_overview}

Sub-question answers:
{sub_q_answers}

Key equations:
{equations}

Numerical results:
{numerical}

Methodology:
{methodology}

Key findings:
{key_findings}

Limitations:
{limitations}

Instruments used:
{instruments}

Verbatim key snippet from the paper:
"{key_snippet}"

═══ TASK ═══
Produce TWO summaries:

1. QUERY-FOCUSED SUMMARY (~250 words, 3-4 paragraphs). Explain specifically
   how this paper answers the user's query. Cite section names when possible.
   Include the quantitative values, equations, and observational evidence
   that address the query. Distinguish what the paper explicitly claims
   from what is inferred. Make it clear WHY this paper was selected as
   the answer to this specific query.

2. GENERALIZED PAPER SUMMARY (~250 words, 3-4 paragraphs). Describe what
   the paper is about ON ITS OWN, independent of the user's query. Cover:
   the research question or hypothesis, the methodology used, the primary
   results with quantitative values, and the acknowledged limitations. Write
   it as a proper technical abstract — a researcher reading this alone
   should understand the paper's contribution to the field.

Respond with ONLY this JSON structure:
{{
  "query_focused": "<detailed query-focused summary>",
  "generalized":   "<detailed generalized summary>"
}}"""


# ══════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════

def _get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _format_sub_q_answers(entry: dict) -> str:
    """Serialize sub_question_answers to compact text."""
    sqa = entry.get("sub_question_answers", {}) or {}
    if not sqa:
        return "(none)"
    lines = []
    labels = {"Q1": "Mechanism", "Q2": "Evidence", "Q3": "Quantitative"}
    for qk, val in sqa.items():
        if not isinstance(val, dict):
            continue
        label = labels.get(qk, qk)
        section = val.get("found_in_section", "unknown")
        text = val.get("answer_text", "")
        answered = val.get("answered", False)
        prefix = "✓" if answered else "✗"
        lines.append(f"  {prefix} [{label}, section '{section}']: {text[:400]}")
    return "\n".join(lines) if lines else "(none)"


def _format_equations(entry: dict) -> str:
    eqs = entry.get("key_equations", []) or []
    if not eqs:
        return "(none)"
    return "\n".join(
        f"  ({i+1}) {eq.get('equation', '')} — {eq.get('meaning', eq.get('variables', ''))}"
        for i, eq in enumerate(eqs[:8])
    )


def _format_numerical(entry: dict) -> str:
    nrs = entry.get("numerical_results", []) or []
    if not nrs:
        return "(none)"
    return "\n".join(
        f"  {nr.get('quantity', '')} = {nr.get('value', '')} "
        f"± {nr.get('uncertainty', '')} {nr.get('unit', '')}".strip()
        for nr in nrs[:15]
    )


# ══════════════════════════════════════════════════════════
# main entry point
# ══════════════════════════════════════════════════════════

def generate_dual_summary(
    library_entry: dict,
    model: str = "llama-3.3-70b-versatile",
) -> DualSummary:
    """Generate both summaries for one library entry."""

    arxiv_id = library_entry.get("arxiv_id", "")

    if not HAS_GROQ:
        return DualSummary(
            arxiv_id=arxiv_id,
            error="groq package not installed",
        )

    client = _get_groq_client()
    if client is None:
        return DualSummary(
            arxiv_id=arxiv_id,
            error="GROQ_API_KEY not configured",
        )

    user_prompt = _USER_TEMPLATE.format(
        query           = library_entry.get("original_query", "") or "(none)",
        arxiv_id        = arxiv_id or "(unknown)",
        paper_overview  = library_entry.get("paper_overview", "") or "(none)",
        sub_q_answers   = _format_sub_q_answers(library_entry),
        equations       = _format_equations(library_entry),
        numerical       = _format_numerical(library_entry),
        methodology     = library_entry.get("methodology", "") or "(none)",
        key_findings    = "\n".join(
            f"  - {f}" for f in (library_entry.get("key_findings", []) or [])[:8]
        ) or "(none)",
        limitations     = "\n".join(
            f"  - {l}" for l in (library_entry.get("limitations", []) or [])[:6]
        ) or "(none)",
        instruments     = ", ".join(library_entry.get("instruments", []) or []) or "(none)",
        key_snippet     = (library_entry.get("key_snippet", "") or "")[:500],
    )

    try:
        response = client.chat.completions.create(
            model       = model,
            messages    = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = 0.2,
            max_tokens  = 2500,
        )
        raw = response.choices[0].message.content.strip()
        payload = _extract_json(raw)

        return DualSummary(
            arxiv_id        = arxiv_id,
            query_focused   = str(payload.get("query_focused", "")).strip(),
            generalized     = str(payload.get("generalized", "")).strip(),
            model_used      = model,
        )
    except Exception as e:
        # fallback to 8B on rate-limit or error
        try:
            response = client.chat.completions.create(
                model       = "llama-3.1-8b-instant",
                messages    = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature = 0.2,
                max_tokens  = 2000,
            )
            raw = response.choices[0].message.content.strip()
            payload = _extract_json(raw)
            return DualSummary(
                arxiv_id      = arxiv_id,
                query_focused = str(payload.get("query_focused", "")).strip(),
                generalized   = str(payload.get("generalized", "")).strip(),
                model_used    = "llama-3.1-8b-instant (fallback)",
            )
        except Exception as e2:
            return DualSummary(
                arxiv_id=arxiv_id,
                error=f"{type(e).__name__}: {e}",
            )
