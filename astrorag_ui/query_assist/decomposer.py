"""
Live decomposition preview.

Calls the Groq API with the same Stage 0 prompt the production pipeline
uses, so the user sees exactly what AstroRAG will do with their query
before submitting.
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
class DecompositionPreview:
    """Preview of Stage 0 output for a query."""

    q1_mechanism:   str = ""
    q2_evidence:    str = ""
    q3_quantitative: str = ""
    wavelength:     str = ""
    catalogs:       list[str] = None
    error:          str = ""

    def __post_init__(self):
        if self.catalogs is None:
            self.catalogs = []


_DECOMPOSER_SYSTEM = (
    "You are an astrophysics query decomposer. Given a research question, "
    "you produce three sub-questions covering mechanism, evidence, and "
    "quantitative aspects. Also identify likely wavelength and any "
    "relevant catalogs. Respond ONLY with valid JSON."
)


_DECOMPOSER_USER_TEMPLATE = """Decompose this astrophysics research query:

QUERY: {query}

Produce a JSON object with these fields:
{{
  "q1_mechanism":    "<sub-question about the physical mechanism>",
  "q2_evidence":     "<sub-question about observational evidence>",
  "q3_quantitative": "<sub-question about quantitative measurements>",
  "wavelength":      "<X-ray | radio | optical | UV | infrared | mm | GW | multi-wavelength | unknown>",
  "catalogs":        ["<catalog name>", ...]
}}

Keep each sub-question concise (1 sentence). Only include catalogs
explicitly mentioned or clearly implied. Respond with JSON only."""


def _get_groq_client():
    """Get Groq client from env or Streamlit secrets."""
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
    """Best-effort JSON extraction."""
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


def preview_decomposition(query: str) -> DecompositionPreview:
    """Call Groq to preview how AstroRAG will decompose the query."""

    if not HAS_GROQ:
        return DecompositionPreview(error="groq package not installed")

    query = query.strip()
    if not query:
        return DecompositionPreview()

    client = _get_groq_client()
    if client is None:
        return DecompositionPreview(error="GROQ_API_KEY not configured")

    try:
        response = client.chat.completions.create(
            model    = "llama-3.1-8b-instant",
            messages = [
                {"role": "system", "content": _DECOMPOSER_SYSTEM},
                {"role": "user",   "content": _DECOMPOSER_USER_TEMPLATE.format(query=query)},
            ],
            temperature = 0.0,
            max_tokens  = 400,
        )
        raw = response.choices[0].message.content.strip()
        payload = _extract_json(raw)

        return DecompositionPreview(
            q1_mechanism    = str(payload.get("q1_mechanism", "")),
            q2_evidence     = str(payload.get("q2_evidence", "")),
            q3_quantitative = str(payload.get("q3_quantitative", "")),
            wavelength      = str(payload.get("wavelength", "unknown")),
            catalogs        = list(payload.get("catalogs", [])),
        )

    except Exception as e:
        return DecompositionPreview(error=f"{type(e).__name__}: {e}")
