"""
Refinement question generator.

Given a rough query, uses the LLM to propose 2-3 targeted clarifying
questions with 3-4 options each. Users pick options → refined query.
"""

from __future__ import annotations

import json
import os
import re
from   dataclasses import dataclass, field


try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


@dataclass
class RefinementQuestion:
    """One clarifying question with multiple-choice options."""

    question: str
    options:  list[str] = field(default_factory=list)


@dataclass
class RefinementSet:
    """Set of clarifying questions for a query."""

    original_query: str
    questions:      list[RefinementQuestion] = field(default_factory=list)
    error:          str = ""


_REFINER_SYSTEM = (
    "You are an astrophysics research assistant helping a user refine "
    "their query for better retrieval. Given a rough query, produce 2-3 "
    "targeted multiple-choice questions to disambiguate it. "
    "Respond with valid JSON only."
)


_REFINER_USER_TEMPLATE = """Refine this astrophysics query with 2-3 clarifying questions:

QUERY: {query}

Focus on:
- What sub-topic within the broad area
- What observation type or wavelength
- What system class or scale
- What kind of answer (mechanism, measurement, review)

Each question needs 3-4 short concrete options plus "Any/no preference".

Respond as JSON:
{{
  "questions": [
    {{
      "question": "<clarifying question 1>",
      "options": ["<option 1>", "<option 2>", "<option 3>", "Any"]
    }},
    {{
      "question": "<clarifying question 2>",
      "options": ["<option 1>", "<option 2>", "<option 3>", "Any"]
    }}
  ]
}}"""


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


def suggest_refinements(query: str) -> RefinementSet:
    """Get 2-3 clarifying questions for the user's rough query."""

    if not HAS_GROQ:
        return RefinementSet(original_query=query, error="groq not installed")

    query = query.strip()
    if not query:
        return RefinementSet(original_query=query)

    client = _get_groq_client()
    if client is None:
        return RefinementSet(original_query=query,
                             error="GROQ_API_KEY not configured")

    try:
        response = client.chat.completions.create(
            model    = "llama-3.1-8b-instant",
            messages = [
                {"role": "system", "content": _REFINER_SYSTEM},
                {"role": "user",   "content": _REFINER_USER_TEMPLATE.format(query=query)},
            ],
            temperature = 0.3,
            max_tokens  = 500,
        )
        raw = response.choices[0].message.content.strip()
        payload = _extract_json(raw)

        questions = []
        for q in payload.get("questions", []):
            questions.append(RefinementQuestion(
                question = str(q.get("question", "")),
                options  = list(q.get("options", [])),
            ))

        return RefinementSet(original_query=query, questions=questions)

    except Exception as e:
        return RefinementSet(original_query=query,
                             error=f"{type(e).__name__}: {e}")


def build_refined_query(
    original: str,
    selections: dict[str, str],
) -> str:
    """
    Combine the original query with the user's selections into
    a refined query. Uses simple concatenation with 'focusing on X, Y, Z'.
    """
    selected = [v for v in selections.values()
                if v and v.lower() not in ("any", "no preference", "")]
    if not selected:
        return original

    focus = ", ".join(selected)
    if any(w in original.lower() for w in ["how", "what", "why"]):
        return f"{original.rstrip('?').strip()}, focusing on {focus}?"
    return f"{original} — focusing on {focus}"
