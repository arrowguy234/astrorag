"""
Smart section-aware context retrieval.

Instead of always shipping the same truncated paper text, this module
selects which parts of the paper to include based on the question's
intent. Currently uses simple keyword matching against section headers,
but the interface allows a future upgrade to embeddings-based retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PaperContext:
    """Assembled context block for the LLM prompt."""

    arxiv_id:       str
    query:          str
    paper_overview: str
    equations:      list[dict]
    numerical:      list[dict]
    sub_q_answers:  dict
    methodology:    str
    key_snippet:    str
    instruments:    list[str]
    evidence_type:  str
    key_findings:   list[str]

    # optional: sections retrieved from full paper text
    relevant_sections: list[dict] = None

    def to_prompt_block(self, max_chars: int = 8000) -> str:
        """Serialize as a text block for the LLM."""
        parts = [
            f"=== PAPER: arXiv:{self.arxiv_id} ===",
            f"USER'S ORIGINAL QUERY: {self.query}",
            "",
            "=== OVERVIEW ===",
            self.paper_overview[:800] if self.paper_overview else "(none)",
            "",
        ]

        if self.evidence_type:
            parts.append(f"EVIDENCE TYPE: {self.evidence_type}")
        if self.instruments:
            parts.append(f"INSTRUMENTS: {', '.join(self.instruments)}")
        if self.evidence_type or self.instruments:
            parts.append("")

        # equations
        if self.equations:
            parts.append("=== KEY EQUATIONS ===")
            for i, eq in enumerate(self.equations[:12], 1):
                parts.append(
                    f"  ({i}) {eq.get('equation', '')} "
                    f"[vars: {eq.get('variables', '')}]"
                )
            parts.append("")

        # numerical
        if self.numerical:
            parts.append("=== NUMERICAL RESULTS ===")
            for nr in self.numerical[:20]:
                parts.append(
                    f"  {nr.get('quantity', '')} = {nr.get('value', '')} "
                    f"± {nr.get('uncertainty', '')} "
                    f"{nr.get('unit', '')}"
                )
            parts.append("")

        # sub-question answers
        if self.sub_q_answers:
            parts.append("=== ANSWERS TO PIPELINE SUB-QUESTIONS ===")
            labels = {"Q1": "Mechanism", "Q2": "Evidence", "Q3": "Quantitative"}
            for qk, sqa in self.sub_q_answers.items():
                if isinstance(sqa, dict) and sqa.get("answered"):
                    label = labels.get(qk, qk)
                    section = sqa.get("section", "unknown")
                    text = sqa.get("answer_text", "")[:500]
                    parts.append(f"  [{label}, from section '{section}']: {text}")
            parts.append("")

        # methodology
        if self.methodology:
            parts.append("=== METHODOLOGY SUMMARY ===")
            parts.append(self.methodology[:1200])
            parts.append("")

        # key findings
        if self.key_findings:
            parts.append("=== KEY FINDINGS ===")
            for f in self.key_findings[:10]:
                parts.append(f"  - {f}")
            parts.append("")

        # key snippet
        if self.key_snippet:
            parts.append("=== VERBATIM KEY SNIPPET FROM PAPER ===")
            parts.append(f"\"{self.key_snippet}\"")
            parts.append("")

        # relevant sections
        if self.relevant_sections:
            parts.append("=== RELEVANT SECTIONS ===")
            for sec in self.relevant_sections[:3]:
                parts.append(
                    f"[Section '{sec.get('title', 'unknown')}']:\n"
                    f"{sec.get('text', '')[:1500]}"
                )
            parts.append("")

        block = "\n".join(parts)

        # cap the block size to avoid rate-limit issues
        if len(block) > max_chars:
            block = block[:max_chars] + "\n\n[...context truncated for length...]"

        return block


# ══════════════════════════════════════════════════════════
# retrieval — pick the right sections given the question
# ══════════════════════════════════════════════════════════

_INTENT_KEYWORDS = {
    "methodology": ["method", "sample", "selection", "calibration", "pipeline",
                    "observ", "instrument", "data"],
    "results":     ["result", "measure", "value", "correlation", "scaling",
                    "significance"],
    "derivation":  ["derivation", "equation", "formula", "physical", "assume"],
    "comparison":  ["compare", "versus", "similar", "differ", "prior"],
    "critique":    ["limitation", "bias", "systematic", "caveat", "assumption",
                    "uncertain"],
    "extension":   ["future", "extend", "follow-up", "improve", "apply"],
}


def retrieve_paper_context(entry: dict, question: str, intent: str = "general") -> PaperContext:
    """
    Build a PaperContext for the given library entry and question.

    Handles both dict format (from context_library.json) and object-style
    LibraryEntry.
    """
    def _get(k, default=None):
        if isinstance(entry, dict):
            return entry.get(k, default)
        return getattr(entry, k, default)

    return PaperContext(
        arxiv_id       = _get("arxiv_id", ""),
        query          = _get("original_query", ""),
        paper_overview = _get("paper_overview", "") or "",
        equations      = list(_get("key_equations", []) or []),
        numerical      = list(_get("numerical_results", []) or []),
        sub_q_answers  = dict(_get("sub_question_answers", {}) or {}),
        methodology    = _get("methodology", "") or "",
        key_snippet    = _get("key_snippet", "") or "",
        instruments    = list(_get("instruments", []) or []),
        evidence_type  = _get("evidence_type", "") or "",
        key_findings   = list(_get("key_findings", []) or []),
        relevant_sections = None,  # will be populated when full text is available
    )
