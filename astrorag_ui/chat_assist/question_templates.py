"""
Predefined research-grade question templates by category.

These are the questions a researcher actually asks — surfacing them
in the UI lowers the barrier to depth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuestionTemplate:
    """One research question with its intent category."""

    text:     str
    intent:   str            # matches keys in system_prompts.INTENT_PROMPTS
    depth:    str = "medium" # basic | medium | advanced


QUESTION_TEMPLATES: dict[str, list[QuestionTemplate]] = {

    "🔬 Methodology": [
        QuestionTemplate(
            "What is the sample selection procedure and what biases might it introduce?",
            "methodology", "advanced"),
        QuestionTemplate(
            "What instruments and observation setups were used, and what are their key limitations?",
            "methodology", "medium"),
        QuestionTemplate(
            "What data processing and calibration steps were applied?",
            "methodology", "medium"),
        QuestionTemplate(
            "What are the systematic uncertainties in the measurements?",
            "methodology", "advanced"),
        QuestionTemplate(
            "What are the key physical assumptions underlying the analysis?",
            "methodology", "advanced"),
        QuestionTemplate(
            "How were the observations cross-validated or checked for robustness?",
            "methodology", "advanced"),
    ],

    "📊 Results": [
        QuestionTemplate(
            "What are the main quantitative results with uncertainties?",
            "results", "basic"),
        QuestionTemplate(
            "What scaling relations or correlations are reported?",
            "results", "medium"),
        QuestionTemplate(
            "What is the significance of the primary measurement?",
            "results", "medium"),
        QuestionTemplate(
            "How do the reported values compare to previously published measurements?",
            "results", "advanced"),
        QuestionTemplate(
            "What are the redshift range, mass range, or parameter space covered?",
            "results", "medium"),
    ],

    "📐 Derivations & Physics": [
        QuestionTemplate(
            "Walk through the derivation of the key equation in this paper.",
            "derivation", "advanced"),
        QuestionTemplate(
            "What physical mechanism does this paper propose or test?",
            "derivation", "medium"),
        QuestionTemplate(
            "Explain the assumptions behind the main equation.",
            "derivation", "advanced"),
        QuestionTemplate(
            "What order-of-magnitude estimate justifies the result?",
            "derivation", "medium"),
        QuestionTemplate(
            "How do the physical parameters combine in the reported scaling?",
            "derivation", "advanced"),
    ],

    "⚖️ Comparisons": [
        QuestionTemplate(
            "How does this method compare with the earlier approach used in similar studies?",
            "comparison", "advanced"),
        QuestionTemplate(
            "What are the strengths of this approach relative to alternatives?",
            "comparison", "medium"),
        QuestionTemplate(
            "Where do the results agree or disagree with previous work?",
            "comparison", "advanced"),
        QuestionTemplate(
            "What competing models or interpretations exist for these observations?",
            "comparison", "advanced"),
    ],

    "❓ Critical Analysis": [
        QuestionTemplate(
            "What are the main limitations of this study?",
            "critique", "medium"),
        QuestionTemplate(
            "Under what conditions would the reported result fail?",
            "critique", "advanced"),
        QuestionTemplate(
            "What alternative explanations could account for these findings?",
            "critique", "advanced"),
        QuestionTemplate(
            "What sources of systematic error are the largest?",
            "critique", "advanced"),
        QuestionTemplate(
            "Are the statistical claims robust to the sample size?",
            "critique", "advanced"),
    ],

    "🚀 Extensions": [
        QuestionTemplate(
            "What would extend this work to higher redshift or a larger sample?",
            "extension", "medium"),
        QuestionTemplate(
            "How could this method be applied to a related phenomenon?",
            "extension", "medium"),
        QuestionTemplate(
            "What data would resolve the open questions raised in this paper?",
            "extension", "advanced"),
        QuestionTemplate(
            "What follow-up observations does the paper suggest?",
            "extension", "medium"),
    ],
}


def get_templates_by_category() -> dict[str, list[QuestionTemplate]]:
    return QUESTION_TEMPLATES


def all_templates() -> list[QuestionTemplate]:
    """Flattened list of all templates."""
    out = []
    for tmpls in QUESTION_TEMPLATES.values():
        out.extend(tmpls)
    return out


def infer_intent(question: str) -> str:
    """
    Heuristic intent classification from question text.
    Used to select the right refined system prompt.
    """
    q = question.lower()

    if any(kw in q for kw in ["method", "sample", "selection", "calibration",
                              "systematic", "assumption", "pipeline",
                              "observ", "instrument"]):
        return "methodology"

    if any(kw in q for kw in ["derive", "derivation", "equation", "formula",
                              "step by step", "explain how", "physical mechanism"]):
        return "derivation"

    if any(kw in q for kw in ["compare", "versus", "vs", "difference",
                              "similar to", "prior work", "previously"]):
        return "comparison"

    if any(kw in q for kw in ["limitation", "problem", "wrong", "fail",
                              "systematic error", "bias", "caveat",
                              "weakness", "critique"]):
        return "critique"

    if any(kw in q for kw in ["extend", "future", "follow-up", "next step",
                              "improve", "apply to"]):
        return "extension"

    if any(kw in q for kw in ["value", "measure", "result", "number",
                              "uncertainty", "significance", "correlation",
                              "scaling"]):
        return "results"

    return "general"
