"""
Query quality assessment.

Fast heuristic checks that flag common problems and suggest
improvements. No LLM needed — pure rule-based.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class QueryQualityReport:
    """Assessment of a query's suitability for AstroRAG."""

    score:         int      = 0                              # 0-100
    grade:         str      = "F"                            # A/B/C/D/F
    issues:        list[str] = field(default_factory=list)
    suggestions:   list[str] = field(default_factory=list)
    strengths:     list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
# heuristic checks
# ══════════════════════════════════════════════════════════

_QUESTION_STARTS = ("how", "what", "why", "when", "where", "which", "who",
                    "does", "do", "did", "is", "are", "was", "were", "can")

_INSTRUMENTS = {
    "jwst", "hubble", "hst", "chandra", "xmm", "swift", "fermi",
    "alma", "vla", "sdss", "desi", "euclid", "roman", "spitzer",
    "planck", "wmap", "ligo", "virgo", "icecube", "kagra",
    "spectroscopy", "photometry", "imaging", "interferometry",
}

_ASTRO_TERMS = {
    "galaxy", "galaxies", "cluster", "clusters", "star", "stars",
    "supernova", "supernovae", "agn", "quasar", "black", "hole",
    "dark", "matter", "energy", "cosmology", "cosmological", "cmb",
    "redshift", "hubble", "big bang", "gravitational", "wave",
    "waves", "jet", "jets", "accretion", "disk", "planet",
    "exoplanet", "solar", "cosmic", "ray", "rays", "nebula",
    "pulsar", "magnetar", "neutron", "collapse", "formation",
    "evolution", "feedback", "cooling", "heating", "turbulence",
    "magnetic", "field", "fields", "instability", "spectrum",
    "spectra", "line", "lines", "absorption", "emission",
    "metallicity", "abundance", "mass", "luminosity", "temperature",
}

_QUANTITATIVE = {
    "how much", "how many", "value", "measured", "measurement",
    "typical", "average", "mean", "median", "distribution",
    "correlation", "scaling", "relation", "ratio",
}

_MECHANISTIC = {
    "how does", "how do", "why does", "why do", "mechanism",
    "process", "drive", "cause", "produce", "generate", "trigger",
    "regulate", "control", "suppress", "enhance",
}


def assess_query_quality(query: str) -> QueryQualityReport:
    """Assess a query's suitability for the AstroRAG pipeline."""

    report = QueryQualityReport()
    q = query.strip().lower()

    if not q:
        return report

    words = re.findall(r"\w+", q)
    n_words = len(words)

    # ── length checks ─────────────────────────────
    if n_words < 4:
        report.issues.append("Query is very short (< 4 words)")
        report.suggestions.append(
            "Add more detail — mechanism, evidence type, or system context"
        )
    elif n_words < 8:
        report.issues.append("Query is short (< 8 words)")
        report.suggestions.append(
            "Consider adding a system type (e.g., 'in galaxy clusters') "
            "or measurement context"
        )
    else:
        report.strengths.append(f"Adequate length ({n_words} words)")

    if n_words > 40:
        report.issues.append("Query is very long — may lose focus")
        report.suggestions.append("Trim to core question + one constraint")

    # ── question form ─────────────────────────────
    if not any(q.startswith(w) for w in _QUESTION_STARTS):
        report.issues.append("Not phrased as a question")
        report.suggestions.append(
            "Reframe as a question starting with 'How', 'What', or 'Why'"
        )
    else:
        first_word = q.split()[0] if q.split() else ""
        report.strengths.append(f"Question form ('{first_word}...')")

    # ── astrophysics vocabulary ───────────────────
    astro_words = [w for w in words if w in _ASTRO_TERMS]
    if len(astro_words) < 1:
        report.issues.append("No recognizable astrophysics terms")
        report.suggestions.append(
            "Include specific astrophysics terminology "
            "(e.g., 'galaxy cluster', 'AGN', 'dark matter')"
        )
    else:
        report.strengths.append(
            f"Uses astro terms: {', '.join(set(astro_words[:4]))}"
        )

    # ── instrument context ────────────────────────
    instruments = [w for w in words if w in _INSTRUMENTS]
    if instruments:
        report.strengths.append(
            f"Mentions instrument/method: {', '.join(set(instruments))}"
        )
    else:
        report.suggestions.append(
            "(Optional) Specify observation type — "
            "'X-ray observations', 'JWST spectra', etc."
        )

    # ── query type detection ──────────────────────
    is_mechanism    = any(p in q for p in _MECHANISTIC)
    is_quantitative = any(p in q for p in _QUANTITATIVE)

    if is_mechanism:
        report.strengths.append("Mechanism-focused (matches Q1 sub-question)")
    if is_quantitative:
        report.strengths.append("Quantitative-focused (matches Q3 sub-question)")

    # ── vague terms warning ──────────────────────
    vague_terms = ["stuff", "things", "something", "somehow", "kind of"]
    if any(v in q for v in vague_terms):
        report.issues.append("Contains vague terms")
        report.suggestions.append("Replace vague terms with specific physics language")

    # ── scoring ──────────────────────────────────
    score = 100
    score -= len(report.issues) * 15
    score += min(len(report.strengths) * 5, 25)
    score = max(0, min(100, score))
    report.score = score

    if   score >= 85: report.grade = "A"
    elif score >= 70: report.grade = "B"
    elif score >= 55: report.grade = "C"
    elif score >= 40: report.grade = "D"
    else:             report.grade = "F"

    return report
