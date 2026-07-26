"""
Query template definitions.

Astrophysics research questions typically fall into a small number
of patterns. Each template gives the user a scaffold to fill in.
"""

from __future__ import annotations

from dataclasses import dataclass
from   typing    import Any


@dataclass
class QueryTemplate:
    """One query template with placeholders."""

    id:               str
    label:            str
    icon:             str
    template:        str
    description:     str
    example_filled:  str
    fields:          list[dict[str, Any]]


# ══════════════════════════════════════════════════════════
# templates
# ══════════════════════════════════════════════════════════

QUERY_TEMPLATES: list[QueryTemplate] = [

    QueryTemplate(
        id       = "mechanism",
        label    = "Mechanism",
        icon     = "⚙️",
        template = "How does {agent} {action} {target} in {context}?",
        description = "Investigate a physical mechanism or causal relationship.",
        example_filled = "How do AGN jets suppress star formation in massive elliptical galaxies?",
        fields = [
            {"name": "agent",   "label": "Physical agent",
             "placeholder": "e.g. AGN jets, supernovae, dark matter",
             "examples": ["AGN jets", "supernova feedback", "gravitational waves",
                          "cosmic rays", "magnetic fields"]},
            {"name": "action",  "label": "Action / effect",
             "placeholder": "e.g. suppress, drive, regulate",
             "examples": ["suppress", "drive", "regulate", "trigger",
                          "heat", "compress", "accelerate"]},
            {"name": "target",  "label": "Target quantity",
             "placeholder": "e.g. star formation",
             "examples": ["star formation", "molecular gas",
                          "stellar mass assembly", "galaxy morphology"]},
            {"name": "context", "label": "System / context",
             "placeholder": "e.g. massive elliptical galaxies",
             "examples": ["massive elliptical galaxies", "galaxy clusters",
                          "protoplanetary disks", "high-redshift galaxies"]},
        ],
    ),

    QueryTemplate(
        id       = "observation",
        label    = "Observation",
        icon     = "🔭",
        template = "What do {instrument} observations reveal about {phenomenon}?",
        description = "Report findings from a specific instrument or survey.",
        example_filled = "What do JWST NIRSpec observations reveal about high-redshift galaxy spectra?",
        fields = [
            {"name": "instrument", "label": "Instrument / survey",
             "placeholder": "e.g. JWST, ALMA, LIGO",
             "examples": ["JWST NIRSpec", "ALMA", "Chandra", "LIGO-Virgo",
                          "DESI", "Hubble Space Telescope", "Euclid"]},
            {"name": "phenomenon", "label": "Phenomenon of interest",
             "placeholder": "e.g. high-redshift galaxy spectra",
             "examples": ["z > 10 galaxy properties", "gravitational wave signals",
                          "dust emission", "molecular clouds",
                          "high-energy cosmic rays"]},
        ],
    ),

    QueryTemplate(
        id       = "quantitative",
        label    = "Quantitative",
        icon     = "🔢",
        template = "What is the measured value of {quantity} in {system}?",
        description = "Find specific numerical measurements.",
        example_filled = "What is the measured value of the Hubble constant from CMB observations?",
        fields = [
            {"name": "quantity", "label": "Quantity",
             "placeholder": "e.g. the Hubble constant",
             "examples": ["the Hubble constant", "supermassive BH mass",
                          "jet power", "cluster mass", "the Sigma-8 parameter"]},
            {"name": "system",   "label": "System / measurement type",
             "placeholder": "e.g. CMB observations",
             "examples": ["CMB observations", "type Ia supernovae",
                          "gravitational lensing", "BAO measurements"]},
        ],
    ),

    QueryTemplate(
        id       = "comparison",
        label    = "Comparison",
        icon     = "⚖️",
        template = "How do {a} and {b} compare in their effect on {target}?",
        description = "Compare two mechanisms, models, or systems.",
        example_filled = "How do quasar-mode and radio-mode AGN feedback compare in their effect on the ICM?",
        fields = [
            {"name": "a",       "label": "First system / mechanism",
             "placeholder": "e.g. quasar-mode feedback",
             "examples": ["quasar-mode feedback", "single-star formation",
                          "cold dark matter", "MHD turbulence"]},
            {"name": "b",       "label": "Second system / mechanism",
             "placeholder": "e.g. radio-mode feedback",
             "examples": ["radio-mode feedback", "binary star formation",
                          "warm dark matter", "hydrodynamic turbulence"]},
            {"name": "target",  "label": "Effect on",
             "placeholder": "e.g. the intracluster medium",
             "examples": ["the intracluster medium", "star formation efficiency",
                          "galaxy morphology", "halo structure"]},
        ],
    ),

    QueryTemplate(
        id       = "review",
        label    = "Review",
        icon     = "📚",
        template = "What is our current understanding of {topic}?",
        description = "Get a broad review of a topic (returns foundational papers).",
        example_filled = "What is our current understanding of dark matter halo structure?",
        fields = [
            {"name": "topic", "label": "Topic",
             "placeholder": "e.g. dark matter halo structure",
             "examples": ["dark matter halo structure",
                          "the origin of gamma-ray bursts",
                          "black hole scaling relations",
                          "the initial mass function"]},
        ],
    ),
]


# ══════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════

def get_template(template_id: str) -> QueryTemplate | None:
    """Get a template by ID."""
    for t in QUERY_TEMPLATES:
        if t.id == template_id:
            return t
    return None


def fill_template(template: QueryTemplate, values: dict[str, str]) -> str:
    """Fill a template with user-provided values."""
    filled = template.template
    for field in template.fields:
        name = field["name"]
        val  = values.get(name, "").strip() or f"[{field['label']}]"
        filled = filled.replace("{" + name + "}", val)
    return filled
