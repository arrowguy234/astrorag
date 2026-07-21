"""
Query → section routing.

Given a query with technical keywords, determine which paper sections
are most likely to contain the answer. Priority sections load first
in the LLM prompt so their tokens receive maximum attention.
"""

from __future__ import annotations

from astrorag.extraction.equations import extract_equations, extract_measurements
from astrorag.extraction.tables    import extract_tables


# ══════════════════════════════════════════════════════════
# routing table
# ══════════════════════════════════════════════════════════

SECTION_ROUTING: dict[str, list[str]] = {
    # equation-oriented
    "equation":     ["Methods", "Methodology", "Analysis", "Results", "Introduction"],
    "derivation":   ["Methods", "Methodology", "Analysis", "Introduction"],
    "formula":      ["Methods", "Methodology", "Analysis", "Results"],

    # measurement-oriented
    "measurement":  ["Results", "Observations", "Data", "Analysis"],
    "value":        ["Results", "Observations", "Analysis"],
    "quantity":     ["Results", "Analysis"],
    "statistic":    ["Results", "Analysis", "Methods"],
    "uncertainty":  ["Results", "Methods", "Analysis"],

    # table-oriented
    "table":        ["Results", "Observations", "Data", "Analysis"],

    # instrument-oriented
    "instrument":   ["Observations", "Data", "Methods"],
    "sample":       ["Observations", "Data", "Sample Selection"],
    "calibration":  ["Observations", "Methods", "Data"],

    # method-oriented
    "algorithm":    ["Methods", "Methodology", "Analysis"],
    "simulation":   ["Methods", "Analysis", "Results"],
    "model":        ["Methods", "Introduction", "Discussion"],
    "parameter":    ["Results", "Methods", "Analysis"],

    # physical quantities
    "mass":         ["Results", "Observations", "Analysis"],
    "luminosity":   ["Results", "Observations", "Analysis"],
    "temperature":  ["Results", "Observations", "Analysis"],
    "redshift":     ["Results", "Observations", "Data"],
    "energy":       ["Results", "Analysis"],
    "velocity":     ["Results", "Analysis"],
    "density":      ["Results", "Analysis"],
    "pressure":     ["Results", "Analysis"],

    # observation-oriented
    "observation":  ["Observations", "Data", "Results"],
    "evidence":     ["Results", "Observations", "Discussion"],
    "detection":    ["Results", "Observations"],

    # interpretive
    "mechanism":    ["Introduction", "Discussion", "Results"],
    "process":      ["Introduction", "Discussion", "Results"],
    "conclusion":   ["Conclusion", "Conclusions", "Discussion", "Summary"],
    "limitation":   ["Discussion", "Conclusion", "Conclusions"],
    "comparison":   ["Discussion", "Results", "Introduction"],
}


# ══════════════════════════════════════════════════════════
# routing logic
# ══════════════════════════════════════════════════════════

def detect_question_type(query: str) -> list[str]:
    """
    Return a prioritised list of section names for this query.

    Matches query keywords against SECTION_ROUTING and combines
    the priority lists preserving order.
    """
    q_lower  = query.lower()
    priority: list[str] = []
    seen:     set[str]  = set()

    for keyword, sections in SECTION_ROUTING.items():
        if keyword in q_lower:
            for s in sections:
                if s not in seen:
                    priority.append(s)
                    seen.add(s)

    # sensible default when no keywords match
    if not priority:
        priority = ["Results", "Methods", "Introduction", "Discussion"]

    return priority


# ══════════════════════════════════════════════════════════
# technical context builder
# ══════════════════════════════════════════════════════════

def build_technical_context(
    sections:        dict[str, "Section"],
    query:           str,
    full_text:       str,
    section_budget:  int = 2000,
    priority_budget: int = 4000,
    equations_max:   int = 30,
    tables_max:      int = 3,
    total_budget:    int = 5500,
) -> str:
    """
    Build a technically dense context block for the LLM prompt.

    Steps:
      1. Determine priority section ordering from query keywords
      2. Load priority sections first with full budget
      3. Fill remaining budget with other sections
      4. Append extracted equations block
      5. Append extracted tables block

    Args:
        sections:        Dict of section_name → Section from Stage 4.
        query:           Original research query.
        full_text:       Full paper text (for equation/table extraction).
        section_budget:  Char limit per non-priority section.
        priority_budget: Char limit per priority section.
        equations_max:   Max equations to append.
        tables_max:      Max tables to append.
        total_budget:    Overall char budget for the returned string.

    Returns:
        Formatted context string ready for LLM prompt.
    """
    priority = detect_question_type(query)

    parts:     list[str] = []
    used_keys: set[str]  = set()
    total_len            = 0

    # ── pass 1: priority sections ───────────────────────
    for target in priority:
        if total_len >= total_budget:
            break
        target_l = target.lower()
        for key, sec in sections.items():
            if target_l in key.lower() and key not in used_keys:
                if not sec.text.strip():
                    continue
                block = f"=== {key.upper()} ===\n{sec.text[:priority_budget]}"
                parts.append(block)
                total_len += len(block)
                used_keys.add(key)
                break

    # ── pass 2: remaining sections ──────────────────────
    for key, sec in sections.items():
        if total_len >= total_budget:
            break
        if key in used_keys or not sec.text.strip():
            continue
        block = f"=== {key.upper()} ===\n{sec.text[:section_budget]}"
        parts.append(block)
        total_len += len(block)

    # ── extracted equations ─────────────────────────────
    equations = extract_equations(full_text, max_results=equations_max)
    if equations:
        eq_block = "=== EXTRACTED EQUATIONS ===\n" + "\n".join(equations)
        parts.append(eq_block)

    # ── extracted measurements ──────────────────────────
    measurements = extract_measurements(full_text, max_results=25)
    if measurements:
        m_block = "=== EXTRACTED MEASUREMENTS ===\n" + "\n".join(measurements)
        parts.append(m_block)

    # ── extracted tables ────────────────────────────────
    tables = extract_tables(full_text, max_tables=tables_max)
    if tables:
        t_block = "=== EXTRACTED TABLES ===\n\n" + "\n\n".join(tables[:tables_max])
        parts.append(t_block)

    return "\n\n".join(parts)