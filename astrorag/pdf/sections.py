"""
Section header detection and splitting for astronomy papers.

Standard section names appear in various formats:
    "1. Introduction"
    "I. INTRODUCTION"
    "Introduction"
    "1 Introduction"

We match by substring on lowercased text and use position ordering
to determine section boundaries.
"""

from __future__ import annotations

import re

from astrorag.pdf.models import Section

# ══════════════════════════════════════════════════════════
# section header vocabulary
# ══════════════════════════════════════════════════════════

# ordered by typical position in astronomy papers
SECTION_HEADERS: list[str] = [
    "Abstract",
    "Introduction",
    "Background",
    "Observations",
    "Data Reduction",
    "Data Analysis",
    "Data",
    "Methods",
    "Methodology",
    "Analysis",
    "Sample Selection",
    "Sample",
    "Results",
    "Simulations",
    "Discussion",
    "Interpretation",
    "Conclusion",
    "Conclusions",
    "Summary",
    "Acknowledgements",
    "Acknowledgments",
    "References",
    "Bibliography",
    "Appendix",
]

# section importance for character budget allocation
IMPORTANT_SECTIONS: set[str] = {
    "results", "methods", "methodology", "analysis",
    "observations", "data", "data reduction",
}


# ══════════════════════════════════════════════════════════
# section detection
# ══════════════════════════════════════════════════════════

def detect_sections(full_text: str) -> list[tuple[int, str]]:
    """
    Find (position, name) of each section header in the text.

    Uses a case-insensitive substring search but requires that each
    header be at or near the beginning of a line (preceded by whitespace
    or start of text). Returns positions sorted by occurrence.
    """
    if not full_text:
        return []

    text_lower = full_text.lower()
    positions: list[tuple[int, str]] = []
    seen_names: set[str] = set()

    for header in SECTION_HEADERS:
        header_lower = header.lower()
        # find first occurrence
        idx = text_lower.find(header_lower)
        if idx < 0:
            continue

        # heuristic: header should be at start of a line, or after
        # a number/roman numeral, to avoid matching header words inside
        # sentences (e.g., "the results section" should not trigger)
        if idx > 0:
            preceding = text_lower[max(0, idx - 4):idx]
            # accept if preceded by newline, digit, roman numeral, or period+space
            if not re.search(r"[\n\r]|\d\.?\s|[ivx]+\.?\s|\A", preceding):
                continue

        if header_lower not in seen_names:
            positions.append((idx, header))
            seen_names.add(header_lower)

    positions.sort()
    return positions


def split_by_sections(
    full_text:  str,
    max_default: int = 2000,
    max_important: int = 4000,
) -> dict[str, Section]:
    """
    Split full paper text into named sections.

    Uses detected header positions to determine boundaries. Each section
    is capped at max_default or max_important characters depending on
    its role.

    Args:
        full_text:      Full extracted paper text.
        max_default:    Char budget for standard sections.
        max_important:  Char budget for Results, Methods, etc.

    Returns:
        Dict of section_name → Section object.
    """
    positions = detect_sections(full_text)
    sections: dict[str, Section] = {}

    if not positions:
        # no headers detected — return everything as "Full text"
        sections["Full text"] = Section(
            name       = "Full text",
            text       = full_text[:max_important * 2],
            char_start = 0,
            char_end   = min(len(full_text), max_important * 2),
        )
        return sections

    for i, (start, name) in enumerate(positions):
        # end = start of next header, or 3000 chars after start
        if i + 1 < len(positions):
            end = positions[i + 1][0]
        else:
            end = start + max_important

        # determine budget for this section
        is_important = name.lower() in IMPORTANT_SECTIONS
        budget = max_important if is_important else max_default

        raw_text = full_text[start:end]
        text = raw_text[:budget]

        sections[name] = Section(
            name       = name,
            text       = text,
            char_start = start,
            char_end   = start + len(text),
        )

    return sections