"""
Equation and measurement extraction from raw PDF text.

Astro-ph PDFs have equations that get scrambled during two-column
extraction. This module uses regex patterns to identify equation-like
strings regardless of their position in the text stream.
"""

from __future__ import annotations

import re


# ══════════════════════════════════════════════════════════
# pattern definitions
# ══════════════════════════════════════════════════════════

EQUATION_PATTERNS: list[tuple[str, str]] = [
    # (name, regex)
    ("display_math",       r"\$\$[^$]{3,200}\$\$"),
    ("inline_math",        r"\$[^$\n]{3,150}\$"),
    ("latex_equation_env", r"\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}"),
    ("latex_align_env",    r"\\begin\{align\*?\}(.+?)\\end\{align\*?\}"),
    ("variable_assignment", r"[A-Za-z_][\w]*\s*=\s*[^=\n]{5,80}"),
    ("proportionality",    r"[A-Za-z_][\w]*\s*[∝\\propto]\s*[A-Za-z_][\w]*[^,\.\n]{0,40}"),
    ("measurement_pm",     r"[-+]?\d+\.?\d*\s*[±\\pm]\s*\d+\.?\d*"),
    ("scientific_notation", r"[-+]?\d+\.?\d*\s*[×xX]\s*10\^?[\{]?[-+]?\d+"),
]

UNIT_PATTERN = re.compile(
    r"\d+\.?\d*(?:\s*[×xX]\s*10\^?[\{]?[-+]?\d+[\}]?)?\s*"
    r"(?:erg(?:/s)?(?:/cm[²2])?|keV|eV|MeV|GeV|TeV|"
    r"kpc|pc|Mpc|Gpc|AU|"
    r"km/s|m/s|c|"
    r"M_?sun|M_?☉|L_?sun|L_?☉|"
    r"yr|Myr|Gyr|s|"
    r"K|Hz|GHz|MHz|"
    r"Jy|mJy|mag|"
    r"arcsec|arcmin|degree|deg|rad|"
    r"cm[²2]|cm[³3]|"
    r"eV/cm[³3]|erg/cm[²2]|erg/cm[³3])"
)


# ══════════════════════════════════════════════════════════
# main extractors
# ══════════════════════════════════════════════════════════

def extract_equations(text: str, max_results: int = 30) -> list[str]:
    """
    Extract equation-like strings from raw text.

    Runs all pattern classes and combines unique matches. Deduplicates
    while preserving discovery order.

    Args:
        text:        Raw PDF text.
        max_results: Cap on returned items.

    Returns:
        List of unique equation strings.
    """
    if not text:
        return []

    found:  list[str] = []
    seen:   set[str]  = set()

    for _, pattern in EQUATION_PATTERNS:
        try:
            matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        except re.error:
            continue

        for m in matches:
            # tuple results from grouped patterns
            candidate = m if isinstance(m, str) else "".join(m)
            candidate = candidate.strip()
            if len(candidate) < 4 or len(candidate) > 200:
                continue
            # deduplicate normalised
            key = re.sub(r"\s+", " ", candidate.lower())
            if key in seen:
                continue
            seen.add(key)
            found.append(candidate)
            if len(found) >= max_results:
                return found

    return found


def extract_measurements(text: str, max_results: int = 30) -> list[str]:
    """
    Extract numeric quantities with units.

    Returns strings like "1.5 keV", "1.2e44 erg/s", "3.5 kpc".
    """
    if not text:
        return []

    matches = UNIT_PATTERN.findall(text)
    unique: list[str] = []
    seen:   set[str]  = set()

    # findall returns unit only in this pattern, so re-find with full match
    for match in UNIT_PATTERN.finditer(text):
        span = match.group(0).strip()
        if len(span) < 3 or len(span) > 60:
            continue
        norm = re.sub(r"\s+", " ", span.lower())
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(span)
        if len(unique) >= max_results:
            break

    return unique