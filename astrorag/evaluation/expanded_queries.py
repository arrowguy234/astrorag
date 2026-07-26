"""
Expanded 45-query evaluation set for final ground-truth study.

Original 20 queries retained. 25 new queries added covering:
- MHD, plasma physics, radiative transfer
- New instruments: JWST, ALMA, LIGO/Virgo, Euclid, Roman
- Exoplanets, solar physics, cosmic rays
- Historical / classical (pre-2000) topics
- Very recent (2024-2025) topics
- Review, comparative, and methodological questions
"""

from __future__ import annotations

from astrorag.evaluation.queries import (
    EvaluationQuery,
    DEFAULT_QUERY_SET,
)


# ══════════════════════════════════════════════════════════
# new queries (starting from idx 21)
# ══════════════════════════════════════════════════════════

NEW_QUERIES: list[EvaluationQuery] = [
    # ── MHD / plasma ──────────────────────────────────
    EvaluationQuery(21,
        "How do magnetohydrodynamic instabilities regulate accretion "
        "in black hole accretion disks?",
        "MHD/plasma"),
    EvaluationQuery(22,
        "What role does the magnetorotational instability play in "
        "angular momentum transport in protoplanetary disks?",
        "MHD/plasma"),

    # ── Radiative transfer ────────────────────────────
    EvaluationQuery(23,
        "How does radiative transfer through dust affect infrared "
        "emission from star-forming regions?",
        "radiative transfer"),
    EvaluationQuery(24,
        "What are the observational signatures of Lyman-alpha "
        "radiative transfer in high-redshift galaxies?",
        "radiative transfer"),

    # ── Nuclear astrophysics ──────────────────────────
    EvaluationQuery(25,
        "How does nucleosynthesis in core-collapse supernovae produce "
        "the abundance patterns observed in metal-poor stars?",
        "nuclear astro"),
    EvaluationQuery(26,
        "What is the origin of the p-process elements in stellar "
        "explosions?",
        "nuclear astro"),

    # ── JWST-specific ─────────────────────────────────
    EvaluationQuery(27,
        "What have JWST NIRSpec observations revealed about the "
        "spectra of galaxies at z > 10?",
        "JWST"),
    EvaluationQuery(28,
        "How do JWST MIRI observations constrain dust properties in "
        "nearby galaxies?",
        "JWST"),

    # ── ALMA / mm ─────────────────────────────────────
    EvaluationQuery(29,
        "What does ALMA reveal about molecular gas kinematics in "
        "galaxy centers?",
        "mm astronomy"),
    EvaluationQuery(30,
        "How do ALMA observations of dust continuum trace star "
        "formation in protoplanetary disks?",
        "mm astronomy"),

    # ── LIGO / GW ─────────────────────────────────────
    EvaluationQuery(31,
        "What are the constraints on neutron star equations of state "
        "from LIGO-Virgo binary neutron star merger observations?",
        "gravitational waves"),
    EvaluationQuery(32,
        "How do gravitational wave detections inform our understanding "
        "of black hole formation channels?",
        "gravitational waves"),

    # ── Exoplanets ────────────────────────────────────
    EvaluationQuery(33,
        "What determines the atmospheric composition of hot Jupiters "
        "as measured by transmission spectroscopy?",
        "exoplanets"),
    EvaluationQuery(34,
        "How does atmospheric escape affect the demographics of "
        "sub-Neptune exoplanets?",
        "exoplanets"),

    # ── Solar physics ─────────────────────────────────
    EvaluationQuery(35,
        "What triggers solar coronal mass ejections and how are they "
        "predicted?",
        "solar"),
    EvaluationQuery(36,
        "How does helioseismology constrain the internal structure "
        "and rotation of the Sun?",
        "solar"),

    # ── Cosmic rays / high-energy ─────────────────────
    EvaluationQuery(37,
        "What are the astrophysical sources of high-energy neutrinos "
        "detected by IceCube?",
        "cosmic rays"),
    EvaluationQuery(38,
        "How does the cosmic ray spectrum reveal information about "
        "propagation through the galaxy?",
        "cosmic rays"),

    # ── Atomic / spectral processes ───────────────────
    EvaluationQuery(39,
        "What atomic processes drive the observed line ratios in "
        "photoionized nebulae?",
        "atomic processes"),
    EvaluationQuery(40,
        "How are recombination lines used to measure temperatures "
        "and densities in HII regions?",
        "atomic processes"),

    # ── Historical / methodological ───────────────────
    EvaluationQuery(41,
        "How did early observations of gravitational lensing test "
        "general relativity?",
        "GR tests"),
    EvaluationQuery(42,
        "What is the role of Bayesian inference in modern "
        "gravitational wave parameter estimation?",
        "methodology"),

    # ── Comparative / cross-cutting ───────────────────
    EvaluationQuery(43,
        "How do observations from different wavelength regimes "
        "combine to constrain AGN spectral energy distributions?",
        "multi-wavelength"),
    EvaluationQuery(44,
        "What are the tensions between different measurements of "
        "the Hubble constant and how might they be resolved?",
        "cosmology tensions"),
    EvaluationQuery(45,
        "How do simulations and observations disagree on the number "
        "of Milky Way satellite galaxies?",
        "cosmology tensions"),
]


# ══════════════════════════════════════════════════════════
# combined 45-query set
# ══════════════════════════════════════════════════════════

EXPANDED_QUERY_SET: list[EvaluationQuery] = list(DEFAULT_QUERY_SET) + NEW_QUERIES


def get_expanded_set(
    n:            int | None = None,
    only_new:     bool       = False,
    subdomains:   list[str] | None = None,
) -> list[EvaluationQuery]:
    """
    Return the expanded query set with optional filtering.

    Args:
        n:          If given, return at most n queries.
        only_new:   If True, return only the 25 new queries (idx 21+).
        subdomains: If given, filter to only these subdomains.
    """
    queries = NEW_QUERIES if only_new else EXPANDED_QUERY_SET
    if subdomains:
        keep = set(subdomains)
        queries = [q for q in queries if q.subdomain in keep]
    if n is not None:
        queries = queries[:n]
    return queries
