"""
Curated evaluation query set.

20 queries spanning subdomains of astrophysics. Chosen for:
- domain diversity
- realistic research questions (not toy questions)
- coverage of mechanism/evidence/quantitative axes
- mix of well-established and cutting-edge topics
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationQuery:
    idx:       int
    query:     str
    subdomain: str


# ══════════════════════════════════════════════════════════
# canonical query set
# ══════════════════════════════════════════════════════════

DEFAULT_QUERY_SET: list[EvaluationQuery] = [
    # ── AGN and cluster physics ────────────────────────
    EvaluationQuery(1,
        "How do AGN jets suppress star formation in massive elliptical galaxies "
        "through X-ray cavity observations?",
        "AGN feedback"),
    EvaluationQuery(2,
        "How does the intracluster medium cool in galaxy clusters and what "
        "regulates the cooling flow?",
        "ICM cooling"),
    EvaluationQuery(3,
        "What is the observed relationship between AGN jet power and X-ray "
        "cavity enthalpy in cluster central galaxies?",
        "AGN feedback"),

    # ── black hole physics ─────────────────────────────
    EvaluationQuery(4,
        "What is the relationship between galaxy stellar mass and central "
        "supermassive black hole mass?",
        "BH scaling"),
    EvaluationQuery(5,
        "What determines the M-sigma relation between supermassive black hole "
        "mass and host galaxy velocity dispersion?",
        "BH scaling"),

    # ── cosmology ──────────────────────────────────────
    EvaluationQuery(6,
        "How are photometric redshifts calibrated using spectroscopic survey "
        "data from SDSS and DESI?",
        "cosmology"),
    EvaluationQuery(7,
        "How do cosmic microwave background measurements constrain the "
        "Hubble constant?",
        "cosmology"),
    EvaluationQuery(8,
        "What role does baryon acoustic oscillations play in measuring "
        "dark energy equation of state?",
        "cosmology"),

    # ── galaxy formation ───────────────────────────────
    EvaluationQuery(9,
        "What mechanisms quench star formation in massive early-type galaxies "
        "at high redshift?",
        "galaxy formation"),
    EvaluationQuery(10,
        "How does gas accretion drive galaxy stellar mass assembly across "
        "cosmic time?",
        "galaxy formation"),
    EvaluationQuery(11,
        "What are the properties of Milky Way progenitor galaxies observed "
        "with JWST at high redshift?",
        "galaxy formation"),

    # ── stellar physics ────────────────────────────────
    EvaluationQuery(12,
        "How do binary neutron star mergers produce heavy elements through "
        "the r-process?",
        "stellar"),
    EvaluationQuery(13,
        "What determines the initial mass function in molecular cloud "
        "star formation?",
        "stellar"),

    # ── high-energy ────────────────────────────────────
    EvaluationQuery(14,
        "How are ultra-high-energy cosmic rays accelerated in astrophysical "
        "sources?",
        "high energy"),
    EvaluationQuery(15,
        "What is the origin of gamma-ray bursts and their host galaxy "
        "environments?",
        "high energy"),

    # ── compact objects ────────────────────────────────
    EvaluationQuery(16,
        "How are gravitational wave signals from binary black hole mergers "
        "extracted from LIGO detector noise?",
        "compact objects"),
    EvaluationQuery(17,
        "What is the mass distribution of stellar-mass black holes measured "
        "through gravitational wave observations?",
        "compact objects"),

    # ── dark matter ────────────────────────────────────
    EvaluationQuery(18,
        "How is dark matter distributed in dwarf spheroidal galaxies "
        "as inferred from stellar kinematics?",
        "dark matter"),
    EvaluationQuery(19,
        "What are the constraints on WIMP dark matter cross-sections from "
        "direct detection experiments?",
        "dark matter"),

    # ── ISM ────────────────────────────────────────────
    EvaluationQuery(20,
        "How does the interstellar magnetic field affect molecular cloud "
        "collapse and star formation efficiency?",
        "ISM"),
]

QUERY_SUBDOMAINS: list[str] = sorted({q.subdomain for q in DEFAULT_QUERY_SET})


def get_query_set(
    n:           int | None = None,
    subdomains:  list[str] | None = None,
) -> list[EvaluationQuery]:
    """
    Return a filtered subset of the default query set.

    Args:
        n:          If given, return at most n queries.
        subdomains: If given, filter to only these subdomains.
    """
    queries = list(DEFAULT_QUERY_SET)
    if subdomains:
        keep = set(subdomains)
        queries = [q for q in queries if q.subdomain in keep]
    if n is not None:
        queries = queries[:n]
    return queries