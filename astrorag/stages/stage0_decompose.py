"""
Stage 0 — Query Decomposition.

Decomposes a natural language query into three targeted sub-questions
and extracts structured metadata (wavelength regime, instrument/survey
catalogs, query type).

Strategy:
  1. Attempt LLM decomposition via Groq LLaMA-3.1-8B
  2. On failure fall back to rule-based decomposition
  3. Enrich with wavelength and catalog detection regardless of method

The result is a QueryDecomposition object consumed by every downstream
stage (Stage 1 uses the query, Stages 3 and 5 use the sub-questions to
anchor generation).
"""

from __future__ import annotations

import re
from   dataclasses import dataclass

from astrorag.config      import Settings, get_settings
from astrorag.llm         import LLMClient, get_llm_client
from astrorag.llm.models  import LLMResponse, QueryDecomposition
from astrorag.logger      import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# wavelength and catalog detection
# ══════════════════════════════════════════════════════════

WAVELENGTH_KEYWORDS: dict[str, list[str]] = {
    "X-ray": [
        "x-ray", "xray", "chandra", "xmm", "xmm-newton",
        "nustar", "swift", "athena", "rosat", "einstein", "keV",
    ],
    "radio": [
        "radio", "vla", "alma", "ska", "meerkat", "lofar",
        "gmrt", "ngvla", "atca", "jansky", "GHz",
    ],
    "optical": [
        "optical", "sdss", "desi", "hst", "hubble",
        "keck", "vlt", "gemini", "subaru", "eso",
    ],
    "infrared": [
        "infrared", "spitzer", "herschel", "jwst",
        "wise", "2mass", "sofia", "mid-ir", "far-ir",
    ],
    "ultraviolet": [
        "ultraviolet", "uv", "galex", "hst-uv", "iue",
    ],
    "gamma-ray": [
        "gamma-ray", "gamma ray", "fermi", "cta",
        "veritas", "hess", "magic", "TeV",
    ],
    "gravitational-wave": [
        "gravitational wave", "ligo", "virgo", "kagra",
        "lisa", "et-explorer",
    ],
}

CATALOG_KEYWORDS: list[str] = [
    "chandra", "xmm-newton", "xmm",
    "sdss", "gaia", "desi",
    "hst", "jwst", "spitzer", "herschel",
    "alma", "vla", "ska", "meerkat", "lofar",
    "fermi", "nustar", "swift",
    "wise", "2mass", "galex", "planck",
    "wmap", "cobe", "iras", "rosat",
    "keck", "vlt", "gemini", "subaru",
    "tess", "kepler", "corot", "plato",
    "ligo", "virgo",
]

QUERY_TYPE_KEYWORDS: dict[str, list[str]] = {
    "observational": [
        "observation", "observational", "measurement",
        "detected", "measured", "spectra",
    ],
    "theoretical": [
        "theoretical", "theory", "prediction", "predicted",
        "model", "simulation", "simulated",
    ],
    "comparative": [
        "compare", "comparison", "versus", "vs", "relative to",
        "difference between",
    ],
}


def detect_wavelength(query: str) -> str:
    """Rule-based wavelength regime detection."""
    q_low = query.lower()
    for regime, keywords in WAVELENGTH_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in q_low:
                return regime
    return "multi-wavelength"


def detect_catalogs(query: str) -> list[str]:
    """Extract instrument or survey names mentioned in query."""
    q_low = query.lower()
    found: list[str] = []
    for name in CATALOG_KEYWORDS:
        if name.lower() in q_low and name.upper() not in found:
            found.append(name.upper())
    return found


def detect_query_type(query: str) -> str:
    """Classify query intent."""
    q_low = query.lower()
    for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in q_low:
                return qtype
    return "general"


# ══════════════════════════════════════════════════════════
# rule-based fallback
# ══════════════════════════════════════════════════════════

def _rule_based_decompose(query: str) -> QueryDecomposition:
    """
    Generate a decomposition from keyword rules only.

    Used when the LLM fails or is unavailable. Sub-questions
    follow the standard mechanism/evidence/measurement pattern.
    """
    wl        = detect_wavelength(query)
    catalogs  = detect_catalogs(query)
    qtype     = detect_query_type(query)
    truncated = query[:80]

    sub_qs = {
        "Q1": f"What is the primary physical mechanism at play in: {truncated}?",
        "Q2": (f"What observational evidence supports this in {wl} observations?"
               if wl != "multi-wavelength"
               else "What observational evidence supports the claims made?"),
        "Q3": "What quantitative measurements or numerical results are reported?",
    }

    return QueryDecomposition(
        original_query = query,
        sub_questions  = sub_qs,
        wavelength     = wl,
        catalogs       = catalogs,
        query_type     = qtype,
    )


# ══════════════════════════════════════════════════════════
# LLM decomposition
# ══════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are an expert astrophysicist assisting with scientific literature retrieval. "
    "You decompose complex research queries into three targeted sub-questions "
    "that together capture the full information need of the original query. "
    "Return valid JSON only. No markdown fences, no commentary."
)


def _build_user_prompt(query: str, wl: str, catalogs: list[str]) -> str:
    catalog_hint = (
        f"Detected instruments/surveys: {', '.join(catalogs)}. "
        if catalogs else ""
    )
    wl_hint = (
        f"Wavelength regime appears to be: {wl}. "
        if wl != "multi-wavelength" else ""
    )
    return (
        f'Decompose this astrophysics research query into three sub-questions.\n\n'
        f'Query: "{query}"\n\n'
        f'{wl_hint}{catalog_hint}\n\n'
        f'Requirements:\n'
        f'  Q1 must target the primary physical mechanism or process.\n'
        f'  Q2 must target the observational evidence or data.\n'
        f'  Q3 must target the quantitative measurements or numerical results.\n\n'
        f'Each sub-question must be specific and directly answerable from a single '
        f'astrophysics paper. Avoid vague or overly broad questions.\n\n'
        f'Return this exact JSON schema:\n'
        f'{{\n'
        f'  "original_query": "<the original query verbatim>",\n'
        f'  "sub_questions": {{\n'
        f'    "Q1": "<mechanism sub-question>",\n'
        f'    "Q2": "<evidence sub-question>",\n'
        f'    "Q3": "<quantitative sub-question>"\n'
        f'  }},\n'
        f'  "wavelength": "{wl}",\n'
        f'  "catalogs": {catalogs if catalogs else "[]"},\n'
        f'  "query_type": "observational | theoretical | comparative | general"\n'
        f'}}'
    )


# ══════════════════════════════════════════════════════════
# main stage class
# ══════════════════════════════════════════════════════════

@dataclass
class Stage0Result:
    """Wrapper around the decomposition with call telemetry."""
    decomposition:  QueryDecomposition
    llm_response:   LLMResponse | None = None
    fallback_used:  bool = False
    total_time_s:   float = 0.0


class Stage0Decompose:
    """
    Stage 0 — Query Decomposition.

    Usage:
        stage0 = Stage0Decompose()
        result = stage0.run("How do AGN jets suppress star formation?")
        print(result.decomposition.summary())
    """

    def __init__(
        self,
        settings:   Settings   | None = None,
        llm_client: LLMClient  | None = None,
    ) -> None:
        self.settings   = settings or get_settings()
        self._llm       = llm_client
        # cache LLM availability check
        self._llm_available: bool | None = None

    # ── lazy LLM client property ────────────────────────
    @property
    def llm(self) -> LLMClient | None:
        if self._llm is not None:
            return self._llm
        try:
            self._llm = get_llm_client()
            self._llm_available = True
            return self._llm
        except Exception as e:
            logger.warning(f"LLM client unavailable: {e}")
            self._llm_available = False
            return None

    # ── main entry point ────────────────────────────────
    def run(
        self,
        query:              str,
        use_llm:            bool = True,
        rule_based_only:    bool = False,
    ) -> Stage0Result:
        """
        Decompose a query into sub-questions.

        Args:
            query:           Natural language research query.
            use_llm:         If True, attempt LLM decomposition first.
            rule_based_only: If True, skip LLM and use only rules.

        Returns:
            Stage0Result with decomposition and telemetry.
        """
        import time
        t0 = time.time()

        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        logger.info(f"Stage 0 — decomposing: {query[:80]}")

        # ── run rule-based extraction first (always used) ─
        wl        = detect_wavelength(query)
        catalogs  = detect_catalogs(query)

        # ── try LLM decomposition ───────────────────────
        if use_llm and not rule_based_only:
            client = self.llm
            if client is not None:
                try:
                    llm_result = self._run_llm(query, wl, catalogs, client)
                    result = Stage0Result(
                        decomposition = llm_result.data,
                        llm_response  = llm_result,
                        fallback_used = False,
                        total_time_s  = time.time() - t0,
                    )
                    self._log_result(result, source="LLM")
                    return result
                except Exception as e:
                    logger.warning(
                        f"LLM decomposition failed, falling back to rules: {e}"
                    )

        # ── rule-based fallback ─────────────────────────
        decomposition = _rule_based_decompose(query)
        result = Stage0Result(
            decomposition = decomposition,
            llm_response  = None,
            fallback_used = True,
            total_time_s  = time.time() - t0,
        )
        self._log_result(result, source="rules")
        return result

    # ── internal: LLM path ──────────────────────────────
    def _run_llm(
        self,
        query:    str,
        wl:       str,
        catalogs: list[str],
        client:   LLMClient,
    ) -> LLMResponse:
        user_prompt = _build_user_prompt(query, wl, catalogs)
        response = client.chat_json(
            system      = _SYSTEM_PROMPT,
            user        = user_prompt,
            schema      = QueryDecomposition,
            max_tokens  = 500,
            stage_name  = "stage0",
        )
        # ensure original_query matches (LLM sometimes rephrases)
        response.data.original_query = query
        # ensure wavelength and catalogs consistent with rules
        if response.data.wavelength == "multi-wavelength" and wl != "multi-wavelength":
            response.data.wavelength = wl
        if catalogs and not response.data.catalogs:
            response.data.catalogs = catalogs
        return response

    # ── logging helper ──────────────────────────────────
    def _log_result(self, result: Stage0Result, source: str) -> None:
        d = result.decomposition
        logger.info(
            f"Stage 0 done in {result.total_time_s:.2f}s "
            f"[{source}] wl={d.wavelength} cats={d.catalogs}"
        )
        logger.debug(f"  Q1: {d.sub_questions['Q1']}")
        logger.debug(f"  Q2: {d.sub_questions['Q2']}")
        logger.debug(f"  Q3: {d.sub_questions['Q3']}")


# ══════════════════════════════════════════════════════════
# convenience function
# ══════════════════════════════════════════════════════════

def decompose_query(
    query:           str,
    use_llm:         bool = True,
    rule_based_only: bool = False,
) -> QueryDecomposition:
    """
    One-shot query decomposition — convenience wrapper.

    Args:
        query:           Natural language research query.
        use_llm:         If True, use LLM (falls back to rules on error).
        rule_based_only: If True, skip LLM entirely.

    Returns:
        QueryDecomposition object.
    """
    stage = Stage0Decompose()
    return stage.run(
        query           = query,
        use_llm         = use_llm,
        rule_based_only = rule_based_only,
    ).decomposition