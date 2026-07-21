"""
Ablation configurations.

Each AblationVariant defines which pipeline components are ENABLED
for a given run. The pipeline stages themselves are unchanged — the
ablation runner conditionally bypasses their outputs.

Variants (name → what is disabled from full pipeline):
    full            all stages enabled (control, matches Step 9 baseline)
    no_graph        Stage 2 outputs ignored; Stage 3 seeded with BM25 only
    no_llm_rerank   Stage 3 skipped; Stage 4 fed top-BM25 paper directly
    no_pdf          Stages 4-5 skipped; summary from abstract only
    no_quality_gate Stage 5 first attempt always accepted (no re-selection)
    bm25_only       Naive baseline — top-BM25 abstract as final answer
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════
# variant specification
# ══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AblationVariant:
    """
    One ablation configuration.

    Each flag is True when the component IS ENABLED. Setting a flag
    to False bypasses that component while leaving upstream stages
    running unchanged (so their timing and outputs remain measurable).
    """

    name:                  str
    description:           str

    # feature flags — True = component enabled
    use_graph:             bool = True   # Stage 2 PPR reranking applied to Stage 3 input
    use_llm_rerank:        bool = True   # Stage 3 LLM selects; else top-BM25 chosen
    use_pdf:               bool = True   # Stage 4-5 run on PDF; else abstract only
    use_quality_gate:      bool = True   # Re-selection on quality failure allowed
    use_full_summary:      bool = True   # Stage 5 runs; else return raw abstract

    # provenance
    is_baseline: bool = False    # marks the control run


# ══════════════════════════════════════════════════════════
# canonical variant set
# ══════════════════════════════════════════════════════════

ABLATION_VARIANTS: list[AblationVariant] = [
    AblationVariant(
        name             = "full",
        description      = "Full pipeline — all components enabled (control)",
        is_baseline      = True,
    ),
    AblationVariant(
        name             = "no_graph",
        description      = "Graph disabled — Stage 3 sees BM25-only ordering",
        use_graph        = False,
    ),
    AblationVariant(
        name             = "no_llm_rerank",
        description      = "LLM reranking disabled — top-BM25 paper selected directly",
        use_llm_rerank   = False,
    ),
    AblationVariant(
        name             = "no_pdf",
        description      = "PDF extraction disabled — summarisation from abstract only",
        use_pdf          = False,
    ),
    AblationVariant(
        name             = "no_quality_gate",
        description      = "Quality gate off — first Stage 5 output always accepted",
        use_quality_gate = False,
    ),
    AblationVariant(
        name             = "bm25_only",
        description      = "Baseline — top-BM25 abstract returned as answer",
        use_graph        = False,
        use_llm_rerank   = False,
        use_pdf          = False,
        use_quality_gate = False,
        use_full_summary = False,
    ),
]


def get_variant(name: str) -> AblationVariant:
    """Look up a variant by name."""
    for v in ABLATION_VARIANTS:
        if v.name == name:
            return v
    raise ValueError(
        f"Unknown ablation variant: {name}. "
        f"Available: {[v.name for v in ABLATION_VARIANTS]}"
    )


def get_all_variant_names() -> list[str]:
    return [v.name for v in ABLATION_VARIANTS]