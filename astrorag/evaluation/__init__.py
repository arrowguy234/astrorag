"""
Evaluation harness — runs the full pipeline on a query set,
records structured traces, and computes aggregate metrics.

Includes ablation runner for comparing pipeline component variants
without modifying the original pipeline code.
"""

from astrorag.evaluation.ablation import (
    AblationVariant,
    ABLATION_VARIANTS,
    get_variant,
    get_all_variant_names,
)
from astrorag.evaluation.ablation_metrics import (
    VariantComparison,
    compute_variant_comparison,
    format_ablation_table,
    load_all_variants,
)
from astrorag.evaluation.ablation_runner import AblationRunner
from astrorag.evaluation.metrics import (
    EvaluationMetrics,
    compute_metrics,
    format_metrics_table,
)
from astrorag.evaluation.models  import (
    QueryTrace,
    EvaluationResult,
    Stage0Trace,
    Stage1Trace,
    Stage2Trace,
    Stage3Trace,
    Stage4Trace,
    Stage5Trace,
)
from astrorag.evaluation.queries import (
    DEFAULT_QUERY_SET,
    QUERY_SUBDOMAINS,
    get_query_set,
)
from astrorag.evaluation.runner  import EvaluationRunner

__all__ = [
    # base evaluation
    "EvaluationRunner",
    "EvaluationResult",
    "EvaluationMetrics",
    "QueryTrace",
    "Stage0Trace",
    "Stage1Trace",
    "Stage2Trace",
    "Stage3Trace",
    "Stage4Trace",
    "Stage5Trace",
    "DEFAULT_QUERY_SET",
    "QUERY_SUBDOMAINS",
    "get_query_set",
    "compute_metrics",
    "format_metrics_table",
    # ablation
    "AblationVariant",
    "ABLATION_VARIANTS",
    "get_variant",
    "get_all_variant_names",
    "AblationRunner",
    "VariantComparison",
    "compute_variant_comparison",
    "format_ablation_table",
    "load_all_variants",
]