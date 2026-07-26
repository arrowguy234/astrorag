"""
Ground truth evaluation subpackage.

Three studies for validating AstroRAG results beyond internal metrics:

1. Multi-judge: use N different LLMs to score existing library entries;
   report inter-judge agreement (Cohen's kappa, mean vote).

2. LLM swap: rerun Stages 3 and 5 with different base LLMs, measure
   how sensitive retrieval and quality are to model choice.

3. Retrieval ground truth: a stronger judge LLM independently ranks
   BM25 top-10 candidates for each query; measure how often AstroRAG
   Stage 3 selection matches the judge's top-K.
"""

from astrorag.ground_truth.judges import (
    Judge,
    JudgeVerdict,
    MultiJudgePanel,
    JUDGE_PROFILES,
)
from astrorag.ground_truth.llm_swap import (
    LLMSwapExperiment,
    SwapResult,
)
from astrorag.ground_truth.models import (
    ComparisonReport,
    GroundTruthConfig,
)
from astrorag.ground_truth.retrieval_eval import (
    RetrievalGroundTruth,
    RetrievalAgreement,
)

__all__ = [
    "Judge", "JudgeVerdict", "MultiJudgePanel", "JUDGE_PROFILES",
    "LLMSwapExperiment", "SwapResult",
    "ComparisonReport", "GroundTruthConfig",
    "RetrievalGroundTruth", "RetrievalAgreement",
]
