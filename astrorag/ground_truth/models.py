"""Data models for ground truth evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime    import datetime
from typing      import Any


# ══════════════════════════════════════════════════════════
# judge verdict
# ══════════════════════════════════════════════════════════

@dataclass
class JudgeVerdict:
    """One judge's assessment of a summary."""

    judge_name:         str
    arxiv_id:           str
    query:              str

    # scores 1-5
    faithfulness:       float = 0.0   # do claims match paper?
    coverage:           float = 0.0   # are sub-questions answered?
    technical_accuracy: float = 0.0   # equations/values correct?

    # composite (mean of three)
    overall:            float = 0.0

    # decision: accept | needs_revision | reject
    verdict:            str = ""

    # short justification
    rationale:          str = ""

    # meta
    judged_at:          str = ""
    llm_error:          str = ""

    def __post_init__(self):
        if not self.judged_at:
            self.judged_at = datetime.now().isoformat()
        if self.overall == 0.0:
            valid = [x for x in (self.faithfulness, self.coverage,
                                  self.technical_accuracy) if x > 0]
            self.overall = sum(valid) / len(valid) if valid else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════
# aggregate report
# ══════════════════════════════════════════════════════════

@dataclass
class ComparisonReport:
    """Aggregated cross-judge or cross-LLM comparison report."""

    study_name:      str  # "multi_judge" | "llm_swap" | "retrieval_gt"
    n_papers:        int
    n_judges:        int  = 0
    n_variants:      int  = 0
    started_at:      str  = ""
    finished_at:     str  = ""

    per_paper:       list[dict] = field(default_factory=list)
    per_judge:       list[dict] = field(default_factory=list)
    aggregate:       dict       = field(default_factory=dict)

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════
# config
# ══════════════════════════════════════════════════════════

@dataclass
class GroundTruthConfig:
    """Config for ground truth studies."""

    judge_models:  list[str] = field(default_factory=lambda: [
        "llama-3.3-70b-versatile",     # larger Llama
        "llama-3.1-8b-instant",        # baseline for reference
        "qwen/qwen3.6-27b",         # different family
    ])

    swap_models:   list[str] = field(default_factory=lambda: [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
    ])

    # strong model used for retrieval ground truth
    retrieval_gt_model: str = "llama-3.3-70b-versatile"

    # sleep between LLM calls to respect rate limits
    sleep_between:     float = 15.0
    max_retries:       int   = 2
