"""
Quality gate for structured summaries from Stage 5.

Composite Q_total = 0.40 * Q_f + 0.35 * Q_c + 0.25 * Q_i

Q_f (faithfulness): fraction of answer claims verified against paper text
Q_c (coverage):     fraction of sub-questions answered
Q_i (consistency):  snippet overlap, evidence type validity, technical depth

Decision logic:
    Q_total >= 0.75 → ACCEPT
    Q_total >= 0.50 → RETRY (re-run same paper)
    Q_total <  0.50 → RE-SELECT (next paper from Stage 3 pool)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum        import Enum

from astrorag.config     import Settings, get_settings
from astrorag.llm.models import StructuredSummary
from astrorag.logger     import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# result types
# ══════════════════════════════════════════════════════════

class QualityDecision(str, Enum):
    ACCEPT    = "ACCEPT"
    RETRY     = "RETRY"
    RE_SELECT = "RE-SELECT"


@dataclass
class QualityScores:
    """Individual quality component scores."""
    Q_f:                    float
    Q_c:                    float
    Q_i:                    float
    Q_total:                float
    n_claims_total:         int   = 0
    n_claims_verified:      int   = 0
    snippet_overlap:        float = 0.0
    has_equations:          bool  = False
    has_numerical_results:  bool  = False
    evidence_type_valid:    bool  = False


@dataclass
class QualityAssessment:
    """Full quality gate output."""
    scores:    QualityScores
    decision:  QualityDecision
    reason:    str = ""

    def summary(self) -> str:
        s = self.scores
        return (
            f"Q_f={s.Q_f:.3f} ({s.n_claims_verified}/{s.n_claims_total} claims)  "
            f"Q_c={s.Q_c:.3f}  Q_i={s.Q_i:.3f}\n"
            f"Q_total={s.Q_total:.3f} → {self.decision.value}\n"
            f"Equations: {s.has_equations}  Numbers: {s.has_numerical_results}"
        )


# ══════════════════════════════════════════════════════════
# scoring helpers
# ══════════════════════════════════════════════════════════

VALID_EVIDENCE_TYPES = {
    "observational", "simulation", "theoretical",
    "statistical", "review",
}


def _compute_faithfulness(
    summary:    StructuredSummary,
    paper_text: str,
    threshold:  float = 0.40,
) -> tuple[float, int, int]:
    """
    Q_f — fraction of claim sentences whose words overlap with paper.

    A "claim" is any sentence >10 chars in any sub-question answer.
    A claim is "verified" if word overlap ratio > threshold.
    """
    paper_words = set(paper_text.lower().split())
    if not paper_words:
        return 0.0, 0, 0

    claims: list[str] = []
    for ans in summary.sub_question_answers.values():
        for sent in ans.answer_text.split("."):
            s = sent.strip()
            if len(s) > 10:
                claims.append(s)

    if not claims:
        return 0.0, 0, 0

    verified = 0
    for c in claims:
        words = set(c.lower().split())
        if not words:
            continue
        overlap = len(words & paper_words) / len(words)
        if overlap > threshold:
            verified += 1

    return verified / len(claims), verified, len(claims)


def _compute_coverage(summary: StructuredSummary) -> float:
    """Q_c — fraction of sub-questions marked as answered."""
    answers = summary.sub_question_answers
    if not answers:
        return 0.0
    n_answered = sum(1 for ans in answers.values() if ans.answered)
    return n_answered / len(answers)


def _compute_consistency(
    summary:    StructuredSummary,
    paper_text: str,
) -> tuple[float, float, bool, bool, bool]:
    """
    Q_i — consistency and technical depth.

    Starts at 1.0, subtracts penalties:
    -0.25 if key_snippet overlap with paper < 0.65
    -0.20 if evidence_type not in valid set
    -0.15 if no equations AND no numerical results
    """
    paper_words = set(paper_text.lower().split())

    penalty = 0.0

    # snippet overlap
    snippet = summary.key_snippet or ""
    snippet_words = set(snippet.lower().split())
    if snippet_words:
        overlap = len(snippet_words & paper_words) / len(snippet_words)
    else:
        overlap = 0.0

    if overlap < 0.65:
        penalty += 0.25

    # evidence type
    ev_valid = summary.evidence_type in VALID_EVIDENCE_TYPES
    if not ev_valid:
        penalty += 0.20

    # technical depth
    has_eq  = len(summary.key_equations)     > 0
    has_num = len(summary.numerical_results) > 0
    if not has_eq and not has_num:
        penalty += 0.15

    Q_i = max(0.0, 1.0 - penalty)
    return Q_i, overlap, has_eq, has_num, ev_valid


# ══════════════════════════════════════════════════════════
# main entry
# ══════════════════════════════════════════════════════════

def assess_quality(
    summary:    StructuredSummary,
    paper_text: str,
    settings:   Settings | None = None,
) -> QualityAssessment:
    """
    Score a structured summary and decide accept/retry/re-select.

    Args:
        summary:    Stage 5 structured output.
        paper_text: Full paper text for claim verification.
        settings:   Configuration (uses defaults if None).

    Returns:
        QualityAssessment with component scores and decision.
    """
    settings = settings or get_settings()

    Q_f, n_verified, n_total = _compute_faithfulness(summary, paper_text)
    Q_c = _compute_coverage(summary)
    Q_i, snippet_overlap, has_eq, has_num, ev_valid = _compute_consistency(
        summary, paper_text
    )

    Q_total = (
        settings.q_weight_faithfulness * Q_f
        + settings.q_weight_coverage     * Q_c
        + settings.q_weight_consistency  * Q_i
    )

    # decision
    if Q_total >= settings.q_accept_threshold:
        decision = QualityDecision.ACCEPT
        reason = f"Q_total {Q_total:.3f} ≥ accept threshold {settings.q_accept_threshold}"
    elif Q_total >= settings.q_retry_threshold:
        decision = QualityDecision.RETRY
        reason = f"Q_total {Q_total:.3f} in retry band"
    else:
        decision = QualityDecision.RE_SELECT
        reason = f"Q_total {Q_total:.3f} below retry threshold — re-select"

    scores = QualityScores(
        Q_f                     = Q_f,
        Q_c                     = Q_c,
        Q_i                     = Q_i,
        Q_total                 = Q_total,
        n_claims_total          = n_total,
        n_claims_verified       = n_verified,
        snippet_overlap         = snippet_overlap,
        has_equations           = has_eq,
        has_numerical_results   = has_num,
        evidence_type_valid     = ev_valid,
    )

    return QualityAssessment(
        scores   = scores,
        decision = decision,
        reason   = reason,
    )