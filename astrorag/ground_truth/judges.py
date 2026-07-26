"""
LLM-as-judge module.

Given a library entry (summary + paper text), asks N different LLMs
to independently score faithfulness, coverage, and technical accuracy.
Aggregates verdicts across judges with inter-judge agreement metrics.
"""

from __future__ import annotations

import json
import re
import time
from   dataclasses import dataclass

from astrorag.chat.library      import LibraryEntry
from astrorag.config            import Settings, get_settings
from astrorag.ground_truth.models import (
    JudgeVerdict,
    ComparisonReport,
)
from astrorag.llm               import LLMClient
from astrorag.logger            import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# judge profiles — model + label
# ══════════════════════════════════════════════════════════

JUDGE_PROFILES: dict[str, str] = {
    "llama-3.1-8b-instant":     "Llama-3.1-8B (baseline)",
    "llama-3.3-70b-versatile":  "Llama-3.3-70B (larger)",
    "openai/gpt-oss-120b":      "GPT-OSS-120B (different family)",
    "moonshotai/kimi-k2-instruct": "Kimi K2 (Moonshot)",
    "deepseek-r1-distill-llama-70b": "DeepSeek-R1-Distill",
    "qwen/qwen3.6-27b":         "Qwen3.6-27B (different family)",
}


# ══════════════════════════════════════════════════════════
# judge prompt
# ══════════════════════════════════════════════════════════

_JUDGE_SYSTEM = (
    "You are an expert astrophysics reviewer. You will evaluate a "
    "structured summary produced by an AI retrieval-augmented system. "
    "You must assess the summary against the paper's actual content. "
    "Respond ONLY with valid JSON matching the requested schema."
)


def _build_judge_prompt(entry: LibraryEntry, paper_snippet: str = "") -> str:
    """Build the judge user prompt."""

    # format equations and numerical results
    eq_lines = "\n".join(
        f"  - {e.get('equation', '')} (vars: {e.get('variables', '')})"
        for e in entry.key_equations[:8]
    ) or "  (none)"

    num_lines = "\n".join(
        f"  - {n.get('quantity', '')} = {n.get('value', '')} "
        f"± {n.get('uncertainty', '')} {n.get('unit', '')}"
        for n in entry.numerical_results[:10]
    ) or "  (none)"

    sqa_lines = []
    for qk in ["Q1", "Q2", "Q3"]:
        ans = entry.sub_question_answers.get(qk, {})
        if ans.get("answered"):
            sqa_lines.append(
                f"  {qk} (section={ans.get('section', 'unknown')}): "
                f"{ans.get('answer_text', '')[:300]}"
            )
    sqa_block = "\n".join(sqa_lines) or "  (no sub-question answers)"

    paper_ctx = ""
    if paper_snippet:
        paper_ctx = (
            f"\n\nPAPER TEXT (excerpt for reference):\n"
            f"{paper_snippet[:3000]}\n"
        )

    return f"""Evaluate this AI-generated summary of arXiv:{entry.arxiv_id}.

USER QUERY: {entry.original_query}

SUMMARY OVERVIEW:
{entry.paper_overview}

SUB-QUESTION ANSWERS:
{sqa_block}

EXTRACTED EQUATIONS:
{eq_lines}

EXTRACTED NUMERICAL RESULTS:
{num_lines}

INSTRUMENTS IDENTIFIED: {", ".join(entry.instruments) or "(none)"}
EVIDENCE TYPE: {entry.evidence_type or "unspecified"}
{paper_ctx}

Score the summary on three axes (each 1.0 to 5.0):

1. FAITHFULNESS: Do the claims (equations, values, mechanisms) plausibly
   come from an astrophysics paper on this topic? Are the equations
   physically meaningful?
2. COVERAGE: Are the three sub-questions (mechanism, evidence, quantitative)
   substantively addressed?
3. TECHNICAL_ACCURACY: Are the extracted equations and numerical values
   dimensionally consistent, correctly formatted, and physically plausible?

Then give an overall verdict: "accept" | "needs_revision" | "reject"
and a one-sentence rationale.

Respond as JSON:
{{"faithfulness": <1-5>, "coverage": <1-5>, "technical_accuracy": <1-5>,
  "verdict": "<accept|needs_revision|reject>", "rationale": "<one sentence>"}}"""


# ══════════════════════════════════════════════════════════
# single judge
# ══════════════════════════════════════════════════════════

class Judge:
    """Single LLM judge for scoring library entries."""

    def __init__(
        self,
        model:    str,
        settings: Settings | None = None,
    ) -> None:
        self.model    = model
        self.label    = JUDGE_PROFILES.get(model, model)
        self.settings = settings or get_settings()

        # instantiate a dedicated LLM client with this model
        # copy settings and override the model
        import copy
        judge_settings = copy.copy(self.settings)
        judge_settings.groq_model = model
        self.client = LLMClient(settings=judge_settings)

    def score(
        self,
        entry:         LibraryEntry,
        paper_snippet: str = "",
    ) -> JudgeVerdict:
        """Score one library entry."""

        user_prompt = _build_judge_prompt(entry, paper_snippet)

        try:
            raw, _ = self.client.chat(
                system      = _JUDGE_SYSTEM,
                user        = user_prompt,
                temperature = 0.0,
                max_tokens  = 400,
                stage_name  = f"judge_{self.model}",
            )
            payload = _extract_json(raw)

            return JudgeVerdict(
                judge_name         = self.label,
                arxiv_id           = entry.arxiv_id,
                query              = entry.original_query,
                faithfulness       = float(payload.get("faithfulness", 0)),
                coverage           = float(payload.get("coverage", 0)),
                technical_accuracy = float(payload.get("technical_accuracy", 0)),
                verdict            = str(payload.get("verdict", "")),
                rationale          = str(payload.get("rationale", "")),
            )

        except Exception as e:
            logger.warning(f"Judge {self.model} failed on {entry.arxiv_id}: {e}")
            return JudgeVerdict(
                judge_name = self.label,
                arxiv_id   = entry.arxiv_id,
                query      = entry.original_query,
                llm_error  = f"{type(e).__name__}: {e}",
            )


def _extract_json(raw: str) -> dict:
    """Best-effort JSON extraction from LLM output."""
    # try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # find first {...} block
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # last resort: regex extract fields
    result = {}
    for field in ["faithfulness", "coverage", "technical_accuracy"]:
        m = re.search(rf'"{field}"\s*:\s*(\d+\.?\d*)', raw)
        if m:
            result[field] = float(m.group(1))
    for field in ["verdict", "rationale"]:
        m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', raw)
        if m:
            result[field] = m.group(1)
    return result


# ══════════════════════════════════════════════════════════
# multi-judge panel
# ══════════════════════════════════════════════════════════

class MultiJudgePanel:
    """Run N judges against a set of library entries."""

    def __init__(
        self,
        judge_models: list[str],
        settings:     Settings | None = None,
        sleep_between: float = 15.0,
    ) -> None:
        self.judges = [Judge(m, settings) for m in judge_models]
        self.sleep_between = sleep_between

    def evaluate(
        self,
        entries: list[LibraryEntry],
    ) -> ComparisonReport:
        """Run all judges over all entries, produce aggregate report."""

        all_verdicts: list[JudgeVerdict] = []

        for i, entry in enumerate(entries):
            logger.info(
                f"Judging {i+1}/{len(entries)}: arXiv:{entry.arxiv_id}"
            )

            for judge in self.judges:
                verdict = judge.score(entry)
                all_verdicts.append(verdict)
                if not verdict.llm_error:
                    logger.info(
                        f"  {judge.label}: overall={verdict.overall:.2f} "
                        f"({verdict.verdict})"
                    )
                if self.sleep_between > 0:
                    time.sleep(self.sleep_between)

        return _build_report(entries, all_verdicts)


def _build_report(
    entries:  list[LibraryEntry],
    verdicts: list[JudgeVerdict],
) -> ComparisonReport:
    """Aggregate verdicts into a ComparisonReport."""

    # group by paper
    by_paper: dict[str, list[JudgeVerdict]] = {}
    for v in verdicts:
        by_paper.setdefault(v.arxiv_id, []).append(v)

    per_paper = []
    for aid, vs in by_paper.items():
        valid = [v for v in vs if not v.llm_error]
        if not valid:
            continue
        per_paper.append({
            "arxiv_id":  aid,
            "n_judges":  len(valid),
            "mean_faithfulness":       _mean([v.faithfulness for v in valid]),
            "mean_coverage":           _mean([v.coverage for v in valid]),
            "mean_technical_accuracy": _mean([v.technical_accuracy for v in valid]),
            "mean_overall":            _mean([v.overall for v in valid]),
            "verdict_distribution":    _count_verdicts(valid),
            "judges":                  [v.to_dict() for v in valid],
        })

    # group by judge
    by_judge: dict[str, list[JudgeVerdict]] = {}
    for v in verdicts:
        by_judge.setdefault(v.judge_name, []).append(v)

    per_judge = []
    for name, vs in by_judge.items():
        valid = [v for v in vs if not v.llm_error]
        per_judge.append({
            "judge":                 name,
            "n_papers_judged":       len(valid),
            "n_errors":              len(vs) - len(valid),
            "mean_faithfulness":     _mean([v.faithfulness for v in valid]),
            "mean_coverage":         _mean([v.coverage for v in valid]),
            "mean_technical_accuracy": _mean([v.technical_accuracy for v in valid]),
            "mean_overall":          _mean([v.overall for v in valid]),
            "verdict_distribution":  _count_verdicts(valid),
        })

    # inter-judge agreement (Cohen's kappa on verdicts, pairwise)
    aggregate = {
        "n_verdicts_total": len(verdicts),
        "n_verdicts_ok":    sum(1 for v in verdicts if not v.llm_error),
        "n_verdicts_err":   sum(1 for v in verdicts if v.llm_error),
        "mean_overall_all_judges": _mean([v.overall for v in verdicts
                                          if not v.llm_error]),
        "pairwise_agreement":      _pairwise_agreement(verdicts),
    }

    return ComparisonReport(
        study_name  = "multi_judge",
        n_papers    = len(entries),
        n_judges    = len(by_judge),
        per_paper   = per_paper,
        per_judge   = per_judge,
        aggregate   = aggregate,
        finished_at = "",  # set by caller
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _count_verdicts(vs: list[JudgeVerdict]) -> dict[str, int]:
    counts = {"accept": 0, "needs_revision": 0, "reject": 0, "other": 0}
    for v in vs:
        key = v.verdict if v.verdict in counts else "other"
        counts[key] += 1
    return counts


def _pairwise_agreement(verdicts: list[JudgeVerdict]) -> dict:
    """Simple pairwise verdict agreement rate."""
    # group verdicts by paper, then check pairwise agreement
    by_paper: dict[str, list[JudgeVerdict]] = {}
    for v in verdicts:
        if v.llm_error:
            continue
        by_paper.setdefault(v.arxiv_id, []).append(v)

    total_pairs = 0
    agree_pairs = 0
    for vs in by_paper.values():
        for i in range(len(vs)):
            for j in range(i+1, len(vs)):
                total_pairs += 1
                if vs[i].verdict == vs[j].verdict:
                    agree_pairs += 1

    return {
        "n_pairs":    total_pairs,
        "n_agree":    agree_pairs,
        "agreement":  agree_pairs / total_pairs if total_pairs else 0.0,
    }
