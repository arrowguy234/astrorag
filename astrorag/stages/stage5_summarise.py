"""
Stage 5 — Deep Technical Summarisation with Quality Gate.

The most important stage in terms of output quality. Reads the parsed
sections from Stage 4, routes the query to the right sections, extracts
equations and tables, and produces a structured summary with dedicated
fields for quantitative content.

Iterative re-selection: if quality is insufficient, picks the next
paper from Stage 3's fallback pool, re-runs Stage 4, and re-runs
Stage 5. Up to max_reselect_attempts.
"""

from __future__ import annotations

import time
from   dataclasses import dataclass, field

from astrorag.config           import Settings, get_settings
from astrorag.extraction       import (
    QualityAssessment,
    QualityDecision,
    assess_quality,
    build_technical_context,
    detect_question_type,
    extract_equations,
    extract_measurements,
    extract_tables,
)
from astrorag.llm              import LLMClient, get_llm_client
from astrorag.llm.models       import (
    LLMResponse,
    QueryDecomposition,
    StructuredSummary,
    SubQuestionAnswer,
)
from astrorag.logger           import get_logger
from astrorag.pdf              import PDFDocument
from astrorag.retrieval        import RetrievalResult, RetrievalRun
from astrorag.stages.stage3_rerank import Stage3Result
from astrorag.stages.stage4_pdf    import Stage4PDF

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# result container
# ══════════════════════════════════════════════════════════

@dataclass
class Stage5Result:
    """Output of Stage 5 — summary + quality gate outcome."""

    selected_arxiv_id:   str
    summary:             StructuredSummary
    quality:             QualityAssessment
    pdf_doc:             PDFDocument
    llm_response:        LLMResponse | None = None
    n_attempts:          int   = 1
    n_reselections:      int   = 0
    n_retries:           int   = 0
    fallback_pool_used:  list[int] = field(default_factory=list)
    total_time_s:        float = 0.0

    @property
    def accepted(self) -> bool:
        return self.quality.decision == QualityDecision.ACCEPT


# ══════════════════════════════════════════════════════════
# prompt construction
# ══════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are an expert astrophysicist reading a research paper to produce "
    "a technically precise structured summary. "
    "You will be given three sub-questions and the paper's full technical "
    "context (routed sections, extracted equations, extracted numerical "
    "measurements, extracted tables). "
    "Answer with quantitative precision: always include exact equations with "
    "variable definitions, always include exact numerical values with units "
    "and uncertainties, and never give vague qualitative answers when "
    "quantitative data exists in the paper. "
    "If an equation appears in the extracted equations block, quote it exactly. "
    "If a number with units appears in the measurements block, use that exact number. "
    "Return valid JSON only. No markdown, no commentary."
)


def _build_user_prompt(
    query:            str,
    decomposition:    QueryDecomposition,
    paper_title:      str,
    paper_abstract:   str,
    technical_context: str,
) -> str:
    """Build the Stage 5 user prompt."""
    sq = decomposition.sub_questions
    sub_q_block = (
        f"Q1 (mechanism)     : {sq['Q1']}\n"
        f"Q2 (evidence)      : {sq['Q2']}\n"
        f"Q3 (quantitative)  : {sq['Q3']}"
    )

    schema_example = """
{
  "paper_overview": "3-4 sentence technical summary",
  "sub_question_answers": {
    "Q1": {
      "answered": true,
      "answer_text": "full technical answer with equations and exact values",
      "section": "Methods",
      "equations": ["exact equation 1"],
      "values": ["value with unit"]
    },
    "Q2": {
      "answered": true,
      "answer_text": "...",
      "section": "Observations",
      "equations": [],
      "values": []
    },
    "Q3": {
      "answered": true,
      "answer_text": "...",
      "section": "Results",
      "equations": [],
      "values": []
    }
  },
  "evidence_type": "observational",
  "instruments": ["Chandra", "XMM-Newton"],
  "key_equations": [
    {"equation": "E_cav = 4PV", "variables": "P=pressure, V=cavity volume", "section": "Methods"}
  ],
  "numerical_results": [
    {"quantity": "jet power", "value": "1.2e44", "uncertainty": "±0.3e44", "unit": "erg/s"}
  ],
  "key_findings": ["quantitative finding 1", "quantitative finding 2", "quantitative finding 3"],
  "methodology": "step by step technical description",
  "limitations": ["specific limitation 1"],
  "key_snippet": "verbatim sentence from paper with most important quantitative result"
}
""".strip()

    return (
        f"SUB-QUESTIONS to answer with technical precision:\n{sub_q_block}\n\n"
        f"PAPER TITLE: {paper_title}\n"
        f"ABSTRACT: {paper_abstract}\n\n"
        f"PAPER TECHNICAL CONTEXT (priority sections routed by query keywords, "
        f"plus separately extracted equations, measurements, and tables):\n\n"
        f"{technical_context}\n\n"
        f"Task:\n"
        f"- Answer each sub-question with full technical detail from the "
        f"paper content above.\n"
        f"- Populate key_equations with variable definitions.\n"
        f"- Populate numerical_results with quantity, exact value, uncertainty, unit.\n"
        f"- Cite the section each answer came from.\n"
        f"- key_snippet must be a verbatim sentence from the paper.\n\n"
        f"Return valid JSON matching this exact schema:\n{schema_example}"
    )


# ══════════════════════════════════════════════════════════
# main stage class
# ══════════════════════════════════════════════════════════

class Stage5Summarise:
    """
    Stage 5 — Deep Technical Summarisation with Quality Gate.

    Usage:
        stage5 = Stage5Summarise()
        result = stage5.run(
            decomposition   = stage0_out,
            retrieval       = stage1_out,
            stage3_result   = stage3_out,
            initial_pdf     = stage4_out,
        )
        print(result.summary.paper_overview)
        print(result.quality.summary())
    """

    def __init__(
        self,
        settings:   Settings   | None = None,
        llm_client: LLMClient  | None = None,
        stage4:     Stage4PDF  | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm     = llm_client
        self.stage4   = stage4 or Stage4PDF(settings=self.settings)

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    # ══════════════════════════════════════════════════
    # main entry with quality-gated loop
    # ══════════════════════════════════════════════════

    def run(
        self,
        decomposition: QueryDecomposition,
        retrieval:     RetrievalRun,
        stage3_result: Stage3Result,
        initial_pdf:   PDFDocument,
    ) -> Stage5Result:
        """
        Run Stage 5 with iterative re-selection on quality failure.

        Args:
            decomposition: Stage 0 output.
            retrieval:     Stage 1 output (needed for fallback pool).
            stage3_result: Stage 3 output with fallback pool.
            initial_pdf:   Stage 4 output for the selected paper.

        Returns:
            Stage5Result with final summary and quality assessment.
        """
        t0 = time.time()

        current_arxiv_id = stage3_result.selected_result.arxiv_id
        current_pdf      = initial_pdf
        fallback_pool    = list(stage3_result.fallback_pool)
        used_pool_idxs:  list[int] = []

        n_attempts    = 0
        n_retries     = 0
        n_reselects   = 0

        summary:  StructuredSummary | None = None
        quality:  QualityAssessment | None = None
        llm_resp: LLMResponse | None       = None

        while n_attempts < self.settings.max_reselect_attempts:
            n_attempts += 1
            logger.info(
                f"Stage 5 attempt {n_attempts}/"
                f"{self.settings.max_reselect_attempts} for {current_arxiv_id}"
            )

            # ── run summarisation ───────────────────────
            try:
                summary, llm_resp = self._summarise(
                    decomposition = decomposition,
                    paper_title   = stage3_result.selected_result.title
                                   if current_pdf.arxiv_id == stage3_result.selected_result.arxiv_id
                                   else current_arxiv_id,
                    paper_abstract= stage3_result.selected_result.abstract
                                   if current_pdf.arxiv_id == stage3_result.selected_result.arxiv_id
                                   else "",
                    pdf_doc       = current_pdf,
                    query         = decomposition.original_query,
                )
            except Exception as e:
                logger.error(f"Summarisation error: {e}")
                # try re-selection
                if fallback_pool:
                    n_reselects += 1
                    next_idx = fallback_pool.pop(0)
                    used_pool_idxs.append(next_idx)
                    next_paper = retrieval.results[next_idx]
                    logger.info(
                        f"Re-selecting due to LLM error: paper #{next_idx} "
                        f"({next_paper.arxiv_id})"
                    )
                    current_arxiv_id = next_paper.arxiv_id
                    current_pdf      = self.stage4.run(next_paper)
                    continue
                else:
                    raise

            # ── assess quality ──────────────────────────
            quality = assess_quality(
                summary    = summary,
                paper_text = current_pdf.full_text,
                settings   = self.settings,
            )
            logger.info(
                f"Quality: Q_f={quality.scores.Q_f:.3f} "
                f"Q_c={quality.scores.Q_c:.3f} "
                f"Q_i={quality.scores.Q_i:.3f} "
                f"Q_total={quality.scores.Q_total:.3f} "
                f"→ {quality.decision.value}"
            )

            # ── decision dispatch ───────────────────────
            if quality.decision == QualityDecision.ACCEPT:
                break

            if quality.decision == QualityDecision.RETRY:
                if n_retries < 2:
                    n_retries += 1
                    logger.info(f"Retrying same paper (retry {n_retries}/2)")
                    continue
                # after 2 retries, fall through to re-select

            # RE-SELECT or exhausted retries
            if fallback_pool:
                n_reselects += 1
                next_idx = fallback_pool.pop(0)
                used_pool_idxs.append(next_idx)
                next_paper = retrieval.results[next_idx]
                logger.info(
                    f"Re-selecting: paper #{next_idx} ({next_paper.arxiv_id}) "
                    f"from fallback pool"
                )
                current_arxiv_id = next_paper.arxiv_id
                current_pdf      = self.stage4.run(next_paper)
                if not current_pdf.success:
                    logger.warning(
                        f"Re-selected paper PDF fetch failed, trying next"
                    )
                    continue
                n_retries = 0   # reset retry counter for new paper
                continue

            # no more fallback available
            logger.warning(
                f"Exhausted fallback pool at attempt {n_attempts}, "
                f"accepting current summary at Q_total={quality.scores.Q_total:.3f}"
            )
            break

        assert summary is not None and quality is not None

        result = Stage5Result(
            selected_arxiv_id  = current_arxiv_id,
            summary            = summary,
            quality            = quality,
            pdf_doc            = current_pdf,
            llm_response       = llm_resp,
            n_attempts         = n_attempts,
            n_reselections     = n_reselects,
            n_retries          = n_retries,
            fallback_pool_used = used_pool_idxs,
            total_time_s       = time.time() - t0,
        )

        logger.info(
            f"Stage 5 done in {result.total_time_s:.2f}s — "
            f"{'ACCEPTED' if result.accepted else 'FORCED'} "
            f"Q_total={quality.scores.Q_total:.3f} "
            f"({n_attempts} attempts, {n_reselects} re-selects)"
        )
        return result

    # ══════════════════════════════════════════════════
    # single summarisation call
    # ══════════════════════════════════════════════════

    def _summarise(
        self,
        decomposition:  QueryDecomposition,
        paper_title:    str,
        paper_abstract: str,
        pdf_doc:        PDFDocument,
        query:          str,
    ) -> tuple[StructuredSummary, LLMResponse]:
        """Run one LLM call to produce a StructuredSummary."""

        # ── build technical context ─────────────────────
        technical_context = build_technical_context(
            sections  = pdf_doc.sections,
            query     = query,
            full_text = pdf_doc.full_text,
        )

        logger.debug(
            f"Technical context: {len(technical_context):,} chars, "
            f"sections used: {list(pdf_doc.sections.keys())[:6]}"
        )

        user_prompt = _build_user_prompt(
            query             = query,
            decomposition     = decomposition,
            paper_title       = paper_title,
            paper_abstract    = paper_abstract,
            technical_context = technical_context,
        )

        response = self.llm.chat_json(
            system      = _SYSTEM_PROMPT,
            user        = user_prompt,
            schema      = StructuredSummary,
            max_tokens  = self.settings.groq_max_tokens_stage5,
            stage_name  = "stage5",
        )
        return response.data, response