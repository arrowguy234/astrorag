"""
Stage 3 — Graph-Primed LLM Reranking.

Takes the graph context from Stage 2 and asks the LLM to select the
single most relevant paper. The prompt is architected to leverage
transformer self-attention:

    1. Cluster summary       (graph context — LLM primed here)
    2. Query + sub-questions (anchors what to look for)
    3. Annotated abstracts   ([PPR=x.xx | cluster=N])
    4. Task instruction

By placing cluster summary first every subsequent token attends
back to it, biasing selection toward graph-identified relevant papers
while allowing the LLM to reject obvious semantic mismatches.
"""

from __future__ import annotations

import time
from   dataclasses import dataclass, field

import numpy as np

from astrorag.config      import Settings, get_settings
from astrorag.data        import CorpusData
from astrorag.graph       import GraphContext
from astrorag.llm         import LLMClient, get_llm_client
from astrorag.llm.models  import LLMResponse, QueryDecomposition, RerankDecision
from astrorag.logger      import get_logger
from astrorag.retrieval   import RetrievalResult, RetrievalRun

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# result container
# ══════════════════════════════════════════════════════════

@dataclass
class Stage3Result:
    """Output of Stage 3 — selected paper + fallback pool + telemetry."""

    selected_result:  RetrievalResult                   # the chosen paper
    selected_idx:     int                                # index in candidate list
    fallback_pool:    list[int]     = field(default_factory=list)
    confidence:       float         = 0.0
    reason:           str           = ""
    graph_adj_score:  float         = 0.0                # confidence * (0.85 + 0.15 * ppr)
    llm_response:     LLMResponse | None = None
    fallback_used:    bool          = False
    total_time_s:     float         = 0.0


# ══════════════════════════════════════════════════════════
# prompt construction
# ══════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are an expert astrophysicist performing scientific literature reranking. "
    "You will be given a research query decomposed into three sub-questions, "
    "a graph cluster summary showing how candidate papers are related, and a list "
    "of candidate paper abstracts each annotated with a Personalised PageRank score "
    "(PPR) and cluster label. Your task is to select the single most relevant paper "
    "that directly answers the query and its sub-questions. "
    "Use the cluster summary as field-level context but always verify against the "
    "individual abstract — a paper with high PPR that is off-topic must not be selected. "
    "Return valid JSON only. No markdown, no commentary."
)


def _format_annotated_abstracts(
    results:    list[RetrievalResult],
    ppr_scores: np.ndarray,
    max_chars:  int = 250,
) -> str:
    """
    Format candidate papers as annotated abstracts for the LLM.

    Each is prefixed with [PAPER #i | PPR=x.xx | cluster=N] header
    followed by title (or first 60 chars of abstract if no title)
    and truncated abstract text.
    """
    blocks = []
    for i, r in enumerate(results):
        title = r.title.strip() if r.title else r.abstract[:60].strip()
        abstract = r.abstract[:max_chars].strip()
        blocks.append(
            f"\n[PAPER #{i} | PPR={ppr_scores[i]:.2f} | "
            f"cluster={r.cluster}]\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}"
        )
    return "".join(blocks)


def _build_user_prompt(
    query:           str,
    decomposition:   QueryDecomposition,
    cluster_summary: str,
    annotated:       str,
    n_candidates:    int,
) -> str:
    """Build the full user prompt for Stage 3."""
    sq = decomposition.sub_questions
    sub_q_block = (
        f"Q1 (mechanism)   : {sq['Q1']}\n"
        f"Q2 (evidence)    : {sq['Q2']}\n"
        f"Q3 (quantitative): {sq['Q3']}"
    )

    return (
        f"{cluster_summary}\n\n"
        f"RESEARCH QUERY:\n{query}\n\n"
        f"SUB-QUESTIONS:\n{sub_q_block}\n\n"
        f"CANDIDATE PAPERS (indices 0-{n_candidates - 1}):\n"
        f"{annotated}\n\n"
        f"Instructions:\n"
        f"- Select the single best paper that directly answers the query "
        f"and all three sub-questions.\n"
        f"- High PPR indicates the paper is graph-central to this topic — "
        f"prefer high-PPR papers among semantically appropriate ones.\n"
        f"- REJECT papers that are off-topic even if PPR is high (verify "
        f"the abstract matches the query domain).\n"
        f"- Return top-5 in ranked order for fallback if the top pick fails "
        f"downstream quality gates.\n\n"
        f'Return JSON matching this exact schema:\n'
        f'{{\n'
        f'  "best_paper_idx": <integer 0 to {n_candidates - 1}>,\n'
        f'  "confidence": <float 0.0 to 1.0>,\n'
        f'  "reason": "<one-sentence justification>",\n'
        f'  "top5": [<5 integers in ranked order>]\n'
        f'}}'
    )


# ══════════════════════════════════════════════════════════
# main stage class
# ══════════════════════════════════════════════════════════

class Stage3Rerank:
    """
    Stage 3 — Graph-primed LLM reranking.

    Usage:
        stage3 = Stage3Rerank()
        result = stage3.run(
            retrieval     = stage1_run,
            graph_context = stage2_context,
            decomposition = stage0_decomposition,
        )
        print(result.selected_result.title)
        print("Reason:", result.reason)
    """

    def __init__(
        self,
        settings:   Settings   | None = None,
        llm_client: LLMClient  | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm     = llm_client

    @property
    def llm(self) -> LLMClient | None:
        if self._llm is not None:
            return self._llm
        try:
            self._llm = get_llm_client()
            return self._llm
        except Exception as e:
            logger.warning(f"LLM client unavailable for Stage 3: {e}")
            return None

    # ══════════════════════════════════════════════════
    # main run
    # ══════════════════════════════════════════════════

    def run(
        self,
        retrieval:     RetrievalRun,
        graph_context: GraphContext,
        decomposition: QueryDecomposition,
        use_llm:       bool = True,
    ) -> Stage3Result:
        """
        Select the best paper from Stage 1 candidates.

        Args:
            retrieval:     Stage 1 output with top-K candidates.
            graph_context: Stage 2 output with PPR scores and cluster summary.
            decomposition: Stage 0 output with sub-questions.
            use_llm:       If False, skip LLM and pick highest PPR directly.

        Returns:
            Stage3Result with selected paper, fallback pool, and telemetry.
        """
        results     = retrieval.results
        ppr_scores  = graph_context.ppr_scores
        n           = len(results)

        if n < 1:
            raise ValueError("Stage 3 requires at least 1 candidate paper")

        logger.info(f"Stage 3 — reranking {n} candidates")

        t0 = time.time()

        # ── LLM path ────────────────────────────────────
        if use_llm:
            client = self.llm
            if client is not None:
                try:
                    return self._run_llm(
                        client        = client,
                        retrieval     = retrieval,
                        graph_context = graph_context,
                        decomposition = decomposition,
                        t_start       = t0,
                    )
                except Exception as e:
                    logger.warning(
                        f"Stage 3 LLM failed, falling back to PPR: {e}"
                    )

        # ── fallback: highest PPR ───────────────────────
        return self._fallback_by_ppr(
            results     = results,
            ppr_scores  = ppr_scores,
            t_start     = t0,
        )

    # ── LLM implementation ──────────────────────────────
    def _run_llm(
        self,
        client:        LLMClient,
        retrieval:     RetrievalRun,
        graph_context: GraphContext,
        decomposition: QueryDecomposition,
        t_start:       float,
    ) -> Stage3Result:
        results          = retrieval.results
        ppr_scores       = graph_context.ppr_scores
        cluster_summary  = graph_context.cluster_summary.prompt_text
        n                = len(results)

        annotated = _format_annotated_abstracts(
            results    = results,
            ppr_scores = ppr_scores,
            max_chars  = 250,
        )
        user_prompt = _build_user_prompt(
            query           = decomposition.original_query,
            decomposition   = decomposition,
            cluster_summary = cluster_summary,
            annotated       = annotated,
            n_candidates    = n,
        )

        response = client.chat_json(
            system      = _SYSTEM_PROMPT,
            user        = user_prompt,
            schema      = RerankDecision,
            max_tokens  = self.settings.groq_max_tokens_stage3,
            stage_name  = "stage3",
        )
        decision: RerankDecision = response.data

        # ── clamp best_paper_idx to valid range ─────────
        best_idx = max(0, min(decision.best_paper_idx, n - 1))
        conf     = decision.confidence
        ppr_val  = float(ppr_scores[best_idx])

        # ── graph-adjusted score ────────────────────────
        adj_score = conf * (0.85 + 0.15 * ppr_val)

        # ── build fallback pool: use LLM top5, clamped ──
        fallback_pool = [
            max(0, min(i, n - 1))
            for i in decision.top5
            if isinstance(i, int)
        ]
        # ensure best_idx not in fallback pool
        fallback_pool = [i for i in fallback_pool if i != best_idx]
        # deduplicate while preserving order
        seen = set()
        fallback_pool = [
            i for i in fallback_pool
            if i not in seen and not seen.add(i)
        ]

        selected = results[best_idx]

        result = Stage3Result(
            selected_result = selected,
            selected_idx    = best_idx,
            fallback_pool   = fallback_pool,
            confidence      = conf,
            reason          = decision.reason,
            graph_adj_score = adj_score,
            llm_response    = response,
            fallback_used   = False,
            total_time_s    = time.time() - t_start,
        )

        logger.info(
            f"Stage 3 done in {result.total_time_s:.3f}s [LLM] — "
            f"selected #{best_idx} ({selected.arxiv_id}) "
            f"conf={conf:.3f} PPR={ppr_val:.3f} adj={adj_score:.3f}"
        )
        return result

    # ── fallback implementation ─────────────────────────
    def _fallback_by_ppr(
        self,
        results:    list[RetrievalResult],
        ppr_scores: np.ndarray,
        t_start:    float,
    ) -> Stage3Result:
        """Fallback: select highest-PPR paper, top-5 as pool."""
        n = len(results)
        top5 = np.argsort(ppr_scores)[::-1][:6].tolist()
        best_idx = int(top5[0])
        fallback_pool = [int(i) for i in top5[1:]]

        ppr_val = float(ppr_scores[best_idx])
        selected = results[best_idx]

        result = Stage3Result(
            selected_result = selected,
            selected_idx    = best_idx,
            fallback_pool   = fallback_pool,
            confidence      = ppr_val,
            reason          = "Fallback: LLM unavailable, selected highest-PPR paper",
            graph_adj_score = ppr_val,
            llm_response    = None,
            fallback_used   = True,
            total_time_s    = time.time() - t_start,
        )

        logger.info(
            f"Stage 3 done in {result.total_time_s:.3f}s [FALLBACK] — "
            f"selected #{best_idx} ({selected.arxiv_id}) PPR={ppr_val:.3f}"
        )
        return result