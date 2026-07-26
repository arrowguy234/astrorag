"""
Retrieval ground truth via a stronger LLM judge.

For each query, take BM25 top-10 candidates. Ask a strong LLM to
independently rank the top-3 most relevant. Measure how often
AstroRAG's Stage 3 selection appears in the judge's top-3.

This is a soft gold standard — the judge is another LLM, not a
human expert. But using a substantially stronger model (70B+ vs
production 8B) gives a defensible robustness check.
"""

from __future__ import annotations

import copy
import json
import re
import time
from   dataclasses import dataclass, field
from typing        import Any

from astrorag.config  import Settings, get_settings
from astrorag.data    import CorpusData
from astrorag.ground_truth.judges import _extract_json
from astrorag.llm     import LLMClient
from astrorag.logger  import get_logger
from astrorag.stages  import Stage0Decompose, Stage1BM25

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# result
# ══════════════════════════════════════════════════════════

@dataclass
class RetrievalAgreement:
    """Retrieval agreement between AstroRAG and a strong judge."""

    query_idx:            int
    query:                str
    astrorag_selected:    str
    judge_top3:           list[str]
    astrorag_in_judge_top1:  bool
    astrorag_in_judge_top3:  bool
    judge_rationale:      str = ""
    error:                str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_idx":              self.query_idx,
            "query":                  self.query,
            "astrorag_selected":      self.astrorag_selected,
            "judge_top3":             list(self.judge_top3),
            "astrorag_in_judge_top1": self.astrorag_in_judge_top1,
            "astrorag_in_judge_top3": self.astrorag_in_judge_top3,
            "judge_rationale":        self.judge_rationale,
            "error":                  self.error,
        }


# ══════════════════════════════════════════════════════════
# strong-judge ranking prompt
# ══════════════════════════════════════════════════════════

_RETRIEVAL_SYSTEM = (
    "You are an expert astrophysicist selecting the most relevant paper "
    "for a research query from a set of candidates. Read each candidate's "
    "title and abstract carefully. Respond ONLY with valid JSON."
)


def _build_retrieval_prompt(query: str, candidates: list) -> str:
    lines = [f"QUERY: {query}", "", "CANDIDATES:"]
    for i, r in enumerate(candidates):
        lines.append(f"[PAPER #{i}] arXiv:{r.arxiv_id}")
        lines.append(f"  Title: {r.title[:200]}")
        lines.append(f"  Abstract: {r.abstract[:400]}")
        lines.append("")

    lines.append(
        "Rank the top 3 most relevant candidates by index (0-based). "
        "Respond as JSON:\n"
        '{"top3_indices": [<int>, <int>, <int>], '
        '"rationale": "<one sentence why these three>"}'
    )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# runner
# ══════════════════════════════════════════════════════════

class RetrievalGroundTruth:
    """
    Compare AstroRAG's Stage 3 selection against a stronger LLM judge.
    """

    def __init__(
        self,
        corpus:      CorpusData,
        judge_model: str = "llama-3.3-70b-versatile",
        settings:    Settings | None = None,
    ) -> None:
        self.corpus   = corpus
        self.settings = settings or get_settings()

        # judge LLM with the stronger model
        judge_settings = copy.copy(self.settings)
        judge_settings.groq_model = judge_model
        self.judge_client = LLMClient(settings=judge_settings)
        self.judge_model  = judge_model

        # BM25 for retrieval
        self.stage1 = Stage1BM25(corpus=corpus, settings=self.settings)

    def evaluate_query(
        self,
        query_idx:        int,
        query:            str,
        astrorag_selected: str,
        top_k:            int = 10,
    ) -> RetrievalAgreement:
        """Rank BM25 top-K with strong judge, compare to AstroRAG choice."""

        try:
            s1 = self.stage1.run(query, top_k=top_k)
            candidates = s1.results[:top_k]

            prompt = _build_retrieval_prompt(query, candidates)

            raw, _ = self.judge_client.chat(
                system      = _RETRIEVAL_SYSTEM,
                user        = prompt,
                temperature = 0.0,
                max_tokens  = 300,
                stage_name  = "retrieval_gt_judge",
            )
            payload = _extract_json(raw)

            top3_idxs = payload.get("top3_indices", [])
            # ensure int
            top3_idxs = [int(i) for i in top3_idxs if isinstance(i, (int, float))
                         and 0 <= int(i) < len(candidates)]
            top3_arxiv = [candidates[i].arxiv_id for i in top3_idxs[:3]]

            in_top1 = (astrorag_selected == top3_arxiv[0]) if top3_arxiv else False
            in_top3 = astrorag_selected in top3_arxiv

            return RetrievalAgreement(
                query_idx             = query_idx,
                query                 = query,
                astrorag_selected     = astrorag_selected,
                judge_top3            = top3_arxiv,
                astrorag_in_judge_top1= in_top1,
                astrorag_in_judge_top3= in_top3,
                judge_rationale       = str(payload.get("rationale", ""))[:400],
            )

        except Exception as e:
            logger.error(f"Retrieval GT failed on Q{query_idx}: {e}")
            return RetrievalAgreement(
                query_idx             = query_idx,
                query                 = query,
                astrorag_selected     = astrorag_selected,
                judge_top3            = [],
                astrorag_in_judge_top1= False,
                astrorag_in_judge_top3= False,
                error                 = f"{type(e).__name__}: {e}",
            )

