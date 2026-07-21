"""
Stage 6 — Persistent Context Library integration.

Wraps ContextLibrary in the pipeline stage interface. Called after
Stage 5: takes the Stage5Result and either creates a new library
entry or merges into an existing one.

Cache-first mode: before Stages 4 and 5 run, callers can query the
library for an existing entry. If present with high enough quality,
skip re-analysis entirely.
"""

from __future__ import annotations

import time
from   dataclasses import dataclass

from astrorag.library         import ContextLibrary, LibraryEntry, get_library
from astrorag.llm.models      import StructuredSummary
from astrorag.logger          import get_logger
from astrorag.stages.stage5_summarise import Stage5Result

logger = get_logger(__name__)


@dataclass
class Stage6Result:
    """Output of Stage 6."""

    arxiv_id:      str
    entry:         LibraryEntry
    was_new:       bool
    was_merged:    bool
    was_cached:    bool = False    # True if we skipped Stage 5
    total_time_s:  float = 0.0


class Stage6Library:
    """
    Stage 6 — Library integration.

    Usage:
        stage6 = Stage6Library()

        # after Stage 5
        s6 = stage6.run(stage5_result, query=query)
        print(f"{s6.arxiv_id}: {s6.entry.n_analyses} analyses on file")

        # cache-first check before Stages 4-5
        cached_entry = stage6.check_cached(
            arxiv_id="1701.06747",
            min_quality=0.75,
        )
        if cached_entry is not None:
            # skip Stages 4-5, use cached knowledge
            print(cached_entry.summary.paper_overview)
    """

    def __init__(
        self,
        library: ContextLibrary | None = None,
    ) -> None:
        self.library = library or get_library()

    # ══════════════════════════════════════════════════
    # main entry — persist Stage 5 result
    # ══════════════════════════════════════════════════

    def run(
        self,
        stage5_result: Stage5Result,
        query:         str,
    ) -> Stage6Result:
        """
        Persist a Stage5Result to the library.

        If no entry exists for this arxiv_id, create one.
        If entry exists, merge new summary in.
        """
        t0 = time.time()
        aid = stage5_result.selected_arxiv_id
        q_total = stage5_result.quality.scores.Q_total

        existing = self.library.get(aid)

        if existing is None:
            entry = self.library.add_from_summary(
                arxiv_id      = aid,
                summary       = stage5_result.summary,
                query         = query,
                q_total       = q_total,
                pdf_path      = str(stage5_result.pdf_doc.pdf_path or ""),
                n_pages       = stage5_result.pdf_doc.n_pages,
                section_names = list(stage5_result.pdf_doc.sections.keys()),
            )
            was_new    = True
            was_merged = False
        else:
            entry = self.library.update_from_summary(
                arxiv_id = aid,
                summary  = stage5_result.summary,
                query    = query,
                q_total  = q_total,
            )
            was_new    = False
            was_merged = True

        result = Stage6Result(
            arxiv_id     = aid,
            entry        = entry,
            was_new      = was_new,
            was_merged   = was_merged,
            was_cached   = False,
            total_time_s = time.time() - t0,
        )

        logger.info(
            f"Stage 6 done in {result.total_time_s*1000:.1f}ms — "
            f"{aid} ({'NEW' if was_new else 'MERGED'}, "
            f"{entry.n_analyses} total analyses)"
        )
        return result

    # ══════════════════════════════════════════════════
    # cache-first lookup
    # ══════════════════════════════════════════════════

    def check_cached(
        self,
        arxiv_id:    str,
        min_quality: float = 0.75,
    ) -> LibraryEntry | None:
        """
        Return existing entry if it meets min_quality, else None.

        Callers use this to short-circuit Stages 4-5 when a paper
        has already been analysed at sufficient quality.
        """
        entry = self.library.get(arxiv_id)
        if entry is None:
            return None
        if entry.best_q_total < min_quality:
            return None
        return entry