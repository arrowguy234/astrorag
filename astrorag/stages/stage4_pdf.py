"""
Stage 4 — PDF Fetch and Section Parsing.

Downloads the selected paper's PDF from arXiv, extracts text with
column-aware handling, and splits into named sections for use by
Stage 5.
"""

from __future__ import annotations

import time
from   dataclasses import dataclass

from astrorag.config     import Settings, get_settings
from astrorag.logger     import get_logger
from astrorag.pdf        import (
    PDFDocument,
    Section,
    extract_text_with_fallback,
    fetch_arxiv_pdf,
    split_by_sections,
)
from astrorag.retrieval  import RetrievalResult
from astrorag.stages.stage3_rerank import Stage3Result

logger = get_logger(__name__)


class Stage4PDF:
    """
    Stage 4 — Fetch PDF and parse into sections.

    Usage:
        stage4 = Stage4PDF()
        pdf_doc = stage4.run(stage3_result)
        results_text = pdf_doc.get_section("Results")
        full_text = pdf_doc.full_text
    """

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()

    # ══════════════════════════════════════════════════
    # main entry
    # ══════════════════════════════════════════════════

    def run(
        self,
        selected: Stage3Result | RetrievalResult,
    ) -> PDFDocument:
        """
        Fetch and parse the selected paper.

        Args:
            selected: Either the full Stage3Result or a RetrievalResult
                      pointing to the target paper.

        Returns:
            PDFDocument with full text, sections, and metadata.
        """
        # handle both input types
        if isinstance(selected, Stage3Result):
            paper = selected.selected_result
        else:
            paper = selected

        arxiv_id = paper.arxiv_id
        logger.info(f"Stage 4 — fetching PDF for {arxiv_id}")

        # ── fetch PDF ───────────────────────────────────
        t_fetch = time.time()
        pdf_path, from_cache, err = fetch_arxiv_pdf(
            arxiv_id = arxiv_id,
            timeout  = 60,
        )
        fetch_seconds = time.time() - t_fetch

        if pdf_path is None:
            logger.error(f"PDF fetch failed for {arxiv_id}: {err}")
            return PDFDocument(
                arxiv_id      = arxiv_id,
                pdf_path      = None,
                full_text     = "",
                sections      = {},
                n_pages       = 0,
                extractor     = "none",
                fetch_seconds = fetch_seconds,
                success       = False,
                error         = err,
            )

        logger.debug(
            f"PDF ready: {pdf_path.name} "
            f"(cache={from_cache}, {fetch_seconds:.2f}s)"
        )

        # ── extract text ────────────────────────────────
        t_parse = time.time()
        full_text, n_pages, extractor = extract_text_with_fallback(pdf_path)

        if not full_text or len(full_text.strip()) < 200:
            logger.warning(
                f"Extraction yielded only {len(full_text)} chars from "
                f"{arxiv_id}"
            )

        # ── split into sections ─────────────────────────
        sections = split_by_sections(full_text)
        parse_seconds = time.time() - t_parse

        n_chars = len(full_text)

        pdf_doc = PDFDocument(
            arxiv_id      = arxiv_id,
            pdf_path      = pdf_path,
            full_text     = full_text,
            sections      = sections,
            n_pages       = n_pages,
            n_chars_total = n_chars,
            extractor     = extractor,
            fetch_seconds = fetch_seconds,
            parse_seconds = parse_seconds,
            from_cache    = from_cache,
            success       = n_chars > 200,
            error         = "" if n_chars > 200 else "extraction produced no usable text",
        )

        logger.info(
            f"Stage 4 done — {arxiv_id}: {n_pages} pages, "
            f"{n_chars:,} chars, {len(sections)} sections "
            f"(fetch={fetch_seconds:.2f}s parse={parse_seconds:.2f}s "
            f"extractor={extractor})"
        )

        return pdf_doc