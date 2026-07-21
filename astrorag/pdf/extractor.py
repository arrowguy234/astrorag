"""
PDF text extraction with column-aware handling.

Astro-ph papers are typeset in two columns. Standard extractors
read left-to-right across the page, interleaving columns and
destroying equation structure. This module extracts blocks with
bounding box coordinates and reorders them by column.
"""

from __future__ import annotations

import time
from   pathlib import Path

from astrorag.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# PyMuPDF (fitz) — primary extractor
# ══════════════════════════════════════════════════════════

def extract_text_pymupdf(pdf_path: Path) -> tuple[str, int]:
    """
    Extract text from PDF using PyMuPDF with column awareness.

    For each page:
      1. Get all text blocks with their bounding boxes
      2. Classify blocks by x-coordinate (left half vs right half)
      3. Sort each column by y-coordinate (top to bottom)
      4. Left column first, then right column

    This preserves reading order for two-column layouts.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Tuple of:
            full_text  Extracted text as single string.
            n_pages    Number of pages processed.
    """
    import fitz

    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    parts: list[str] = []

    for page in doc:
        page_width = page.rect.width
        midpoint   = page_width / 2

        # get blocks: list of (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks", sort=True)

        left_blocks:  list[tuple[float, str]] = []
        right_blocks: list[tuple[float, str]] = []

        for b in blocks:
            if len(b) < 5:
                continue
            x0, y0, _, _, text = b[0], b[1], b[2], b[3], b[4]
            if not text or not text.strip():
                continue
            if x0 < midpoint:
                left_blocks.append((y0, text))
            else:
                right_blocks.append((y0, text))

        # sort each column by y-coordinate (top to bottom)
        left_blocks.sort(key=lambda x: x[0])
        right_blocks.sort(key=lambda x: x[0])

        # append left column first, then right column
        for _, text in left_blocks:
            parts.append(text)
        for _, text in right_blocks:
            parts.append(text)

    doc.close()
    full_text = "".join(parts)
    return full_text, n_pages


# ══════════════════════════════════════════════════════════
# pdfplumber — fallback extractor
# ══════════════════════════════════════════════════════════

def extract_text_pdfplumber(pdf_path: Path) -> tuple[str, int]:
    """
    Extract text using pdfplumber (fallback when PyMuPDF fails).

    Less column-aware but more robust to unusual PDF formats.
    """
    import pdfplumber

    parts:   list[str] = []
    n_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text + "\n")

    full_text = "\n".join(parts)
    return full_text, n_pages


# ══════════════════════════════════════════════════════════
# extractor selection with fallback
# ══════════════════════════════════════════════════════════

def extract_text_with_fallback(pdf_path: Path) -> tuple[str, int, str]:
    """
    Try PyMuPDF first, fall back to pdfplumber on failure.

    Returns:
        Tuple of:
            full_text    Extracted text.
            n_pages      Page count.
            extractor    Name of extractor that succeeded.
    """
    # try PyMuPDF first
    try:
        text, pages = extract_text_pymupdf(pdf_path)
        if text and len(text.strip()) > 200:
            return text, pages, "pymupdf"
        logger.warning(
            f"PyMuPDF extracted only {len(text)} chars from "
            f"{pdf_path.name}, trying pdfplumber"
        )
    except Exception as e:
        logger.warning(f"PyMuPDF failed on {pdf_path.name}: {e}")

    # fall back to pdfplumber
    try:
        text, pages = extract_text_pdfplumber(pdf_path)
        if text and len(text.strip()) > 200:
            return text, pages, "pdfplumber"
        logger.warning(
            f"pdfplumber extracted only {len(text)} chars from "
            f"{pdf_path.name}"
        )
    except Exception as e:
        logger.error(f"pdfplumber failed on {pdf_path.name}: {e}")

    return "", 0, "failed"