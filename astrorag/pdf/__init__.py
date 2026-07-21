"""
PDF handling subpackage — fetch arXiv PDFs and parse into sections.
"""

from astrorag.pdf.extractor import (
    extract_text_pymupdf,
    extract_text_pdfplumber,
    extract_text_with_fallback,
)
from astrorag.pdf.fetcher   import (
    fetch_arxiv_pdf,
    build_arxiv_urls,
    normalize_arxiv_id_for_url,
)
from astrorag.pdf.models    import PDFDocument, Section
from astrorag.pdf.sections  import (
    detect_sections,
    split_by_sections,
    SECTION_HEADERS,
)

__all__ = [
    "PDFDocument", "Section",
    "fetch_arxiv_pdf", "build_arxiv_urls", "normalize_arxiv_id_for_url",
    "extract_text_pymupdf", "extract_text_pdfplumber", "extract_text_with_fallback",
    "detect_sections", "split_by_sections", "SECTION_HEADERS",
]