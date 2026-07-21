"""
Data models for PDF documents and sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path


@dataclass
class Section:
    """One named section extracted from a paper PDF."""

    name:        str
    text:        str
    char_start:  int  = 0
    char_end:    int  = 0

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass
class PDFDocument:
    """
    Complete PDF extraction result for one paper.

    Contains the raw full text, parsed sections, source PDF path, and
    diagnostic metadata (extractor used, timing, page count).
    """

    arxiv_id:      str
    pdf_path:      Path
    full_text:     str
    sections:      dict[str, Section]
    n_pages:       int = 0
    n_chars_total: int = 0
    extractor:     str = ""
    fetch_seconds: float = 0.0
    parse_seconds: float = 0.0
    from_cache:    bool = False
    success:       bool = True
    error:         str  = ""

    @property
    def section_names(self) -> list[str]:
        return list(self.sections.keys())

    @property
    def has_results(self) -> bool:
        return any("result" in n.lower() for n in self.sections)

    @property
    def has_methods(self) -> bool:
        return any(n.lower() in ("methods", "methodology", "analysis")
                   for n in self.sections)

    def get_section(self, name: str, default: str = "") -> str:
        """Case-insensitive section lookup."""
        target = name.lower()
        for n, s in self.sections.items():
            if n.lower() == target:
                return s.text
        return default

    def summary(self) -> str:
        return (
            f"PDF     : {self.arxiv_id}  ({self.n_pages} pages, "
            f"{self.n_chars_total:,} chars)\n"
            f"Extractor: {self.extractor}\n"
            f"Sections : {' | '.join(self.section_names) or '(none detected)'}\n"
            f"Fetch    : {self.fetch_seconds:.2f}s  "
            f"Parse: {self.parse_seconds:.2f}s"
        )