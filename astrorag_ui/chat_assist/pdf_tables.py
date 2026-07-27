"""
On-demand table retrieval from arXiv PDFs.

When the user asks a table-related question about a paper, this module:
1. Fetches the PDF from arxiv.org
2. Extracts tables using pdfplumber (structure-preserving)
3. Returns them as markdown-formatted strings
4. Optionally scores tables by relevance to the user's question
"""

from __future__ import annotations

import io
import re
from   dataclasses import dataclass, field

import requests


try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


# ══════════════════════════════════════════════════════════
# data models
# ══════════════════════════════════════════════════════════

@dataclass
class ExtractedTable:
    """One table extracted from a PDF."""

    page_num:      int
    table_num:     int
    n_rows:        int
    n_cols:        int
    header:        list[str] = field(default_factory=list)
    rows:          list[list[str]] = field(default_factory=list)
    caption:       str = ""

    def to_markdown(self, max_col_width: int = 30) -> str:
        """Render the table as GitHub-flavored markdown."""
        if not self.rows:
            return "_(empty table)_"

        def trim(s: str) -> str:
            s = (s or "").strip().replace("\n", " ")
            return s if len(s) <= max_col_width else s[:max_col_width - 1] + "…"

        header = self.header or [f"col{i+1}" for i in range(self.n_cols)]
        header = [trim(h) for h in header]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in self.rows:
            lines.append("| " + " | ".join(trim(c) for c in row) + " |")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "page_num":  self.page_num,
            "table_num": self.table_num,
            "n_rows":    self.n_rows,
            "n_cols":    self.n_cols,
            "header":    self.header,
            "rows":      self.rows,
            "caption":   self.caption,
        }


@dataclass
class TableSet:
    """All tables found in a paper."""

    arxiv_id:      str
    n_tables:      int
    tables:        list[ExtractedTable] = field(default_factory=list)
    error:         str = ""
    n_pages:       int = 0


# ══════════════════════════════════════════════════════════
# arxiv URL construction
# ══════════════════════════════════════════════════════════

def _normalize_arxiv_id_for_url(arxiv_id: str) -> str:
    """Convert an arxiv ID to the format used in URLs."""
    if arxiv_id.startswith("astro-ph-"):
        return "astro-ph/" + arxiv_id[len("astro-ph-"):]
    if arxiv_id.startswith("hep-ph-"):
        return "hep-ph/" + arxiv_id[len("hep-ph-"):]
    if arxiv_id.startswith("gr-qc-"):
        return "gr-qc/" + arxiv_id[len("gr-qc-"):]
    return arxiv_id


def _arxiv_pdf_url(arxiv_id: str) -> str:
    normalized = _normalize_arxiv_id_for_url(arxiv_id)
    return f"https://arxiv.org/pdf/{normalized}.pdf"


# ══════════════════════════════════════════════════════════
# fetch and extract
# ══════════════════════════════════════════════════════════

def fetch_pdf_bytes(arxiv_id: str, timeout: int = 30) -> bytes | None:
    """Fetch the raw PDF bytes from arxiv.org."""
    url = _arxiv_pdf_url(arxiv_id)
    try:
        resp = requests.get(
            url,
            timeout = timeout,
            headers = {"User-Agent": "AstroRAG/1.0 (research tool)"},
        )
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        pass
    return None


def extract_tables_from_bytes(pdf_bytes: bytes,
                              arxiv_id: str = "") -> TableSet:
    """Extract all tables from a PDF byte buffer."""
    if not HAS_PDFPLUMBER:
        return TableSet(
            arxiv_id=arxiv_id,
            n_tables=0,
            error="pdfplumber not installed",
        )

    tables_out: list[ExtractedTable] = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            n_pages = len(pdf.pages)

            for page_idx, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables() or []

                for tbl_idx, tbl in enumerate(page_tables, start=1):
                    if not tbl or not tbl[0]:
                        continue

                    cleaned_rows = []
                    for row in tbl:
                        cleaned = [(c or "").strip() for c in row]
                        if any(cell for cell in cleaned):
                            cleaned_rows.append(cleaned)

                    if len(cleaned_rows) < 2:
                        continue

                    header = cleaned_rows[0]
                    rows = cleaned_rows[1:]

                    n_cols = len(header)
                    n_rows = len(rows)
                    if n_cols < 2 or n_rows < 1:
                        continue

                    caption = _find_caption_on_page(page, tbl_idx)

                    tables_out.append(ExtractedTable(
                        page_num  = page_idx,
                        table_num = tbl_idx,
                        n_rows    = n_rows,
                        n_cols    = n_cols,
                        header    = header,
                        rows      = rows,
                        caption   = caption,
                    ))
    except Exception as e:
        return TableSet(
            arxiv_id=arxiv_id,
            n_tables=0,
            error=f"{type(e).__name__}: {e}",
        )

    return TableSet(
        arxiv_id = arxiv_id,
        n_tables = len(tables_out),
        tables   = tables_out,
        n_pages  = n_pages,
    )


def _find_caption_on_page(page, table_idx: int) -> str:
    """Best-effort: find a 'Table N.' caption on the page."""
    try:
        text = page.extract_text() or ""
        patterns = [
            rf"Table\s+{table_idx}\s*[:\.]\s*([^\n\r]{{0,200}})",
            rf"TABLE\s+{table_idx}\s*[:\.]\s*([^\n\r]{{0,200}})",
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    return ""


# ══════════════════════════════════════════════════════════
# high-level entry point
# ══════════════════════════════════════════════════════════

def get_paper_tables(arxiv_id: str) -> TableSet:
    """Fetch and extract tables for a paper."""
    pdf_bytes = fetch_pdf_bytes(arxiv_id)
    if pdf_bytes is None:
        return TableSet(
            arxiv_id=arxiv_id,
            n_tables=0,
            error=f"Could not fetch PDF for arXiv:{arxiv_id}",
        )
    return extract_tables_from_bytes(pdf_bytes, arxiv_id=arxiv_id)


# ══════════════════════════════════════════════════════════
# relevance scoring
# ══════════════════════════════════════════════════════════

def score_table_relevance(table: ExtractedTable, question: str) -> float:
    """Score how relevant a table is to a user question (0-1)."""
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    q_tokens = {t for t in q_tokens if len(t) > 2}
    if not q_tokens:
        return 0.0

    tbl_text = " ".join(table.header + [c for row in table.rows for c in row]
                        + [table.caption])
    tbl_tokens = set(re.findall(r"[a-z0-9]+", tbl_text.lower()))

    if not tbl_tokens:
        return 0.0

    overlap = len(q_tokens & tbl_tokens)
    return overlap / len(q_tokens)


def rank_tables_by_question(tables: list[ExtractedTable],
                            question: str,
                            top_k: int = 3) -> list[tuple[ExtractedTable, float]]:
    """Return the top-K most relevant tables, with scores."""
    scored = [(t, score_table_relevance(t, question)) for t in tables]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# ══════════════════════════════════════════════════════════
# intent detection
# ══════════════════════════════════════════════════════════

_TABLE_INTENT_KEYWORDS = [
    "table", "tables",
    "sample properties",
    "list of",
    "show me the",
    "properties table",
    "measurements table",
]


def is_table_intent(question: str) -> bool:
    """Heuristic: does this question ask for a table?"""
    q = question.lower()
    return any(kw in q for kw in _TABLE_INTENT_KEYWORDS)
