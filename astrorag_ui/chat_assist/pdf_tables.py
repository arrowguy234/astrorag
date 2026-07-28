"""
On-demand table retrieval from arXiv PDFs.

Fetches the PDF, extracts candidate tables with pdfplumber, and applies a
content-quality filter before returning anything. Astrophysics papers are
dense with multi-panel figures whose vector-drawn axis boxes get picked up
by pdfplumber's line-based table detector as false-positive "tables" —
these are filtered out based on empty-cell ratio, cross-cell word
repetition (figure panels tile the same labels across many cells), average
cell length (figure captions bleeding in produce long fragments), value
uniqueness, sub-panel labels like "(a)"/"(b)", arithmetic-sequence numeric
grids (spectral/velocity axis channels), and repeated short source-code
labels tiled as a figure legend rather than a genuine per-row ID column.
"""

from __future__ import annotations

import io
import re
from   collections import Counter
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
    """All tables found in a paper (post-filter)."""

    arxiv_id:      str
    n_tables:      int
    tables:        list[ExtractedTable] = field(default_factory=list)
    error:         str = ""
    n_pages:       int = 0
    n_rejected:    int = 0   # candidates dropped by the quality filter


# ══════════════════════════════════════════════════════════
# arxiv URL construction
# ══════════════════════════════════════════════════════════

def _normalize_arxiv_id_for_url(arxiv_id: str) -> str:
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
# fetch
# ══════════════════════════════════════════════════════════

def fetch_pdf_bytes(arxiv_id: str, timeout: int = 30) -> bytes | None:
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


# ══════════════════════════════════════════════════════════
# quality filter — reject figure-grid false positives
# ══════════════════════════════════════════════════════════

_WORD_RE = re.compile(r"[A-Za-z]{4,}")


def _looks_like_real_table(header: list[str], rows: list[list[str]]) -> bool:
    """
    Heuristic filter distinguishing genuine data tables from pdfplumber
    false positives on multi-panel figure axis grids.

    Rejects tables that show any of the following patterns typical of
    figure-grid misdetections:
      1. high fraction of empty cells (plot interiors are mostly blank)
      2. long average cell length (figure captions bleeding into cells)
      3. a single word repeated across a large fraction of cells (figure
         panel labels like "eclipses" or a planet name tiled many times)
      4. low value uniqueness (repeated identical cell content)
      5. sub-panel labels — "(a)", "(b)", "a", "b" tiled across cells
      6. arithmetic-sequence numeric grid — spectral/velocity channel axes
         tabulate near-uniformly-spaced float values across many cells
      7. repeated short source-code labels (e.g. "N43") tiled as a figure
         legend rather than a genuine per-row ID column
    """
    try:
        all_cells = list(header) + [c for row in rows for c in row]
        non_empty = [c.strip() for c in all_cells if c and c.strip()]
        total = len(all_cells)

        if total == 0 or not non_empty:
            return False

        # 1. empty-cell ratio
        empty_fraction = 1 - (len(non_empty) / total)
        if empty_fraction > 0.35:
            return False

        # 2. average cell length — long fragments suggest caption bleed-through
        avg_len = sum(len(c) for c in non_empty) / len(non_empty)
        if avg_len > 35:
            return False

        # 3. cross-cell word repetition — figure panels tile the same labels
        tokens = []
        for c in non_empty:
            tokens.extend(t.lower() for t in _WORD_RE.findall(c))
        if tokens:
            counts = Counter(tokens)
            _, most_common_n = counts.most_common(1)[0]
            if most_common_n / len(non_empty) > 0.40:
                return False

        # 4. value uniqueness — real tables rarely repeat identical cells
        unique_ratio = len(set(non_empty)) / len(non_empty)
        if unique_ratio < 0.5:
            return False

        # 5. sub-panel labels — "(a)", "(b)", "a", "b" etc. tiled across cells
        panel_label_re = re.compile(r"^\(?[a-hA-H]\)?$")
        n_panel_labels = sum(1 for c in non_empty if panel_label_re.match(c))
        if n_panel_labels / len(non_empty) > 0.3:
            return False

        # 6. arithmetic-sequence numeric grid — spectral/velocity channel axes
        #    tabulate near-uniformly-spaced float values across many cells
        numeric_vals = []
        for c in non_empty:
            try:
                numeric_vals.append(float(c))
            except ValueError:
                continue
        if len(numeric_vals) >= 8 and len(numeric_vals) / len(non_empty) > 0.7:
            sorted_vals = sorted(numeric_vals)
            diffs = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
            if diffs:
                mean_diff = sum(diffs) / len(diffs)
                if mean_diff > 0:
                    variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
                    cv = (variance ** 0.5) / mean_diff  # coefficient of variation
                    if cv < 0.15:  # very uniform spacing → axis, not data
                        return False

        # 7. short alnum source-code labels repeated across cells (e.g. "N43",
        #    "N51" tiled as a figure legend rather than a real ID column)
        code_re = re.compile(r"^[A-Z]{1,3}\d{1,4}[a-zA-Z]?$")
        code_cells = [c for c in non_empty if code_re.match(c)]
        if code_cells:
            code_counts = Counter(code_cells)
            most_common_code, most_common_code_n = code_counts.most_common(1)[0]
            if most_common_code_n >= 2 and len(code_cells) / len(non_empty) > 0.5:
                if most_common_code_n / len(code_cells) > 0.25:
                    return False

        return True

    except Exception:
        # fail closed — reject on any unexpected error rather than risk
        # showing garbled content
        return False


# ══════════════════════════════════════════════════════════
# extraction
# ══════════════════════════════════════════════════════════

def extract_tables_from_bytes(pdf_bytes: bytes,
                              arxiv_id: str = "") -> TableSet:
    if not HAS_PDFPLUMBER:
        return TableSet(
            arxiv_id=arxiv_id, n_tables=0,
            error="pdfplumber not installed",
        )

    tables_out: list[ExtractedTable] = []
    n_rejected = 0

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

                    # apply the quality filter
                    if not _looks_like_real_table(header, rows):
                        n_rejected += 1
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
            arxiv_id=arxiv_id, n_tables=0,
            error=f"{type(e).__name__}: {e}",
        )

    return TableSet(
        arxiv_id   = arxiv_id,
        n_tables   = len(tables_out),
        tables     = tables_out,
        n_pages    = n_pages,
        n_rejected = n_rejected,
    )


def _find_caption_on_page(page, table_idx: int) -> str:
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
    pdf_bytes = fetch_pdf_bytes(arxiv_id)
    if pdf_bytes is None:
        return TableSet(
            arxiv_id=arxiv_id, n_tables=0,
            error=f"Could not fetch PDF for arXiv:{arxiv_id}",
        )
    return extract_tables_from_bytes(pdf_bytes, arxiv_id=arxiv_id)


# ══════════════════════════════════════════════════════════
# relevance scoring
# ══════════════════════════════════════════════════════════

def score_table_relevance(table: ExtractedTable, question: str) -> float:
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
    q = question.lower()
    return any(kw in q for kw in _TABLE_INTENT_KEYWORDS)
