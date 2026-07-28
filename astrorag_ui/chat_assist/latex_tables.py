"""
Extract tables directly from arXiv LaTeX source instead of the rendered PDF.

Astrophysics papers almost universally use AASTeX, whose deluxetable
environment has no vertical rule lines in the rendered PDF -- which is
exactly what defeats pdfplumber's line-based table detector. But the
LaTeX source itself is just `&`-delimited columns and `\\`-terminated
rows, which we can parse directly and far more reliably. Simple
\\newcommand/\\def macros (e.g. \\co for $^{12}$CO shorthand) are expanded
before parsing so table cells resolve to real molecule/line names instead
of vanishing under generic command stripping.
"""

from __future__ import annotations

import io
import re
import tarfile
import gzip
from   dataclasses import dataclass, field

import requests

from chat_assist.pdf_tables import ExtractedTable, TableSet, _normalize_arxiv_id_for_url


def _arxiv_source_url(arxiv_id: str) -> str:
    normalized = _normalize_arxiv_id_for_url(arxiv_id)
    return f"https://arxiv.org/e-print/{normalized}"


def fetch_latex_source(arxiv_id: str, timeout: int = 30) -> str | None:
    """Fetch and concatenate all .tex files from the arXiv source bundle."""
    url = _arxiv_source_url(arxiv_id)
    try:
        resp = requests.get(
            url, timeout=timeout,
            headers={"User-Agent": "AstroRAG/1.0 (research tool)"},
        )
        if resp.status_code != 200 or not resp.content:
            return None
        raw = resp.content
    except Exception:
        return None

    # try as tar.gz bundle (most common for multi-file submissions)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            tex_parts = []
            for member in tar.getmembers():
                if member.name.endswith(".tex") and member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        tex_parts.append(f.read().decode("utf-8", errors="ignore"))
            if tex_parts:
                return "\n\n".join(tex_parts)
    except tarfile.ReadError:
        pass

    # fall back: raw gzip of a single .tex file (common for simple submissions)
    try:
        text = gzip.decompress(raw).decode("utf-8", errors="ignore")
        if "\\documentclass" in text or "\\begin{document}" in text:
            return text
    except OSError:
        pass

    # last resort: maybe it wasn't gzipped at all
    try:
        text = raw.decode("utf-8", errors="ignore")
        if "\\documentclass" in text or "\\begin{document}" in text:
            return text
    except Exception:
        pass

    return None


# ══════════════════════════════════════════════════════════
# macro expansion (\co -> $^{12}$CO, etc.)
# ══════════════════════════════════════════════════════════

_NEWCOMMAND_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand)\*?\{\\([a-zA-Z]+)\}(?:\[\d+\])?\{((?:[^{}]|\{[^{}]*\})*)\}"
)
_DEF_RE = re.compile(r"\\def\\([a-zA-Z]+)\{((?:[^{}]|\{[^{}]*\})*)\}")


def _expand_macros(source: str, max_passes: int = 3) -> str:
    """
    Expand simple \\newcommand/\\def macros so table cells that reference
    shorthand like \\co (often defined as $^{12}$CO) resolve to real text
    instead of vanishing when generic \\command stripping runs.
    """
    macros = {}
    for pat in (_NEWCOMMAND_RE, _DEF_RE):
        for m in pat.finditer(source):
            name, body = m.group(1), m.group(2)
            if len(body) < 60:
                macros[name] = body

    if not macros:
        return source

    for _ in range(max_passes):
        changed = False
        for name, body in macros.items():
            pattern = re.compile(r"\\" + name + r"\b")
            new_source, n = pattern.subn(lambda m:body, source)
            if n:
                source = new_source
                changed = True
        if not changed:
            break

    return source


# ══════════════════════════════════════════════════════════
# LaTeX table-environment parsing
# ══════════════════════════════════════════════════════════

_TABLE_ENV_RE = re.compile(
    r"\\begin\{(deluxetable\*?|table\*?|longtable\*?)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)

_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_MATH_RE = re.compile(r"\$([^$]*)\$")


def _clean_cell(cell: str) -> str:
    """Strip LaTeX formatting commands and math delimiters from one cell."""
    cell = _COMMENT_RE.sub("", cell)

    # common astro molecule notation: ^{12}CO -> 12CO,  _{k} -> k
    cell = re.sub(r"\^\{?(\d+)\}?", r"\1", cell)
    cell = re.sub(r"_\{?([a-zA-Z0-9,]+)\}?", r"_\1", cell)

    cell = _MATH_RE.sub(r"\1", cell)              # $x$ -> x
    cell = re.sub(r"\\tablenotemark\{[^}]*\}", "", cell)
    cell = re.sub(r"\\arcsec", '"', cell)
    cell = re.sub(r"\\arcmin", "'", cell)
    cell = re.sub(r"\\times", "x", cell)
    cell = re.sub(r"\\rm\b", "", cell)
    cell = re.sub(r"\\[a-zA-Z]+", " ", cell)       # drop remaining \commands
    cell = cell.replace("{", "").replace("}", "")
    cell = cell.replace("\\\\", " ").replace("\\", " ")  # stray line-end markers
    cell = re.sub(r"\s+", " ", cell).strip()
    return cell


def _extract_caption(table_body: str) -> str:
    m = re.search(r"\\(?:tablecaption|caption)\{(.*?)\}(?=\s*\\|\s*$)",
                  table_body, re.DOTALL)
    if m:
        return _clean_cell(m.group(1))[:200]
    return ""


def _split_deluxetable(table_body: str) -> tuple[list[str], list[list[str]]]:
    """Parse a deluxetable's \\tablehead and \\startdata...\\enddata blocks."""
    header: list[str] = []
    rows: list[list[str]] = []

    head_m = re.search(r"\\tablehead\{(.*?)\}\s*(?=\\startdata|\\enddata)",
                       table_body, re.DOTALL)
    if head_m:
        head_raw = head_m.group(1)
        # tablehead often has two \\-separated lines: names, then units.
        # Merge them into "Name (unit)" per column instead of treating
        # the units line as extra trailing columns.
        head_lines = [l for l in re.split(r"(?<!\\)\\\\", head_raw) if l.strip()]

        line_cols = []
        for line in head_lines:
            cols = re.split(r"(?<!\\)&", line)
            line_cols.append([_clean_cell(c) for c in cols])

        if len(line_cols) == 1:
            header = line_cols[0]
        elif line_cols:
            n = max(len(l) for l in line_cols)
            merged = []
            for i in range(n):
                parts = [l[i] for l in line_cols if i < len(l) and l[i]]
                merged.append(" ".join(parts).strip())
            header = merged

    data_m = re.search(r"\\startdata(.*?)\\enddata", table_body, re.DOTALL)
    if data_m:
        data_raw = data_m.group(1)
        for line in re.split(r"(?<!\\)\\\\", data_raw):
            line = line.strip()
            if not line:
                continue
            cols = re.split(r"(?<!\\)&", line)
            cleaned = [_clean_cell(c) for c in cols]
            if any(cleaned):
                rows.append(cleaned)

    return header, rows


def _split_standard_tabular(table_body: str) -> tuple[list[str], list[list[str]]]:
    """Parse a standard \\begin{tabular}...\\end{tabular} block."""
    tab_m = re.search(r"\\begin\{tabular\}(?:\{[^}]*\})?(.*?)\\end\{tabular\}",
                      table_body, re.DOTALL)
    if not tab_m:
        return [], []

    body = tab_m.group(1)
    body = re.sub(r"\\(?:hline|toprule|midrule|bottomrule|cline\{[^}]*\})", "",
                  body)

    lines = [l.strip() for l in re.split(r"(?<!\\)\\\\", body) if l.strip()]
    if not lines:
        return [], []

    parsed_rows = []
    for line in lines:
        cols = re.split(r"(?<!\\)&", line)
        cleaned = [_clean_cell(c) for c in cols]
        if any(cleaned):
            parsed_rows.append(cleaned)

    if not parsed_rows:
        return [], []

    return parsed_rows[0], parsed_rows[1:]


def extract_tables_from_latex(source: str, arxiv_id: str = "") -> TableSet:
    """Find and parse all table/deluxetable environments in LaTeX source."""
    tables_out: list[ExtractedTable] = []

    for i, m in enumerate(_TABLE_ENV_RE.finditer(source), start=1):
        env_name, body = m.group(1), m.group(2)
        caption = _extract_caption(body)

        if "deluxetable" in env_name:
            header, rows = _split_deluxetable(body)
        else:
            header, rows = _split_standard_tabular(body)

        if not rows or len(header) < 2:
            continue

        n_cols = len(header)
        norm_rows = []
        for r in rows:
            if len(r) < n_cols:
                r = r + [""] * (n_cols - len(r))
            elif len(r) > n_cols:
                r = r[:n_cols]
            norm_rows.append(r)

        # drop columns that are empty across header AND every row
        keep_idx = [
            j for j in range(n_cols)
            if header[j].strip() or any(row[j].strip() for row in norm_rows)
        ]
        if len(keep_idx) < 2:
            continue

        header = [header[j] for j in keep_idx]
        norm_rows = [[row[j] for j in keep_idx] for row in norm_rows]
        n_cols = len(header)

        # skip degenerate tables that are really just \input{} stubs or
        # empty shells with almost no actual content
        non_empty_data_cells = sum(
            1 for row in norm_rows for c in row if c.strip()
        )
        if non_empty_data_cells < 3:
            continue

        tables_out.append(ExtractedTable(
            page_num  = 0,       # unknown from source alone
            table_num = i,
            n_rows    = len(norm_rows),
            n_cols    = n_cols,
            header    = header,
            rows      = norm_rows,
            caption   = caption,
        ))

    return TableSet(
        arxiv_id = arxiv_id,
        n_tables = len(tables_out),
        tables   = tables_out,
    )


def get_paper_tables_from_source(arxiv_id: str) -> TableSet:
    """High-level entry point: fetch LaTeX source and extract tables."""
    source = fetch_latex_source(arxiv_id)
    if source is None:
        return TableSet(
            arxiv_id=arxiv_id, n_tables=0,
            error=f"Could not fetch/parse LaTeX source for arXiv:{arxiv_id}",
        )
    source = _expand_macros(source)
    return extract_tables_from_latex(source, arxiv_id=arxiv_id)