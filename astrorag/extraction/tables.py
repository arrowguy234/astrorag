"""
Table detection in raw PDF text.

PDFs extracted from two-column layouts often contain tabular data
that appears as consecutive lines of numbers or pipe-separated
values. We detect these groups and extract them as table blocks.
"""

from __future__ import annotations

import re


NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")


def is_table_row(line: str, min_numeric_tokens: int = 3) -> bool:
    """
    Detect whether a line looks like a table row.

    Rules:
      1. Contains min_numeric_tokens or more numeric tokens, OR
      2. Uses pipe (|) separators with at least 3 fields, OR
      3. Uses tab (\\t) separators with at least 3 fields.
    """
    if not line or not line.strip():
        return False

    stripped = line.strip()

    # pipe-separated
    if stripped.count("|") >= 3:
        return True

    # tab-separated
    if stripped.count("\t") >= 2:
        parts = stripped.split("\t")
        if sum(1 for p in parts if p.strip()) >= 3:
            return True

    # numeric-heavy row
    numeric_tokens = NUMERIC_TOKEN_RE.findall(stripped)
    return len(numeric_tokens) >= min_numeric_tokens


def extract_tables(
    text:       str,
    max_tables: int = 5,
    min_rows:   int = 2,
) -> list[str]:
    """
    Extract table blocks from raw text.

    Groups consecutive table-like lines into blocks. Returns each
    block as a joined string.

    Args:
        text:        Raw PDF text.
        max_tables:  Maximum tables to return.
        min_rows:    Minimum consecutive rows to count as a table.

    Returns:
        List of table block strings.
    """
    if not text:
        return []

    tables:  list[str]       = []
    current: list[str]       = []
    lines                    = text.split("\n")

    for line in lines:
        if is_table_row(line):
            current.append(line.rstrip())
        else:
            if len(current) >= min_rows:
                tables.append("\n".join(current))
                if len(tables) >= max_tables:
                    return tables
            current = []

    # flush last block
    if len(current) >= min_rows:
        tables.append("\n".join(current))

    return tables[:max_tables]