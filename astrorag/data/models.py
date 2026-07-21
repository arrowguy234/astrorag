"""
Pydantic data models for AstroRAG corpus data.

These models validate the structure of records read from the raw
dataset files and provide a typed interface for downstream stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing      import Any

import numpy as np
import pandas as pd
from   pydantic import BaseModel, ConfigDict, Field, field_validator


# ══════════════════════════════════════════════════════════
# per-record models
# ══════════════════════════════════════════════════════════

class Paper(BaseModel):
    """
    Single paper record from abstracts_all.jsonl.gz.

    Fields present in the raw JSONL are captured here; extra fields
    are ignored to allow schema evolution.
    """

    model_config = ConfigDict(
        extra              = "allow",
        str_strip_whitespace = True,
    )

    paper_idx: int   = Field(
        description = "Integer index within the corpus (0-based)",
    )
    arxiv_id:  str   = Field(
        default     = "",
        description = "arXiv identifier (e.g. '0709.2152' or '2301.07688')",
    )
    title:     str   = Field(
        default     = "",
        description = "Paper title",
    )
    abstract:  str   = Field(
        default     = "",
        description = "Paper abstract text",
    )

    @field_validator("arxiv_id", mode="before")
    @classmethod
    def _stringify_arxiv_id(cls, v: Any) -> str:
        """Ensure arxiv_id is always a stripped string."""
        return str(v).strip() if v is not None else ""

    def searchable_text(self, concept_labels: list[str] | None = None,
                        max_concepts: int = 15) -> str:
        """
        Return concatenated searchable text: title + abstract + concepts.

        Used by BM25 index construction to include concept labels
        in the searchable representation.
        """
        text = f"{self.title} {self.abstract}"
        if concept_labels:
            text += " " + " ".join(concept_labels[:max_concepts])
        return text.strip()


class Concept(BaseModel):
    """Single concept from concepts_vocabulary.csv.gz."""

    model_config = ConfigDict(extra="allow")

    concept_idx: int
    label:       str
    domain:      str = ""
    subdomain:   str = ""


class CitationRecord(BaseModel):
    """Single citation record from citations_indexed.jsonl.gz."""

    model_config = ConfigDict(extra="allow")

    paper_idx:  int
    arxiv_id:   str            = ""
    references: list[str]      = Field(default_factory=list)
    cited_by:   list[str]      = Field(default_factory=list)


# ══════════════════════════════════════════════════════════
# corpus statistics
# ══════════════════════════════════════════════════════════

class CorpusStats(BaseModel):
    """Aggregate statistics about the loaded corpus."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_papers:                  int   = 0
    n_concepts:                int   = 0
    n_papers_with_concepts:    int   = 0
    n_papers_with_citations:   int   = 0
    concept_emb_dim:           int   = 0
    total_paper_concept_edges: int   = 0
    avg_concepts_per_paper:    float = 0.0
    avg_refs_per_paper:        float = 0.0
    avg_citers_per_paper:      float = 0.0
    load_time_seconds:         float = 0.0
    memory_usage_mb:           float = 0.0

    def summary(self) -> str:
        return f"""
Corpus Statistics
{'═' * 60}
  Papers loaded            : {self.n_papers:>12,}
  Papers with concepts     : {self.n_papers_with_concepts:>12,}
  Papers with citations    : {self.n_papers_with_citations:>12,}
  Concept vocabulary size  : {self.n_concepts:>12,}
  Concept embedding dim    : {self.concept_emb_dim:>12,}
  Paper-concept edges      : {self.total_paper_concept_edges:>12,}
  Avg concepts per paper   : {self.avg_concepts_per_paper:>12.2f}
  Avg references per paper : {self.avg_refs_per_paper:>12.2f}
  Avg citers per paper     : {self.avg_citers_per_paper:>12.2f}
  Load time                : {self.load_time_seconds:>12.1f}s
  Memory usage             : {self.memory_usage_mb:>12.1f} MB
{'═' * 60}
"""


# ══════════════════════════════════════════════════════════
# load configuration
# ══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LoadConfig:
    """
    Configuration for data loading behaviour.

    Controls what fraction of the corpus loads and whether cache
    is used or regenerated.
    """
    sample_size:      int  = 408_590
    use_cache:        bool = True
    force_reload:     bool = False
    show_progress:    bool = True
    validate_records: bool = False