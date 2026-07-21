"""
Pydantic models for LLM responses.

Every LLM call in the pipeline returns JSON validated against one of
these schemas. This prevents malformed LLM output from silently
corrupting downstream stages.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ══════════════════════════════════════════════════════════
# base response wrapper
# ══════════════════════════════════════════════════════════

class LLMResponse(BaseModel):
    """
    Wrapper capturing metadata about every LLM call.

    Used for telemetry — logging cost, latency, retry counts.
    Downstream code accesses .data for the parsed response.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data:            Any
    model:           str   = ""
    latency_seconds: float = 0.0
    input_tokens:    int   = 0
    output_tokens:   int   = 0
    retries:         int   = 0
    from_cache:      bool  = False
    fallback_used:   bool  = False


# ══════════════════════════════════════════════════════════
# stage 0 — query decomposition
# ══════════════════════════════════════════════════════════

class QueryDecomposition(BaseModel):
    """
    Result of Stage 0 query decomposition.

    Contains three sub-questions targeting distinct information
    needs, plus optional structured metadata about the query.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    # ── original query ──────────────────────────────────
    original_query: str

    # ── sub-questions (required) ────────────────────────
    sub_questions: dict[str, str] = Field(
        description = "Q1, Q2, Q3 mapped to sub-question text",
    )

    # ── inferred structured metadata ────────────────────
    wavelength: str        = Field(
        default = "multi-wavelength",
        description = "X-ray | optical | radio | infrared | multi-wavelength",
    )
    catalogs: list[str]    = Field(
        default_factory = list,
        description = "Detected instrument or survey names",
    )
    query_type: str        = Field(
        default = "general",
        description = "observational | theoretical | comparative | general",
    )

    @field_validator("sub_questions")
    @classmethod
    def _validate_sub_questions(cls, v: dict[str, str]) -> dict[str, str]:
        """Ensure Q1, Q2, Q3 keys all present with non-empty values."""
        required_keys = {"Q1", "Q2", "Q3"}
        missing       = required_keys - set(v.keys())
        if missing:
            raise ValueError(
                f"sub_questions missing keys: {missing}. "
                f"Got: {list(v.keys())}"
            )
        empty = [k for k in required_keys if not v.get(k, "").strip()]
        if empty:
            raise ValueError(f"sub_questions has empty values: {empty}")
        return {k: v[k].strip() for k in required_keys}

    def summary(self) -> str:
        return (
            f"Query decomposition:\n"
            f"  Query      : {self.original_query[:80]}\n"
            f"  Q1         : {self.sub_questions['Q1']}\n"
            f"  Q2         : {self.sub_questions['Q2']}\n"
            f"  Q3         : {self.sub_questions['Q3']}\n"
            f"  Wavelength : {self.wavelength}\n"
            f"  Catalogs   : {', '.join(self.catalogs) or '(none)'}\n"
            f"  Type       : {self.query_type}"
        )


# ══════════════════════════════════════════════════════════
# stage 3 — LLM reranking
# ══════════════════════════════════════════════════════════

class RerankDecision(BaseModel):
    """Result of Stage 3 LLM reranking."""

    model_config = ConfigDict(str_strip_whitespace=True)

    best_paper_idx: int
    confidence:     float = Field(ge=0.0, le=1.0)
    reason:         str
    top5:           list[int] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════
# stage 5 — structured summarisation
# ══════════════════════════════════════════════════════════

class SubQuestionAnswer(BaseModel):
    """Answer to one sub-question with extracted technical detail."""

    answered:    bool
    answer_text: str        = ""
    section:     str        = ""
    equations:   list[str]  = Field(default_factory=list)
    values:      list[str]  = Field(default_factory=list)


class KeyEquation(BaseModel):
    equation:  str
    variables: str = ""
    section:   str = ""


class NumericalResult(BaseModel):
    quantity:    str
    value:       str = ""
    uncertainty: str = ""
    unit:        str = ""


class StructuredSummary(BaseModel):
    """Full structured summary from Stage 5."""

    model_config = ConfigDict(str_strip_whitespace=True)

    paper_overview:        str
    sub_question_answers:  dict[str, SubQuestionAnswer]
    evidence_type:         str        = "observational"
    instruments:           list[str]  = Field(default_factory=list)
    key_equations:         list[KeyEquation]    = Field(default_factory=list)
    numerical_results:     list[NumericalResult] = Field(default_factory=list)
    key_findings:          list[str]  = Field(default_factory=list)
    methodology:           str        = ""
    limitations:           list[str]  = Field(default_factory=list)
    key_snippet:           str        = ""