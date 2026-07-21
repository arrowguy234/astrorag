"""
Content extraction subpackage.

Provides utilities for extracting equations, tables, and section
routing decisions from raw PDF text. Used by Stage 5 to build
technically dense context blocks for the LLM.
"""

from astrorag.extraction.equations import (
    extract_equations,
    extract_measurements,
    EQUATION_PATTERNS,
)
from astrorag.extraction.tables import (
    extract_tables,
    is_table_row,
)
from astrorag.extraction.routing import (
    detect_question_type,
    build_technical_context,
    SECTION_ROUTING,
)
from astrorag.extraction.quality import (
    QualityAssessment,
    QualityDecision,
    QualityScores,
    assess_quality,
)

__all__ = [
    "extract_equations",
    "extract_measurements",
    "extract_tables",
    "is_table_row",
    "detect_question_type",
    "build_technical_context",
    "assess_quality",
    "QualityAssessment",
    "QualityDecision",
    "QualityScores",
    "EQUATION_PATTERNS",
    "SECTION_ROUTING",
]