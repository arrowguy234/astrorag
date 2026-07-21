"""
AstroRAG chat interface subpackage.

Provides:
- Persistent context library (JSON-backed)
- Rich output formatting for pipeline results
- Follow-up Q&A on the selected paper
- Ipywidgets-based interactive UI
"""

from astrorag.chat.formatter import (
    format_summary_markdown,
    format_equations_table,
    format_numerical_results_table,
    format_quality_scores,
    format_stage_timings,
)
from astrorag.chat.library import (
    ContextLibrary,
    LibraryEntry,
    get_library,
)
from astrorag.chat.models import ChatMessage, ChatSession
from astrorag.chat.qa import PaperQA

__all__ = [
    "ContextLibrary",
    "LibraryEntry",
    "get_library",
    "ChatMessage",
    "ChatSession",
    "PaperQA",
    "format_summary_markdown",
    "format_equations_table",
    "format_numerical_results_table",
    "format_quality_scores",
    "format_stage_timings",
]