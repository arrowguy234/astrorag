"""
Query Assistance module — helps users construct well-formed queries
for AstroRAG. This is UI-only; nothing here touches the production
pipeline. All modules assume access to Groq via GROQ_API_KEY.
"""

from astrorag_ui.query_assist.templates      import QUERY_TEMPLATES, get_template
from astrorag_ui.query_assist.decomposer     import preview_decomposition
from astrorag_ui.query_assist.similar        import find_similar_queries
from astrorag_ui.query_assist.quality_check  import assess_query_quality
from astrorag_ui.query_assist.refiner        import suggest_refinements

__all__ = [
    "QUERY_TEMPLATES", "get_template",
    "preview_decomposition",
    "find_similar_queries",
    "assess_query_quality",
    "suggest_refinements",
]
