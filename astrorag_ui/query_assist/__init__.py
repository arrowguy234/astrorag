"""
Query Assistance module — helps users construct well-formed queries
for AstroRAG. This is UI-only; nothing here touches the production
pipeline.
"""

from query_assist.templates      import QUERY_TEMPLATES, get_template, fill_template
from query_assist.decomposer     import preview_decomposition
from query_assist.similar        import find_similar_queries
from query_assist.quality_check  import assess_query_quality
from query_assist.refiner        import suggest_refinements, build_refined_query

__all__ = [
    "QUERY_TEMPLATES", "get_template", "fill_template",
    "preview_decomposition",
    "find_similar_queries",
    "assess_query_quality",
    "suggest_refinements", "build_refined_query",
]