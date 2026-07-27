"""
Chat Assistance module — research-grade Q&A over a selected paper.
"""

from chat_assist.system_prompts     import RESEARCH_SYSTEM_PROMPT, get_prompt_for_intent
from chat_assist.question_templates import QUESTION_TEMPLATES, get_templates_by_category
from chat_assist.context_retrieval  import retrieve_paper_context
from chat_assist.qa_engine          import ResearchPaperQA
from chat_assist.dual_summary       import generate_dual_summary, DualSummary

__all__ = [
    "RESEARCH_SYSTEM_PROMPT", "get_prompt_for_intent",
    "QUESTION_TEMPLATES", "get_templates_by_category",
    "retrieve_paper_context",
    "ResearchPaperQA",
    "generate_dual_summary", "DualSummary",
]