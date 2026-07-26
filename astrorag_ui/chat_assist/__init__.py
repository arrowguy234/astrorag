"""
Chat Assistance module — research-grade Q&A over a selected paper.

Focus areas:
- Deep methodology, results, and derivation questions
- Comparative and critical questions
- Section-aware context retrieval
- Transparent citation of paper sections in answers
"""

from chat_assist.system_prompts     import RESEARCH_SYSTEM_PROMPT, get_prompt_for_intent
from chat_assist.question_templates import QUESTION_TEMPLATES, get_templates_by_category
from chat_assist.context_retrieval  import retrieve_paper_context
from chat_assist.qa_engine          import ResearchPaperQA

__all__ = [
    "RESEARCH_SYSTEM_PROMPT", "get_prompt_for_intent",
    "QUESTION_TEMPLATES", "get_templates_by_category",
    "retrieve_paper_context",
    "ResearchPaperQA",
]