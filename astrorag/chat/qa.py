"""
Follow-up Q&A on a selected paper.

Once the pipeline selects and summarises a paper, the user can ask
follow-up questions. This module handles those follow-ups by giving
the LLM the paper's full text plus the summary as context.
"""

from __future__ import annotations

import time

from astrorag.chat.library import LibraryEntry
from astrorag.chat.models  import ChatMessage, ChatSession
from astrorag.config       import Settings, get_settings
from astrorag.llm          import LLMClient, get_llm_client
from astrorag.logger       import get_logger

logger = get_logger(__name__)


_QA_SYSTEM_PROMPT = (
    "You are an expert astrophysicist answering follow-up questions about "
    "a specific paper the user has already retrieved. "
    "Use the paper's summary, equations, and numerical results provided "
    "in the context to answer precisely and quantitatively. "
    "If the answer is not in the context, say so clearly. "
    "Do not invent data. Cite section names when relevant. "
    "Keep answers focused and technical."
)


class PaperQA:
    """
    Follow-up question answerer for a specific paper.

    Usage:
        qa = PaperQA(entry=library_entry)
        answer = qa.ask("What is the value of the cavity power?")
        session = qa.session   # accumulated messages
    """

    def __init__(
        self,
        entry:      LibraryEntry,
        settings:   Settings   | None = None,
        llm_client: LLMClient  | None = None,
        paper_text: str = "",
    ) -> None:
        self.entry      = entry
        self.settings   = settings or get_settings()
        self._llm       = llm_client
        self.paper_text = paper_text

        self.session = ChatSession(
            arxiv_id       = entry.arxiv_id,
            paper_title    = entry.title,
            original_query = entry.original_query,
        )

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    def _build_context(self) -> str:
        """Build the paper context block for the LLM prompt."""
        e = self.entry
        parts = [
            f"PAPER: arXiv:{e.arxiv_id}",
            f"TITLE: {e.title}",
            f"",
            f"OVERVIEW: {e.paper_overview}",
            f"",
            f"EVIDENCE TYPE: {e.evidence_type}",
        ]
        if e.instruments:
            parts.append(f"INSTRUMENTS: {', '.join(e.instruments)}")
        parts.append("")

        if e.key_equations:
            parts.append("KEY EQUATIONS EXTRACTED:")
            for eq in e.key_equations[:10]:
                parts.append(
                    f"  - {eq.get('equation', '')} "
                    f"(variables: {eq.get('variables', '')})"
                )
            parts.append("")

        if e.numerical_results:
            parts.append("NUMERICAL RESULTS EXTRACTED:")
            for nr in e.numerical_results[:15]:
                parts.append(
                    f"  - {nr.get('quantity', '')} = "
                    f"{nr.get('value', '')} "
                    f"± {nr.get('uncertainty', '')} "
                    f"{nr.get('unit', '')}"
                )
            parts.append("")

        if e.sub_question_answers:
            parts.append("SUB-QUESTION ANSWERS:")
            for qk, ans in e.sub_question_answers.items():
                if ans.get("answered"):
                    parts.append(
                        f"  {qk} ({ans.get('section', '')}): "
                        f"{ans.get('answer_text', '')[:400]}"
                    )
            parts.append("")

        if e.methodology:
            parts.append(f"METHODOLOGY: {e.methodology[:600]}")
            parts.append("")

        if e.key_snippet:
            parts.append(f"KEY QUOTE: \"{e.key_snippet}\"")
            parts.append("")

        # if we have full paper text, append truncated
        if self.paper_text:
            parts.append("PAPER FULL TEXT (truncated):")
            parts.append(self.paper_text[:4000])

        return "\n".join(parts)

    def ask(self, question: str) -> str:
        """
        Ask a follow-up question and return the LLM's answer.

        The full conversation history is included in the LLM prompt
        so the LLM can maintain conversational context.
        """
        question = question.strip()
        if not question:
            return "Please enter a question."

        self.session.add_user(question)

        context = self._build_context()

        # build conversation history for LLM
        conversation_history = []
        for m in self.session.messages[:-1]:  # all but the just-added user msg
            conversation_history.append(f"{m.role.upper()}: {m.content}")
        history_block = "\n\n".join(conversation_history) if conversation_history else "(none yet)"

        user_prompt = (
            f"CONTEXT (about the paper the user is asking):\n"
            f"{context}\n\n"
            f"CONVERSATION HISTORY:\n{history_block}\n\n"
            f"CURRENT QUESTION: {question}\n\n"
            f"Answer precisely using only the paper context above. "
            f"If a value or equation is in the context, quote it exactly. "
            f"If not, say so."
        )

        try:
            raw, telemetry = self.llm.chat(
                system      = _QA_SYSTEM_PROMPT,
                user        = user_prompt,
                temperature = 0.0,
                max_tokens  = 800,
                stage_name  = "paper_qa",
            )
            answer = raw.strip()
        except Exception as e:
            answer = f"⚠ LLM error: {type(e).__name__}: {e}"
            logger.error(f"Paper QA failed: {e}")

        self.session.add_assistant(answer)
        return answer

    def reset(self) -> None:
        """Reset the session (start a new conversation)."""
        self.session = ChatSession(
            arxiv_id       = self.entry.arxiv_id,
            paper_title    = self.entry.title,
            original_query = self.entry.original_query,
        )