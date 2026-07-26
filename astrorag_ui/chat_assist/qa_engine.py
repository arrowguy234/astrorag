"""
Research-grade QA engine.

Given a paper's library entry and a user question, this module:
1. Infers the question's intent (methodology / results / etc.)
2. Retrieves the most relevant paper context for that intent
3. Selects the appropriate system prompt
4. Calls the LLM with much larger max_tokens for detailed answers
5. Returns the answer plus metadata about how it was produced
"""

from __future__ import annotations

import os
from   dataclasses import dataclass


try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


from chat_assist.context_retrieval  import retrieve_paper_context, PaperContext
from chat_assist.question_templates import infer_intent
from chat_assist.system_prompts     import get_prompt_for_intent


@dataclass
class AnswerResult:
    """Result of a research-grade answer."""

    answer:         str
    intent:         str
    context_chars:  int
    model_used:     str
    error:          str = ""


class ResearchPaperQA:
    """
    Enhanced QA engine for research-depth paper analysis.

    Usage:
        qa = ResearchPaperQA(library_entry=entry_dict)
        result = qa.ask("Walk me through the derivation of E_cav = 4PV")
        print(result.answer)
    """

    def __init__(
        self,
        library_entry: dict,
        history:       list[dict] = None,
        model:         str = "llama-3.3-70b-versatile",
    ) -> None:
        self.entry   = library_entry
        self.history = history or []
        self.model   = model

    def ask(self, question: str) -> AnswerResult:
        """Ask a research-grade question about the paper."""

        question = question.strip()
        if not question:
            return AnswerResult(
                answer="", intent="general", context_chars=0,
                model_used=self.model, error="empty question",
            )

        if not HAS_GROQ:
            return AnswerResult(
                answer="", intent="general", context_chars=0,
                model_used=self.model,
                error="groq package not installed",
            )

        # ── infer intent ─────────────────────────────
        intent = infer_intent(question)

        # ── build context ────────────────────────────
        context = retrieve_paper_context(self.entry, question, intent)
        context_block = context.to_prompt_block(max_chars=8000)

        # ── choose system prompt ─────────────────────
        system_prompt = get_prompt_for_intent(intent)

        # ── build conversation ───────────────────────
        # Include prior turns so the assistant maintains conversation flow.
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"CONTEXT FOR THIS PAPER:\n\n{context_block}\n\n"
                    f"I will now ask you research-oriented questions about "
                    f"this paper. Answer based ONLY on the context above. "
                    f"Cite sections and quote equations when relevant."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Understood. I have the paper's overview, equations, "
                    "numerical results, methodology summary, and sub-question "
                    "answers. Ask your research question."
                ),
            },
        ]

        # append prior history
        for m in self.history[-6:]:  # last 3 turns to keep context tight
            messages.append({"role": m["role"], "content": m["content"]})

        messages.append({"role": "user", "content": question})

        # ── call LLM ─────────────────────────────────
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                pass

        if not api_key:
            return AnswerResult(
                answer="", intent=intent, context_chars=len(context_block),
                model_used=self.model, error="GROQ_API_KEY not configured",
            )

        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                temperature = 0.2,       # slight creativity for reasoning
                max_tokens  = 1800,      # much larger for detailed answers
            )
            answer = response.choices[0].message.content.strip()
            return AnswerResult(
                answer         = answer,
                intent         = intent,
                context_chars  = len(context_block),
                model_used     = self.model,
            )
        except Exception as e:
            # fall back to production 8B if 70B fails or hits rate limit
            try:
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model       = "llama-3.1-8b-instant",
                    messages    = messages,
                    temperature = 0.2,
                    max_tokens  = 1500,
                )
                answer = response.choices[0].message.content.strip()
                return AnswerResult(
                    answer         = answer,
                    intent         = intent,
                    context_chars  = len(context_block),
                    model_used     = "llama-3.1-8b-instant (fallback)",
                )
            except Exception as e2:
                return AnswerResult(
                    answer="", intent=intent,
                    context_chars=len(context_block),
                    model_used=self.model,
                    error=f"{type(e).__name__}: {e}",
                )
