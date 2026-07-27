"""
Research-grade QA engine.

Given a paper's library entry and a user question, this module:
1. Infers the question's intent (methodology / results / etc.)
2. Retrieves the most relevant paper context for that intent
3. On table-intent questions, fetches PDF and extracts tables on demand
4. Selects the appropriate system prompt
5. Calls the LLM with larger max_tokens for detailed answers
6. Returns the answer plus metadata about how it was produced
"""

from __future__ import annotations

import os
from   dataclasses import dataclass, field


try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


from chat_assist.context_retrieval  import retrieve_paper_context
from chat_assist.question_templates import infer_intent
from chat_assist.system_prompts     import get_prompt_for_intent
from chat_assist.pdf_tables         import (
    get_paper_tables,
    rank_tables_by_question,
    is_table_intent,
)


@dataclass
class AnswerResult:
    """Result of a research-grade answer."""

    answer:         str
    intent:         str
    context_chars:  int
    model_used:     str
    error:          str = ""
    tables:         list[dict] = field(default_factory=list)


class ResearchPaperQA:
    """
    Enhanced QA engine for research-depth paper analysis.
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

        # ── table intent: fetch and extract on demand ─
        rendered_tables: list[dict] = []

        if intent == "tables" or is_table_intent(question):
            if isinstance(self.entry, dict):
                arxiv_id = self.entry.get("arxiv_id", "")
            else:
                arxiv_id = getattr(self.entry, "arxiv_id", "")

            if arxiv_id:
                table_set = get_paper_tables(arxiv_id)
                if not table_set.error and table_set.n_tables > 0:
                    top_tables = rank_tables_by_question(
                        table_set.tables, question, top_k=3,
                    )
                    for tbl, score in top_tables:
                        rendered_tables.append({
                            "page_num":  tbl.page_num,
                            "table_num": tbl.table_num,
                            "caption":   tbl.caption,
                            "n_rows":    tbl.n_rows,
                            "n_cols":    tbl.n_cols,
                            "markdown":  tbl.to_markdown(),
                            "relevance": round(score, 3),
                        })

        # ── build context ────────────────────────────
        context = retrieve_paper_context(self.entry, question, intent)
        context_block = context.to_prompt_block(max_chars=8000)

        # append rendered tables to context if we retrieved any
        if rendered_tables:
            tbl_block = ["\n=== TABLES EXTRACTED FROM THE PDF ===\n"]
            for t in rendered_tables[:3]:
                tbl_block.append(
                    f"[Table {t['table_num']} on page {t['page_num']}] "
                    f"{t['caption'] or '(no caption)'}\n"
                    f"{t['markdown']}\n"
                )
            context_block = context_block + "\n" + "\n".join(tbl_block)

        # ── choose system prompt ─────────────────────
        system_prompt = get_prompt_for_intent(intent)

        if rendered_tables:
            system_prompt = system_prompt + (
                "\n\nADDITIONAL INSTRUCTION: The user is asking about tables. "
                "Tables extracted directly from the PDF are provided below in "
                "markdown format. Refer to them by page and table number. "
                "You may summarize their content or point out what they contain, "
                "but the tables themselves will be shown to the user separately."
            )

        # ── build conversation ───────────────────────
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
                    "answers"
                    + (", plus tables extracted directly from the PDF" if rendered_tables else "")
                    + ". Ask your research question."
                ),
            },
        ]

        # append prior history (last 3 turns to keep prompt tight)
        for m in self.history[-6:]:
            messages.append({"role": m["role"], "content": m["content"]})

        messages.append({"role": "user", "content": question})

        # ── resolve API key ──────────────────────────
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
                tables=rendered_tables,
            )

        # ── call LLM ─────────────────────────────────
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                temperature = 0.2,
                max_tokens  = 1800,
            )
            answer = response.choices[0].message.content.strip()
            return AnswerResult(
                answer         = answer,
                intent         = intent,
                context_chars  = len(context_block),
                model_used     = self.model,
                tables         = rendered_tables,
            )
        except Exception as e:
            # fallback to 8B on rate-limit or error
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
                    tables         = rendered_tables,
                )
            except Exception as e2:
                return AnswerResult(
                    answer="", intent=intent,
                    context_chars=len(context_block),
                    model_used=self.model,
                    error=f"{type(e).__name__}: {e}",
                    tables=rendered_tables,
                )
