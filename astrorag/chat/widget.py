"""
Ipywidgets-based interactive UI for AstroRAG.

Provides tabs for:
- Query input and pipeline execution
- Structured summary display
- Follow-up Q&A on the selected paper
- Context library browser
"""

from __future__ import annotations

from typing import Any

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML, Markdown
    HAS_IPYWIDGETS = True
except ImportError:
    HAS_IPYWIDGETS = False

from astrorag.chat.formatter import (
    format_summary_markdown,
    format_equations_table,
    format_numerical_results_table,
    format_quality_scores,
    format_stage_timings,
    format_library_grid,
)
from astrorag.chat.library   import ContextLibrary, LibraryEntry, get_library
from astrorag.chat.qa        import PaperQA
from astrorag.logger         import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# main widget
# ══════════════════════════════════════════════════════════

class AstroRAGChatbot:
    """
    Full interactive AstroRAG UI.

    Instantiate in a Jupyter notebook and call `.show()` to display
    the tabbed interface.

    Usage:
        from astrorag.chat.widget import AstroRAGChatbot
        bot = AstroRAGChatbot(corpus=corpus)
        bot.show()
    """

    def __init__(
        self,
        corpus,
        library:  ContextLibrary | None = None,
    ) -> None:
        if not HAS_IPYWIDGETS:
            raise ImportError(
                "ipywidgets is required. Install with: pip install ipywidgets"
            )

        # lazy import to avoid loading heavy stages at import time
        from astrorag.stages import (
            Stage0Decompose, Stage1BM25, Stage2Graph,
            Stage3Rerank, Stage4PDF, Stage5Summarise,
        )

        self.corpus  = corpus
        self.library = library or get_library()

        # instantiate stages once
        logger.info("Initialising stages...")
        self.stage0 = Stage0Decompose()
        self.stage1 = Stage1BM25(corpus=corpus)
        self.stage2 = Stage2Graph(corpus=corpus)
        self.stage3 = Stage3Rerank()
        self.stage4 = Stage4PDF()
        self.stage5 = Stage5Summarise(stage4=self.stage4)

        self.current_entry: LibraryEntry | None = None
        self.current_qa:    PaperQA      | None = None
        self.current_pdf_text: str = ""

        self._build_ui()

    # ══════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════

    def _build_ui(self) -> None:
        # ── Tab 1: Query ──────────────────────────────
        self.query_input = widgets.Textarea(
            value="",
            placeholder="Enter astrophysics research question here...",
            description="Query:",
            layout=widgets.Layout(width="98%", height="80px"),
        )
        self.run_button = widgets.Button(
            description="🚀 Run Pipeline",
            button_style="primary",
            layout=widgets.Layout(width="180px"),
        )
        self.run_button.on_click(self._on_run_pipeline)

        self.progress_output = widgets.Output(
            layout=widgets.Layout(border="1px solid #dee2e6",
                                  padding="10px",
                                  min_height="150px"),
        )

        query_tab = widgets.VBox([
            widgets.HTML("<h3>🔬 Ask AstroRAG</h3>"),
            self.query_input,
            self.run_button,
            widgets.HTML("<h4>Progress</h4>"),
            self.progress_output,
        ])

        # ── Tab 2: Results ────────────────────────────
        self.results_output = widgets.Output()
        results_tab = widgets.VBox([
            widgets.HTML("<h3>📊 Selected Paper Results</h3>"),
            self.results_output,
        ])

        # ── Tab 3: Q&A ────────────────────────────────
        self.qa_input = widgets.Text(
            value="",
            placeholder="Ask a follow-up question about the current paper...",
            layout=widgets.Layout(width="82%"),
        )
        self.qa_ask_button = widgets.Button(
            description="Ask",
            button_style="success",
            layout=widgets.Layout(width="80px"),
        )
        self.qa_reset_button = widgets.Button(
            description="🔄",
            tooltip="Reset conversation",
            layout=widgets.Layout(width="50px"),
        )
        self.qa_ask_button.on_click(self._on_ask_question)
        self.qa_reset_button.on_click(self._on_reset_qa)

        self.qa_conversation = widgets.Output(
            layout=widgets.Layout(border="1px solid #dee2e6",
                                  padding="10px",
                                  max_height="500px",
                                  overflow="auto"),
        )

        qa_tab = widgets.VBox([
            widgets.HTML("<h3>💬 Chat with Selected Paper</h3>"),
            widgets.HBox([self.qa_input, self.qa_ask_button, self.qa_reset_button]),
            self.qa_conversation,
        ])

        # ── Tab 4: Library ────────────────────────────
        self.library_search = widgets.Text(
            value="",
            placeholder="Filter by keyword, arxiv ID, or subdomain...",
            layout=widgets.Layout(width="70%"),
        )
        self.library_search.observe(self._on_library_search, names="value")

        self.library_refresh_button = widgets.Button(
            description="🔄 Refresh",
            layout=widgets.Layout(width="120px"),
        )
        self.library_refresh_button.on_click(lambda _: self._render_library())

        self.library_stats_output = widgets.Output()
        self.library_list_output  = widgets.Output()

        self.library_paper_selector = widgets.Dropdown(
            options=[("(select paper)", None)],
            description="Load:",
            layout=widgets.Layout(width="60%"),
        )
        self.library_load_button = widgets.Button(
            description="📂 Load into Results",
            button_style="info",
            layout=widgets.Layout(width="200px"),
        )
        self.library_load_button.on_click(self._on_load_from_library)

        library_tab = widgets.VBox([
            widgets.HTML("<h3>📚 Context Library</h3>"),
            self.library_stats_output,
            widgets.HBox([self.library_search, self.library_refresh_button]),
            self.library_list_output,
            widgets.HTML("<hr>"),
            widgets.HBox([self.library_paper_selector, self.library_load_button]),
        ])

        # ── main tabs ─────────────────────────────────
        self.tabs = widgets.Tab()
        self.tabs.children = [query_tab, results_tab, qa_tab, library_tab]
        self.tabs.set_title(0, "🔬 Query")
        self.tabs.set_title(1, "📊 Results")
        self.tabs.set_title(2, "💬 Chat")
        self.tabs.set_title(3, "📚 Library")

        # init library display
        self._render_library()

    # ══════════════════════════════════════════════════
    # display entry point
    # ══════════════════════════════════════════════════

    def show(self) -> None:
        """Display the widget."""
        display(widgets.HTML("""
<div style="background:linear-gradient(90deg, #1e3a5f, #2c5282);
            color:white; padding:15px; border-radius:8px; margin-bottom:15px;">
  <h1 style="margin:0;">🌌 AstroRAG</h1>
  <p style="margin:5px 0 0 0; opacity:0.9;">
    Evidence-Aware Retrieval over 408,590 arXiv Astrophysics Papers
  </p>
</div>
"""))
        display(self.tabs)

    # ══════════════════════════════════════════════════
    # pipeline execution
    # ══════════════════════════════════════════════════

    def _on_run_pipeline(self, _btn) -> None:
        query = self.query_input.value.strip()
        if not query:
            with self.progress_output:
                clear_output()
                print("⚠ Please enter a query.")
            return

        with self.progress_output:
            clear_output()
            print("🚀 Starting pipeline...\n")

            try:
                # ── Stage 0 ─────────────────────────
                print("Stage 0: Decomposing query...")
                s0 = self.stage0.run(query)
                print(f"  Q1: {s0.decomposition.sub_questions['Q1'][:80]}")
                print(f"  Q2: {s0.decomposition.sub_questions['Q2'][:80]}")
                print(f"  Q3: {s0.decomposition.sub_questions['Q3'][:80]}\n")

                # ── Stage 1 ─────────────────────────
                print("Stage 1: BM25 retrieval...")
                s1 = self.stage1.run(query, top_k=50)
                print(f"  Retrieved {len(s1.results)} candidates "
                      f"(top BM25={s1.top_score:.2f})\n")

                # ── Stage 2 ─────────────────────────
                print("Stage 2: Graph + PPR...")
                s2 = self.stage2.run(s1)
                print(f"  Graph: {s2.n_nodes} nodes, "
                      f"density={s2.signals.density:.1%}\n")

                # ── Stage 3 ─────────────────────────
                print("Stage 3: LLM reranking...")
                s3 = self.stage3.run(
                    retrieval     = s1,
                    graph_context = s2,
                    decomposition = s0.decomposition,
                )
                print(f"  Selected: {s3.selected_result.arxiv_id} "
                      f"(BM25 rank #{s3.selected_result.rank}, "
                      f"conf={s3.confidence:.2f})\n")

                # ── Stage 4 ─────────────────────────
                print("Stage 4: PDF fetch and parsing...")
                s4 = self.stage4.run(s3)
                if not s4.success:
                    print(f"  ⚠ PDF failed: {s4.error}")
                    # try fallback
                    for pool_idx in list(s3.fallback_pool):
                        next_paper = s1.results[pool_idx]
                        print(f"  Trying fallback: {next_paper.arxiv_id}...")
                        s4 = self.stage4.run(next_paper)
                        if s4.success:
                            s3.selected_result = next_paper
                            s3.fallback_pool.remove(pool_idx)
                            break
                if not s4.success:
                    print("  ✗ All fallbacks exhausted.")
                    return
                print(f"  ✓ {s4.n_pages} pages, {s4.n_chars_total:,} chars, "
                      f"{len(s4.sections)} sections\n")

                # ── Stage 5 ─────────────────────────
                print("Stage 5: Summarisation with quality gate...")
                s5 = self.stage5.run(
                    decomposition = s0.decomposition,
                    retrieval     = s1,
                    stage3_result = s3,
                    initial_pdf   = s4,
                )
                print(f"  Q_total={s5.quality.scores.Q_total:.3f} "
                      f"→ {s5.quality.decision.value}")
                print(f"  Attempts={s5.n_attempts}, "
                      f"Time={s5.total_time_s:.1f}s\n")

                # ── save to library ─────────────────
                entry = self.library.add_from_stage5(query, s5)
                self.current_entry     = entry
                self.current_pdf_text  = s5.pdf_doc.full_text
                self.current_qa        = PaperQA(
                    entry      = entry,
                    paper_text = self.current_pdf_text,
                )

                print(f"✅ Pipeline complete. Saved to library as "
                      f"arXiv:{entry.arxiv_id}")
                print(f"   Switch to '📊 Results' tab to view.")

            except Exception as e:
                print(f"\n✗ Pipeline error: {type(e).__name__}: {e}")
                logger.error(f"Pipeline failed: {e}", exc_info=True)
                return

        # populate results and refresh library
        self._render_results()
        self._render_qa_intro()
        self._render_library()

        # switch to results tab
        self.tabs.selected_index = 1

    # ══════════════════════════════════════════════════
    # results rendering
    # ══════════════════════════════════════════════════

    def _render_results(self) -> None:
        with self.results_output:
            clear_output()
            if self.current_entry is None:
                print("No paper loaded. Run a query first.")
                return

            e = self.current_entry

            # Header
            display(HTML(f"""
<div style="background:#f8f9fa; padding:15px; border-radius:8px;
            border-left:4px solid #1e3a5f; margin-bottom:15px;">
  <h2 style="margin:0;">arXiv:{e.arxiv_id}</h2>
  <p style="margin:5px 0 0 0; color:#495057;">
    <em>Query: {e.original_query}</em>
  </p>
  <p style="margin:5px 0 0 0;">
    📄 {e.pdf_pages} pages &nbsp;•&nbsp;
    📚 {e.n_sections} sections &nbsp;•&nbsp;
    ⏱ {e.total_seconds:.1f}s
  </p>
</div>
"""))

            # Quality gate
            display(HTML(format_quality_scores(e)))

            # Summary
            display(Markdown(format_summary_markdown(e)))

            # Equations
            display(HTML("<h2>📐 Extracted Equations</h2>"))
            display(HTML(format_equations_table(e)))

            # Numerical results
            display(HTML("<h2>🔢 Numerical Results</h2>"))
            display(HTML(format_numerical_results_table(e)))

    # ══════════════════════════════════════════════════
    # Q&A
    # ══════════════════════════════════════════════════

    def _render_qa_intro(self) -> None:
        with self.qa_conversation:
            clear_output()
            if self.current_qa is None:
                display(HTML("<em>No paper loaded. Run a query first.</em>"))
                return
            display(HTML(f"""
<div style="background:#e3f2fd; padding:10px; border-radius:6px;
            margin-bottom:10px;">
  <strong>💬 Chat with arXiv:{self.current_entry.arxiv_id}</strong>
  <div style="font-size:12px; color:#495057; margin-top:4px;">
    Ask follow-up questions about this paper. The LLM has access to the
    paper's summary, extracted equations, numerical results, and truncated
    full text.
  </div>
</div>
"""))

    def _on_ask_question(self, _btn) -> None:
        if self.current_qa is None:
            with self.qa_conversation:
                clear_output()
                display(HTML(
                    "<em>No paper loaded. Run a query first.</em>"
                ))
            return

        question = self.qa_input.value.strip()
        if not question:
            return

        # clear the input
        self.qa_input.value = ""

        # get answer
        answer = self.current_qa.ask(question)

        # render conversation
        with self.qa_conversation:
            clear_output()
            display(HTML(f"""
<div style="background:#e3f2fd; padding:10px; border-radius:6px;
            margin-bottom:10px;">
  <strong>💬 arXiv:{self.current_entry.arxiv_id}</strong>
</div>
"""))
            for m in self.current_qa.session.messages:
                if m.role == "user":
                    display(HTML(f"""
<div style="background:#e7f3ff; padding:10px; border-radius:6px;
            margin:8px 0; border-left:3px solid #1e3a5f;">
  <strong>You:</strong><br>{m.content}
</div>
"""))
                else:
                    display(HTML(f"""
<div style="background:#f8f9fa; padding:10px; border-radius:6px;
            margin:8px 0; border-left:3px solid #28a745;">
  <strong>🤖 AstroRAG:</strong><br>{m.content.replace(chr(10), '<br>')}
</div>
"""))

        # save session to library
        self.library.add_chat_session(
            self.current_entry.arxiv_id,
            self.current_qa.session,
        )

    def _on_reset_qa(self, _btn) -> None:
        if self.current_qa is not None:
            self.current_qa.reset()
        self._render_qa_intro()

    # ══════════════════════════════════════════════════
    # library rendering
    # ══════════════════════════════════════════════════

    def _render_library(self) -> None:
        entries = self.library.list_all()
        stats   = self.library.stats()

        with self.library_stats_output:
            clear_output()
            if stats.get("n_entries", 0) == 0:
                display(HTML(
                    "<p><em>Library is empty. Run a query to populate.</em></p>"
                ))
            else:
                subs = ", ".join(stats.get("subdomains", [])) or "n/a"
                display(HTML(f"""
<div style="background:#f0f8ff; padding:10px; border-radius:6px;
            margin-bottom:10px;">
  <strong>📊 Library Statistics</strong><br>
  <span style="font-size:13px;">
    {stats['n_entries']} papers  •
    {stats['n_with_equations']} with equations  •
    {stats['n_with_numerical']} with numerical results  •
    {stats['total_chat_sessions']} chat sessions  •
    Mean Q_total: {stats['mean_q_total']:.3f}
  </span>
</div>
"""))

        # populate dropdown
        options = [("(select paper)", None)]
        for e in sorted(entries, key=lambda x: x.updated_at, reverse=True):
            label = f"{e.arxiv_id} — {e.title[:60] or e.original_query[:60]}"
            options.append((label, e.arxiv_id))
        self.library_paper_selector.options = options

        # render grid
        self._render_library_list(entries)

    def _render_library_list(self, entries: list[LibraryEntry]) -> None:
        with self.library_list_output:
            clear_output()
            display(HTML(format_library_grid(entries)))

    def _on_library_search(self, change) -> None:
        keyword = change["new"]
        entries = self.library.search(keyword)
        self._render_library_list(entries)

    def _on_load_from_library(self, _btn) -> None:
        arxiv_id = self.library_paper_selector.value
        if not arxiv_id:
            return
        entry = self.library.get(arxiv_id)
        if entry is None:
            return

        self.current_entry    = entry
        self.current_pdf_text = ""    # would need to re-fetch PDF
        self.current_qa       = PaperQA(entry=entry)

        self._render_results()
        self._render_qa_intro()

        # switch to results tab
        self.tabs.selected_index = 1