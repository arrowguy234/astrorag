"""
Rich output formatting for pipeline results.

Converts Stage 5 output and library entries into Markdown, HTML,
and structured data suitable for display in Jupyter notebooks.
"""

from __future__ import annotations

from typing import Any

from astrorag.chat.library import LibraryEntry


# ══════════════════════════════════════════════════════════
# summary as markdown
# ══════════════════════════════════════════════════════════

def format_summary_markdown(entry: LibraryEntry) -> str:
    """Format a library entry as rich Markdown."""

    lines = [
        f"# 📄 arXiv:{entry.arxiv_id}",
        "",
        f"**Query:** {entry.original_query}",
        "",
        "## Paper Overview",
        "",
        entry.paper_overview or "_No overview available_",
        "",
    ]

    # Sub-question answers
    if entry.sub_question_answers:
        lines.append("## Sub-Question Answers")
        lines.append("")
        for qk in ["Q1", "Q2", "Q3"]:
            if qk not in entry.sub_question_answers:
                continue
            ans = entry.sub_question_answers[qk]
            check = "✅" if ans.get("answered") else "❌"
            section = ans.get("section", "unknown")
            label = {
                "Q1": "Mechanism",
                "Q2": "Evidence",
                "Q3": "Quantitative",
            }.get(qk, qk)
            lines.append(f"### {check} {qk} — {label} _(Section: {section})_")
            lines.append("")
            lines.append(ans.get("answer_text", "_No answer_"))
            lines.append("")

    # Evidence / instruments
    if entry.evidence_type or entry.instruments:
        lines.append("## Study Context")
        lines.append("")
        if entry.evidence_type:
            lines.append(f"- **Evidence type:** {entry.evidence_type}")
        if entry.instruments:
            lines.append(f"- **Instruments:** {', '.join(entry.instruments)}")
        lines.append("")

    # Methodology
    if entry.methodology:
        lines.append("## Methodology")
        lines.append("")
        lines.append(entry.methodology)
        lines.append("")

    # Key findings
    if entry.key_findings:
        lines.append("## Key Findings")
        lines.append("")
        for f in entry.key_findings:
            lines.append(f"- {f}")
        lines.append("")

    # Key snippet
    if entry.key_snippet:
        lines.append("## Key Snippet")
        lines.append("")
        lines.append(f"> {entry.key_snippet}")
        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# equations table (HTML)
# ══════════════════════════════════════════════════════════

def format_equations_table(entry: LibraryEntry) -> str:
    """Format extracted equations as an HTML table."""
    if not entry.key_equations:
        return "<p><em>No equations extracted.</em></p>"

    rows = "\n".join(
        f"<tr><td><code>{eq.get('equation', '')}</code></td>"
        f"<td>{eq.get('variables', '')}</td></tr>"
        for eq in entry.key_equations
    )

    return f"""
<table style="width:100%; border-collapse:collapse; margin-top:10px;">
  <thead style="background:#1e3a5f; color:white;">
    <tr>
      <th style="padding:8px; text-align:left; width:45%;">Equation</th>
      <th style="padding:8px; text-align:left;">Variables</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
"""


# ══════════════════════════════════════════════════════════
# numerical results table (HTML)
# ══════════════════════════════════════════════════════════

def format_numerical_results_table(entry: LibraryEntry) -> str:
    """Format extracted numerical results as an HTML table."""
    if not entry.numerical_results:
        return "<p><em>No numerical results extracted.</em></p>"

    rows = "\n".join(
        f"<tr>"
        f"<td>{nr.get('quantity', '')}</td>"
        f"<td style='text-align:right;'><strong>{nr.get('value', '')}</strong></td>"
        f"<td style='text-align:right;'>{nr.get('uncertainty', '')}</td>"
        f"<td>{nr.get('unit', '')}</td>"
        f"</tr>"
        for nr in entry.numerical_results
    )

    return f"""
<table style="width:100%; border-collapse:collapse; margin-top:10px;">
  <thead style="background:#1e3a5f; color:white;">
    <tr>
      <th style="padding:8px; text-align:left;">Quantity</th>
      <th style="padding:8px; text-align:right;">Value</th>
      <th style="padding:8px; text-align:right;">Uncertainty</th>
      <th style="padding:8px; text-align:left;">Unit</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
"""


# ══════════════════════════════════════════════════════════
# quality scores panel (HTML)
# ══════════════════════════════════════════════════════════

def format_quality_scores(entry: LibraryEntry) -> str:
    """Format quality scores as a color-coded HTML panel."""

    decision_colors = {
        "ACCEPT":    "#28a745",
        "RETRY":     "#ffc107",
        "RE-SELECT": "#dc3545",
    }
    color = decision_colors.get(entry.decision, "#6c757d")

    def bar(value: float, label: str) -> str:
        pct = int(value * 100)
        bar_color = "#28a745" if value >= 0.75 else \
                    "#ffc107" if value >= 0.50 else "#dc3545"
        return f"""
<div style="margin-bottom:10px;">
  <div style="display:flex; justify-content:space-between;">
    <span><strong>{label}</strong></span>
    <span>{value:.3f}</span>
  </div>
  <div style="background:#e9ecef; border-radius:4px; height:12px;">
    <div style="width:{pct}%; height:12px; background:{bar_color};
                border-radius:4px;"></div>
  </div>
</div>
"""

    return f"""
<div style="background:#f8f9fa; padding:15px; border-radius:8px; border-left:4px solid {color};">
  <div style="display:flex; justify-content:space-between; margin-bottom:15px;">
    <h3 style="margin:0;">Quality Gate</h3>
    <span style="background:{color}; color:white; padding:4px 12px;
                 border-radius:12px; font-weight:bold;">
      {entry.decision}
    </span>
  </div>
  {bar(entry.q_f,     "Q_f — Faithfulness (0.40 weight)")}
  {bar(entry.q_c,     "Q_c — Coverage (0.35 weight)")}
  {bar(entry.q_i,     "Q_i — Consistency (0.25 weight)")}
  <hr>
  {bar(entry.q_total, "Q_total (composite)")}
</div>
"""


# ══════════════════════════════════════════════════════════
# stage timings (HTML)
# ══════════════════════════════════════════════════════════

def format_stage_timings(
    stage_timings: dict[str, float],
    total: float,
) -> str:
    """Format per-stage timings as an HTML table."""
    rows = "\n".join(
        f"<tr>"
        f"<td>{stage}</td>"
        f"<td style='text-align:right;'>{seconds:.2f}</td>"
        f"<td style='text-align:right;'>{seconds/total*100:.1f}%</td>"
        f"</tr>"
        for stage, seconds in stage_timings.items()
    )
    return f"""
<table style="width:100%; border-collapse:collapse; margin-top:10px;">
  <thead style="background:#495057; color:white;">
    <tr>
      <th style="padding:6px; text-align:left;">Stage</th>
      <th style="padding:6px; text-align:right;">Seconds</th>
      <th style="padding:6px; text-align:right;">% of total</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
  <tfoot style="background:#e9ecef;">
    <tr>
      <td><strong>Total</strong></td>
      <td style="text-align:right;"><strong>{total:.2f}</strong></td>
      <td style="text-align:right;"><strong>100.0%</strong></td>
    </tr>
  </tfoot>
</table>
"""


# ══════════════════════════════════════════════════════════
# library list (HTML card grid)
# ══════════════════════════════════════════════════════════

def format_library_grid(entries: list[LibraryEntry]) -> str:
    """Format library entries as an HTML card grid."""
    if not entries:
        return """
<div style="text-align:center; padding:40px; color:#6c757d;">
  <h3>Context Library Empty</h3>
  <p>Run a query to populate the library.</p>
</div>
"""

    cards = []
    for e in sorted(entries, key=lambda x: x.updated_at, reverse=True):
        decision_color = {
            "ACCEPT":    "#28a745",
            "RETRY":     "#ffc107",
            "RE-SELECT": "#dc3545",
        }.get(e.decision, "#6c757d")

        n_eq = len(e.key_equations)
        n_num = len(e.numerical_results)
        n_chat = len(e.chat_sessions)

        cards.append(f"""
<div style="border:1px solid #dee2e6; border-radius:8px; padding:12px;
            margin-bottom:10px; background:white;">
  <div style="display:flex; justify-content:space-between;
              align-items:start; margin-bottom:8px;">
    <div>
      <strong style="color:#1e3a5f;">arXiv:{e.arxiv_id}</strong>
      {f'<span style="color:#6c757d; margin-left:8px;">[{e.subdomain}]</span>' if e.subdomain else ''}
    </div>
    <span style="background:{decision_color}; color:white;
                 padding:2px 8px; border-radius:8px; font-size:11px;">
      Q={e.q_total:.2f} {e.decision}
    </span>
  </div>
  <div style="font-size:13px; color:#495057; margin-bottom:6px;">
    <em>{e.original_query[:100]}{"..." if len(e.original_query) > 100 else ""}</em>
  </div>
  <div style="font-size:12px; color:#6c757d;">
    📐 {n_eq} equations &nbsp;•&nbsp;
    🔢 {n_num} values &nbsp;•&nbsp;
    💬 {n_chat} chat sessions &nbsp;•&nbsp;
    👁 viewed {e.view_count}×
  </div>
</div>
""")

    return f"""
<div style="max-height:500px; overflow-y:auto; padding:5px;">
  {"".join(cards)}
</div>
"""