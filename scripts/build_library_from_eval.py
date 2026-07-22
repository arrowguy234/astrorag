#!/usr/bin/env python
"""
Build a context_library.json from eval_full.json.

The evaluation traces contain per-stage outputs for each query.
We map those into the LibraryEntry format so the Streamlit UI has
real papers to display without rerunning the pipeline.

Note: the eval traces preserve aggregate quality metrics and counts,
but not the full LLM-generated answer text. For a richer library
with full sub-question answers, equations with variables, and
detailed numerical results, rerun the pipeline through the chat
interface which populates the library directly.
"""

from __future__ import annotations

import json
from   datetime import datetime
from   pathlib  import Path


def main():
    root      = Path(__file__).resolve().parent.parent
    eval_path = root / "results" / "eval_full.json"
    out_path  = root / "data"    / "context_library.json"

    if not eval_path.exists():
        raise SystemExit(f"eval_full.json not found at {eval_path}")

    with open(eval_path, encoding="utf-8") as fh:
        eval_data = json.load(fh)

    now = datetime.now().isoformat()
    entries: dict[str, dict] = {}

    for trace in eval_data.get("traces", []):
        if not trace.get("success"):
            continue
        s5 = trace.get("stage5") or {}
        s4 = trace.get("stage4") or {}
        s3 = trace.get("stage3") or {}
        s1 = trace.get("stage1") or {}

        arxiv_id = s5.get("final_arxiv_id") or s3.get("selected_arxiv_id", "")
        if not arxiv_id:
            continue

        n_eq   = s5.get("n_equations", 0)
        n_num  = s5.get("n_numerical_results", 0)
        n_inst = s5.get("n_instruments", 0)

        # Reconstruct placeholder sub-question answers from trace counts
        sub_q_answers = {
            qk: {
                "answered":    True,
                "answer_text": ("Detailed answer available in original pipeline "
                                "output. This library entry reconstructed from "
                                "benchmark trace metadata."),
                "section":     "unknown",
            }
            for qk in ["Q1", "Q2", "Q3"]
        }

        # placeholder equations, numerical, instruments (counts preserved)
        key_equations = [
            {"equation": f"[Equation {i+1} — see original paper]",
             "variables": ""}
            for i in range(n_eq)
        ]
        numerical_results = [
            {"quantity":    f"Numerical result {i+1}",
             "value":       "see paper",
             "uncertainty": "",
             "unit":        ""}
            for i in range(n_num)
        ]
        instruments = [f"Instrument {i+1}" for i in range(n_inst)]

        entry = {
            "arxiv_id":       arxiv_id,
            "title":          "",
            "abstract":       "",
            "original_query": trace.get("query", ""),
            "subdomain":      trace.get("subdomain", ""),

            "paper_overview": s5.get("paper_overview")
                              or ("Paper selected via full six-stage pipeline. "
                                  "Detailed overview available in original "
                                  "Stage 5 output; not preserved in benchmark trace."),
            "evidence_type":  s5.get("evidence_type", ""),
            "instruments":    instruments,
            "key_equations":  key_equations,
            "numerical_results": numerical_results,
            "sub_question_answers": sub_q_answers,
            "key_snippet":    s5.get("key_snippet", ""),
            "key_findings":   [],
            "methodology":    "",

            "q_total":  s5.get("q_total", 0),
            "q_f":      s5.get("q_f", 0),
            "q_c":      s5.get("q_c", 0),
            "q_i":      s5.get("q_i", 0),
            "decision": s5.get("decision", ""),

            "total_seconds": trace.get("total_seconds", 0),
            "pdf_pages":     s4.get("n_pages", 0),
            "n_sections":    s4.get("n_sections", 0),

            "added_at":      now,
            "updated_at":    now,
            "view_count":    0,
            "chat_sessions": [],
        }
        entries[arxiv_id] = entry

    payload = {
        "version":    1,
        "updated_at": now,
        "n_entries":  len(entries),
        "entries":    entries,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Built library with {len(entries)} entries → {out_path}")


if __name__ == "__main__":
    main()
