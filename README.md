# 🌌 AstroRAG

**Evidence-Aware Graph-Augmented Retrieval over 408,590 arXiv Astrophysics Papers**

A six-stage retrieval-augmented generation pipeline that combines BM25 lexical retrieval, a four-signal citation graph with Personalised PageRank, LLM-based reranking, column-aware PDF extraction, and quality-gated structured summarisation with iterative re-selection.

**Author:** Surinder Singh Chhabra
**Institution:** San Diego State University, Computational Science Research Center
**Contact:** schhabra@sdsu.edu

## 🎬 Live Demo

Interactive UI: [astrorag.streamlit.app](https://astrorag.streamlit.app) (link updates after Streamlit Cloud deploy)

## 📊 Benchmark Results

On a curated 20-query benchmark spanning 10 astrophysics subdomains:

| Metric | Value |
|--------|-------|
| Success rate | 100% (20/20) |
| Q_total mean | 0.981 |
| Papers with equations | 100% |
| Papers with numerical results | 100% |
| Median latency | 78.9s |

Six-variant ablation study confirms:
- PDF extraction essential: ΔQ_total = −0.657 without
- BM25-only baseline collapses to Q_total = 0.303
- LLM rerank changes selection in 70% of queries at only −0.030 quality cost

## 🏗 Architecture

```
Query
  ↓
Stage 0 — Query decomposition (Q1 / Q2 / Q3)
  ↓
Stage 1 — BM25 top-50 over 408,590 papers
  ↓
Stage 2 — Four-signal graph + Personalised PageRank
  ↓
Stage 3 — Graph-primed LLM reranking
  ↓
Stage 4 — arXiv PDF fetch + column-aware parsing
  ↓
Stage 5 — Structured summarisation + quality gate
  ↓
Structured summary with equations, values, instruments
```

## 📁 Repository Structure

```
astrorag/
├── astrorag/                # main package
│   ├── config.py            # Pydantic settings
│   ├── data/                # corpus loading
│   ├── llm/                 # Groq client wrapper
│   ├── retrieval/           # BM25 index
│   ├── graph/               # signals + PPR
│   ├── pdf/                 # column-aware extraction
│   ├── extraction/          # equations, measurements, quality
│   ├── stages/              # Stage 0-5 orchestration
│   ├── evaluation/          # benchmark + ablation
│   └── chat/                # context library + Q&A
├── tests/                   # unit + integration tests
├── scripts/                 # CLI entry points
├── paper/                   # IEEE LaTeX paper
├── results/                 # benchmark + ablation traces
└── notebooks/               # Jupyter interfaces
```

## ⚙️ Installation

Requires Python 3.11+. Dataset files (~450 MB compressed) must be downloaded separately.

```bash
git clone https://github.com/arrowguy234/astrorag.git
cd astrorag

# install
pip install -e ".[dev]"

# set your Groq API key
cp .env.example .env
# edit .env and add GROQ_API_KEY
```

## 🚀 Usage

### Run the full pipeline

```bash
python scripts/run_stage5.py --query "How do AGN jets suppress star formation?"
```

### Run the 20-query benchmark

```bash
python scripts/run_evaluation.py --output results/eval_full.json
python scripts/report_evaluation.py results/eval_full.json --traces
```

### Run the ablation study

```bash
python scripts/run_ablation.py --all --output-dir results/ablation_full
python scripts/report_ablation.py --dir results/ablation_full
```

## 📄 Paper

Full IEEE-format paper in `paper/astrorag.pdf`. Source: `paper/astrorag.tex`.

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 License

MIT License — see LICENSE.

## 🙏 Acknowledgments

Thanks to the SDSU Computational Science Research Center for compute infrastructure and dataset access.
