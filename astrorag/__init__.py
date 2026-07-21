"""AstroRAG — Evidence-Aware Astrophysics Literature Retrieval."""

__version__     = "1.0.0"
__author__      = "Surinder Singh Chhabra"
__email__       = "schhabra@sdsu.edu"
__license__     = "MIT"
__description__ = (
    "Evidence-Aware Astrophysics Literature Retrieval — "
    "Graph-Augmented RAG over 408K arXiv papers"
)

# ── foundation ────────────────────────────────────────────
from astrorag.config import Settings, get_settings
from astrorag.logger import get_logger, setup_logging
from astrorag.paths  import ProjectPaths, get_paths

# ── data loading ──────────────────────────────────────────
from astrorag.data import (
    CorpusData,
    DataLoader,
    LoadConfig,
    load_corpus,
)

# ── LLM client ────────────────────────────────────────────
from astrorag.llm import (
    LLMClient,
    get_llm_client,
    QueryDecomposition,
    RerankDecision,
    StructuredSummary,
)

# ── retrieval ─────────────────────────────────────────────
from astrorag.retrieval import (
    BM25Index,
    RetrievalResult,
    RetrievalRun,
    build_bm25_index,
    load_bm25_index,
    tokenize,
    normalize_arxiv_id,
)

# ── graph construction ────────────────────────────────────
from astrorag.graph import (
    GraphContext,
    SignalMatrices,
    ClusterSummary,
    ClusterInfo,
    compute_signal_matrices,
    personalized_pagerank,
    build_cluster_summary,
)

# ── PDF handling ──────────────────────────────────────────
from astrorag.pdf import (
    PDFDocument,
    Section,
    fetch_arxiv_pdf,
    extract_text_with_fallback,
    split_by_sections,
)

# ── extraction ────────────────────────────────────────────
from astrorag.extraction import (
    extract_equations,
    extract_measurements,
    extract_tables,
    detect_question_type,
    build_technical_context,
    assess_quality,
    QualityAssessment,
    QualityDecision,
    QualityScores,
)

# ── pipeline stages ───────────────────────────────────────
from astrorag.stages import (
    Stage0Decompose,
    Stage1BM25,
    Stage2Graph,
    Stage3Rerank,
    Stage3Result,
    Stage4PDF,
    Stage5Summarise,
    Stage5Result,
    decompose_query,
)

__all__ = [
    # meta
    "__version__",

    # foundation
    "Settings", "get_settings",
    "get_logger", "setup_logging",
    "ProjectPaths", "get_paths",

    # data
    "CorpusData", "DataLoader", "LoadConfig", "load_corpus",

    # LLM
    "LLMClient", "get_llm_client",
    "QueryDecomposition", "RerankDecision", "StructuredSummary",

    # retrieval
    "BM25Index", "RetrievalResult", "RetrievalRun",
    "build_bm25_index", "load_bm25_index",
    "tokenize", "normalize_arxiv_id",

    # graph
    "GraphContext", "SignalMatrices", "ClusterSummary", "ClusterInfo",
    "compute_signal_matrices", "personalized_pagerank",
    "build_cluster_summary",

    # PDF
    "PDFDocument", "Section",
    "fetch_arxiv_pdf", "extract_text_with_fallback", "split_by_sections",

    # extraction
    "extract_equations", "extract_measurements", "extract_tables",
    "detect_question_type", "build_technical_context",
    "assess_quality", "QualityAssessment", "QualityDecision", "QualityScores",

    # pipeline stages
    "Stage0Decompose",
    "Stage1BM25",
    "Stage2Graph",
    "Stage3Rerank", "Stage3Result",
    "Stage4PDF",
    "Stage5Summarise", "Stage5Result",
    "decompose_query",
    # chat
    "ContextLibrary", "LibraryEntry", "get_library",
    "ChatMessage", "ChatSession", "PaperQA",
]
from astrorag.evaluation import (
    EvaluationRunner,
    EvaluationResult,
    EvaluationMetrics,
    compute_metrics,
    format_metrics_table,
    get_query_set,
    DEFAULT_QUERY_SET,
    # ablation
    AblationVariant,
    ABLATION_VARIANTS,
    AblationRunner,
    VariantComparison,
    compute_variant_comparison,
    format_ablation_table,
    load_all_variants,
    get_variant,
    get_all_variant_names,
)
from astrorag.chat import (
    ContextLibrary,
    LibraryEntry,
    get_library,
    ChatMessage,
    ChatSession,
    PaperQA,
)