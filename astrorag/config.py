"""
Configuration management for AstroRAG.

Uses pydantic-settings to load configuration from environment
variables (via .env file) with validation and type checking.

All pipeline parameters are centralised here. Individual stages
import Settings and read their configuration from it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib   import Path

from pydantic          import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from astrorag.paths import get_paths


class Settings(BaseSettings):
    """
    AstroRAG settings loaded from environment variables.

    Environment variables prefixed with ASTRORAG_ override defaults.
    A .env file at the project root is loaded automatically.
    """

    model_config = SettingsConfigDict(
        env_file            = ".env",
        env_file_encoding   = "utf-8",
        env_prefix          = "ASTRORAG_",
        extra               = "ignore",
        case_sensitive      = False,
    )

    # ══════════════════════════════════════════════════════
    # LLM API keys — required for pipeline operation
    # ══════════════════════════════════════════════════════
    groq_api_key: str = Field(
        default = "",
        description = "Groq API key for LLaMA-3.1-8B inference",
        alias   = "GROQ_API_KEY",
    )
    gemini_api_key: str = Field(
        default = "",
        description = "Google Gemini API key (optional alternative)",
        alias   = "GEMINI_API_KEY",
    )

    # ══════════════════════════════════════════════════════
    # dataset paths
    # ══════════════════════════════════════════════════════
    data_dir: str = Field(
        default = "~/noteboom/astro-ph_knowledge_graph",
        description = "Absolute path to arXiv astro-ph dataset directory",
    )

    # ══════════════════════════════════════════════════════
    # pipeline scale
    # ══════════════════════════════════════════════════════
    sample_size: int = Field(
        default = 408_590,
        ge      = 1,
        le      = 500_000,
        description = "Number of papers to index (use full 408590 for final runs)",
    )

    top_k: int = Field(
        default = 50,
        ge      = 5,
        le      = 200,
        description = "BM25 top-K candidates per query",
    )

    # ══════════════════════════════════════════════════════
    # BM25 parameters
    # ══════════════════════════════════════════════════════
    bm25_k1: float = Field(
        default = 1.5,
        ge      = 0.5,
        le      = 3.0,
        description = "BM25 term frequency saturation parameter",
    )
    bm25_b: float = Field(
        default = 0.75,
        ge      = 0.0,
        le      = 1.0,
        description = "BM25 length normalisation parameter",
    )

    # ══════════════════════════════════════════════════════
    # Graph parameters (Stage 2)
    # ══════════════════════════════════════════════════════
    edge_threshold: float = Field(
        default = 0.25,
        ge      = 0.0,
        le      = 1.0,
        description = "Threshold below which edge weights are set to zero",
    )
    n_clusters: int = Field(
        default = 3,
        ge      = 2,
        le      = 10,
        description = "Number of K-means clusters for graph summary",
    )
    w_s1_concept: float = Field(
        default = 0.35,
        ge      = 0.0,
        le      = 1.0,
        description = "Signal 1 weight — concept embedding cosine similarity",
    )
    w_s2_biblio: float = Field(
        default = 0.30,
        ge      = 0.0,
        le      = 1.0,
        description = "Signal 2 weight — bibliographic coupling (Jaccard)",
    )
    w_s3_cocitation: float = Field(
        default = 0.20,
        ge      = 0.0,
        le      = 1.0,
        description = "Signal 3 weight — co-citation strength",
    )
    w_s4_domain: float = Field(
        default = 0.15,
        ge      = 0.0,
        le      = 1.0,
        description = "Signal 4 weight — domain hierarchy match",
    )

    # ══════════════════════════════════════════════════════
    # PageRank parameters
    # ══════════════════════════════════════════════════════
    ppr_alpha: float = Field(
        default = 0.85,
        ge      = 0.0,
        le      = 1.0,
        description = "PageRank damping factor (probability of following edge)",
    )
    ppr_max_iter: int = Field(
        default = 200,
        ge      = 50,
        le      = 1000,
        description = "Maximum PageRank iterations before forced convergence",
    )
    ppr_tol: float = Field(
        default = 1e-6,
        gt      = 0.0,
        description = "PageRank convergence tolerance (L1 norm)",
    )

    # ══════════════════════════════════════════════════════
    # Quality gate (Stage 5)
    # ══════════════════════════════════════════════════════
    q_accept_threshold: float = Field(
        default = 0.75,
        ge      = 0.0,
        le      = 1.0,
        description = "Q_total >= this → ACCEPT",
    )
    q_retry_threshold: float = Field(
        default = 0.50,
        ge      = 0.0,
        le      = 1.0,
        description = "Q_total >= this → RETRY, below → RE-SELECT",
    )
    q_weight_faithfulness: float = Field(
        default = 0.40,
        ge      = 0.0,
        le      = 1.0,
        description = "Weight for Q_f in Q_total",
    )
    q_weight_coverage: float = Field(
        default = 0.35,
        ge      = 0.0,
        le      = 1.0,
        description = "Weight for Q_c in Q_total",
    )
    q_weight_consistency: float = Field(
        default = 0.25,
        ge      = 0.0,
        le      = 1.0,
        description = "Weight for Q_i in Q_total",
    )
    max_reselect_attempts: int = Field(
        default = 5,
        ge      = 1,
        le      = 20,
        description = "Maximum re-selection attempts before forced acceptance",
    )

    # ══════════════════════════════════════════════════════
    # LLM parameters
    # ══════════════════════════════════════════════════════
    groq_model: str = Field(
        default = "llama-3.1-8b-instant",
        description = "Groq model identifier for all LLM calls",
    )
    groq_temperature: float = Field(
        default = 0.0,
        ge      = 0.0,
        le      = 2.0,
        description = "LLM sampling temperature (0.0 = deterministic)",
    )
    groq_max_tokens_stage3: int = Field(
        default = 300,
        ge      = 100,
        le      = 4000,
        description = "Max output tokens for Stage 3 reranking",
    )
    groq_max_tokens_stage5: int = Field(
        default = 2000,
        ge      = 500,
        le      = 8000,
        description = "Max output tokens for Stage 5 summarisation",
    )
    groq_max_tokens_stage6: int = Field(
        default = 3000,
        ge      = 500,
        le      = 8000,
        description = "Max output tokens for Stage 6 context library",
    )
    groq_timeout_seconds: int = Field(
        default = 60,
        ge      = 10,
        le      = 300,
        description = "Timeout for individual LLM API calls",
    )
    groq_max_retries: int = Field(
        default = 3,
        ge      = 0,
        le      = 10,
        description = "Retries on transient LLM API failures",
    )

    # ══════════════════════════════════════════════════════
    # Logging
    # ══════════════════════════════════════════════════════
    log_level: str = Field(
        default = "INFO",
        description = "Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    log_file: str = Field(
        default = "logs/astrorag.log",
        description = "Path to log file relative to project root",
    )

    # ══════════════════════════════════════════════════════
    # validators
    # ══════════════════════════════════════════════════════
    @field_validator("data_dir")
    @classmethod
    def _expand_data_dir(cls, v: str) -> str:
        """Expand ~ to absolute home path."""
        return str(Path(v).expanduser().resolve())

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Ensure log level is a valid Python logging level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(
                f"log_level must be one of {valid}, got '{v}'"
            )
        return upper

    # ══════════════════════════════════════════════════════
    # derived properties
    # ══════════════════════════════════════════════════════
    @property
    def data_path(self) -> Path:
        """Absolute Path to dataset directory."""
        return Path(self.data_dir)

    @property
    def dataset_files(self) -> dict[str, Path]:
        """Return expected dataset files with their absolute paths."""
        d = self.data_path
        return {
            "abstracts":       d / "abstracts_all.jsonl.gz",
            "concept_emb":     d / "concepts_embeddings.npz",
            "paper_concepts":  d / "papers_concepts_mapping.csv.gz",
            "vocabulary":      d / "concepts_vocabulary.csv.gz",
            "citations":       d / "citations_indexed.jsonl.gz",
            "index_mapping":   d / "papers_index_mapping.csv.gz",
            "years":           d / "papers_years.npy",
            "identifier_map":  d / "identifier_mapping_arxiv.csv.gz",
        }

    @property
    def signal_weights_sum_valid(self) -> bool:
        """Check that S1 + S2 + S3 + S4 weights sum to approximately 1.0."""
        total = (
            self.w_s1_concept
            + self.w_s2_biblio
            + self.w_s3_cocitation
            + self.w_s4_domain
        )
        return abs(total - 1.0) < 0.01

    @property
    def quality_weights_sum_valid(self) -> bool:
        """Check that Q_f + Q_c + Q_i weights sum to approximately 1.0."""
        total = (
            self.q_weight_faithfulness
            + self.q_weight_coverage
            + self.q_weight_consistency
        )
        return abs(total - 1.0) < 0.01

    def summary(self) -> str:
        """Return a formatted summary of current settings."""
        paths = get_paths()
        return f"""
AstroRAG Configuration Summary
{'═' * 60}
  Sample size      : {self.sample_size:,} papers
  BM25 top-K       : {self.top_k}
  Data directory   : {self.data_path}
  PDF directory    : {paths.pdf_dir}
  Results          : {paths.results_dir}
  Logs             : {paths.logs_dir}

  Signal weights   : S1={self.w_s1_concept:.2f}  S2={self.w_s2_biblio:.2f}  S3={self.w_s3_cocitation:.2f}  S4={self.w_s4_domain:.2f}
  Weights valid    : {self.signal_weights_sum_valid}

  PPR alpha        : {self.ppr_alpha}
  Edge threshold   : {self.edge_threshold}

  Quality accept   : Q ≥ {self.q_accept_threshold}
  Quality retry    : Q ≥ {self.q_retry_threshold}
  Quality weights  : Qf={self.q_weight_faithfulness:.2f}  Qc={self.q_weight_coverage:.2f}  Qi={self.q_weight_consistency:.2f}
  Weights valid    : {self.quality_weights_sum_valid}

  LLM model        : {self.groq_model}
  LLM temperature  : {self.groq_temperature}
  API key set      : {'Yes' if self.groq_api_key else 'NO — MISSING'}
{'═' * 60}
"""


# ── singleton accessor ────────────────────────────────────
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Uses lru_cache to ensure a single instance is created and shared
    across all modules. Call this instead of instantiating Settings
    directly.
    """
    return Settings()