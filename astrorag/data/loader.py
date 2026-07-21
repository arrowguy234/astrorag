"""
Main data loader for AstroRAG.

Schema (from the actual dataset):
  papers_index_mapping.csv.gz    paper_idx (int)  ↔ arxiv_id (str)
  abstracts_all.jsonl.gz         {arxiv_id, abstract}
  concepts_vocabulary.csv.gz     label (int), class (str), concept (str), description
  papers_concepts_mapping.csv.gz arxiv_id (str), label (int)
  concepts_embeddings.npz        shape (9999, 3072) — indexed by label
  citations_indexed.jsonl.gz     {paper_idx, references[int], citations[int], ...}
  papers_years.npy               array of length 408,590 (year per paper_idx)

The loader normalises everything to arxiv_id-keyed lookups since
that is the primary identifier used by downstream stages.
"""

from __future__ import annotations

import sys
import time
from   dataclasses import dataclass, field
from   pathlib     import Path

import numpy as np
import pandas as pd
from   tqdm.auto import tqdm

from astrorag.config          import Settings, get_settings
from astrorag.data.cache      import CacheManager, get_cache_manager
from astrorag.data.models     import CorpusStats, LoadConfig
from astrorag.data.streaming  import iter_abstracts, iter_citations
from astrorag.logger          import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# CorpusData container
# ══════════════════════════════════════════════════════════

@dataclass
class CorpusData:
    """
    Container for all loaded corpus data.

    Every downstream stage receives a CorpusData instance and accesses
    fields directly rather than re-parsing raw files.
    """

    # ── raw records ─────────────────────────────────────
    papers:      list[dict]                        # each has arxiv_id, paper_idx, abstract, title
    concepts_df: pd.DataFrame                      # label, class, concept, description
    pc_mapping:  pd.DataFrame                      # arxiv_id, label

    # ── numerical arrays ────────────────────────────────
    concept_emb: np.ndarray                        # (N_labels, embedding_dim)

    # ── column name conventions (kept for backward compat) ─
    pid_col: str = "arxiv_id"                      # papers keyed by arxiv_id
    cid_col: str = "label"                         # concept label int
    lbl_col: str = "concept"                       # human-readable concept name

    # ── derived lookups ─────────────────────────────────
    arxiv_to_labels:   dict[str, list[int]]  = field(default_factory=dict)  # concept label ints
    arxiv_to_cidx:     dict[str, list[int]]  = field(default_factory=dict)  # alias for stage 2
    paper_to_concepts: dict[str, list[str]]  = field(default_factory=dict)  # concept names
    paper_to_classes:  dict[str, list[str]]  = field(default_factory=dict)  # domain classes
    paper_idx_to_arxiv: dict[int, str]       = field(default_factory=dict)
    arxiv_to_paper_idx: dict[str, int]       = field(default_factory=dict)
    paper_refs:        dict[str, set[str]]   = field(default_factory=dict)
    paper_cited_by:    dict[str, set[str]]   = field(default_factory=dict)

    # ── metadata ────────────────────────────────────────
    stats:        CorpusStats = field(default_factory=CorpusStats)
    load_config:  LoadConfig  = field(default_factory=LoadConfig)

    # ── convenience accessors ───────────────────────────
    def get_paper(self, index: int) -> dict:
        return self.papers[index]

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> dict | None:
        aid = str(arxiv_id).strip()
        for p in self.papers:
            if str(p.get("arxiv_id", "")).strip() == aid:
                return p
        return None

    def get_paper_vector(self, arxiv_id: str) -> np.ndarray:
        """
        Mean concept embedding vector for a paper.
        Concept embeddings are indexed by label (int).
        """
        labels = self.arxiv_to_labels.get(str(arxiv_id).strip(), [])
        if not labels:
            return np.zeros(self.concept_emb.shape[1], dtype=np.float32)
        valid = [l for l in labels if 0 <= l < self.concept_emb.shape[0]]
        if not valid:
            return np.zeros(self.concept_emb.shape[1], dtype=np.float32)
        return self.concept_emb[valid].mean(axis=0)

    def has_concept_data(self, arxiv_id: str) -> bool:
        return str(arxiv_id).strip() in self.arxiv_to_labels

    def has_citation_data(self, arxiv_id: str) -> bool:
        return str(arxiv_id).strip() in self.paper_refs

    def n_papers(self) -> int:
        return len(self.papers)


# ══════════════════════════════════════════════════════════
# main loader class
# ══════════════════════════════════════════════════════════

class DataLoader:
    """
    Loads the full AstroRAG corpus.
    """

    def __init__(
        self,
        settings: Settings   | None = None,
        config:   LoadConfig | None = None,
        cache:    CacheManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config   = config   or LoadConfig(
            sample_size = self.settings.sample_size,
        )
        self.cache    = cache    or get_cache_manager()

    # ── high-level entry ────────────────────────────────
    def load(self) -> CorpusData:
        t0 = time.time()

        cache_key = self.cache.make_key(
            sample_size  = self.config.sample_size,
            source_paths = self.settings.dataset_files,
        )

        if self.config.use_cache and not self.config.force_reload:
            cached = self.cache.load(cache_key)
            if cached is not None:
                data = self._corpus_from_cache(cached)
                data.stats.load_time_seconds = time.time() - t0
                logger.info(
                    f"Corpus ready from cache: {data.n_papers():,} papers "
                    f"in {data.stats.load_time_seconds:.1f}s"
                )
                return data

        logger.info(
            f"Loading corpus fresh — sample size "
            f"{self.config.sample_size:,} papers"
        )
        data = self._load_from_source()
        data.stats.load_time_seconds = time.time() - t0

        if self.config.use_cache:
            self.cache.save(cache_key, self._corpus_to_cache(data))

        logger.info(
            f"Corpus loaded: {data.n_papers():,} papers "
            f"in {data.stats.load_time_seconds:.1f}s"
        )
        return data

    # ══════════════════════════════════════════════════
    # individual load steps
    # ══════════════════════════════════════════════════

    def _load_from_source(self) -> CorpusData:
        files = self.settings.dataset_files

        # 1. paper index mapping (arxiv_id ↔ paper_idx)
        pidx_to_aid, aid_to_pidx = self._load_paper_index(
            files["index_mapping"]
        )

        # 2. abstracts
        papers = self._load_papers(
            files["abstracts"], aid_to_pidx
        )

        # 3. embeddings (indexed by concept label)
        emb = self._load_embeddings(files["concept_emb"])

        # 4. vocabulary
        vocab_df = self._load_vocabulary(files["vocabulary"])

        # 5. paper-concept mapping (arxiv_id → label)
        pc_df = self._load_paper_concepts(files["paper_concepts"])

        # 6. concept lookups
        (arxiv_to_labels,
         paper_to_concepts,
         paper_to_classes) = self._build_concept_lookups(
            pc_df    = pc_df,
            vocab_df = vocab_df,
            n_emb    = emb.shape[0],
        )

        # 7. citations
        arxiv_set = {str(p["arxiv_id"]) for p in papers}
        refs, cited_by = self._load_citations(
            files["citations"],
            arxiv_set    = arxiv_set,
            pidx_to_aid  = pidx_to_aid,
        )

        # 8. stats
        stats = self._compute_stats(
            papers            = papers,
            concept_emb       = emb,
            vocab_df          = vocab_df,
            arxiv_to_labels   = arxiv_to_labels,
            paper_to_concepts = paper_to_concepts,
            paper_refs        = refs,
            paper_cited_by    = cited_by,
        )

        return CorpusData(
            papers             = papers,
            concepts_df        = vocab_df,
            pc_mapping         = pc_df,
            concept_emb        = emb,
            pid_col            = "arxiv_id",
            cid_col            = "label",
            lbl_col            = "concept",
            arxiv_to_labels    = arxiv_to_labels,
            arxiv_to_cidx      = arxiv_to_labels,   # backward-compat alias
            paper_to_concepts  = paper_to_concepts,
            paper_to_classes   = paper_to_classes,
            paper_idx_to_arxiv = pidx_to_aid,
            arxiv_to_paper_idx = aid_to_pidx,
            paper_refs         = refs,
            paper_cited_by     = cited_by,
            stats              = stats,
            load_config        = self.config,
        )

    # ── step 1: paper index mapping ─────────────────────
    def _load_paper_index(
        self, path: Path
    ) -> tuple[dict[int, str], dict[str, int]]:
        logger.info(f"Loading paper index mapping from {path.name}...")
        df = pd.read_csv(
            path,
            dtype = {"paper_idx": "int64", "arxiv_id": "string"},
            low_memory = False,
        )
        df["arxiv_id"] = df["arxiv_id"].astype(str).str.strip()

        pidx_to_aid: dict[int, str] = dict(
            zip(df["paper_idx"].values, df["arxiv_id"].values)
        )
        aid_to_pidx: dict[str, int] = dict(
            zip(df["arxiv_id"].values, df["paper_idx"].values)
        )
        logger.info(f"  Index mapping: {len(pidx_to_aid):,} papers")
        return pidx_to_aid, aid_to_pidx

    # ── step 2: abstracts ───────────────────────────────
    def _load_papers(
        self,
        path:        Path,
        aid_to_pidx: dict[str, int],
    ) -> list[dict]:
        logger.info(f"Loading abstracts from {path.name}...")
        papers: list[dict] = []
        for rec in iter_abstracts(
            path          = path,
            limit         = self.config.sample_size,
            show_progress = self.config.show_progress,
        ):
            aid       = str(rec.get("arxiv_id", "")).strip()
            paper_idx = aid_to_pidx.get(aid, len(papers))
            papers.append({
                "arxiv_id":  aid,
                "paper_idx": paper_idx,
                "abstract":  str(rec.get("abstract", "")),
                "title":     str(rec.get("title", "")),
            })
        logger.info(f"  Loaded {len(papers):,} papers")
        return papers

    # ── step 3: embeddings ──────────────────────────────
    def _load_embeddings(self, path: Path) -> np.ndarray:
        logger.info(f"Loading concept embeddings from {path.name}...")
        arr = np.load(path)
        key = arr.files[0]
        emb = arr[key].astype(np.float32)
        logger.info(f"  Shape: {emb.shape} dtype: {emb.dtype}")
        return emb

    # ── step 4: vocabulary ──────────────────────────────
    def _load_vocabulary(self, path: Path) -> pd.DataFrame:
        logger.info(f"Loading concept vocabulary from {path.name}...")
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]

        required = {"label", "concept"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Vocabulary missing required columns {missing}. "
                f"Found: {list(df.columns)}"
            )
        if "class" not in df.columns:
            df["class"] = "Unknown"
        if "description" not in df.columns:
            df["description"] = ""

        logger.info(
            f"  {len(df):,} concepts | domains: "
            f"{df['class'].nunique()}"
        )
        return df

    # ── step 5: paper-concept mapping ───────────────────
    def _load_paper_concepts(self, path: Path) -> pd.DataFrame:
        logger.info(f"Loading paper-concept mapping from {path.name}...")
        df = pd.read_csv(
            path,
            dtype = {"arxiv_id": "string", "label": "int64"},
            low_memory = False,
        )
        df.columns = [c.lower().strip() for c in df.columns]

        required = {"arxiv_id", "label"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"paper_concepts missing required columns {missing}. "
                f"Found: {list(df.columns)}"
            )

        df["arxiv_id"] = df["arxiv_id"].astype(str).str.strip()

        logger.info(
            f"  {len(df):,} paper-concept edges | "
            f"unique papers: {df['arxiv_id'].nunique():,} | "
            f"unique labels: {df['label'].nunique():,}"
        )
        return df

    # ── step 6: build concept lookups ───────────────────
    def _build_concept_lookups(
        self,
        pc_df:    pd.DataFrame,
        vocab_df: pd.DataFrame,
        n_emb:    int,
    ) -> tuple[dict[str, list[int]],
               dict[str, list[str]],
               dict[str, list[str]]]:
        """
        Build:
            arxiv_to_labels   arxiv_id → list of concept label ints
            paper_to_concepts arxiv_id → list of concept names
            paper_to_classes  arxiv_id → list of domain class names
        """
        logger.info("Building concept lookups...")

        # ── build vocab lookup tables ────────────────────
        vocab_map_name = dict(zip(
            vocab_df["label"].astype(int).values,
            vocab_df["concept"].astype(str).values,
        ))
        vocab_map_class = dict(zip(
            vocab_df["label"].astype(int).values,
            vocab_df["class"].astype(str).values,
        ))
        logger.info(f"  Vocabulary index size: {len(vocab_map_name):,}")

        arxiv_to_labels:   dict[str, list[int]] = {}
        paper_to_concepts: dict[str, list[str]] = {}
        paper_to_classes:  dict[str, list[str]] = {}

        arxiv_arr = pc_df["arxiv_id"].values
        label_arr = pc_df["label"].values.astype(np.int64)

        iterator = tqdm(
            range(len(pc_df)),
            desc    = "Concept lookups",
            unit    = "edges",
            disable = not self.config.show_progress,
        )
        for i in iterator:
            aid   = str(arxiv_arr[i]).strip()
            label = int(label_arr[i])

            if 0 <= label < n_emb:
                arxiv_to_labels.setdefault(aid, []).append(label)

            name = vocab_map_name.get(label)
            if name is not None:
                paper_to_concepts.setdefault(aid, []).append(name)

            klass = vocab_map_class.get(label)
            if klass is not None:
                paper_to_classes.setdefault(aid, []).append(klass)

        logger.info(
            f"  arxiv_to_labels  : "
            f"{len(arxiv_to_labels):,} papers with concept labels"
        )
        logger.info(
            f"  paper_to_concepts: "
            f"{len(paper_to_concepts):,} papers with concept names"
        )
        return arxiv_to_labels, paper_to_concepts, paper_to_classes

    # ── step 7: citations ───────────────────────────────
    def _load_citations(
        self,
        path:        Path,
        arxiv_set:   set[str],
        pidx_to_aid: dict[int, str],
    ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """
        Load references and citations. References/citations lists in
        the source data are integer paper_idx values — we convert
        them to arxiv_ids using pidx_to_aid.
        """
        logger.info(f"Loading citations from {path.name}...")

        refs:     dict[str, set[str]] = {}
        cited_by: dict[str, set[str]] = {}

        for rec in iter_citations(
            path          = path,
            keep_ids      = None,   # will filter by arxiv_id below
            show_progress = self.config.show_progress,
        ):
            paper_idx = rec.get("paper_idx")
            if paper_idx is None:
                continue
            aid = pidx_to_aid.get(int(paper_idx))
            if aid is None or aid not in arxiv_set:
                continue

            # convert reference paper_idxs → arxiv_ids
            ref_aids: set[str] = set()
            for ridx in rec.get("references", []):
                try:
                    r_aid = pidx_to_aid.get(int(ridx))
                    if r_aid:
                        ref_aids.add(r_aid)
                except (TypeError, ValueError):
                    continue
            refs[aid] = ref_aids

            # note: field is called "citations" in this dataset schema
            cite_aids: set[str] = set()
            for cidx in rec.get("citations", rec.get("cited_by", [])):
                try:
                    c_aid = pidx_to_aid.get(int(cidx))
                    if c_aid:
                        cite_aids.add(c_aid)
                except (TypeError, ValueError):
                    continue
            cited_by[aid] = cite_aids

        logger.info(f"  {len(refs):,} papers with citation data")
        return refs, cited_by

    # ── step 8: stats ───────────────────────────────────
    def _compute_stats(
        self,
        papers:            list[dict],
        concept_emb:       np.ndarray,
        vocab_df:          pd.DataFrame,
        arxiv_to_labels:   dict[str, list[int]],
        paper_to_concepts: dict[str, list[str]],
        paper_refs:        dict[str, set[str]],
        paper_cited_by:    dict[str, set[str]],
    ) -> CorpusStats:
        def obj_mb(o) -> float:
            return sys.getsizeof(o) / (1024 * 1024)

        n_pc_edges = sum(len(v) for v in arxiv_to_labels.values())
        n_refs     = sum(len(v) for v in paper_refs.values())
        n_citers   = sum(len(v) for v in paper_cited_by.values())

        n_papers = len(papers)
        avg_c    = n_pc_edges / max(n_papers, 1)
        avg_r    = n_refs     / max(len(paper_refs), 1)
        avg_ci   = n_citers   / max(len(paper_cited_by), 1)

        mem_mb = (
            obj_mb(papers)
            + concept_emb.nbytes / (1024 * 1024)
            + obj_mb(arxiv_to_labels)
            + obj_mb(paper_to_concepts)
            + obj_mb(paper_refs)
            + obj_mb(paper_cited_by)
        )

        return CorpusStats(
            n_papers                  = n_papers,
            n_concepts                = len(vocab_df),
            n_papers_with_concepts    = len(arxiv_to_labels),
            n_papers_with_citations   = len(paper_refs),
            concept_emb_dim           = concept_emb.shape[1],
            total_paper_concept_edges = n_pc_edges,
            avg_concepts_per_paper    = avg_c,
            avg_refs_per_paper        = avg_r,
            avg_citers_per_paper      = avg_ci,
            memory_usage_mb           = mem_mb,
        )

    # ══════════════════════════════════════════════════
    # cache serialisation
    # ══════════════════════════════════════════════════

    def _corpus_to_cache(self, data: CorpusData) -> dict:
        return {
            "papers":             data.papers,
            "concepts_df":        data.concepts_df,
            "pc_mapping":         data.pc_mapping,
            "concept_emb":        data.concept_emb,
            "pid_col":            data.pid_col,
            "cid_col":            data.cid_col,
            "lbl_col":            data.lbl_col,
            "arxiv_to_labels":    data.arxiv_to_labels,
            "paper_to_concepts":  data.paper_to_concepts,
            "paper_to_classes":   data.paper_to_classes,
            "paper_idx_to_arxiv": data.paper_idx_to_arxiv,
            "arxiv_to_paper_idx": data.arxiv_to_paper_idx,
            "paper_refs":         data.paper_refs,
            "paper_cited_by":     data.paper_cited_by,
            "stats":              data.stats.model_dump(),
        }

    def _corpus_from_cache(self, cache: dict) -> CorpusData:
        return CorpusData(
            papers             = cache["papers"],
            concepts_df        = cache["concepts_df"],
            pc_mapping         = cache["pc_mapping"],
            concept_emb        = cache["concept_emb"],
            pid_col            = cache["pid_col"],
            cid_col            = cache["cid_col"],
            lbl_col            = cache["lbl_col"],
            arxiv_to_labels    = cache["arxiv_to_labels"],
            arxiv_to_cidx      = cache["arxiv_to_labels"],
            paper_to_concepts  = cache["paper_to_concepts"],
            paper_to_classes   = cache["paper_to_classes"],
            paper_idx_to_arxiv = cache["paper_idx_to_arxiv"],
            arxiv_to_paper_idx = cache["arxiv_to_paper_idx"],
            paper_refs         = cache["paper_refs"],
            paper_cited_by     = cache["paper_cited_by"],
            stats              = CorpusStats(**cache["stats"]),
            load_config        = self.config,
        )


# ══════════════════════════════════════════════════════════
# convenience one-shot function
# ══════════════════════════════════════════════════════════

def load_corpus(
    sample_size:  int  | None = None,
    use_cache:    bool = True,
    force_reload: bool = False,
) -> CorpusData:
    settings = get_settings()
    config   = LoadConfig(
        sample_size  = sample_size or settings.sample_size,
        use_cache    = use_cache,
        force_reload = force_reload,
    )
    return DataLoader(settings=settings, config=config).load()