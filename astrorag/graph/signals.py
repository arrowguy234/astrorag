"""
Four-signal matrix computation for the AstroRAG context graph.

S1 — concept embedding cosine similarity      w=0.35
S2 — bibliographic coupling (Jaccard)          w=0.30
S3 — co-citation strength (geom mean norm)     w=0.20
S4 — domain hierarchy match                    w=0.15

Each signal captures a different dimension of scientific relevance;
their weighted combination produces the graph edge weights.
"""

from __future__ import annotations

import math

import numpy as np
from   sklearn.metrics.pairwise import cosine_similarity

from astrorag.config    import Settings, get_settings
from astrorag.data      import CorpusData
from astrorag.graph.models import SignalMatrices
from astrorag.logger    import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# Signal 1 — concept embedding cosine similarity
# ══════════════════════════════════════════════════════════

def signal_1_concept_similarity(
    arxiv_ids: list[str],
    corpus:    CorpusData,
) -> np.ndarray:
    """
    Compute pairwise concept embedding cosine similarity.

    For each paper i, we compute its mean concept embedding vector
    v_i = mean of concept embeddings assigned to paper i. Then
    S1[i,j] = cosine similarity between v_i and v_j.

    This captures semantic relatedness at the concept level,
    which is domain-aware because concept embeddings are trained
    on the astro-ph corpus itself.

    Args:
        arxiv_ids: List of arxiv_ids for the K candidate papers.
        corpus:    Loaded CorpusData for embedding lookup.

    Returns:
        (K, K) symmetric similarity matrix with zero diagonal.
    """
    n = len(arxiv_ids)
    dim = corpus.concept_emb.shape[1]

    # build (K, dim) matrix of paper vectors
    vecs = np.zeros((n, dim), dtype=np.float32)
    for i, aid in enumerate(arxiv_ids):
        vecs[i] = corpus.get_paper_vector(aid)

    # cosine similarity — handles zero vectors safely
    S1 = cosine_similarity(vecs).astype(np.float32)
    np.fill_diagonal(S1, 0.0)
    # clip small negatives from floating point to zero
    S1 = np.clip(S1, 0.0, 1.0)
    return S1


# ══════════════════════════════════════════════════════════
# Signal 2 — bibliographic coupling
# ══════════════════════════════════════════════════════════

def signal_2_bibliographic_coupling(
    arxiv_ids: list[str],
    corpus:    CorpusData,
) -> np.ndarray:
    """
    Compute Jaccard similarity of reference sets.

    S2[i,j] = |R_i ∩ R_j| / |R_i ∪ R_j|

    Where R_i is the set of papers cited by paper i. Two papers
    that cite the same foundational works are studying related
    phenomena, even if they never cite each other directly. This
    is Kessler's 1963 bibliographic coupling.

    Args:
        arxiv_ids: List of arxiv_ids for the K candidate papers.
        corpus:    Loaded CorpusData for citation lookup.

    Returns:
        (K, K) symmetric similarity matrix.
    """
    n = len(arxiv_ids)
    S2 = np.zeros((n, n), dtype=np.float32)

    # pre-fetch reference sets
    ref_sets: list[set[str]] = [
        corpus.paper_refs.get(aid, set()) for aid in arxiv_ids
    ]

    for i in range(n):
        R_i = ref_sets[i]
        if not R_i:
            continue
        for j in range(i + 1, n):
            R_j = ref_sets[j]
            if not R_j:
                continue
            union = R_i | R_j
            if not union:
                continue
            jaccard = len(R_i & R_j) / len(union)
            S2[i, j] = S2[j, i] = jaccard

    return S2


# ══════════════════════════════════════════════════════════
# Signal 3 — co-citation strength
# ══════════════════════════════════════════════════════════

def signal_3_cocitation(
    arxiv_ids: list[str],
    corpus:    CorpusData,
) -> np.ndarray:
    """
    Compute co-citation strength with geometric mean normalization.

    S3[i,j] = |C_i ∩ C_j| / sqrt(|C_i| * |C_j|)

    Where C_i is the set of papers that cite paper i. Two papers
    frequently cited together are treated as related by the
    community — this is Small's 1973 co-citation analysis.

    Geometric mean normalization prevents highly cited papers
    from dominating (a paper cited 10,000 times would otherwise
    show artificial high co-citation with every other paper).

    Args:
        arxiv_ids: List of arxiv_ids for the K candidate papers.
        corpus:    Loaded CorpusData for reverse-citation lookup.

    Returns:
        (K, K) symmetric similarity matrix.
    """
    n = len(arxiv_ids)
    S3 = np.zeros((n, n), dtype=np.float32)

    # pre-fetch citer sets
    citer_sets: list[set[str]] = [
        corpus.paper_cited_by.get(aid, set()) for aid in arxiv_ids
    ]
    sizes = [len(s) for s in citer_sets]

    for i in range(n):
        if sizes[i] == 0:
            continue
        for j in range(i + 1, n):
            if sizes[j] == 0:
                continue
            intersect = citer_sets[i] & citer_sets[j]
            if not intersect:
                continue
            denom = math.sqrt(sizes[i] * sizes[j])
            score = len(intersect) / denom
            S3[i, j] = S3[j, i] = score

    return S3


# ══════════════════════════════════════════════════════════
# Signal 4 — domain hierarchy match
# ══════════════════════════════════════════════════════════

def _get_paper_domain_class(
    arxiv_id: str,
    corpus:   CorpusData,
) -> str:
    """
    Return the majority-vote domain class for a paper.

    Uses paper_to_classes if available (per-paper class list),
    else derives from concepts_df lookup on the paper's concepts.
    """
    # try direct lookup first
    classes = corpus.paper_to_classes.get(arxiv_id, [])
    if not classes:
        return "Unknown"
    # majority vote
    from collections import Counter
    counter = Counter(classes)
    return counter.most_common(1)[0][0]


def signal_4_domain_hierarchy(
    arxiv_ids: list[str],
    corpus:    CorpusData,
) -> np.ndarray:
    """
    Compute domain hierarchy match signal.

    S4[i,j] = 1.0 if same class ("Cosmology & Nongalactic Physics"),
              0.0 otherwise.

    (The dataset has only "class" as a single-level hierarchy;
    if a subdomain field is added later, this can extend to
    the tri-level scheme 1.0/0.5/0.0.)

    Args:
        arxiv_ids: List of arxiv_ids for the K candidate papers.
        corpus:    Loaded CorpusData for domain lookup.

    Returns:
        (K, K) symmetric similarity matrix.
    """
    n = len(arxiv_ids)
    S4 = np.zeros((n, n), dtype=np.float32)

    domains = [_get_paper_domain_class(aid, corpus) for aid in arxiv_ids]

    for i in range(n):
        di = domains[i]
        if di == "Unknown":
            continue
        for j in range(i + 1, n):
            dj = domains[j]
            if dj == "Unknown":
                continue
            if di == dj:
                S4[i, j] = S4[j, i] = 1.0
            # else already 0

    return S4


# ══════════════════════════════════════════════════════════
# combine all four signals
# ══════════════════════════════════════════════════════════

def compute_signal_matrices(
    arxiv_ids: list[str],
    corpus:    CorpusData,
    settings:  Settings | None = None,
) -> SignalMatrices:
    """
    Compute all four signal matrices and their weighted combination.

    Steps:
      1. Compute S1, S2, S3, S4 independently
      2. Combine as W_raw = w1*S1 + w2*S2 + w3*S3 + w4*S4
      3. Normalise W_raw by its maximum
      4. Threshold: entries below `edge_threshold` set to zero
      5. Zero the diagonal

    Args:
        arxiv_ids: K candidate paper arxiv_ids.
        corpus:    Loaded CorpusData.
        settings:  Configuration (uses defaults if None).

    Returns:
        SignalMatrices with all four raw signals and combined W.
    """
    settings = settings or get_settings()
    n = len(arxiv_ids)
    if n < 2:
        raise ValueError(f"Need at least 2 papers, got {n}")

    logger.debug(f"Computing signal matrices for {n} papers")

    # ── compute all four signals ────────────────────────
    S1 = signal_1_concept_similarity(arxiv_ids, corpus)
    S2 = signal_2_bibliographic_coupling(arxiv_ids, corpus)
    S3 = signal_3_cocitation(arxiv_ids, corpus)
    S4 = signal_4_domain_hierarchy(arxiv_ids, corpus)

    # ── log per-signal density ──────────────────────────
    for name, S in (("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4)):
        mean_nonzero = (S > 0).sum() // 2
        logger.debug(
            f"  {name}: mean={S.mean():.3f} "
            f"max={S.max():.3f} nonzero_pairs={mean_nonzero}"
        )

    # ── weighted combination ────────────────────────────
    W_raw = (
        settings.w_s1_concept    * S1
        + settings.w_s2_biblio     * S2
        + settings.w_s3_cocitation * S3
        + settings.w_s4_domain     * S4
    ).astype(np.float32)

    # ── normalise by max ────────────────────────────────
    max_val = W_raw.max()
    if max_val > 0:
        W_norm = W_raw / max_val
    else:
        W_norm = W_raw

    # ── count edges before threshold ────────────────────
    n_before = int((W_norm > 0).sum()) // 2

    # ── threshold ───────────────────────────────────────
    W = W_norm.copy()
    W[W < settings.edge_threshold] = 0.0
    np.fill_diagonal(W, 0.0)

    n_after = int((W > 0).sum()) // 2

    logger.debug(
        f"  W_combined: {n_before} → {n_after} edges "
        f"after threshold {settings.edge_threshold}"
    )

    return SignalMatrices(
        S1_concept     = S1,
        S2_biblio      = S2,
        S3_cocitation  = S3,
        S4_domain      = S4,
        W_combined     = W,
        n_edges_before = n_before,
        n_edges_after  = n_after,
    )