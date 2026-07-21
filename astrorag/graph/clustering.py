"""
K-means cluster summary of the top-K candidate papers.

The cluster summary is passed to Stage 3 as the first element of the
LLM prompt, giving the model field-level context before it reads
individual abstracts.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from   sklearn.cluster import KMeans

from astrorag.config       import Settings, get_settings
from astrorag.data         import CorpusData
from astrorag.graph.models import ClusterInfo, ClusterSummary
from astrorag.logger       import get_logger
from astrorag.retrieval    import RetrievalResult

logger = get_logger(__name__)


def kmeans_cluster(
    arxiv_ids: list[str],
    corpus:    CorpusData,
    n_clusters: int | None = None,
    settings:  Settings | None = None,
) -> tuple[np.ndarray, KMeans]:
    """
    Run K-means clustering over paper concept embeddings.

    Args:
        arxiv_ids:  List of K candidate arxiv_ids.
        corpus:     Loaded CorpusData for embedding lookup.
        n_clusters: Number of clusters (default from settings).
        settings:   Configuration.

    Returns:
        Tuple of:
            labels  (K,) cluster assignments
            km      fitted KMeans object
    """
    settings   = settings or get_settings()
    n_clusters = n_clusters or settings.n_clusters
    n          = len(arxiv_ids)
    k          = min(n_clusters, n)

    dim  = corpus.concept_emb.shape[1]
    vecs = np.zeros((n, dim), dtype=np.float32)
    for i, aid in enumerate(arxiv_ids):
        vecs[i] = corpus.get_paper_vector(aid)

    km = KMeans(
        n_clusters   = k,
        random_state = 42,
        n_init       = 10,
    )
    labels = km.fit_predict(vecs)
    return labels, km


def build_cluster_summary(
    results:     list[RetrievalResult],
    ppr_scores:  np.ndarray,
    corpus:      CorpusData,
    settings:    Settings | None = None,
) -> ClusterSummary:
    """
    Build the ClusterSummary passed to Stage 3.

    Steps:
      1. Run K-means over paper concept embeddings
      2. For each cluster:
           - Identify hub (highest PPR)
           - Extract top-4 most frequent concepts
           - Identify member domains
      3. Format as prompt text for LLM consumption

    Args:
        results:    Stage 1 retrieval results.
        ppr_scores: PPR scores per paper (from Stage 2).
        corpus:     Loaded CorpusData.
        settings:   Configuration.

    Returns:
        ClusterSummary object with prompt-ready text.
    """
    settings  = settings or get_settings()
    arxiv_ids = [r.arxiv_id for r in results]
    n         = len(results)

    labels, km = kmeans_cluster(arxiv_ids, corpus, settings=settings)
    k          = int(labels.max() + 1)

    clusters: list[ClusterInfo] = []
    for cid in range(k):
        member_mask = labels == cid
        member_idxs = np.where(member_mask)[0].tolist()
        if not member_idxs:
            continue

        # hub = member with highest PPR
        hub_idx = int(max(member_idxs, key=lambda i: ppr_scores[i]))

        # collect concepts and domains
        all_concepts: list[str] = []
        domains_seen: set[str]  = set()
        for i in member_idxs:
            all_concepts.extend(results[i].concepts[:10])
            aid = results[i].arxiv_id
            domain_classes = corpus.paper_to_classes.get(aid, [])
            if domain_classes:
                domains_seen.add(domain_classes[0])

        top_concepts = [
            c for c, _ in Counter(all_concepts).most_common(4)
        ]

        hub_result = results[hub_idx]
        clusters.append(ClusterInfo(
            cluster_id   = cid,
            n_papers     = len(member_idxs),
            member_idxs  = member_idxs,
            top_concepts = top_concepts,
            domains      = sorted(domains_seen),
            hub_idx      = hub_idx,
            hub_arxiv_id = hub_result.arxiv_id,
            hub_ppr      = float(ppr_scores[hub_idx]),
            hub_title    = (hub_result.title or hub_result.abstract[:60])[:60],
        ))

    prompt_text = _format_cluster_prompt(clusters)

    return ClusterSummary(
        n_clusters     = len(clusters),
        clusters       = clusters,
        cluster_labels = labels.astype(np.int32),
        prompt_text    = prompt_text,
    )


def _format_cluster_prompt(clusters: list[ClusterInfo]) -> str:
    """Render clusters as human-readable text for the Stage 3 LLM."""
    lines = ["CONTEXT GRAPH CLUSTER SUMMARY:"]
    for c in clusters:
        lines.extend([
            f"",
            f"Cluster {c.cluster_id}  [{c.n_papers} papers]",
            f"  Concepts : {' · '.join(c.top_concepts) or '(none extracted)'}",
            f"  Domain   : {' / '.join(c.domains) or 'Unknown'}",
            f"  Hub      : #{c.hub_idx}  PPR={c.hub_ppr:.2f}",
            f"  Title    : {c.hub_title}",
        ])
    return "\n".join(lines)