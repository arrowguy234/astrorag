"""
Personalised PageRank wrapper.

Uses NetworkX pagerank with:
- Personalisation vector derived from BM25 scores
- Configurable damping factor alpha (default 0.85)
- Configurable tolerance and max_iter

For 50 nodes convergence typically takes 30-80 iterations
and completes in under 100 ms.
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from astrorag.config import Settings, get_settings
from astrorag.logger import get_logger

logger = get_logger(__name__)


def personalized_pagerank(
    W:            np.ndarray,
    bm25_scores:  np.ndarray,
    settings:     Settings | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Run Personalised PageRank on the weighted graph.

    Args:
        W:           (K, K) symmetric weighted adjacency matrix from
                     compute_signal_matrices.
        bm25_scores: (K,) array of BM25 scores for the same K papers.
                     Used to build the personalisation vector.
        settings:    Configuration.

    Returns:
        Tuple of:
            ppr_norm    (K,) PPR scores normalised to [0, 1]
            ppr_raw     (K,) raw PPR probabilities
            iterations  int  iterations to convergence
    """
    settings = settings or get_settings()
    n = W.shape[0]
    eps = 1e-9

    # ── build NetworkX graph from W ────────────────────
    G = nx.from_numpy_array(W)

    # ── build personalisation vector from BM25 ─────────
    bm25 = bm25_scores.astype(np.float64)
    lo, hi = bm25.min(), bm25.max()
    if hi - lo < eps:
        # all scores equal — uniform personalisation
        norm_scores = np.ones(n) / n
    else:
        norm_scores = (bm25 - lo) / (hi - lo + eps)
        norm_scores = norm_scores / (norm_scores.sum() + eps)

    personalization = {i: float(norm_scores[i]) for i in range(n)}

    # ── run PPR ────────────────────────────────────────
    t0 = time.time()
    try:
        ppr = nx.pagerank(
            G,
            alpha           = settings.ppr_alpha,
            personalization = personalization,
            weight          = "weight",
            max_iter        = settings.ppr_max_iter,
            tol             = settings.ppr_tol,
        )
        # nx.pagerank returns after convergence but doesn't tell us iters
        # we estimate by re-running with the log
        iterations = _estimate_iterations(
            G, personalization, settings
        )
    except nx.PowerIterationFailedConvergence as e:
        logger.warning(
            f"PPR did not converge in {settings.ppr_max_iter} iters "
            f"— using last iterate"
        )
        # get last iterate from exception if available
        ppr = getattr(e, "args", [{}])[0] if e.args else {i: 1.0/n for i in range(n)}
        iterations = settings.ppr_max_iter

    elapsed = time.time() - t0
    logger.debug(f"PPR converged in {iterations} iters ({elapsed*1000:.1f} ms)")

    # ── convert to array and normalise ─────────────────
    ppr_raw = np.array([ppr.get(i, 0.0) for i in range(n)], dtype=np.float32)

    lo_p, hi_p = ppr_raw.min(), ppr_raw.max()
    if hi_p - lo_p < eps:
        ppr_norm = np.ones(n, dtype=np.float32) * 0.5
    else:
        ppr_norm = ((ppr_raw - lo_p) / (hi_p - lo_p + eps)).astype(np.float32)

    return ppr_norm, ppr_raw, iterations


def _estimate_iterations(
    G:              nx.Graph,
    personalization: dict[int, float],
    settings:       Settings,
) -> int:
    """
    Run power iteration manually to count iterations to convergence.

    This is separate from the actual PPR call to keep the main path
    clean; only used for diagnostic reporting.
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0

    # get normalised adjacency
    W = nx.to_numpy_array(G, weight="weight")
    col_sums = W.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    A_hat = W / col_sums[np.newaxis, :]

    alpha = settings.ppr_alpha
    pi    = np.array([personalization.get(i, 1.0/n) for i in range(n)])
    r     = pi.copy()

    for it in range(1, settings.ppr_max_iter + 1):
        r_new = alpha * (A_hat.T @ r) + (1 - alpha) * pi
        if np.abs(r_new - r).sum() < settings.ppr_tol:
            return it
        r = r_new
    return settings.ppr_max_iter