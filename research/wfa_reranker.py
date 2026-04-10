"""
research/wfa_reranker.py — DTW post-retrieval reranker for E3.

Reranks top-50 Euclidean neighbours by DTW distance on the 8 return
columns only (indices 0:8). Candlestick columns (8:23) are excluded —
bounded ratios have no meaningful temporal warping.

Usage:
    reranked_idx, reranked_dists = dtw_rerank(query_23d, nbrs_23d, indices, k=20)
"""
from __future__ import annotations
import numpy as np

RETURN_COLS_SLICE = slice(0, 8)  # Only the 8 return columns


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic DTW distance between two equal-length 1D sequences."""
    try:
        from dtw import dtw as dtw_fn
        result = dtw_fn(a, b)
        return float(result.distance)
    except ImportError:
        pass
    # Fallback: dtaidistance
    try:
        from dtaidistance import dtw as dtai_dtw
        return float(dtai_dtw.distance(a, b))
    except ImportError:
        pass
    # Last resort: Euclidean distance (no warping — will likely trigger Spearman fast-fail)
    return float(np.linalg.norm(a - b))


def dtw_rerank(
    query_23d: np.ndarray,
    neighbours_23d: np.ndarray,
    neighbour_indices: np.ndarray,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Rerank top-N neighbours by DTW on return columns.

    Args:
        query_23d:          (D,) full feature vector (D >= 8).
        neighbours_23d:     (N, D) top-N neighbour features.
        neighbour_indices:  (N,) original training indices.
        k:                  Number of neighbours to return.

    Returns:
        (reranked_indices, reranked_dtw_distances): both shape (k,).
    """
    query_ret = query_23d[RETURN_COLS_SLICE]         # (8,)
    nbr_ret   = neighbours_23d[:, RETURN_COLS_SLICE] # (N, 8)
    n = len(neighbour_indices)

    dtw_dists = np.array([
        _dtw_distance(query_ret, nbr_ret[i]) for i in range(n)
    ])

    order = np.argsort(dtw_dists)[:k]
    return neighbour_indices[order], dtw_dists[order]
