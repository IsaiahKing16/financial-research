"""
wfa_reranker.py — DTW-based reranker for HNSW top-K candidates (SLE-73).

Applied as a *post-retrieval* reranking step on the HNSW top-K candidates.
Euclidean distance treats the feature vector as an unordered point cloud;
DTW (Dynamic Time Warping) accounts for temporal warping — it finds the
optimal alignment between two sequences, tolerating slight shifts in timing
that Euclidean distance would penalise.

Pipeline position:
    HNSW kneighbors → post_filter → (WFA rerank, optional) → Stage 5

Design:
    - Pure NumPy — no fastdtw/tslearn dependency.
    - Constrained DTW with Sakoe-Chiba band (window=2 by default) for O(N·W)
      vs the naive O(N²).
    - Feature flag: `getattr(config, 'use_wfa_rerank', False)` — False by
      default so baseline behavior is always restored when flag is off.
    - Reranking does NOT change which candidates are included or excluded;
      it only reorders the top_k survivors so Stage 5 picks the most
      temporally-aligned ones first.

Linear: SLE-73
"""

from __future__ import annotations

import math

import numpy as np

# ─── Constrained DTW ──────────────────────────────────────────────────────────

def _dtw_distance(a: np.ndarray, b: np.ndarray, window: int = 2) -> float:
    """Compute constrained DTW distance between two 1-D sequences.

    Uses the Sakoe-Chiba band constraint (|i - j| <= window) to keep
    computation linear in practice for short sequences (D <= 8 features).

    Args:
        a:      (D,) float64 sequence.
        b:      (D,) float64 sequence.
        window: Maximum allowed warp shift (default 2 → ±2 positions).

    Returns:
        DTW distance (float, always >= 0).
    """
    n, m = len(a), len(b)
    w = max(window, abs(n - m))   # must cover the length difference
    inf = float("inf")
    # DP table
    dtw = np.full((n + 1, m + 1), inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        for j in range(j_lo, j_hi + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dtw[i, j] = cost + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1],
            )
    return math.sqrt(dtw[n, m]) if dtw[n, m] < inf else inf


def _dtw_distance_batch(
    query: np.ndarray,
    candidates: np.ndarray,
    window: int = 2,
) -> np.ndarray:
    """Compute DTW distance from one query to each candidate row.

    Args:
        query:      (D,) float64 query feature vector.
        candidates: (K, D) float64 candidate feature matrix.
        window:     Sakoe-Chiba band width.

    Returns:
        (K,) float64 DTW distances.
    """
    K = len(candidates)
    dists = np.empty(K, dtype=np.float64)
    for k in range(K):
        dists[k] = _dtw_distance(query, candidates[k], window=window)
    return dists


# ─── WFAReranker ──────────────────────────────────────────────────────────────

class WFAReranker:
    """Rerank HNSW top-K candidates by DTW distance after post-filtering.

    The reranker operates on the *surviving* candidates (those that passed
    the distance / ticker / sector / regime / SAX filters) and returns a new
    ordering that places the most temporally-aligned candidates first.

    The primary Euclidean ordering is replaced with the DTW ordering for
    Stage 5 probability/return estimation.  Since Stage 5 uses the top_k
    mask (not raw ordering), only the *relative ranking* within the surviving
    set matters here — re-sorted candidates don't affect the count.

    Args:
        window:  Sakoe-Chiba band width (default 2).  Larger values allow
                 more warping but increase compute time O(D × window).
    """

    def __init__(self, window: int = 2) -> None:
        if window < 0:
            raise ValueError(f"window must be >= 0, got {window}")
        self.window = window
        self._train_features: np.ndarray | None = None

    def fit(self, X_train_scaled: np.ndarray) -> WFAReranker:
        """Cache the scaled training feature matrix for reranking.

        Args:
            X_train_scaled: (N, D) scaled training features (same as index).

        Returns:
            self (for chaining).
        """
        self._train_features = X_train_scaled
        return self

    def rerank(
        self,
        query_vec: np.ndarray,
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        """Return candidate_indices reordered by ascending DTW distance.

        Args:
            query_vec:         (D,) scaled query feature vector.
            candidate_indices: (K,) int training row indices (survivors only).

        Returns:
            (K,) int — same indices reordered closest-DTW-first.
        """
        if self._train_features is None:
            raise RuntimeError("WFAReranker.fit() must be called before rerank()")
        if len(candidate_indices) == 0:
            return candidate_indices

        cand_feats = self._train_features[candidate_indices]   # (K, D)
        dtw_dists = _dtw_distance_batch(query_vec, cand_feats, window=self.window)
        order = np.argsort(dtw_dists)
        return candidate_indices[order]

    def rerank_mask(
        self,
        X_query_batch: np.ndarray,        # (B, D)
        candidate_indices_b: np.ndarray,  # (B, n_probe) — original HNSW indices
        top_mask: np.ndarray,             # (B, n_probe) bool — surviving candidates
        top_k: int,
    ) -> np.ndarray:
        """Rerank survivors within the top_mask and return updated mask.

        For each query row b:
          1. Extract surviving candidate indices (where top_mask[b] is True).
          2. Reorder by DTW distance.
          3. Re-build a mask keeping only the first top_k by DTW order.

        Args:
            X_query_batch:        (B, D) scaled query features.
            candidate_indices_b:  (B, n_probe) training row indices.
            top_mask:             (B, n_probe) bool — pre-filter survivors.
            top_k:                Maximum candidates to keep.

        Returns:
            (B, n_probe) bool — reranked mask (same shape; same survivors,
            re-ordered so early positions hold DTW-nearest candidates).
        """
        if self._train_features is None:
            raise RuntimeError("WFAReranker.fit() must be called before rerank_mask()")

        B, n_probe = top_mask.shape
        new_mask = np.zeros_like(top_mask)

        for b in range(B):
            surviving = np.where(top_mask[b])[0]   # positions in [0, n_probe)
            if len(surviving) == 0:
                continue
            surviving_train_idx = candidate_indices_b[b, surviving]
            reranked_train_idx = self.rerank(X_query_batch[b], surviving_train_idx)

            # Map reranked training indices back to positions in the probe array
            # Build reverse lookup: train_idx → probe_position
            idx_to_pos = {ti: pos for pos, ti in zip(surviving, surviving_train_idx)}
            reranked_positions = np.array(
                [idx_to_pos[ti] for ti in reranked_train_idx], dtype=np.int64
            )
            # Keep at most top_k survivors in DTW order
            keep = reranked_positions[:top_k]
            new_mask[b, keep] = True

        return new_mask
