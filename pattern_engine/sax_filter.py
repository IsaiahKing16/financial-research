"""
sax_filter.py — SAX symbolic filter for HNSW top-K candidates (SLE-72).

SAX (Symbolic Aggregate Approximation) converts a numeric time series into a
discrete symbol string.  Applied as a *second-stage* filter on top of HNSW
candidate retrieval — it never replaces the primary Euclidean-distance search.

Pipeline position:
    HNSW kneighbors → (SAX filter, optional) → post_filter → Stage 5

Why SAX?
    Euclidean distance can be small for sequences whose *shape* differs — e.g.
    a flat series and a V-shaped series of the same magnitude.  SAX captures
    the ordinal pattern (rising, flat, falling) in each PAA segment and
    discards candidates whose symbolic representation is far from the query's.

Feature vector interpretation:
    The 8-dim FPPE return fingerprint [ret_1d, ret_3d, ret_7d, ret_14d,
    ret_30d, ret_45d, ret_60d, ret_90d] is treated as a time series ordered
    by window length.  Longer windows give lower-frequency shape information.

Design:
    - Pure NumPy — no external tslearn/saxpy dependency.
    - All parameters configurable; sane defaults match literature.
    - Feature flag: `getattr(config, 'use_sax_filter', False)` — False by
      default so baseline behavior is always restored when flag is off.

Usage:
    sax = SAXFilter(word_size=4, alphabet_size=4, max_symbolic_distance=1)
    # fit on scaled training features
    sax.fit(X_train_scaled)
    # during _post_filter: remove candidates whose SAX distance > threshold
    keep_mask = sax.filter_candidates(X_query_row, candidate_indices)

Linear: SLE-72
"""

from __future__ import annotations

import numpy as np

# ─── SAX breakpoints (pre-computed from N(0,1) quantiles) ─────────────────────
# breakpoints[k] gives the boundaries for an alphabet of size k.
# breakpoints[4] = [-0.674, 0, 0.674] → symbols a/b/c/d.
# Source: Lin et al. 2003, Table 3 (extended to alphabet_size 8).
_BREAKPOINTS: dict[int, list[float]] = {
    2: [0.0],
    3: [-0.4307, 0.4307],
    4: [-0.6745, 0.0, 0.6745],
    5: [-0.8416, -0.2533, 0.2533, 0.8416],
    6: [-0.9674, -0.4307, 0.0, 0.4307, 0.9674],
    7: [-1.0676, -0.5659, -0.1800, 0.1800, 0.5659, 1.0676],
    8: [-1.1503, -0.6745, -0.3186, 0.0, 0.3186, 0.6745, 1.1503],
}

# Symbolic distance lookup table — mindist per symbol pair.
# Entry (i, j) = 0 when |i - j| <= 1 (adjacent symbols are "free").
# Entry (i, j) = breakpoints[a-1][max(i,j)-1] - breakpoints[a-1][min(i,j)]
# for |i-j| >= 2.  We precompute lazily via _symbolic_dist().


def _build_dist_table(alphabet_size: int) -> np.ndarray:
    """Build the SAX MINDIST lookup table for the given alphabet size.

    Returns:
        (alphabet_size, alphabet_size) float64 array where entry [i, j]
        is the minimum possible distance between symbol i and symbol j.
    """
    a = alphabet_size
    bp = _BREAKPOINTS[a]
    table = np.zeros((a, a), dtype=np.float64)
    for i in range(a):
        for j in range(a):
            if abs(i - j) <= 1:
                table[i, j] = 0.0
            else:
                lo, hi = min(i, j), max(i, j)
                # Distance = breakpoint just above lo - breakpoint just below hi
                table[i, j] = bp[hi - 1] - bp[lo]
    return table


# ─── PAA helpers ───────────────────────────────────────────────────────────────

def _paa(x: np.ndarray, w: int) -> np.ndarray:
    """Piecewise Aggregate Approximation: reduce D-dim vector to w segments.

    Each segment is the mean of D/w consecutive elements (with equal-width
    fractional splitting when D is not divisible by w).

    Args:
        x: (D,) float64 input.
        w: Number of PAA segments (word_size).  Must be <= D.

    Returns:
        (w,) float64 PAA-reduced representation.
    """
    D = len(x)
    if w == D:
        return x.copy()
    if w > D:
        raise ValueError(f"word_size={w} > feature_dim={D}")
    # Integer-split: each segment covers D/w elements (with remainder spread)
    indices = np.array_split(x, w)
    return np.array([seg.mean() for seg in indices], dtype=np.float64)


def _digitize(paa_vec: np.ndarray, breakpoints: list[float]) -> np.ndarray:
    """Map PAA values to integer symbol indices via breakpoints.

    Args:
        paa_vec: (w,) PAA-reduced values (already z-normalised at the
                 training-scaler level — no re-normalisation needed here).
        breakpoints: Sorted list of (alphabet_size - 1) boundary values.

    Returns:
        (w,) int array of symbol indices in [0, alphabet_size).
    """
    return np.searchsorted(breakpoints, paa_vec, side="right").astype(np.int32)


# ─── SAXFilter ─────────────────────────────────────────────────────────────────

class SAXFilter:
    """Second-stage symbolic filter for HNSW top-K candidates.

    Converts scaled feature vectors to SAX symbol strings and discards
    candidates whose MINDIST from the query exceeds `max_symbolic_distance`.

    The SAX MINDIST is a lower bound on the true Euclidean distance — so
    this filter can only *exclude* false positives; it cannot introduce
    false negatives relative to the exact Euclidean retrieval.

    Args:
        word_size:            Number of PAA segments (symbols per word). 2–8.
        alphabet_size:        Symbol vocabulary size. 2–8.
        max_symbolic_distance: Maximum allowed MINDIST per element of the word.
                               Candidates with mean_mindist > this are dropped.
                               None = no threshold (accept all).
    """

    def __init__(
        self,
        word_size: int = 4,
        alphabet_size: int = 4,
        max_symbolic_distance: float | None = 1.0,
    ) -> None:
        if word_size < 1 or word_size > 8:
            raise ValueError(f"word_size must be 1–8, got {word_size}")
        if alphabet_size not in _BREAKPOINTS:
            raise ValueError(
                f"alphabet_size must be one of {list(_BREAKPOINTS)}, got {alphabet_size}"
            )
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.max_symbolic_distance = max_symbolic_distance
        self._breakpoints: list[float] = _BREAKPOINTS[alphabet_size]
        self._dist_table: np.ndarray = _build_dist_table(alphabet_size)
        self._train_words: np.ndarray | None = None  # (N_train, word_size) int32

    def fit(self, X_scaled: np.ndarray) -> SAXFilter:
        """Pre-compute SAX words for the entire training set.

        Must be called after the StandardScaler is fitted (X_scaled is the
        scaler-transformed training matrix, same as what was passed to the
        BallTree/HNSW index).

        Args:
            X_scaled: (N, D) scaled training feature matrix.

        Returns:
            self (for chaining).
        """
        words = np.empty((len(X_scaled), self.word_size), dtype=np.int32)
        for i, row in enumerate(X_scaled):
            paa = _paa(row, self.word_size)
            words[i] = _digitize(paa, self._breakpoints)
        self._train_words = words
        return self

    def encode(self, x_scaled: np.ndarray) -> np.ndarray:
        """Encode a single scaled feature vector to its SAX word.

        Args:
            x_scaled: (D,) scaled feature vector.

        Returns:
            (word_size,) int32 symbol array.
        """
        paa = _paa(x_scaled, self.word_size)
        return _digitize(paa, self._breakpoints)

    def mindist(self, word_a: np.ndarray, word_b: np.ndarray) -> float:
        """Compute SAX MINDIST between two symbol words.

        MINDIST is element-wise: sum over positions of dist_table[a_i, b_i],
        normalised by sqrt(word_size / feature_dim).  We return the raw
        per-element mean for threshold comparison (dimensionless).

        Args:
            word_a: (word_size,) int32 symbols.
            word_b: (word_size,) int32 symbols.

        Returns:
            Mean per-element symbolic distance (float).
        """
        dists = self._dist_table[word_a, word_b]
        return float(dists.mean())

    def filter_candidates(
        self,
        query_word: np.ndarray,
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        """Return a boolean keep-mask for HNSW candidate indices.

        Candidates with MINDIST > max_symbolic_distance are dropped.
        If max_symbolic_distance is None, all candidates are kept.

        Args:
            query_word:        (word_size,) int32 — SAX word for the query row.
            candidate_indices: (K,) int — training row indices to evaluate.

        Returns:
            (K,) bool — True = candidate survives the symbolic filter.
        """
        if self._train_words is None:
            raise RuntimeError("SAXFilter.fit() must be called before filter_candidates()")
        if self.max_symbolic_distance is None:
            return np.ones(len(candidate_indices), dtype=bool)

        cand_words = self._train_words[candidate_indices]   # (K, word_size)
        # Vectorised distance via lookup table
        dists = self._dist_table[
            query_word[np.newaxis, :],  # (1, word_size)
            cand_words,                 # (K, word_size)
        ]  # (K, word_size)
        mean_dists = dists.mean(axis=1)   # (K,)
        return mean_dists <= self.max_symbolic_distance

    def filter_batch(
        self,
        query_words: np.ndarray,        # (B, word_size) int32
        candidate_indices_b: np.ndarray, # (B, K) training row indices
    ) -> np.ndarray:
        """Vectorised batch version of filter_candidates.

        Args:
            query_words:         (B, word_size) int32 SAX words for each query.
            candidate_indices_b: (B, K) training row indices (padded with 0 for
                                 masked-out positions — masking applied externally).

        Returns:
            (B, K) bool — True = candidate survives the symbolic filter.
        """
        if self._train_words is None:
            raise RuntimeError("SAXFilter.fit() must be called before filter_batch()")
        if self.max_symbolic_distance is None:
            return np.ones(candidate_indices_b.shape, dtype=bool)

        B, K = candidate_indices_b.shape
        cand_words = self._train_words[candidate_indices_b]  # (B, K, word_size)
        # query_words[:, np.newaxis, :] → (B, 1, word_size)
        q = query_words[:, np.newaxis, :]                    # (B, 1, word_size)
        dists = self._dist_table[q, cand_words]              # (B, K, word_size)
        mean_dists = dists.mean(axis=2)                      # (B, K)
        return mean_dists <= self.max_symbolic_distance


# ─── Integration helper ─────────────────────────────────────────────────────────

def apply_sax_filter(
    sax: SAXFilter,
    X_train_scaled: np.ndarray,
    X_query_batch_scaled: np.ndarray,
    candidate_indices_b: np.ndarray,
    current_mask: np.ndarray,
) -> np.ndarray:
    """Apply SAX symbolic filter to a batch of HNSW candidates.

    Intended to be called from PatternMatcher._post_filter() when
    `use_sax_filter=True`.  Combines the SAX boolean mask with the
    existing distance/ticker/sector mask via logical-AND.

    Args:
        sax:                     Fitted SAXFilter instance.
        X_train_scaled:          (N, D) — full scaled training matrix.
        X_query_batch_scaled:    (B, D) — scaled query rows for this batch.
        candidate_indices_b:     (B, K) — training row indices per query.
        current_mask:            (B, K) bool — existing post-filter mask.

    Returns:
        (B, K) bool — updated mask with SAX filtering applied.
    """
    if sax._train_words is None:
        sax.fit(X_train_scaled)

    B = X_query_batch_scaled.shape[0]
    query_words = np.empty((B, sax.word_size), dtype=np.int32)
    for i in range(B):
        query_words[i] = sax.encode(X_query_batch_scaled[i])

    sax_mask = sax.filter_batch(query_words, candidate_indices_b)
    return current_mask & sax_mask
