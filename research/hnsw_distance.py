"""
hnsw_distance.py — HNSW approximate nearest-neighbour index.

Wraps hnswlib.Index with a sklearn-compatible interface (fit / kneighbors)
so it can be used as a drop-in replacement for sklearn NearestNeighbors in
Matcher.fit() when EngineConfig.use_hnsw=True.

Also subclasses BaseDistanceMetric to satisfy the research ABC.

Design notes
------------
- space='l2'    : matches production Euclidean (L2) metric exactly.
- num_threads=1 : respects the nn_jobs=1 rule (HNSW manages its own
                  threading internally; joblib parallelism is forbidden).
- hnswlib returns SQUARED L2 — we np.sqrt() so distances match sklearn's
  NearestNeighbors output (max_distance=1.1019 filter stays correct).
- knn_query() returns (labels, sq_dists) — note reversed order vs sklearn.
  kneighbors() returns (distances, indices) to match sklearn convention.
- ef_construction=200, M=16: proven defaults for recall@50 ≥ 0.95 on
  8-dimensional unit-scaled fingerprints up to 200k points.
- ef at query time = max(n_neighbors * 2, 100) — balances recall vs speed.

Install
-------
    python -m pip install hnswlib

Fallback
--------
If hnswlib is not installed, importing this module still works but
instantiating HNSWIndex raises ImportError with an install hint.
"""

import numpy as np

from research import BaseDistanceMetric

try:
    import hnswlib as _hnswlib
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _hnswlib = None
    _HNSWLIB_AVAILABLE = False

# Build-time hyperparameters (not user-tunable without new benchmark evidence)
_EF_CONSTRUCTION = 200
_M = 16


class HNSWIndex(BaseDistanceMetric):
    """Approximate NN index using an HNSW graph (hnswlib backend).

    Sklearn-compatible interface:
        index.fit(X_train)                       → self
        index.kneighbors(X_query, n_neighbors=k) → (distances, indices)

    BaseDistanceMetric ABC:
        index.fit(X_train)                       → self
        index.compute(query, candidates)          → distances  (ABC method;
            candidates ignored — index is queried directly)
        index.fitted                              → bool

    Usage in Matcher (via EngineConfig.use_hnsw=True):
        The Matcher calls fit() once on X_train_weighted, then kneighbors()
        per batch. No other changes to Matcher are required.
    """

    def __init__(
        self,
        n_neighbors: int = 150,
        dim: int = 8,
        ef_construction: int = _EF_CONSTRUCTION,
        M: int = _M,
        num_threads: int = 1,
    ):
        """
        Args:
            n_neighbors: k returned per query (set to top_k * 3 in Matcher,
                         matching the NearestNeighbors n_neighbors default).
            dim: feature dimensionality. Must match X_train column count.
            ef_construction: index build quality — higher = better recall,
                             slower build. 200 is the standard recommendation.
            M: number of bidirectional links per node. 16 is the standard
               for moderate-dimension data.
            num_threads: always 1 (nn_jobs=1 rule). HNSW's internal threading
                         is separate from joblib and safe on Windows/Py3.12.
        """
        if not _HNSWLIB_AVAILABLE:
            raise ImportError(
                "hnswlib is required for HNSWIndex. "
                "Install with: python -m pip install hnswlib"
            )
        self.n_neighbors = n_neighbors
        self.dim = dim
        self.ef_construction = ef_construction
        self.M = M
        self.num_threads = num_threads
        self._index = None
        self._n_train: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # BaseDistanceMetric ABC
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray) -> "HNSWIndex":
        """Build HNSW index on training fingerprints.

        Args:
            X_train: (N, dim) array — already scaled and weighted by Matcher.

        Returns:
            self (for chaining)
        """
        X = np.asarray(X_train, dtype=np.float32)
        N, d = X.shape

        if d != self.dim:
            raise ValueError(
                f"HNSWIndex.fit: expected dim={self.dim}, got {d}. "
                "Re-instantiate with the correct dim."
            )

        self._index = _hnswlib.Index(space="l2", dim=d)
        self._index.init_index(
            max_elements=N,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self._index.add_items(X, num_threads=self.num_threads)
        self._n_train = N
        self._fitted = True
        return self

    def compute(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Return L2 distances from query to its k nearest neighbours.

        NOTE: `candidates` is ignored — the HNSW index is queried directly.
        This satisfies the BaseDistanceMetric ABC; the primary interface for
        Matcher is kneighbors().

        Args:
            query: (dim,) or (1, dim) single query fingerprint.
            candidates: ignored (index is queried internally).

        Returns:
            (k,) L2 distances to the k nearest neighbours.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before compute().")
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        k = min(self.n_neighbors, self._n_train)
        self._index.set_ef(max(k * 2, 100))
        labels, sq_dists = self._index.knn_query(q, k=k, num_threads=self.num_threads)
        return np.sqrt(sq_dists[0])  # squared L2 → L2

    @property
    def fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Sklearn-compatible interface (used by Matcher.query() unchanged)
    # ------------------------------------------------------------------

    def kneighbors(
        self, X: np.ndarray, n_neighbors: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query HNSW index — sklearn NearestNeighbors.kneighbors() API.

        Args:
            X: (N_queries, dim) query matrix (float32 preferred).
            n_neighbors: k to return. Defaults to self.n_neighbors.

        Returns:
            (distances, indices):
                distances: (N_queries, k) L2 distances (sqrt applied)
                indices:   (N_queries, k) integer indices into training set
            Matches sklearn NearestNeighbors.kneighbors() return convention.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before kneighbors().")

        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        k = min(k, self._n_train)

        # ef > k for quality; floor at 100 for stability at small N
        self._index.set_ef(max(k * 2, 100))

        X_f32 = np.asarray(X, dtype=np.float32)
        # hnswlib returns (labels, sq_distances) — reversed vs sklearn
        labels, sq_dists = self._index.knn_query(
            X_f32, k=k, num_threads=self.num_threads
        )

        # sqrt: hnswlib space='l2' returns squared L2
        distances = np.sqrt(sq_dists)
        return distances, labels  # (distances, indices) — sklearn convention
