"""
hnsw_matcher.py — HNSW-based approximate nearest-neighbor matcher.

Wraps hnswlib.Index behind the BaseMatcher ABC. Repositioned from
research/hnsw_distance.py where it was incorrectly subclassing
BaseDistanceMetric.

Performance: 54.5x speedup over BallTree, recall@50 >= 0.9996.
Validated in research/hnsw_distance.py experiments.

Key design decisions:
    - space='l2' (squared Euclidean) — matches production metric
    - sqrt() applied to return true Euclidean distances (sklearn convention)
    - num_threads=1 always (nn_jobs=1 rule, CLAUDE.md)
    - ef_construction=200, M=16 — proven defaults from Phase C research
    - ef at query time = max(k * 2, 100) — balances speed vs recall

Linear: SLE-62
"""

from typing import Dict, Tuple

import numpy as np

from pattern_engine.contracts.matcher import BaseMatcher


class HNSWMatcher(BaseMatcher):
    """Approximate nearest-neighbor search using HNSW (hnswlib).

    Provides ~55x speedup over BallTree with recall@50 >= 0.9996.
    Results are approximate — parity tests verify acceptable deviation.

    Args:
        n_neighbors: Default number of neighbors to return.
        ef_construction: HNSW build-time parameter. Higher = better recall,
                        slower build. 200 is empirically optimal for FPPE.
        M: Number of bi-directional links per node. 16 is standard.
        num_threads: Must be 1 (Windows/Py3.12 joblib deadlock rule).

    Raises:
        ImportError: If hnswlib is not installed.
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        ef_construction: int = 200,
        M: int = 16,
        num_threads: int = 1,
    ) -> None:
        if num_threads != 1:
            raise RuntimeError(
                f"num_threads must be 1 (got {num_threads}). "
                "See CLAUDE.md: multi-threading causes deadlock on Windows/Py3.12."
            )

        # Defer import so hnswlib is only required when this backend is selected
        try:
            import hnswlib  # noqa: F401
        except ImportError:
            raise ImportError(
                "hnswlib is required for HNSWMatcher. "
                "Install with: pip install hnswlib"
            )

        self._n_neighbors = n_neighbors
        self._ef_construction = ef_construction
        self._M = M
        self._num_threads = num_threads
        self._index = None
        self._n_features: int | None = None
        self._n_samples: int = 0

    def fit(self, X: np.ndarray) -> None:
        """Build HNSW index from training data.

        Args:
            X: Training features, shape (n_samples, n_features).
               Must be pre-scaled and feature-weighted.
               Converted to float32 internally (hnswlib requirement).
        """
        import hnswlib

        if X.ndim != 2:
            raise RuntimeError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] == 0:
            raise RuntimeError("X is empty — cannot fit on zero samples")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise RuntimeError("X contains NaN or Inf values")

        n_samples, n_features = X.shape

        # hnswlib requires float32
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)

        self._index = hnswlib.Index(space="l2", dim=n_features)
        self._index.init_index(
            max_elements=n_samples,
            ef_construction=self._ef_construction,
            M=self._M,
        )
        self._index.add_items(X_f32, num_threads=self._num_threads)

        self._n_features = n_features
        self._n_samples = n_samples

    def kneighbors(
        self, X: np.ndarray, n_neighbors: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query HNSW index for approximate nearest neighbors.

        Returns (distances, indices) matching sklearn convention:
            distances: shape (n_queries, n_neighbors), Euclidean (sqrt applied)
            indices: shape (n_queries, n_neighbors), sorted by distance

        Note: hnswlib returns SQUARED L2 distances. We apply sqrt() to return
        true Euclidean distances, matching the BallTreeMatcher contract.
        """
        if self._index is None:
            raise RuntimeError("fit() must be called before kneighbors()")
        if X.ndim != 2:
            raise RuntimeError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self._n_features:
            raise RuntimeError(
                f"Feature dimension mismatch: fit had {self._n_features}, "
                f"query has {X.shape[1]}"
            )

        k = n_neighbors if n_neighbors > 0 else self._n_neighbors
        k = min(k, self._n_samples)

        # Set ef (query-time parameter): higher ef = better recall, slower query.
        # k * 2 restored (was temporarily k * 4): M=32/ef=k*4 test 2026-03-24
        # showed zero BSS improvement vs M=16/ef=k*2. Recall gap vs BallTree
        # is statistical noise at current feature quality. Reverted to baseline.
        self._index.set_ef(max(k * 2, 100))

        # hnswlib requires float32
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)

        # knn_query returns (labels, squared_distances) — note the reversed order!
        labels, sq_distances = self._index.knn_query(X_f32, k=k)

        # CRITICAL: hnswlib returns SQUARED L2. Apply sqrt for Euclidean convention.
        # Clamp negative values (floating-point artifacts near zero) before sqrt.
        distances = np.sqrt(np.maximum(sq_distances, 0.0))

        return distances, labels

    def get_params(self) -> Dict[str, object]:
        return {
            "backend": "hnsw",
            "n_neighbors": self._n_neighbors,
            "ef_construction": self._ef_construction,
            "M": self._M,
            "num_threads": self._num_threads,
            "n_samples_fitted": self._n_samples,
        }

    @property
    def is_fitted(self) -> bool:
        return self._index is not None
