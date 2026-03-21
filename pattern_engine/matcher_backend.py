"""
matcher_backend.py — backend abstraction for staged Matcher refactor.

This module is intentionally narrow: it only owns index construction and
neighbor queries. The existing Matcher still owns feature preparation,
post-filtering, and projection. That split is the first safe step toward the
Phase 3Z staged architecture:

1. prepare features
2. build backend index
3. query backend
4. post-filter candidates
5. package results

The default sklearn backend preserves current behavior. The optional HNSW
backend exposes the same fit()/kneighbors() surface so it can later be wired
behind an EngineConfig flag without changing Matcher call sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.neighbors import NearestNeighbors


class BaseMatcherBackend(ABC):
    """Minimal backend interface required by Matcher."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseMatcherBackend":
        """Build the search index from a 2D feature matrix."""

    @abstractmethod
    def kneighbors(
        self,
        X: np.ndarray,
        n_neighbors: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) in sklearn-compatible format."""


class SklearnNNBackend(BaseMatcherBackend):
    """Thin wrapper around sklearn NearestNeighbors."""

    def __init__(
        self,
        *,
        metric: str,
        algorithm: str,
        n_jobs: int,
        n_neighbors: int,
    ) -> None:
        self.metric = metric
        self.algorithm = algorithm
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors
        self._index: NearestNeighbors | None = None
        self._row_count = 0

    def fit(self, X: np.ndarray) -> "SklearnNNBackend":
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected 2D feature matrix for backend fit().")

        self._row_count = len(X)
        self._index = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, self._row_count),
            metric=self.metric,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
        )
        self._index.fit(X)
        return self

    def kneighbors(
        self,
        X: np.ndarray,
        n_neighbors: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Call fit() before kneighbors().")

        k = self._resolve_neighbors(n_neighbors)
        return self._index.kneighbors(np.asarray(X), n_neighbors=k)

    def _resolve_neighbors(self, n_neighbors: int | None) -> int:
        if self._row_count == 0:
            raise RuntimeError("Cannot query backend with zero fitted rows.")
        requested = self.n_neighbors if n_neighbors is None else n_neighbors
        return min(requested, self._row_count)


class HNSWMatcherBackend(BaseMatcherBackend):
    """Optional HNSW backend with sklearn-compatible kneighbors output.

    Notes:
    - hnswlib reports squared L2 distance for `space='l2'`
    - sklearn NearestNeighbors(metric='euclidean') reports true Euclidean
    - therefore we apply sqrt() before returning distances
    """

    def __init__(
        self,
        *,
        space: str = "l2",
        n_neighbors: int = 50,
        ef_search: int | None = None,
        ef_construction: int | None = None,
        m: int = 16,
    ) -> None:
        self.space = space
        self.n_neighbors = n_neighbors
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.m = m

        self._index = None
        self._dim: int | None = None
        self._row_count = 0

    def fit(self, X: np.ndarray) -> "HNSWMatcherBackend":
        try:
            import hnswlib
        except ImportError as exc:
            raise ImportError(
                "hnswlib is required for HNSWMatcherBackend. "
                "Install it with `pip install hnswlib`."
            ) from exc

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("Expected 2D feature matrix for backend fit().")

        self._row_count, self._dim = X.shape
        self._index = hnswlib.Index(space=self.space, dim=self._dim)
        self._index.init_index(
            max_elements=self._row_count,
            ef_construction=self.ef_construction or max(self.n_neighbors * 4, 200),
            M=self.m,
        )
        self._index.add_items(X)
        self._index.set_ef(max(self.ef_search or self.n_neighbors * 2, 100))
        return self

    def kneighbors(
        self,
        X: np.ndarray,
        n_neighbors: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._dim is None:
            raise RuntimeError("Call fit() before kneighbors().")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("Expected 2D query matrix for backend kneighbors().")
        if X.shape[1] != self._dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self._dim}, got {X.shape[1]}."
            )

        k = min(self.n_neighbors if n_neighbors is None else n_neighbors, self._row_count)
        indices, distances_sq = self._index.knn_query(X, k=k)
        distances = np.sqrt(distances_sq.astype(np.float64, copy=False))
        return distances, indices.astype(np.int64, copy=False)


def build_matcher_backend(
    *,
    use_hnsw: bool,
    metric: str,
    algorithm: str,
    n_jobs: int,
    n_neighbors: int,
) -> BaseMatcherBackend:
    """Factory for a backend with Matcher-compatible fit()/kneighbors()."""
    if use_hnsw:
        if metric != "euclidean":
            raise ValueError(
                "The initial HNSW backend only supports euclidean distance parity."
            )
        return HNSWMatcherBackend(n_neighbors=n_neighbors)

    return SklearnNNBackend(
        metric=metric,
        algorithm=algorithm,
        n_jobs=n_jobs,
        n_neighbors=n_neighbors,
    )
