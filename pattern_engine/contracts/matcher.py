"""
matcher.py — BaseMatcher ABC for nearest-neighbor retrieval backends.

This replaces the architectural misfit where HNSWIndex inherited BaseDistanceMetric.
A distance metric computes d(x, y); a matcher answers "find the k closest points."
These are fundamentally different abstractions.

Current architecture (wrong):
    BaseDistanceMetric (ABC)
     ├── EuclideanDistance      ← correct: this IS a distance metric
     └── HNSWIndex              ← WRONG: this is a retrieval backend, not a metric

Target architecture (correct):
    BaseMatcher (ABC)
     ├── BallTreeMatcher        ← wraps sklearn NearestNeighbors + EuclideanDistance
     └── HNSWMatcher            ← wraps hnswlib Index, applies sqrt() correction
    BaseDistanceMetric (ABC)
     └── EuclideanDistance      ← unchanged, used by BallTreeMatcher

The key interface contract:
    fit(X)              → build index from training data
    kneighbors(X, k)    → return (distances, indices) matching sklearn convention

Linear: SLE-60
Design doc: docs/rebuild/PHASE_3Z_EXECUTION_PLAN.md §4
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseMatcher(ABC):
    """Abstract base class for nearest-neighbor retrieval backends.

    Implementations must provide:
        - fit(): Build an index from training feature vectors
        - kneighbors(): Query the index for k nearest neighbors
        - get_params(): Return a dict of configuration parameters for provenance

    Return contract for kneighbors():
        distances: np.ndarray of shape (n_queries, n_neighbors)
            Euclidean distances (NOT squared). hnswlib returns squared L2;
            implementations must apply sqrt() before returning.
        indices: np.ndarray of shape (n_queries, n_neighbors)
            Integer indices into the fitted training array.

    Both arrays are sorted by ascending distance (closest first),
    matching the sklearn.neighbors convention.

    Thread safety:
        Implementations must be safe for single-threaded use (nn_jobs=1).
        Multi-threaded query is NOT required for v1.

    Usage:
        matcher = BallTreeMatcher(n_neighbors=50)
        matcher.fit(X_train)
        distances, indices = matcher.kneighbors(X_query, n_neighbors=50)
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Build the nearest-neighbor index from training data.

        Args:
            X: Training feature matrix, shape (n_samples, n_features).
               Must already be scaled and feature-weighted.
               The Matcher owns the index; the caller owns the data.

        Raises:
            RuntimeError: If X is empty, has wrong dtype, or contains NaN/Inf.
        """
        ...

    @abstractmethod
    def kneighbors(
        self, X: np.ndarray, n_neighbors: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the k nearest neighbors for each query point.

        Args:
            X: Query feature matrix, shape (n_queries, n_features).
               Must be scaled and weighted with the same transform as fit().
            n_neighbors: Number of neighbors to return per query.

        Returns:
            Tuple of (distances, indices):
                distances: shape (n_queries, n_neighbors), Euclidean (not squared)
                indices: shape (n_queries, n_neighbors), int indices into fit(X)

        Raises:
            RuntimeError: If fit() has not been called.
            RuntimeError: If X has different n_features than training data.
        """
        ...

    @abstractmethod
    def get_params(self) -> dict[str, object]:
        """Return configuration parameters for logging/provenance.

        Returns:
            Dict with at minimum:
                "backend": str (e.g., "balltree", "hnsw")
                "n_neighbors": int
                Any backend-specific hyperparameters.
        """
        ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether fit() has been called. Must be overridden by concrete subclasses.

        Returns:
            True if fit() has been called and the index is ready for queries.
        """
        ...

    def __repr__(self) -> str:
        params = self.get_params()
        backend = params.get("backend", self.__class__.__name__)
        return f"{backend}({params})"
