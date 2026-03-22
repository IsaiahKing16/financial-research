"""
balltree_matcher.py — BallTree-based exact nearest-neighbor matcher.

Wraps sklearn.neighbors.NearestNeighbors with Euclidean distance.
This is the current production retrieval backend, properly encapsulated
behind the BaseMatcher ABC.

Performance: Exact results, O(n log n) query time. Sufficient for the
current 52-ticker universe (~260k training points per fold).

Linear: SLE-61
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from rebuild_phase_3z.fppe.pattern_engine.contracts.matcher import BaseMatcher


class BallTreeMatcher(BaseMatcher):
    """Exact nearest-neighbor search using sklearn BallTree.

    This is the reference implementation. All parity tests compare
    against BallTreeMatcher results.

    Args:
        n_neighbors: Default number of neighbors to return.
        nn_jobs: Number of parallel jobs. MUST be 1 on Windows/Py3.12
                 to prevent joblib deadlock (CLAUDE.md critical rule #5).
    """

    def __init__(self, n_neighbors: int = 50, nn_jobs: int = 1) -> None:
        if nn_jobs != 1:
            raise RuntimeError(
                f"nn_jobs must be 1 (got {nn_jobs}). "
                "See CLAUDE.md: nn_jobs>1 causes joblib deadlock on Windows/Py3.12."
            )
        self._n_neighbors = n_neighbors
        self._nn_jobs = nn_jobs
        self._nn: NearestNeighbors | None = None
        self._n_features: int | None = None
        self._n_samples: int = 0

    def fit(self, X: np.ndarray) -> None:
        """Build BallTree index from training data.

        Args:
            X: Training features, shape (n_samples, n_features).
               Must be pre-scaled and feature-weighted.
        """
        if X.ndim != 2:
            raise RuntimeError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] == 0:
            raise RuntimeError("X is empty — cannot fit on zero samples")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise RuntimeError("X contains NaN or Inf values")

        self._nn = NearestNeighbors(
            n_neighbors=min(self._n_neighbors, X.shape[0]),
            algorithm="ball_tree",
            metric="euclidean",
            n_jobs=self._nn_jobs,
        )
        self._nn.fit(X)
        self._n_features = X.shape[1]
        self._n_samples = X.shape[0]

    def kneighbors(
        self, X: np.ndarray, n_neighbors: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query BallTree for exact nearest neighbors.

        Returns (distances, indices) matching sklearn convention:
            distances: shape (n_queries, n_neighbors), Euclidean
            indices: shape (n_queries, n_neighbors), sorted by distance
        """
        if self._nn is None:
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

        distances, indices = self._nn.kneighbors(X, n_neighbors=k)
        return distances, indices

    def get_params(self) -> Dict[str, object]:
        return {
            "backend": "balltree",
            "n_neighbors": self._n_neighbors,
            "nn_jobs": self._nn_jobs,
            "algorithm": "ball_tree",
            "metric": "euclidean",
            "n_samples_fitted": self._n_samples,
        }

    @property
    def is_fitted(self) -> bool:
        return self._nn is not None
