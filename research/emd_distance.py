"""
emd_distance.py — Earth Mover's Distance metric for FPPE fingerprints.

Treats each 8-feature return fingerprint as a 2D weighted point set:
  - x-axis: time lag (calendar days) × time_penalty
  - y-axis: return value × price_penalty

Each point carries uniform weight 1/8.  The EMD (Wasserstein-1 distance) between
two fingerprints is the minimum total cost to transport one distribution onto the
other.

Primary backend: POT (Python Optimal Transport) — ot.emd2 with a (8×8) cost matrix.
Fallback: scipy.stats.wasserstein_distance_nd (SciPy ≥ 1.13, uniform weights only).
"""

import numpy as np
from scipy.spatial.distance import cdist

from research import BaseDistanceMetric

# Time lags must match pattern_engine/features.py RETURN_WINDOWS
RETURN_WINDOWS = [1, 3, 7, 14, 30, 45, 60, 90]

try:
    import ot as _ot
    _POT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _POT_AVAILABLE = False


class EMDDistance(BaseDistanceMetric):
    """Earth Mover's Distance between 8-dimensional return fingerprints.

    Each fingerprint is a 1-D array of 8 returns, one per time window in
    RETURN_WINDOWS.  Internally these are embedded as (8, 2) 2D point sets
    before computing the EMD.

    Args:
        time_penalty:  Scaling factor on the time-lag axis (default 0.5).
        price_penalty: Scaling factor on the return-value axis (default 1.0).
    """

    def __init__(self, time_penalty: float = 0.5, price_penalty: float = 1.0) -> None:
        self.time_penalty = time_penalty
        self.price_penalty = price_penalty
        self._X_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # BaseDistanceMetric interface
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray) -> "EMDDistance":
        """Store training data reference.  No-op for EMD — kept for ABC compliance."""
        self._X_train = X_train
        return self

    def compute(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Compute EMD from query fingerprint to each candidate fingerprint.

        Args:
            query:      Shape (8,) — the current fingerprint.
            candidates: Shape (N, 8) — historical fingerprints.

        Returns:
            np.ndarray of shape (N,), one EMD per candidate.
        """
        query = np.asarray(query, dtype=float)
        candidates = np.asarray(candidates, dtype=float)
        current_coords = self._construct_coords(query)  # (8, 2)
        distances = np.empty(len(candidates), dtype=float)
        for i, candidate in enumerate(candidates):
            hist_coords = self._construct_coords(candidate)  # (8, 2)
            distances[i] = self._emd_scalar(current_coords, hist_coords)
        return distances

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _construct_coords(self, fingerprint: np.ndarray) -> np.ndarray:
        """Convert fingerprint (8,) → 2D point set (8, 2).

        Column 0: time_lag × time_penalty
        Column 1: return_value × price_penalty
        """
        time_lags = np.array(RETURN_WINDOWS, dtype=float) * self.time_penalty
        returns = np.asarray(fingerprint, dtype=float) * self.price_penalty
        return np.column_stack([time_lags, returns])  # (8, 2)

    def _emd_scalar(
        self, current_coords: np.ndarray, hist_coords: np.ndarray
    ) -> float:
        """Compute scalar EMD between two (8, 2) point sets with uniform weights."""
        weights = np.full(8, 1.0 / 8.0)
        cost_matrix = cdist(current_coords, hist_coords, "euclidean")  # (8, 8)

        if _POT_AVAILABLE:
            return float(_ot.emd2(weights, weights, cost_matrix))

        # Fallback: scipy.stats.wasserstein_distance_nd (SciPy >= 1.13).
        # Equivalent to POT for the uniform-weight case on 2D coordinates.
        from scipy.stats import wasserstein_distance_nd  # noqa: PLC0415
        return float(wasserstein_distance_nd(current_coords, hist_coords))
