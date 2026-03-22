"""
Tests for research/hnsw_distance.py — HNSWIndex.

Smoke tests validate the sklearn-compatible interface and recall@50.
All tests are skipped gracefully if hnswlib is not installed.
"""

import numpy as np
import pytest

hnswlib = pytest.importorskip("hnswlib", reason="hnswlib not installed")

from research.hnsw_distance import HNSWIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    """300 training points, 8-dim (matches production fingerprint shape)."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((300, 8)).astype(np.float32)
    X_query = rng.standard_normal((20, 8)).astype(np.float32)
    return X_train, X_query


@pytest.fixture
def fitted_index(small_dataset):
    X_train, _ = small_dataset
    return HNSWIndex(n_neighbors=60, dim=8).fit(X_train)


# ---------------------------------------------------------------------------
# Construction and fit
# ---------------------------------------------------------------------------

class TestHNSWIndexConstruction:

    def test_not_fitted_before_fit(self):
        idx = HNSWIndex(n_neighbors=10, dim=8)
        assert not idx.fitted

    def test_fitted_after_fit(self, small_dataset):
        X_train, _ = small_dataset
        idx = HNSWIndex(n_neighbors=10, dim=8).fit(X_train)
        assert idx.fitted

    def test_fit_returns_self(self, small_dataset):
        X_train, _ = small_dataset
        idx = HNSWIndex(n_neighbors=10, dim=8)
        result = idx.fit(X_train)
        assert result is idx

    def test_dim_mismatch_raises(self, small_dataset):
        X_train, _ = small_dataset
        idx = HNSWIndex(n_neighbors=10, dim=4)  # wrong dim
        with pytest.raises(ValueError, match="dim"):
            idx.fit(X_train)

    def test_compute_before_fit_raises(self):
        idx = HNSWIndex(n_neighbors=10, dim=8)
        q = np.zeros(8, dtype=np.float32)
        with pytest.raises(RuntimeError):
            idx.compute(q, candidates=None)

    def test_kneighbors_before_fit_raises(self):
        idx = HNSWIndex(n_neighbors=10, dim=8)
        X = np.zeros((5, 8), dtype=np.float32)
        with pytest.raises(RuntimeError):
            idx.kneighbors(X)


# ---------------------------------------------------------------------------
# kneighbors API — sklearn compatibility
# ---------------------------------------------------------------------------

class TestKneighborsAPI:

    def test_returns_tuple_of_two(self, fitted_index, small_dataset):
        _, X_query = small_dataset
        result = fitted_index.kneighbors(X_query)
        assert isinstance(result, tuple) and len(result) == 2

    def test_distances_shape(self, fitted_index, small_dataset):
        _, X_query = small_dataset
        distances, indices = fitted_index.kneighbors(X_query)
        assert distances.shape == (len(X_query), 60)

    def test_indices_shape(self, fitted_index, small_dataset):
        _, X_query = small_dataset
        distances, indices = fitted_index.kneighbors(X_query)
        assert indices.shape == (len(X_query), 60)

    def test_distances_non_negative(self, fitted_index, small_dataset):
        _, X_query = small_dataset
        distances, _ = fitted_index.kneighbors(X_query)
        assert np.all(distances >= 0)

    def test_distances_sorted_ascending(self, fitted_index, small_dataset):
        _, X_query = small_dataset
        distances, _ = fitted_index.kneighbors(X_query)
        # Each row should be non-decreasing
        assert np.all(np.diff(distances, axis=1) >= -1e-6)

    def test_n_neighbors_override(self, fitted_index, small_dataset):
        _, X_query = small_dataset
        distances, indices = fitted_index.kneighbors(X_query, n_neighbors=10)
        assert distances.shape[1] == 10
        assert indices.shape[1] == 10

    def test_indices_in_valid_range(self, fitted_index, small_dataset):
        X_train, X_query = small_dataset
        _, indices = fitted_index.kneighbors(X_query)
        assert np.all(indices >= 0)
        assert np.all(indices < len(X_train))


# ---------------------------------------------------------------------------
# compute() — BaseDistanceMetric ABC
# ---------------------------------------------------------------------------

class TestComputeABC:

    def test_compute_returns_1d_array(self, fitted_index):
        q = np.zeros(8, dtype=np.float32)
        result = fitted_index.compute(q, candidates=None)
        assert result.ndim == 1

    def test_compute_non_negative(self, fitted_index):
        q = np.zeros(8, dtype=np.float32)
        result = fitted_index.compute(q, candidates=None)
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# recall@50 vs exact L2 (sklearn ball_tree)
# ---------------------------------------------------------------------------

class TestRecall:

    def test_recall_at_50_vs_exact(self):
        """recall@50 ≥ 0.95 on 8-dim data — core SLE-47 success criterion."""
        from sklearn.neighbors import NearestNeighbors

        rng = np.random.default_rng(0)
        N_train, N_query, k = 5000, 100, 50
        X_train = rng.standard_normal((N_train, 8)).astype(np.float32)
        X_query = rng.standard_normal((N_query, 8)).astype(np.float32)

        # Exact ball_tree ground truth
        exact = NearestNeighbors(n_neighbors=k, metric="euclidean",
                                 algorithm="ball_tree", n_jobs=1)
        exact.fit(X_train)
        _, exact_indices = exact.kneighbors(X_query)

        # HNSW approximate
        hnsw = HNSWIndex(n_neighbors=k, dim=8).fit(X_train)
        _, hnsw_indices = hnsw.kneighbors(X_query, n_neighbors=k)

        # recall@50 = mean fraction of true top-50 found by HNSW
        recalls = []
        for i in range(N_query):
            true_set = set(exact_indices[i])
            found_set = set(hnsw_indices[i])
            recalls.append(len(true_set & found_set) / k)
        mean_recall = np.mean(recalls)

        assert mean_recall >= 0.95, (
            f"recall@50 = {mean_recall:.3f} < 0.95 — "
            "increase ef_construction or M in HNSWIndex defaults"
        )

    def test_distances_close_to_exact(self):
        """HNSW distances within 5% of exact L2 distances for top-1."""
        from sklearn.neighbors import NearestNeighbors

        rng = np.random.default_rng(1)
        X_train = rng.standard_normal((2000, 8)).astype(np.float32)
        X_query = rng.standard_normal((50, 8)).astype(np.float32)

        exact = NearestNeighbors(n_neighbors=1, metric="euclidean",
                                 algorithm="ball_tree", n_jobs=1)
        exact.fit(X_train)
        exact_dists, _ = exact.kneighbors(X_query)

        hnsw = HNSWIndex(n_neighbors=1, dim=8).fit(X_train)
        hnsw_dists, _ = hnsw.kneighbors(X_query, n_neighbors=1)

        # Top-1 distances should be within 5% of exact
        rel_errors = np.abs(hnsw_dists[:, 0] - exact_dists[:, 0]) / (exact_dists[:, 0] + 1e-9)
        assert np.mean(rel_errors) < 0.05, (
            f"Mean relative distance error = {np.mean(rel_errors):.3f} ≥ 5%"
        )


# ---------------------------------------------------------------------------
# Matcher integration smoke test
# ---------------------------------------------------------------------------

class TestMatcherIntegration:

    def test_matcher_uses_hnsw_when_configured(self):
        """Matcher.fit() builds HNSWIndex when use_hnsw=True."""
        from pattern_engine.config import EngineConfig
        from pattern_engine.matching import Matcher
        import pandas as pd

        rng = np.random.default_rng(7)
        n = 200
        feature_cols = [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]
        data = {col: rng.standard_normal(n) for col in feature_cols}
        data["Ticker"] = ["AAPL"] * n
        data["fwd_7d_up"] = rng.integers(0, 2, n)
        train_db = pd.DataFrame(data)

        cfg = EngineConfig(use_hnsw=True)
        matcher = Matcher(cfg)
        matcher.fit(train_db, feature_cols)

        assert isinstance(matcher._nn_index, HNSWIndex)
        assert matcher.fitted

    def test_matcher_uses_sklearn_by_default(self):
        """Matcher.fit() uses NearestNeighbors when use_hnsw=False (default)."""
        from sklearn.neighbors import NearestNeighbors
        from pattern_engine.config import EngineConfig
        from pattern_engine.matching import Matcher
        import pandas as pd

        rng = np.random.default_rng(8)
        n = 200
        feature_cols = [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]
        data = {col: rng.standard_normal(n) for col in feature_cols}
        data["Ticker"] = ["AAPL"] * n
        data["fwd_7d_up"] = rng.integers(0, 2, n)
        train_db = pd.DataFrame(data)

        cfg = EngineConfig()  # use_hnsw=False by default
        matcher = Matcher(cfg)
        matcher.fit(train_db, feature_cols)

        assert isinstance(matcher._nn_index, NearestNeighbors)
