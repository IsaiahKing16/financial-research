"""
test_matchers.py — Unit tests for BaseMatcher implementations.

Tests that:
    - BallTreeMatcher and HNSWMatcher satisfy the BaseMatcher contract
    - Both return correct shapes
    - Both raise proper errors for invalid inputs
    - nn_jobs/num_threads != 1 is rejected
    - fit-before-query is enforced

Linear: SLE-61 (BallTree), SLE-62 (HNSW)
"""

import numpy as np
import pytest

from pattern_engine.contracts.matcher import BaseMatcher
from pattern_engine.contracts.matchers.balltree_matcher import BallTreeMatcher

# HNSW is optional — skip tests if hnswlib not installed
try:
    from pattern_engine.contracts.matchers.hnsw_matcher import HNSWMatcher
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False


def _random_data(n_samples=500, n_features=8, seed=42):
    """Generate reproducible random feature matrix."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples, n_features).astype(np.float64)


# ─── BallTreeMatcher ────────────────────────────────────────────

class TestBallTreeMatcher:
    """Tests for the BallTreeMatcher (exact NN search)."""

    def test_is_base_matcher(self):
        assert issubclass(BallTreeMatcher, BaseMatcher)

    def test_fit_and_query(self):
        X_train = _random_data(200, 8)
        X_query = _random_data(10, 8, seed=99)
        matcher = BallTreeMatcher(n_neighbors=50)

        matcher.fit(X_train)
        assert matcher.is_fitted

        distances, indices = matcher.kneighbors(X_query, n_neighbors=50)
        assert distances.shape == (10, 50)
        assert indices.shape == (10, 50)

    def test_distances_are_sorted(self):
        X_train = _random_data(200, 8)
        X_query = _random_data(5, 8, seed=99)
        matcher = BallTreeMatcher(n_neighbors=20)
        matcher.fit(X_train)
        distances, _ = matcher.kneighbors(X_query, n_neighbors=20)
        # Each row should be sorted ascending
        for i in range(distances.shape[0]):
            assert np.all(np.diff(distances[i]) >= 0), f"Row {i} not sorted"

    def test_distances_non_negative(self):
        X_train = _random_data(200, 8)
        X_query = _random_data(5, 8, seed=99)
        matcher = BallTreeMatcher(n_neighbors=20)
        matcher.fit(X_train)
        distances, _ = matcher.kneighbors(X_query, n_neighbors=20)
        assert np.all(distances >= 0)

    def test_query_before_fit_raises(self):
        matcher = BallTreeMatcher()
        X = _random_data(5, 8)
        with pytest.raises(RuntimeError, match="fit"):
            matcher.kneighbors(X, n_neighbors=10)

    def test_empty_training_data_raises(self):
        matcher = BallTreeMatcher()
        with pytest.raises(RuntimeError, match="empty"):
            matcher.fit(np.empty((0, 8)))

    def test_nan_training_data_raises(self):
        X = _random_data(100, 8)
        X[5, 3] = np.nan
        matcher = BallTreeMatcher()
        with pytest.raises(RuntimeError, match="NaN"):
            matcher.fit(X)

    def test_feature_dimension_mismatch_raises(self):
        matcher = BallTreeMatcher()
        matcher.fit(_random_data(100, 8))
        X_wrong = _random_data(5, 10)  # 10 features, trained on 8
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            matcher.kneighbors(X_wrong, n_neighbors=10)

    def test_nn_jobs_must_be_1(self):
        with pytest.raises(RuntimeError, match="nn_jobs must be 1"):
            BallTreeMatcher(nn_jobs=4)

    def test_get_params(self):
        matcher = BallTreeMatcher(n_neighbors=30)
        params = matcher.get_params()
        assert params["backend"] == "balltree"
        assert params["n_neighbors"] == 30
        assert params["metric"] == "euclidean"

    def test_k_capped_at_n_samples(self):
        """If k > n_samples, return n_samples neighbors (not error)."""
        X_train = _random_data(20, 8)
        X_query = _random_data(3, 8, seed=99)
        matcher = BallTreeMatcher(n_neighbors=50)
        matcher.fit(X_train)
        distances, indices = matcher.kneighbors(X_query, n_neighbors=50)
        assert distances.shape == (3, 20)  # capped at 20


# ─── HNSWMatcher ─────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_HNSWLIB, reason="hnswlib not installed")
class TestHNSWMatcher:
    """Tests for the HNSWMatcher (approximate NN search)."""

    def test_is_base_matcher(self):
        assert issubclass(HNSWMatcher, BaseMatcher)

    def test_fit_and_query(self):
        X_train = _random_data(200, 8)
        X_query = _random_data(10, 8, seed=99)
        matcher = HNSWMatcher(n_neighbors=50)

        matcher.fit(X_train)
        assert matcher.is_fitted

        distances, indices = matcher.kneighbors(X_query, n_neighbors=50)
        assert distances.shape == (10, 50)
        assert indices.shape == (10, 50)

    def test_distances_non_negative(self):
        """HNSW should return Euclidean distances (sqrt applied), all >= 0."""
        X_train = _random_data(200, 8)
        X_query = _random_data(5, 8, seed=99)
        matcher = HNSWMatcher(n_neighbors=20)
        matcher.fit(X_train)
        distances, _ = matcher.kneighbors(X_query, n_neighbors=20)
        assert np.all(distances >= 0)

    def test_query_before_fit_raises(self):
        matcher = HNSWMatcher()
        X = _random_data(5, 8)
        with pytest.raises(RuntimeError, match="fit"):
            matcher.kneighbors(X, n_neighbors=10)

    def test_empty_training_data_raises(self):
        matcher = HNSWMatcher()
        with pytest.raises(RuntimeError, match="empty"):
            matcher.fit(np.empty((0, 8)))

    def test_nan_training_data_raises(self):
        X = _random_data(100, 8)
        X[5, 3] = np.nan
        matcher = HNSWMatcher()
        with pytest.raises(RuntimeError, match="NaN"):
            matcher.fit(X)

    def test_feature_dimension_mismatch_raises(self):
        matcher = HNSWMatcher()
        matcher.fit(_random_data(100, 8))
        X_wrong = _random_data(5, 10)
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            matcher.kneighbors(X_wrong, n_neighbors=10)

    def test_num_threads_must_be_1(self):
        with pytest.raises(RuntimeError, match="num_threads must be 1"):
            HNSWMatcher(num_threads=4)

    def test_get_params(self):
        matcher = HNSWMatcher(n_neighbors=30, ef_construction=200, M=16)
        params = matcher.get_params()
        assert params["backend"] == "hnsw"
        assert params["n_neighbors"] == 30
        assert params["ef_construction"] == 200
        assert params["M"] == 16

    def test_k_capped_at_n_samples(self):
        X_train = _random_data(20, 8)
        X_query = _random_data(3, 8, seed=99)
        matcher = HNSWMatcher(n_neighbors=50)
        matcher.fit(X_train)
        distances, indices = matcher.kneighbors(X_query, n_neighbors=50)
        assert distances.shape == (3, 20)
