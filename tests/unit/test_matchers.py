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

    def test_save_index_creates_files(self, tmp_path):
        """save_index creates both binary and companion .meta.json files."""
        X_train = _random_data(200, 8)
        matcher = HNSWMatcher(n_neighbors=50)
        matcher.fit(X_train)

        index_path = tmp_path / "test.index"
        matcher.save_index(index_path)

        assert index_path.exists(), "Binary index file was not created"
        assert (tmp_path / "test.index.meta.json").exists(), "Companion .meta.json was not created"

    def test_save_before_fit_raises(self, tmp_path):
        """save_index on unfitted matcher raises RuntimeError."""
        matcher = HNSWMatcher()
        index_path = tmp_path / "test.index"
        with pytest.raises(RuntimeError, match="save_index\\(\\) called on unfitted"):
            matcher.save_index(index_path)

    def test_load_index_round_trip(self, tmp_path):
        """save then load produces same kneighbors results as original."""
        X_train = _random_data(200, 8)
        X_query = _random_data(10, 8, seed=99)
        original = HNSWMatcher(n_neighbors=50)
        original.fit(X_train)

        index_path = tmp_path / "test.index"
        original.save_index(index_path)

        distances_orig, indices_orig = original.kneighbors(X_query, n_neighbors=50)

        loaded = HNSWMatcher()
        loaded.load_index(index_path)

        assert loaded.is_fitted, "Loaded matcher should report is_fitted == True"

        distances_loaded, indices_loaded = loaded.kneighbors(X_query, n_neighbors=50)

        assert distances_loaded.shape == distances_orig.shape
        assert indices_loaded.shape == indices_orig.shape
        np.testing.assert_array_equal(indices_loaded, indices_orig)

    def test_load_nonexistent_raises(self, tmp_path):
        """load_index raises RuntimeError when companion .meta.json missing."""
        matcher = HNSWMatcher()
        missing_path = tmp_path / "nonexistent.index"
        with pytest.raises(RuntimeError, match="Companion metadata file not found"):
            matcher.load_index(missing_path)

    def test_loaded_matcher_kneighbors_works(self, tmp_path):
        """Loaded matcher returns valid distances (non-negative) and correct shape."""
        X_train = _random_data(200, 8)
        X_query = _random_data(5, 8, seed=77)
        original = HNSWMatcher(n_neighbors=50)
        original.fit(X_train)

        index_path = tmp_path / "test.index"
        original.save_index(index_path)

        loaded = HNSWMatcher()
        loaded.load_index(index_path)

        distances, indices = loaded.kneighbors(X_query, n_neighbors=50)

        assert distances.shape == (5, 50)
        assert indices.shape == (5, 50)
        assert np.all(distances >= 0), "All distances must be non-negative"

    def test_load_restores_all_params(self, tmp_path):
        """load_index restores n_neighbors, ef_construction, M, num_threads from meta."""
        matcher = HNSWMatcher(n_neighbors=10, ef_construction=100, M=8)
        matcher.fit(_random_data(100, 8))
        path = tmp_path / "idx"
        matcher.save_index(path)

        loaded = HNSWMatcher()  # default n_neighbors=50
        loaded.load_index(path)

        params = loaded.get_params()
        assert params["n_neighbors"] == 10
        assert params["ef_construction"] == 100
        assert params["M"] == 8
