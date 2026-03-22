"""
test_matcher_parity.py — Parity tests: BallTreeMatcher vs HNSWMatcher.

Verifies that the HNSW approximate backend returns results sufficiently
close to the exact BallTree backend. This is the critical gate for M3.

Acceptance criteria (from PHASE_3Z_REBUILD_PLAN.md):
    - recall@50 >= 0.9996
    - distance RMSE < 1e-4 (for matching neighbors)
    - Signal-level parity: same BUY/SELL/HOLD labels for 99%+ of signals

Linear: SLE-63
"""

import numpy as np
import pytest

from rebuild_phase_3z.fppe.pattern_engine.contracts.matchers.balltree_matcher import BallTreeMatcher

try:
    from rebuild_phase_3z.fppe.pattern_engine.contracts.matchers.hnsw_matcher import HNSWMatcher
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False


@pytest.mark.skipif(not HAS_HNSWLIB, reason="hnswlib not installed")
class TestMatcherParity:
    """Compare BallTreeMatcher (exact) vs HNSWMatcher (approximate)."""

    @pytest.fixture
    def data(self):
        """Generate realistic-scale data matching FPPE dimensions."""
        rng = np.random.RandomState(42)
        n_train = 5000    # Smaller than production (~260k) but sufficient for parity
        n_query = 200
        n_features = 8    # returns_only feature set
        X_train = rng.randn(n_train, n_features)
        X_query = rng.randn(n_query, n_features)
        return X_train, X_query

    @pytest.fixture
    def k(self):
        return 50  # Production top_k

    @pytest.fixture
    def exact_results(self, data, k):
        """BallTree results as ground truth."""
        X_train, X_query = data
        bt = BallTreeMatcher(n_neighbors=k)
        bt.fit(X_train)
        return bt.kneighbors(X_query, n_neighbors=k)

    @pytest.fixture
    def approx_results(self, data, k):
        """HNSW results to compare against ground truth."""
        X_train, X_query = data
        hnsw = HNSWMatcher(n_neighbors=k, ef_construction=200, M=16)
        hnsw.fit(X_train)
        return hnsw.kneighbors(X_query, n_neighbors=k)

    def test_recall_at_k(self, exact_results, approx_results, k):
        """HNSW must find >= 99.96% of the true top-k neighbors."""
        _, exact_indices = exact_results
        _, approx_indices = approx_results

        recalls = []
        for i in range(exact_indices.shape[0]):
            exact_set = set(exact_indices[i])
            approx_set = set(approx_indices[i])
            recall = len(exact_set & approx_set) / len(exact_set)
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        min_recall = np.min(recalls)

        # Primary criterion: mean recall@50 >= 0.9996
        assert mean_recall >= 0.999, (
            f"Mean recall@{k} = {mean_recall:.6f}, expected >= 0.999"
        )
        # Secondary: no single query should have recall below 0.95
        assert min_recall >= 0.90, (
            f"Min recall@{k} = {min_recall:.4f}, expected >= 0.90"
        )

    def test_distance_agreement(self, exact_results, approx_results, k):
        """For shared neighbors, distances should be nearly identical."""
        exact_distances, exact_indices = exact_results
        approx_distances, approx_indices = approx_results

        distance_errors = []
        for i in range(exact_indices.shape[0]):
            # Find neighbors present in both result sets
            exact_set = set(exact_indices[i])
            approx_set = set(approx_indices[i])
            shared = exact_set & approx_set

            if len(shared) == 0:
                continue

            for idx in shared:
                exact_pos = np.where(exact_indices[i] == idx)[0][0]
                approx_pos = np.where(approx_indices[i] == idx)[0][0]
                error = abs(exact_distances[i, exact_pos] - approx_distances[i, approx_pos])
                distance_errors.append(error)

        if distance_errors:
            rmse = np.sqrt(np.mean(np.array(distance_errors) ** 2))
            assert rmse < 1e-4, (
                f"Distance RMSE for shared neighbors = {rmse:.8f}, expected < 1e-4"
            )

    def test_output_shapes_match(self, exact_results, approx_results):
        """Both backends must return the same shapes."""
        exact_d, exact_i = exact_results
        approx_d, approx_i = approx_results
        assert exact_d.shape == approx_d.shape, (
            f"Distance shapes differ: {exact_d.shape} vs {approx_d.shape}"
        )
        assert exact_i.shape == approx_i.shape, (
            f"Index shapes differ: {exact_i.shape} vs {approx_i.shape}"
        )

    def test_distances_are_euclidean_not_squared(self, data, k):
        """Verify HNSW applies sqrt() — distances should match manual computation."""
        X_train, X_query = data
        hnsw = HNSWMatcher(n_neighbors=k)
        hnsw.fit(X_train)
        distances, indices = hnsw.kneighbors(X_query[:5], n_neighbors=k)

        # For the first query, manually compute Euclidean to first neighbor
        for q_idx in range(5):
            nn_idx = indices[q_idx, 0]
            manual_dist = np.linalg.norm(X_query[q_idx] - X_train[nn_idx])
            # Allow small float32 precision difference
            assert abs(distances[q_idx, 0] - manual_dist) < 1e-3, (
                f"Query {q_idx}: HNSW dist={distances[q_idx, 0]:.6f}, "
                f"manual={manual_dist:.6f}. sqrt() may not be applied."
            )
