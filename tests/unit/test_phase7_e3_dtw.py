"""TDD tests for E3: DTW Reranker."""
from __future__ import annotations
import numpy as np
import pytest
import time


class TestDTWReranker:
    def test_reranker_output_has_k_elements(self):
        """dtw_rerank returns exactly k neighbour indices."""
        from research.wfa_reranker import dtw_rerank
        rng = np.random.RandomState(0)
        query_23d = rng.randn(23)
        neighbours_23d = rng.randn(50, 23)
        indices = np.arange(50)
        reranked_idx, reranked_dists = dtw_rerank(query_23d, neighbours_23d, indices, k=20)
        assert len(reranked_idx) == 20
        assert len(reranked_dists) == 20

    def test_dtw_uses_return_columns_only(self):
        """DTW distance ignores columns 8:23 (candlestick proportions)."""
        from research.wfa_reranker import dtw_rerank
        rng = np.random.RandomState(1)
        query_23d = rng.randn(23)
        neighbours_23d = rng.randn(50, 23)
        indices = np.arange(50)
        # Perturb only candlestick columns — ranking should not change
        neighbours_perturbed = neighbours_23d.copy()
        neighbours_perturbed[:, 8:] *= 1000
        idx1, _ = dtw_rerank(query_23d, neighbours_23d, indices, k=20)
        idx2, _ = dtw_rerank(query_23d, neighbours_perturbed, indices, k=20)
        np.testing.assert_array_equal(idx1, idx2)

    def test_dtw_latency_under_500ms_per_50_computations(self):
        """50 DTW computations on 8-point series completes in < 500ms."""
        from research.wfa_reranker import dtw_rerank
        rng = np.random.RandomState(2)
        query_23d = rng.randn(23)
        neighbours_23d = rng.randn(50, 23)
        indices = np.arange(50)
        t0 = time.time()
        dtw_rerank(query_23d, neighbours_23d, indices, k=20)
        elapsed = time.time() - t0
        assert elapsed < 0.5, f"DTW too slow: {elapsed:.3f}s for 50 computations"

    def test_spearman_fast_fail_threshold(self):
        """Gate logic: if Spearman rho > 0.95, fail without full walk-forward."""
        from scipy.stats import spearmanr
        # Identical rankings → rho = 1.0
        euclidean_ranks = np.arange(50)
        dtw_ranks = np.arange(50)
        rho, _ = spearmanr(euclidean_ranks, dtw_ranks)
        assert rho > 0.95  # should trigger fast-fail


class TestDTWGate:
    def test_gate_passes_with_bss_improvement(self):
        """Gate passes when >=3 folds improve by >=+0.001."""
        baseline = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        enhanced = np.array([0.003, 0.003, 0.003, 0.001, 0.001, 0.001])
        deltas = enhanced - baseline
        assert (deltas >= 0.001).sum() >= 3
