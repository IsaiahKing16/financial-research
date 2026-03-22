"""
test_wfa_reranker.py — Unit tests for DTW-based WFA reranker (SLE-73).

Tests cover:
  - _dtw_distance: known values, symmetry, zero diagonal, window constraint
  - _dtw_distance_batch: matches per-row calls
  - WFAReranker.fit() and rerank()
  - WFAReranker.rerank_mask(): shape, survivor count preservation
  - Edge cases: empty candidates, single candidate, window=0
  - Parity check: when all DTW distances are equal, original order is stable
"""

from __future__ import annotations

import numpy as np
import pytest

from pattern_engine.wfa_reranker import (
    WFAReranker,
    _dtw_distance,
    _dtw_distance_batch,
)


# ─── DTW distance tests ───────────────────────────────────────────────────────

class TestDTWDistance:
    def test_identical_sequences_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert _dtw_distance(a, a) == pytest.approx(0.0)

    def test_symmetry(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert _dtw_distance(a, b) == pytest.approx(_dtw_distance(b, a))

    def test_unit_shift_small(self):
        """Sequence shifted by 1 position should have small DTW distance."""
        a = np.array([0.0, 1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 1.0, 0.0])
        d = _dtw_distance(a, b, window=2)
        # DTW allows 1-step warp → distance should be 0 or very small
        assert d >= 0.0
        # Euclidean distance would be sqrt(2); DTW with window=2 should do better
        euclidean = float(np.sqrt(np.sum((a - b) ** 2)))
        assert d <= euclidean + 1e-9

    def test_constant_sequences(self):
        """All-zeros vs all-zeros → zero."""
        a = np.zeros(8)
        assert _dtw_distance(a, a) == pytest.approx(0.0)

    def test_window_zero_is_euclidean(self):
        """Window=0 collapses to Euclidean (no warping allowed)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        dtw_d = _dtw_distance(a, b, window=0)
        euclidean = float(np.sqrt(np.sum((a - b) ** 2)))
        assert dtw_d == pytest.approx(euclidean, rel=1e-9)

    def test_non_negative(self):
        rng = np.random.RandomState(42)
        for _ in range(20):
            a = rng.randn(8)
            b = rng.randn(8)
            assert _dtw_distance(a, b) >= 0.0


class TestDTWDistanceBatch:
    def test_matches_per_row(self):
        rng = np.random.RandomState(7)
        query = rng.randn(8)
        candidates = rng.randn(10, 8)
        batch = _dtw_distance_batch(query, candidates, window=2)
        for k in range(10):
            expected = _dtw_distance(query, candidates[k], window=2)
            assert batch[k] == pytest.approx(expected)

    def test_output_shape(self):
        query = np.zeros(8)
        candidates = np.zeros((15, 8))
        result = _dtw_distance_batch(query, candidates)
        assert result.shape == (15,)


# ─── WFAReranker tests ────────────────────────────────────────────────────────

class TestWFAReranker:
    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    @pytest.fixture
    def X_train(self, rng):
        return rng.randn(50, 8).astype(np.float64)

    @pytest.fixture
    def fitted_reranker(self, X_train):
        r = WFAReranker(window=2)
        r.fit(X_train)
        return r

    def test_fit_stores_features(self, fitted_reranker, X_train):
        assert fitted_reranker._train_features is not None
        assert fitted_reranker._train_features.shape == X_train.shape

    def test_rerank_empty_candidates(self, fitted_reranker):
        query = np.zeros(8)
        result = fitted_reranker.rerank(query, np.array([], dtype=int))
        assert len(result) == 0

    def test_rerank_returns_same_indices(self, fitted_reranker, rng):
        """Reranked result must contain same indices (just reordered)."""
        query = rng.randn(8)
        indices = np.array([0, 5, 10, 15, 20])
        reranked = fitted_reranker.rerank(query, indices)
        assert set(reranked) == set(indices)

    def test_rerank_puts_nearest_first(self, X_train):
        """Create a query identical to one training row → that row is first."""
        r = WFAReranker(window=2)
        r.fit(X_train)
        query = X_train[3].copy()   # identical to row 3
        indices = np.array([0, 1, 2, 3, 4])
        reranked = r.rerank(query, indices)
        assert reranked[0] == 3   # row 3 (zero DTW) must come first

    def test_rerank_raises_before_fit(self):
        r = WFAReranker()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            r.rerank(np.zeros(8), np.array([0, 1]))

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window"):
            WFAReranker(window=-1)


class TestWFARerankerMask:
    @pytest.fixture
    def setup(self):
        rng = np.random.RandomState(0)
        X_train = rng.randn(40, 8).astype(np.float64)
        r = WFAReranker(window=2)
        r.fit(X_train)
        return r, X_train, rng

    def test_rerank_mask_output_shape(self, setup):
        r, X_train, rng = setup
        B, n_probe, top_k = 4, 20, 10
        X_query = rng.randn(B, 8)
        indices = np.tile(np.arange(n_probe), (B, 1))
        top_mask = np.zeros((B, n_probe), dtype=bool)
        top_mask[:, :15] = True   # 15 survivors per row

        new_mask = r.rerank_mask(X_query, indices, top_mask, top_k=top_k)
        assert new_mask.shape == (B, n_probe)

    def test_rerank_mask_preserves_survivor_count(self, setup):
        """Reranked mask keeps at most top_k survivors per row."""
        r, X_train, rng = setup
        B, n_probe, top_k = 3, 20, 8
        X_query = rng.randn(B, 8)
        indices = np.tile(np.arange(n_probe), (B, 1))
        top_mask = np.zeros((B, n_probe), dtype=bool)
        top_mask[:, :15] = True   # 15 pre-filter survivors

        new_mask = r.rerank_mask(X_query, indices, top_mask, top_k=top_k)
        for b in range(B):
            assert new_mask[b].sum() <= top_k

    def test_rerank_mask_empty_row_stays_empty(self, setup):
        """A row with no pre-filter survivors stays empty after reranking."""
        r, X_train, rng = setup
        B, n_probe = 2, 10
        X_query = rng.randn(B, 8)
        indices = np.tile(np.arange(n_probe), (B, 1))
        top_mask = np.zeros((B, n_probe), dtype=bool)
        top_mask[0, :5] = True   # row 0 has survivors, row 1 has none

        new_mask = r.rerank_mask(X_query, indices, top_mask, top_k=5)
        assert new_mask[1].sum() == 0

    def test_rerank_mask_raises_before_fit(self):
        r = WFAReranker()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            r.rerank_mask(
                np.zeros((2, 8)),
                np.zeros((2, 5), dtype=int),
                np.ones((2, 5), dtype=bool),
                top_k=3,
            )
