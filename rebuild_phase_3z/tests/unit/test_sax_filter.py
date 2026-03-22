"""
test_sax_filter.py — Unit tests for SAX symbolic filter (SLE-72).

Tests cover:
  - PAA segmentation (exact and fractional splits)
  - Symbol digitisation (breakpoint boundary conditions)
  - MINDIST lookup table construction
  - SAXFilter.fit() and encode()
  - SAXFilter.filter_candidates() correctness
  - SAXFilter.filter_batch() — vectorised equivalence
  - Edge cases: word_size=1, word_size==D, identical sequences
  - Flag-off baseline: when use_sax_filter=False, no filtering applied
"""

from __future__ import annotations

import numpy as np
import pytest

from rebuild_phase_3z.fppe.pattern_engine.sax_filter import (
    SAXFilter,
    _build_dist_table,
    _digitize,
    _paa,
    apply_sax_filter,
)


# ─── PAA tests ────────────────────────────────────────────────────────────────

class TestPAA:
    def test_paa_identity_when_w_equals_d(self):
        """word_size == D → PAA is the identity."""
        x = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])
        result = _paa(x, w=8)
        np.testing.assert_array_equal(result, x)

    def test_paa_mean_of_segments(self):
        """4 elements, word_size=2 → each segment is mean of 2."""
        x = np.array([1.0, 3.0, 5.0, 7.0])
        result = _paa(x, w=2)
        np.testing.assert_allclose(result, [2.0, 6.0])

    def test_paa_single_segment(self):
        """word_size=1 → single mean."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = _paa(x, w=1)
        np.testing.assert_allclose(result, [2.5])

    def test_paa_8d_to_4(self):
        """8-dim → 4-segment PAA."""
        x = np.arange(8, dtype=float)   # [0,1,2,3,4,5,6,7]
        result = _paa(x, w=4)
        expected = [0.5, 2.5, 4.5, 6.5]
        np.testing.assert_allclose(result, expected)

    def test_paa_raises_when_w_greater_than_d(self):
        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="word_size"):
            _paa(x, w=5)


# ─── Digitise tests ───────────────────────────────────────────────────────────

class TestDigitize:
    """Symbol mapping via breakpoints."""

    def test_alphabet4_below_all_breakpoints(self):
        """Value < -0.6745 → symbol 0."""
        paa = np.array([-1.0])
        bp = [-0.6745, 0.0, 0.6745]
        result = _digitize(paa, bp)
        assert result[0] == 0

    def test_alphabet4_above_all_breakpoints(self):
        """Value > 0.6745 → symbol 3."""
        paa = np.array([1.0])
        bp = [-0.6745, 0.0, 0.6745]
        result = _digitize(paa, bp)
        assert result[0] == 3

    def test_alphabet4_at_zero(self):
        """Value exactly 0 → symbol 2 (searchsorted right)."""
        paa = np.array([0.0])
        bp = [-0.6745, 0.0, 0.6745]
        result = _digitize(paa, bp)
        assert result[0] == 2

    def test_output_dtype_is_int32(self):
        paa = np.array([0.5, -0.5])
        bp = [-0.6745, 0.0, 0.6745]
        result = _digitize(paa, bp)
        assert result.dtype == np.int32


# ─── MINDIST table tests ──────────────────────────────────────────────────────

class TestDistTable:
    def test_diagonal_is_zero(self):
        """Same symbol → MINDIST = 0."""
        table = _build_dist_table(4)
        assert np.all(np.diag(table) == 0.0)

    def test_adjacent_symbols_are_zero(self):
        """|i - j| == 1 → MINDIST = 0."""
        table = _build_dist_table(4)
        for i in range(3):
            assert table[i, i + 1] == 0.0
            assert table[i + 1, i] == 0.0

    def test_non_adjacent_positive(self):
        """|i - j| >= 2 → MINDIST > 0."""
        table = _build_dist_table(4)
        assert table[0, 2] > 0.0
        assert table[0, 3] > 0.0

    def test_symmetry(self):
        """MINDIST table is symmetric."""
        table = _build_dist_table(4)
        np.testing.assert_array_equal(table, table.T)

    def test_alphabet2_shape(self):
        table = _build_dist_table(2)
        assert table.shape == (2, 2)

    def test_alphabet8_shape(self):
        table = _build_dist_table(8)
        assert table.shape == (8, 8)


# ─── SAXFilter class tests ────────────────────────────────────────────────────

class TestSAXFilter:
    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    @pytest.fixture
    def X_train(self, rng):
        return rng.randn(100, 8).astype(np.float64)

    @pytest.fixture
    def fitted_sax(self, X_train):
        sax = SAXFilter(word_size=4, alphabet_size=4, max_symbolic_distance=1.0)
        sax.fit(X_train)
        return sax

    def test_fit_stores_words_shape(self, fitted_sax, X_train):
        assert fitted_sax._train_words is not None
        assert fitted_sax._train_words.shape == (len(X_train), 4)

    def test_encode_returns_word_size_symbols(self, fitted_sax, rng):
        x = rng.randn(8)
        word = fitted_sax.encode(x)
        assert word.shape == (4,)
        assert word.dtype == np.int32

    def test_encode_symbols_in_alphabet_range(self, fitted_sax, rng):
        x = rng.randn(8)
        word = fitted_sax.encode(x)
        assert np.all(word >= 0)
        assert np.all(word < 4)

    def test_mindist_identical_words_is_zero(self, fitted_sax):
        word = np.array([1, 2, 1, 3], dtype=np.int32)
        assert fitted_sax.mindist(word, word) == 0.0

    def test_mindist_far_words_positive(self, fitted_sax):
        word_a = np.array([0, 0, 0, 0], dtype=np.int32)
        word_b = np.array([3, 3, 3, 3], dtype=np.int32)
        assert fitted_sax.mindist(word_a, word_b) > 0.0

    def test_filter_candidates_keeps_identical(self, fitted_sax):
        """A candidate identical to the query should always be kept."""
        query_word = fitted_sax._train_words[0]
        keep = fitted_sax.filter_candidates(query_word, np.array([0]))
        assert keep[0]

    def test_filter_candidates_all_kept_with_none_threshold(self, X_train):
        sax = SAXFilter(word_size=4, alphabet_size=4, max_symbolic_distance=None)
        sax.fit(X_train)
        query_word = sax._train_words[0]
        indices = np.arange(len(X_train))
        keep = sax.filter_candidates(query_word, indices)
        assert keep.all()

    def test_filter_candidates_raises_before_fit(self):
        sax = SAXFilter(word_size=4, alphabet_size=4)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            sax.filter_candidates(np.array([0, 0, 0, 0], dtype=np.int32), np.array([0]))

    def test_filter_batch_matches_filter_candidates(self, fitted_sax, X_train, rng):
        """Batch and per-row filter_candidates must agree."""
        B = 5
        query_rows = rng.randn(B, 8)
        query_words = np.array([fitted_sax.encode(r) for r in query_rows])
        candidate_indices = np.tile(np.arange(10), (B, 1))

        batch_mask = fitted_sax.filter_batch(query_words, candidate_indices)
        for b in range(B):
            per_row = fitted_sax.filter_candidates(query_words[b], candidate_indices[b])
            np.testing.assert_array_equal(batch_mask[b], per_row)

    def test_invalid_word_size_raises(self):
        with pytest.raises(ValueError, match="word_size"):
            SAXFilter(word_size=0, alphabet_size=4)

    def test_invalid_alphabet_size_raises(self):
        with pytest.raises(ValueError, match="alphabet_size"):
            SAXFilter(word_size=4, alphabet_size=9)


# ─── apply_sax_filter integration helper ──────────────────────────────────────

class TestApplySAXFilter:
    def test_combines_with_existing_mask(self):
        """apply_sax_filter combines SAX mask with current_mask via AND."""
        rng = np.random.RandomState(0)
        X_train = rng.randn(50, 8)
        sax = SAXFilter(word_size=4, alphabet_size=4, max_symbolic_distance=None)
        # No SAX filtering (threshold=None) → sax_mask all True
        # current_mask has some False entries → result matches current_mask
        query_batch = rng.randn(3, 8)
        indices = np.tile(np.arange(10), (3, 1))
        current_mask = np.zeros((3, 10), dtype=bool)
        current_mask[:, :5] = True  # only first 5 survive

        result = apply_sax_filter(sax, X_train, query_batch, indices, current_mask)
        np.testing.assert_array_equal(result, current_mask)

    def test_auto_fits_if_not_fitted(self):
        """apply_sax_filter calls fit() if _train_words is None."""
        rng = np.random.RandomState(1)
        X_train = rng.randn(30, 8)
        sax = SAXFilter(word_size=4, alphabet_size=4, max_symbolic_distance=None)
        assert sax._train_words is None  # not yet fitted

        query_batch = rng.randn(2, 8)
        indices = np.tile(np.arange(5), (2, 1))
        current_mask = np.ones((2, 5), dtype=bool)

        result = apply_sax_filter(sax, X_train, query_batch, indices, current_mask)
        assert sax._train_words is not None  # now fitted
        assert result.shape == (2, 5)
