"""
test_ib_compression.py — Unit tests for Information Bottleneck pilot (SLE-78).

Tests cover:
  - IBCompressor.fit(): stores projection, feature_importance
  - IBCompressor.transform(): output shape, d_out dimensionality
  - IBCompressor.fit_transform(): chained
  - top_features(): ranking order is valid
  - compare_bss_with_ib(): output dict structure (not BSS value — random data)
  - Edge cases: d_out >= D raises, d_out=1 works
"""

from __future__ import annotations

import numpy as np
import pytest

from pattern_engine.ib_compression import (
    IBCompressor,
    compare_bss_with_ib,
)


class TestIBCompressor:
    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    @pytest.fixture
    def data(self, rng):
        N, D = 200, 8
        X = rng.randn(N, D)
        y = rng.randint(0, 2, size=N).astype(float)
        return X, y

    @pytest.fixture
    def fitted_compressor(self, data):
        X, y = data
        c = IBCompressor(d_out=4)
        c.fit(X, y)
        return c

    def test_fit_stores_projection(self, fitted_compressor):
        assert fitted_compressor._projection is not None
        assert fitted_compressor._projection.shape == (8, 4)

    def test_fit_stores_feature_importance(self, fitted_compressor):
        assert fitted_compressor.feature_importance is not None
        assert fitted_compressor.feature_importance.shape == (8,)

    def test_feature_importance_non_negative(self, fitted_compressor):
        assert np.all(fitted_compressor.feature_importance >= 0.0)

    def test_transform_output_shape(self, fitted_compressor, data):
        X, _ = data
        Z = fitted_compressor.transform(X)
        assert Z.shape == (len(X), 4)

    def test_fit_transform_matches_separate(self, data):
        X, y = data
        c1 = IBCompressor(d_out=4)
        Z1 = c1.fit_transform(X, y)
        c2 = IBCompressor(d_out=4)
        c2.fit(X, y)
        Z2 = c2.transform(X)
        np.testing.assert_array_equal(Z1, Z2)

    def test_d_out_1_works(self, data):
        X, y = data
        c = IBCompressor(d_out=1)
        Z = c.fit_transform(X, y)
        assert Z.shape == (len(X), 1)

    def test_d_out_gte_d_raises(self, data):
        X, y = data
        with pytest.raises(ValueError, match="d_out"):
            IBCompressor(d_out=8).fit(X, y)

    def test_d_out_zero_raises(self):
        with pytest.raises(ValueError, match="d_out"):
            IBCompressor(d_out=0)

    def test_transform_raises_before_fit(self, data):
        X, _ = data
        c = IBCompressor(d_out=4)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            c.transform(X)

    def test_top_features_returns_all_names(self, fitted_compressor):
        names = [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]
        ranked = fitted_compressor.top_features(names)
        assert len(ranked) == 8
        assert set(ranked) == set(names)

    def test_top_features_raises_before_fit(self):
        c = IBCompressor(d_out=4)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            c.top_features(["a", "b"])

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            IBCompressor(d_out=4, method="nonlinear_ib")


class TestCompareBSSWithIB:
    def test_output_dict_structure(self):
        rng = np.random.RandomState(0)
        N_tr, N_val, D = 200, 50, 8
        X_train = rng.randn(N_tr, D)
        y_train = rng.randint(0, 2, N_tr).astype(float)
        X_val = rng.randn(N_val, D)
        y_val = rng.randint(0, 2, N_val).astype(float)

        result = compare_bss_with_ib(X_train, y_train, X_val, y_val, d_out=4)
        assert "bss_baseline" in result
        assert "bss_ib" in result
        assert "delta_bss" in result
        assert result["d_in"] == D
        assert result["d_out"] == 4

    def test_delta_is_ib_minus_baseline(self):
        rng = np.random.RandomState(1)
        X_tr = rng.randn(150, 8)
        y_tr = rng.randint(0, 2, 150).astype(float)
        X_val = rng.randn(50, 8)
        y_val = rng.randint(0, 2, 50).astype(float)

        result = compare_bss_with_ib(X_tr, y_tr, X_val, y_val, d_out=4)
        expected_delta = round(result["bss_ib"] - result["bss_baseline"], 6)
        assert result["delta_bss"] == pytest.approx(expected_delta, abs=1e-9)
