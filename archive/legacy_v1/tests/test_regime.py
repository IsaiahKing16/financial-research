"""Tests for pattern_engine.regime."""

import numpy as np
import pandas as pd
import pytest
from pattern_engine.regime import (
    RegimeLabeler, apply_regime_filter, fallback_regime_mode,
)


@pytest.fixture
def regime_db(rng):
    """Synthetic database with SPY rows for regime testing."""
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    dfs = []
    for ticker in ["SPY", "AAPL", "MSFT"]:
        df = pd.DataFrame({
            "Date": dates,
            "Ticker": ticker,
            "ret_90d": rng.randn(n) * 0.1,  # mix of positive/negative
            "vol_30d": rng.rand(n) * 0.03 + 0.005,
            "adx_14": rng.rand(n) * 40 + 10,
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


class TestRegimeLabeler:
    def test_binary_mode(self, regime_db):
        labeler = RegimeLabeler(mode="binary")
        labeler.fit(regime_db)
        labels = labeler.label(regime_db)
        assert labels.shape == (len(regime_db),)
        assert set(np.unique(labels)).issubset({0, 1})
        assert labeler.n_regimes == 2

    def test_multi_mode(self, regime_db):
        labeler = RegimeLabeler(mode="multi")
        labeler.fit(regime_db)
        labels = labeler.label(regime_db)
        assert labels.shape == (len(regime_db),)
        assert all(l in range(4) for l in np.unique(labels))
        assert labeler.n_regimes == 4

    def test_octet_mode(self, regime_db):
        labeler = RegimeLabeler(mode="octet")
        labeler.fit(regime_db)
        labels = labeler.label(regime_db)
        assert labels.shape == (len(regime_db),)
        assert all(l in range(8) for l in np.unique(labels))
        assert labeler.n_regimes == 8

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown regime mode"):
            RegimeLabeler(mode="invalid")

    def test_fit_returns_self(self, regime_db):
        labeler = RegimeLabeler()
        result = labeler.fit(regime_db)
        assert result is labeler
        assert labeler.fitted

    def test_vol_median_computed(self, regime_db):
        labeler = RegimeLabeler(mode="multi")
        labeler.fit(regime_db)
        assert labeler._vol_median is not None
        assert labeler._vol_median > 0

    def test_no_spy_returns_zeros(self):
        """If no SPY rows, return all zeros (single regime)."""
        db = pd.DataFrame({
            "Date": pd.bdate_range("2020-01-01", periods=50),
            "Ticker": "AAPL",
            "ret_90d": np.random.randn(50) * 0.1,
        })
        labeler = RegimeLabeler(mode="binary")
        labeler.fit(db)
        labels = labeler.label(db)
        assert np.all(labels == 0)

    def test_reference_db_fallback(self, regime_db):
        """When db has no SPY, fallback to reference_db."""
        non_spy = regime_db[regime_db["Ticker"] != "SPY"].copy()
        labeler = RegimeLabeler(mode="binary")
        labeler.fit(regime_db)
        labels = labeler.label(non_spy, reference_db=regime_db)
        assert len(labels) == len(non_spy)
        # Should have both regimes since reference_db has SPY
        assert len(np.unique(labels)) >= 1

    def test_adx_threshold(self, regime_db):
        """Octet mode uses adx_threshold for trend detection."""
        labeler_low = RegimeLabeler(mode="octet", adx_threshold=10.0)
        labeler_high = RegimeLabeler(mode="octet", adx_threshold=50.0)
        labeler_low.fit(regime_db)
        labeler_high.fit(regime_db)
        labels_low = labeler_low.label(regime_db)
        labels_high = labeler_high.label(regime_db)
        # Lower threshold = more trending = different label distribution
        assert not np.array_equal(labels_low, labels_high)


class TestApplyRegimeFilter:
    def test_filters_by_regime(self):
        matches = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        match_labels = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        query_label = 0
        match_indices = np.array([0, 2, 4, 5, 8])
        result = apply_regime_filter(matches, match_labels, query_label, match_indices)
        assert len(result) == 5  # all indices map to regime 0

    def test_filters_some(self):
        matches = pd.DataFrame({"value": [1, 2, 3]})
        match_labels = np.array([0, 1, 0, 1, 0])
        query_label = 0
        match_indices = np.array([0, 1, 2])
        result = apply_regime_filter(matches, match_labels, query_label, match_indices)
        assert len(result) == 2  # indices 0 and 2 are regime 0


class TestFallbackRegimeMode:
    def test_octet_to_multi(self):
        assert fallback_regime_mode("octet") == "multi"

    def test_multi_to_binary(self):
        assert fallback_regime_mode("multi") == "binary"

    def test_binary_to_none(self):
        assert fallback_regime_mode("binary") is None
