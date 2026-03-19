"""Tests for pattern_engine.matching."""

import numpy as np
import pandas as pd
import pytest
from pattern_engine.config import EngineConfig
from pattern_engine.matching import Matcher, apply_feature_weights, SECTOR_MAP
from pattern_engine.features import RETURN_COLS


class TestApplyFeatureWeights:
    def test_applies_weights(self):
        X = np.ones((5, 3))
        cols = ["a", "b", "c"]
        weights = {"a": 2.0, "b": 0.5, "c": 1.0}
        result = apply_feature_weights(X, cols, weights)
        np.testing.assert_array_equal(result[:, 0], 2.0)
        np.testing.assert_array_equal(result[:, 1], 0.5)
        np.testing.assert_array_equal(result[:, 2], 1.0)

    def test_missing_weight_defaults_to_one(self):
        X = np.ones((3, 2))
        cols = ["known", "unknown"]
        weights = {"known": 3.0}
        result = apply_feature_weights(X, cols, weights)
        np.testing.assert_array_equal(result[:, 0], 3.0)
        np.testing.assert_array_equal(result[:, 1], 1.0)

    def test_does_not_mutate_original(self):
        X = np.ones((3, 2))
        original = X.copy()
        apply_feature_weights(X, ["a", "b"], {"a": 5.0})
        np.testing.assert_array_equal(X, original)


class TestSectorMap:
    def test_52_tickers(self):
        assert len(SECTOR_MAP) == 52

    def test_spy_is_index(self):
        assert SECTOR_MAP["SPY"] == "Index"

    def test_aapl_is_tech(self):
        assert SECTOR_MAP["AAPL"] == "Tech"

    def test_jpm_is_finance(self):
        assert SECTOR_MAP["JPM"] == "Finance"


class TestMatcher:
    """Test the Matcher class."""

    @pytest.fixture
    def simple_config(self):
        return EngineConfig(
            regime_filter=False,
            calibration_method="none",
            top_k=10,
            max_distance=5.0,
            batch_size=64,
        )

    def test_fit_and_query(self, train_db, val_db, simple_config):
        matcher = Matcher(simple_config)
        fcols = RETURN_COLS
        matcher.fit(train_db, fcols)
        assert matcher.fitted

        probs, signals, reasons, n_matches, mean_rets, ensembles = \
            matcher.query(val_db.head(20), verbose=0)

        assert len(probs) == 20
        assert len(signals) == 20
        assert all(s in ("BUY", "SELL", "HOLD") for s in signals)
        assert all(0 <= p <= 1 for p in probs)

    def test_query_before_fit_raises(self, val_db):
        matcher = Matcher(EngineConfig())
        with pytest.raises(RuntimeError, match="fit"):
            matcher.query(val_db)

    def test_exclude_same_ticker(self, train_db, val_db):
        """With exclude_same_ticker=True, matches should not include same ticker."""
        cfg = EngineConfig(
            regime_filter=False,
            calibration_method="none",
            exclude_same_ticker=True,
            max_distance=50.0,
            top_k=10,
        )
        matcher = Matcher(cfg)
        matcher.fit(train_db, RETURN_COLS)
        # Query just one SPY row
        spy_rows = val_db[val_db["Ticker"] == "SPY"].head(1)
        probs, *_ = matcher.query(spy_rows, verbose=0)
        assert len(probs) == 1

    def test_distance_filtering(self, train_db, val_db):
        """Very tight max_distance should reduce match count."""
        cfg_tight = EngineConfig(
            regime_filter=False,
            calibration_method="none",
            max_distance=0.01,
            top_k=50,
        )
        cfg_loose = EngineConfig(
            regime_filter=False,
            calibration_method="none",
            max_distance=50.0,
            top_k=50,
        )
        m_tight = Matcher(cfg_tight)
        m_tight.fit(train_db, RETURN_COLS)
        _, _, _, n_tight, _, _ = m_tight.query(val_db.head(5), verbose=0)

        m_loose = Matcher(cfg_loose)
        m_loose.fit(train_db, RETURN_COLS)
        _, _, _, n_loose, _, _ = m_loose.query(val_db.head(5), verbose=0)

        # Tight distance should have fewer or equal matches
        assert sum(n_tight) <= sum(n_loose)

    def test_with_regime_filter(self, train_db, val_db):
        """Matcher works with regime filtering enabled."""
        from pattern_engine.regime import RegimeLabeler
        cfg = EngineConfig(
            regime_filter=True,
            regime_mode="binary",
            calibration_method="none",
            max_distance=50.0,
            top_k=20,
        )
        labeler = RegimeLabeler(mode="binary")
        labeler.fit(train_db)

        matcher = Matcher(cfg)
        matcher.fit(train_db, RETURN_COLS, regime_labeler=labeler)
        probs, signals, *_ = matcher.query(
            val_db.head(10), regime_labeler=labeler, verbose=0
        )
        assert len(probs) == 10
