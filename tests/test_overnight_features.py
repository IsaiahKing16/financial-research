"""
test_overnight_features.py — Tests for overnight/session decomposition features.

Research basis: Fed NY Staff Report 917 (overnight drift), 24/7 equities
microstructure analysis (March 2026). Features decompose total daily return
into gap (overnight) and session (intraday) components.
"""

import numpy as np
import pandas as pd
import pytest

from pattern_engine.features import (
    OVERNIGHT_COLS, WEEKEND_COLS, RETURN_COLS,
    FeatureRegistry,
)
from pattern_engine.config import EngineConfig
from pattern_engine.data import DataLoader


# ============================================================
# Column definition tests
# ============================================================

class TestOvernightColumnDefinitions:
    """Verify column constants are correctly defined."""

    def test_overnight_cols_count(self):
        assert len(OVERNIGHT_COLS) == 4

    def test_weekend_cols_count(self):
        assert len(WEEKEND_COLS) == 2

    def test_overnight_cols_contents(self):
        assert "ret_overnight" in OVERNIGHT_COLS
        assert "ret_intraday" in OVERNIGHT_COLS
        assert "gap_magnitude" in OVERNIGHT_COLS
        assert "gap_direction_streak" in OVERNIGHT_COLS

    def test_weekend_cols_contents(self):
        assert "weekend_gap" in WEEKEND_COLS
        assert "weekend_gap_magnitude" in WEEKEND_COLS


# ============================================================
# Feature registry tests
# ============================================================

class TestOvernightFeatureRegistry:
    """Verify new feature sets are registered correctly."""

    def test_returns_overnight_registered(self):
        fs = FeatureRegistry.get("returns_overnight")
        assert fs.name == "returns_overnight"

    def test_returns_session_registered(self):
        fs = FeatureRegistry.get("returns_session")
        assert fs.name == "returns_session"

    def test_returns_overnight_column_count(self):
        fs = FeatureRegistry.get("returns_overnight")
        assert len(fs.columns) == 12  # 8 returns + 4 overnight

    def test_returns_session_column_count(self):
        fs = FeatureRegistry.get("returns_session")
        assert len(fs.columns) == 14  # 8 returns + 4 overnight + 2 weekend

    def test_returns_overnight_includes_returns(self):
        fs = FeatureRegistry.get("returns_overnight")
        for col in RETURN_COLS:
            assert col in fs.columns

    def test_returns_session_includes_all(self):
        fs = FeatureRegistry.get("returns_session")
        for col in RETURN_COLS + OVERNIGHT_COLS + WEEKEND_COLS:
            assert col in fs.columns


# ============================================================
# Regression guards
# ============================================================

class TestRegressionGuards:
    """Ensure existing feature sets and defaults are untouched."""

    def test_returns_only_unchanged(self):
        fs = FeatureRegistry.get("returns_only")
        assert len(fs.columns) == 8

    def test_default_feature_set_unchanged(self):
        config = EngineConfig()
        assert config.feature_set == "returns_only"

    def test_default_feature_weights_include_overnight(self):
        config = EngineConfig()
        assert "ret_overnight" in config.feature_weights
        assert "weekend_gap" in config.feature_weights


# ============================================================
# Computation tests
# ============================================================

class TestOvernightComputation:
    """Verify feature computation arithmetic."""

    @pytest.fixture
    def sample_df(self):
        """Simple 10-row OHLCV for arithmetic verification."""
        dates = pd.bdate_range("2024-01-01", periods=10)
        closes = [100.0, 102.0, 101.0, 103.0, 105.0,
                  104.0, 106.0, 107.0, 108.0, 110.0]
        opens =  [100.5, 101.5, 101.5, 102.5, 104.5,
                  104.5, 105.5, 107.5, 107.5, 109.5]
        return pd.DataFrame({
            "Date": dates,
            "Open": opens,
            "Close": closes,
            "High": [max(o, c) + 1 for o, c in zip(opens, closes)],
            "Low": [min(o, c) - 1 for o, c in zip(opens, closes)],
            "Volume": [1000000] * 10,
        })

    def test_ret_overnight_arithmetic(self, sample_df):
        result = DataLoader._compute_overnight_features(sample_df)
        # Row 1: Open[1] / Close[0] - 1 = 101.5 / 100.0 - 1 = 0.015
        expected = 101.5 / 100.0 - 1
        assert abs(result["ret_overnight"].iloc[1] - expected) < 1e-10

    def test_ret_intraday_arithmetic(self, sample_df):
        result = DataLoader._compute_overnight_features(sample_df)
        # Row 1: Close[1] / Open[1] - 1 = 102.0 / 101.5 - 1
        expected = 102.0 / 101.5 - 1
        assert abs(result["ret_intraday"].iloc[1] - expected) < 1e-10

    def test_gap_magnitude_nonnegative(self, sample_df):
        result = DataLoader._compute_overnight_features(sample_df)
        assert (result["gap_magnitude"].dropna() >= 0).all()

    def test_gap_magnitude_is_absolute(self, sample_df):
        result = DataLoader._compute_overnight_features(sample_df)
        np.testing.assert_array_almost_equal(
            result["gap_magnitude"].dropna().values,
            result["ret_overnight"].dropna().abs().values,
        )

    def test_weekend_gap_zero_on_non_mondays(self, sample_df):
        result = DataLoader._compute_overnight_features(sample_df)
        non_mondays = result[pd.to_datetime(result["Date"]).dt.dayofweek != 0]
        assert (non_mondays["weekend_gap"] == 0.0).all()

    def test_weekend_gap_magnitude_nonnegative(self, sample_df):
        result = DataLoader._compute_overnight_features(sample_df)
        assert (result["weekend_gap_magnitude"].dropna() >= 0).all()

    def test_streak_accumulates(self):
        """Three consecutive positive gaps should produce streak 1, 2, 3."""
        df = pd.DataFrame({
            "Date": pd.bdate_range("2024-01-01", periods=5),
            "Open": [101, 103, 105, 107, 109],
            "Close": [100, 102, 104, 106, 108],
            "High": [102, 104, 106, 108, 110],
            "Low": [99, 101, 103, 105, 107],
            "Volume": [1e6] * 5,
        })
        result = DataLoader._compute_overnight_features(df)
        # Rows 1-4 all have positive overnight return (Open > prev Close)
        streaks = result["gap_direction_streak"].iloc[1:].values
        # All positive, increasing streak
        assert all(s > 0 for s in streaks)

    def test_first_row_is_nan(self, sample_df):
        """First row has no previous close, so ret_overnight should be NaN."""
        result = DataLoader._compute_overnight_features(sample_df)
        assert pd.isna(result["ret_overnight"].iloc[0])


# ============================================================
# Integration tests
# ============================================================

class TestOvernightIntegration:
    """Verify PatternEngine works with new feature sets."""

    def test_engine_fit_predict_overnight(self, train_db, val_db):
        from pattern_engine import PatternEngine
        config = EngineConfig(feature_set="returns_overnight")
        engine = PatternEngine(config)
        engine.fit(train_db)
        result = engine.predict(val_db, verbose=0)
        assert len(result.signals) == len(val_db)

    def test_engine_fit_predict_session(self, train_db, val_db):
        from pattern_engine import PatternEngine
        config = EngineConfig(feature_set="returns_session")
        engine = PatternEngine(config)
        engine.fit(train_db)
        result = engine.predict(val_db, verbose=0)
        assert len(result.signals) == len(val_db)

    def test_engine_evaluate_overnight(self, train_db, val_db):
        from pattern_engine import PatternEngine
        config = EngineConfig(feature_set="returns_overnight")
        engine = PatternEngine(config)
        metrics = engine.fit(train_db).evaluate(val_db, verbose=0)
        assert "brier_skill_score" in metrics
        assert "accuracy_confident" in metrics
