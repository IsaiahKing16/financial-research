"""
tests/unit/test_candlestick.py — Unit tests for pattern_engine.candlestick.

Coverage targets from HANDOFF_phase6-candlestick-features.md §6.

Tests:
    1.  test_bullish_candle               — known-value verification
    2.  test_bearish_candle               — same proportions, direction=0
    3.  test_full_body_no_wicks           — body=1.0, wicks=0, pos=0, dir=1
    4.  test_zero_range_doji              — all proportions 0.0, direction=1
    5.  test_proportion_sum_identity      — upper+body+lower ≈ 1.0 (atol=1e-9)
    6.  test_values_in_range              — 100 random rows all in [0,1] / {0,1}
    7.  test_3day_composite_construction  — verify composite OHLC math
    8.  test_5day_composite_construction  — verify composite OHLC math
    9.  test_composite_nan_at_start       — NaN at start of history
    10. test_multi_ticker_isolation       — rolling windows don't cross boundaries
    11. test_output_column_names          — exactly 15 columns, correct names
    12. test_output_index_preserved       — DatetimeIndex in = DatetimeIndex out
    13. test_negative_range_clamped       — H<L (bad data) → proportions=0.0, warn
    14. test_nan_ohlc_propagates          — NaN high → all 5 features NaN
    15. test_deterministic                — same input twice → identical output
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pattern_engine.candlestick import (
    CANDLE_COLS,
    FEATURE_NAMES,
    TIMEFRAMES,
    compute_candlestick_features,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_df(rows: list[dict], ticker: str = "AAA") -> pd.DataFrame:
    """Build a minimal OHLC DataFrame for testing."""
    df = pd.DataFrame(rows)
    df["Ticker"] = ticker
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="B")
    return df


def _single_row(o, h, l, c, ticker="AAA") -> pd.DataFrame:
    return _make_df([{"Open": o, "High": h, "Low": l, "Close": c}], ticker=ticker)


def _get_1d(result: pd.DataFrame, feat: str):
    """Extract 1-day feature value from the result (single row)."""
    return result[f"candle_1d_{feat}"].iloc[0]


# ─── Test classes ──────────────────────────────────────────────────────────────

class TestKnownValues:
    """Tests 1–5: verify feature math on hand-computed examples."""

    def test_bullish_candle(self):
        """O=100, H=110, L=95, C=108 → body=8/15, upper=2/15, lower=5/15,
        body_position=5/15, direction=1."""
        df = _single_row(100, 110, 95, 108)
        result = compute_candlestick_features(df)

        assert pytest.approx(_get_1d(result, "body_to_range"), abs=1e-9) == 8 / 15
        assert pytest.approx(_get_1d(result, "upper_wick"),    abs=1e-9) == 2 / 15
        assert pytest.approx(_get_1d(result, "lower_wick"),    abs=1e-9) == 5 / 15
        assert pytest.approx(_get_1d(result, "body_position"), abs=1e-9) == 5 / 15
        assert _get_1d(result, "direction") == 1.0

    def test_bearish_candle(self):
        """O=108, H=110, L=95, C=100 → same proportions, direction=0."""
        df = _single_row(108, 110, 95, 100)
        result = compute_candlestick_features(df)

        assert pytest.approx(_get_1d(result, "body_to_range"), abs=1e-9) == 8 / 15
        assert pytest.approx(_get_1d(result, "upper_wick"),    abs=1e-9) == 2 / 15
        assert pytest.approx(_get_1d(result, "lower_wick"),    abs=1e-9) == 5 / 15
        assert pytest.approx(_get_1d(result, "body_position"), abs=1e-9) == 5 / 15
        assert _get_1d(result, "direction") == 0.0

    def test_full_body_no_wicks(self):
        """O=100, H=110, L=100, C=110 → body=1.0, upper=0, lower=0, pos=0, dir=1."""
        df = _single_row(100, 110, 100, 110)
        result = compute_candlestick_features(df)

        assert pytest.approx(_get_1d(result, "body_to_range"), abs=1e-9) == 1.0
        assert pytest.approx(_get_1d(result, "upper_wick"),    abs=1e-9) == 0.0
        assert pytest.approx(_get_1d(result, "lower_wick"),    abs=1e-9) == 0.0
        assert pytest.approx(_get_1d(result, "body_position"), abs=1e-9) == 0.0
        assert _get_1d(result, "direction") == 1.0

    def test_zero_range_doji(self):
        """O=H=L=C=100 → all proportions 0.0, direction=1. No ZeroDivisionError."""
        df = _single_row(100, 100, 100, 100)
        result = compute_candlestick_features(df)

        assert _get_1d(result, "body_to_range") == 0.0
        assert _get_1d(result, "upper_wick")    == 0.0
        assert _get_1d(result, "lower_wick")    == 0.0
        assert _get_1d(result, "body_position") == 0.0
        assert _get_1d(result, "direction")     == 1.0

    def test_proportion_sum_identity(self):
        """For any valid OHLC: upper_wick + body_to_range + lower_wick ≈ 1.0."""
        df = _single_row(100, 110, 95, 108)
        result = compute_candlestick_features(df)

        s = (_get_1d(result, "upper_wick") +
             _get_1d(result, "body_to_range") +
             _get_1d(result, "lower_wick"))
        assert abs(s - 1.0) < 1e-9


class TestRandomValues:
    """Test 6: 100 random rows — all outputs in expected ranges."""

    def test_values_in_range(self):
        """100 random rows → all proportions ∈ [0,1], direction ∈ {0,1}."""
        rng = np.random.RandomState(42)
        n = 100
        closes  = rng.uniform(50, 200, n)
        spreads = rng.uniform(0.5, 5.0, n)
        opens   = closes + rng.uniform(-spreads, spreads, n)
        highs   = np.maximum(opens, closes) + rng.uniform(0, spreads, n)
        lows    = np.minimum(opens, closes) - rng.uniform(0, spreads, n)

        df = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows, "Close": closes,
            "Ticker": "AAA",
        })
        df.index = pd.date_range("2020-01-01", periods=n, freq="B")

        result = compute_candlestick_features(df)

        for feat in ["body_to_range", "upper_wick", "lower_wick", "body_position"]:
            col = f"candle_1d_{feat}"
            vals = result[col].dropna().values
            assert (vals >= 0.0).all(), f"{col} has values < 0"
            assert (vals <= 1.0).all(), f"{col} has values > 1"

        dir_vals = result["candle_1d_direction"].dropna().values
        assert set(dir_vals).issubset({0.0, 1.0}), "direction values outside {0, 1}"


class TestCompositeConstruction:
    """Tests 7–9: multi-timeframe composite OHLC construction."""

    def test_3day_composite_construction(self):
        """5-row DataFrame → verify composite OHLC math at t=2 (first valid 3d row)."""
        rows = [
            {"Open": 100, "High": 105, "Low":  98, "Close": 103},  # t=0
            {"Open": 103, "High": 108, "Low": 101, "Close": 107},  # t=1
            {"Open": 107, "High": 112, "Low": 104, "Close": 110},  # t=2 ← first valid
            {"Open": 110, "High": 115, "Low": 108, "Close": 113},  # t=3
            {"Open": 113, "High": 118, "Low": 111, "Close": 116},  # t=4
        ]
        df = _make_df(rows)
        result = compute_candlestick_features(df)

        # t=0,1 must be NaN for 3d (insufficient lookback)
        assert result["candle_3d_body_to_range"].iloc[0] != result["candle_3d_body_to_range"].iloc[0]  # isnan
        assert result["candle_3d_body_to_range"].iloc[1] != result["candle_3d_body_to_range"].iloc[1]

        # t=2: composite O = open[0]=100, H = max(105,108,112)=112,
        #                    L = min(98,101,104)=98, C = close[2]=110
        comp_o = 100
        comp_h = 112
        comp_l = 98
        comp_c = 110
        body    = abs(comp_c - comp_o)  # 10
        rng_    = comp_h - comp_l       # 14
        expected_body = body / rng_     # 10/14

        assert pytest.approx(
            result["candle_3d_body_to_range"].iloc[2], abs=1e-9
        ) == expected_body

    def test_5day_composite_construction(self):
        """7-row DataFrame → verify composite OHLC at t=4 (first valid 5d row)."""
        rows = [
            {"Open": 100, "High": 105, "Low":  98, "Close": 103},  # t=0
            {"Open": 103, "High": 108, "Low": 101, "Close": 107},  # t=1
            {"Open": 107, "High": 112, "Low": 104, "Close": 110},  # t=2
            {"Open": 110, "High": 115, "Low": 108, "Close": 113},  # t=3
            {"Open": 113, "High": 118, "Low": 111, "Close": 116},  # t=4 ← first valid
            {"Open": 116, "High": 121, "Low": 114, "Close": 119},  # t=5
            {"Open": 119, "High": 124, "Low": 117, "Close": 122},  # t=6
        ]
        df = _make_df(rows)
        result = compute_candlestick_features(df)

        # t=0..3 must be NaN for 5d
        for i in range(4):
            val = result["candle_5d_body_to_range"].iloc[i]
            assert val != val, f"t={i}: expected NaN for 5d feature"

        # t=4: composite O=open[0]=100, H=max(105,108,112,115,118)=118,
        #               L=min(98,101,104,108,111)=98, C=close[4]=116
        comp_o = 100; comp_h = 118; comp_l = 98; comp_c = 116
        body   = abs(comp_c - comp_o)  # 16
        rng_   = comp_h - comp_l       # 20
        expected_body = body / rng_    # 0.8

        assert pytest.approx(
            result["candle_5d_body_to_range"].iloc[4], abs=1e-9
        ) == expected_body

    def test_composite_nan_at_start(self):
        """3-row DataFrame: 5d=NaN everywhere, 3d=NaN for t<2, 1d always valid."""
        rows = [
            {"Open": 100, "High": 105, "Low": 98, "Close": 103},
            {"Open": 103, "High": 108, "Low": 101, "Close": 107},
            {"Open": 107, "High": 112, "Low": 104, "Close": 110},
        ]
        df = _make_df(rows)
        result = compute_candlestick_features(df)

        # 1d: all rows valid
        assert result["candle_1d_body_to_range"].notna().all()

        # 3d: first 2 rows NaN, t=2 valid
        assert pd.isna(result["candle_3d_body_to_range"].iloc[0])
        assert pd.isna(result["candle_3d_body_to_range"].iloc[1])
        assert pd.notna(result["candle_3d_body_to_range"].iloc[2])

        # 5d: all rows NaN (need 5 rows but only 3)
        assert result["candle_5d_body_to_range"].isna().all()


class TestMultiTicker:
    """Test 10: rolling windows must not cross ticker boundaries."""

    def test_multi_ticker_isolation(self):
        """2 tickers × 10 rows: composite windows must not bleed between tickers."""
        n = 10
        rng = np.random.RandomState(7)

        def make_ticker_rows(price_start):
            prices = price_start + np.arange(n) * 0.5
            return [
                {
                    "Open": prices[i],
                    "High": prices[i] + 2,
                    "Low":  prices[i] - 2,
                    "Close": prices[i] + 1,
                }
                for i in range(n)
            ]

        rows_a = [dict(r, Ticker="AAA") for r in make_ticker_rows(100)]
        rows_b = [dict(r, Ticker="BBB") for r in make_ticker_rows(200)]

        # Interleave tickers to stress-test groupby isolation
        rows = [val for pair in zip(rows_a, rows_b) for val in pair]
        df = pd.DataFrame(rows)
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="B")

        result = compute_candlestick_features(df)

        # Build single-ticker results for comparison
        df_a = pd.DataFrame(rows_a)
        df_a.index = pd.date_range("2024-01-01", periods=n, freq="B")
        result_a = compute_candlestick_features(df_a)

        # 3d feature at row 2 of ticker AAA must match the single-ticker result
        aa_rows = result[df["Ticker"] == "AAA"]
        assert pytest.approx(
            aa_rows["candle_3d_body_to_range"].iloc[2], abs=1e-9
        ) == result_a["candle_3d_body_to_range"].iloc[2]


class TestOutputContract:
    """Tests 11–12: column names and index preservation."""

    def test_output_column_names(self):
        """Output has exactly 15 columns with correct naming convention."""
        df = _make_df([{"Open": 100, "High": 105, "Low": 98, "Close": 103}])
        result = compute_candlestick_features(df)

        assert list(result.columns) == CANDLE_COLS
        assert len(result.columns) == 15

        # Verify naming convention: candle_{tf}d_{feat}
        for tf in TIMEFRAMES:
            for feat in FEATURE_NAMES:
                assert f"candle_{tf}d_{feat}" in result.columns

    def test_output_index_preserved(self):
        """DatetimeIndex from input is preserved exactly in output."""
        idx = pd.date_range("2024-06-01", periods=5, freq="B")
        rows = [{"Open": 100 + i, "High": 105 + i, "Low": 98 + i, "Close": 103 + i}
                for i in range(5)]
        df = pd.DataFrame(rows, index=idx)
        df["Ticker"] = "AAA"

        result = compute_candlestick_features(df)
        pd.testing.assert_index_equal(result.index, idx)


class TestEdgeCases:
    """Tests 13–15: negative range, NaN propagation, determinism."""

    def test_negative_range_clamped(self):
        """H < L (bad data) → proportions=0.0 (zero-range treatment), warning emitted."""
        df = _single_row(o=100, h=95, l=100, c=100)  # h < l → bad data

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compute_candlestick_features(df)

        # Should have emitted a UserWarning about high < low
        assert any("high < low" in str(w.message) for w in caught), \
            "Expected UserWarning about high < low"

        # Proportions should be 0.0 (clamped to zero-range)
        assert _get_1d(result, "body_to_range") == 0.0
        assert _get_1d(result, "upper_wick")    == 0.0
        assert _get_1d(result, "lower_wick")    == 0.0

    def test_nan_ohlc_propagates(self):
        """NaN in any OHLC field → all 5 features NaN for that row."""
        df = _make_df([
            {"Open": 100, "High": np.nan, "Low": 98, "Close": 103},  # NaN high
            {"Open": 103, "High": 108,    "Low": 101, "Close": 107},  # valid
        ])
        result = compute_candlestick_features(df)

        for feat in FEATURE_NAMES:
            val = result[f"candle_1d_{feat}"].iloc[0]
            assert pd.isna(val), f"candle_1d_{feat} should be NaN for row with NaN high"

        # Second row should be valid
        assert pd.notna(result["candle_1d_body_to_range"].iloc[1])

    def test_deterministic(self):
        """Same input twice → identical output."""
        rng = np.random.RandomState(99)
        n = 20
        closes = rng.uniform(50, 200, n)
        df = pd.DataFrame({
            "Open":   closes,
            "High":   closes + rng.uniform(0, 3, n),
            "Low":    closes - rng.uniform(0, 3, n),
            "Close":  closes + rng.uniform(-1, 1, n),
            "Ticker": "AAA",
        })
        df.index = pd.date_range("2024-01-01", periods=n, freq="B")

        result1 = compute_candlestick_features(df)
        result2 = compute_candlestick_features(df)

        pd.testing.assert_frame_equal(result1, result2)


class TestFeatureRegistry:
    """Test that FeatureRegistry correctly exposes returns_candle with 23 columns."""

    def test_returns_candle_has_23_columns(self):
        from pattern_engine.features import FeatureRegistry
        fs = FeatureRegistry.get("returns_candle")
        assert len(fs.columns) == 23, f"Expected 23, got {len(fs.columns)}"

    def test_returns_candle_starts_with_vol_norm(self):
        from pattern_engine.features import FeatureRegistry, VOL_NORM_COLS
        fs = FeatureRegistry.get("returns_candle")
        assert fs.columns[:8] == list(VOL_NORM_COLS)

    def test_returns_candle_ends_with_candle_cols(self):
        from pattern_engine.features import FeatureRegistry
        from pattern_engine.candlestick import CANDLE_COLS
        fs = FeatureRegistry.get("returns_candle")
        assert fs.columns[8:] == list(CANDLE_COLS)

    def test_returns_only_unchanged(self):
        """Default feature set still has 8 VOL_NORM_COLS (regression guard)."""
        from pattern_engine.features import FeatureRegistry, VOL_NORM_COLS
        fs = FeatureRegistry.get("returns_only")
        assert fs.columns == list(VOL_NORM_COLS)
        assert len(fs.columns) == 8
