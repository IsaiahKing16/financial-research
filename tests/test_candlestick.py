"""Tests for pattern_engine.candlestick."""

import numpy as np
import pandas as pd
import pytest
from pattern_engine.candlestick import (
    compute_candle_features, compute_composite_ohlc, add_candlestick_features,
)
from pattern_engine.features import CANDLE_1D_COLS, CANDLE_3D_COLS, CANDLE_5D_COLS


class TestComputeCandleFeatures:
    def test_bullish_candle(self):
        """Close > Open → direction = +1."""
        result = compute_candle_features(
            open_=np.array([100.0]),
            high=np.array([110.0]),
            low=np.array([95.0]),
            close=np.array([108.0]),
        )
        assert result["candle_1d_direction"][0] == 1.0
        assert 0 <= result["candle_1d_body_to_range"][0] <= 1
        assert 0 <= result["candle_1d_upper_wick"][0] <= 1
        assert 0 <= result["candle_1d_lower_wick"][0] <= 1

    def test_bearish_candle(self):
        """Close < Open → direction = -1."""
        result = compute_candle_features(
            open_=np.array([108.0]),
            high=np.array([110.0]),
            low=np.array([95.0]),
            close=np.array([100.0]),
        )
        assert result["candle_1d_direction"][0] == -1.0

    def test_doji(self):
        """Open ≈ Close → small body_to_range."""
        result = compute_candle_features(
            open_=np.array([100.0]),
            high=np.array([105.0]),
            low=np.array([95.0]),
            close=np.array([100.001]),
        )
        assert result["candle_1d_body_to_range"][0] < 0.01

    def test_marubozu(self):
        """Open = Low, Close = High → body_to_range ≈ 1, no wicks."""
        result = compute_candle_features(
            open_=np.array([95.0]),
            high=np.array([105.0]),
            low=np.array([95.0]),
            close=np.array([105.0]),
        )
        assert result["candle_1d_body_to_range"][0] > 0.99
        assert result["candle_1d_upper_wick"][0] < 0.01
        assert result["candle_1d_lower_wick"][0] < 0.01

    def test_custom_prefix(self):
        result = compute_candle_features(
            np.array([100.0]), np.array([110.0]),
            np.array([90.0]), np.array([105.0]),
            prefix="candle_3d",
        )
        assert "candle_3d_body_to_range" in result
        assert "candle_3d_direction" in result

    def test_batch_computation(self):
        n = 50
        result = compute_candle_features(
            np.random.rand(n) * 100 + 50,
            np.random.rand(n) * 100 + 60,
            np.random.rand(n) * 100 + 40,
            np.random.rand(n) * 100 + 50,
        )
        assert all(len(v) == n for v in result.values())


class TestComputeCompositeOHLC:
    def test_3day_composite(self, small_ohlcv):
        o, h, l, c = compute_composite_ohlc(small_ohlcv, 3)
        assert len(o) == len(small_ohlcv)
        # First 2 rows should be NaN (insufficient lookback)
        assert np.isnan(o[0])
        assert np.isnan(o[1])
        assert not np.isnan(o[2])
        # High should be max of 3-day window
        assert not np.isnan(h[2])

    def test_5day_composite(self, small_ohlcv):
        o, h, l, c = compute_composite_ohlc(small_ohlcv, 5)
        assert np.isnan(o[3])
        assert not np.isnan(o[4])


class TestAddCandlestickFeatures:
    def test_adds_all_columns(self, small_ohlcv):
        result = add_candlestick_features(small_ohlcv)
        for col in CANDLE_1D_COLS:
            assert col in result.columns
        for col in CANDLE_3D_COLS:
            assert col in result.columns
        for col in CANDLE_5D_COLS:
            assert col in result.columns

    def test_does_not_mutate_input(self, small_ohlcv):
        original_cols = set(small_ohlcv.columns)
        add_candlestick_features(small_ohlcv)
        assert set(small_ohlcv.columns) == original_cols

    def test_1d_features_no_nan(self, small_ohlcv):
        """1-day features should have no NaN (no lookback needed)."""
        result = add_candlestick_features(small_ohlcv)
        for col in CANDLE_1D_COLS:
            assert not result[col].isna().any()
