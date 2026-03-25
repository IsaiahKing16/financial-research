"""Tests for MomentumSignalFilter."""
import numpy as np
import pandas as pd
import pytest
from datetime import date


def _make_val_db():
    """Val DataFrame with ticker, date, and 7d return columns."""
    return pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "JPM", "BAC"],
        "Date": [pd.Timestamp("2024-01-05")] * 4,
        "ret_7d": [0.05, 0.02, -0.03, -0.01],
    })


def test_fit_computes_sector_averages():
    """fit() computes rolling sector averages from training data."""
    from pattern_engine.momentum_signal import MomentumSignalFilter
    sector_map = {"AAPL": "Tech", "MSFT": "Tech", "JPM": "Finance", "BAC": "Finance"}
    filt = MomentumSignalFilter(sector_map, lookback_col="ret_7d")
    train_db = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "JPM", "BAC"] * 10,
        "Date": pd.date_range("2023-01-01", periods=40),
        "ret_7d": np.array([0.03, 0.02, -0.01, -0.02] * 10),
    })
    filt.fit(train_db)
    assert "Tech" in filt.sector_mean_returns_
    assert "Finance" in filt.sector_mean_returns_


def test_momentum_agrees_with_buy():
    """BUY signal is kept when ticker outperforms sector by threshold."""
    from pattern_engine.momentum_signal import MomentumSignalFilter
    sector_map = {"AAPL": "Tech", "MSFT": "Tech"}
    filt = MomentumSignalFilter(sector_map, lookback_col="ret_7d", min_outperformance=0.02)
    filt.sector_mean_returns_ = {"Tech": 0.01}

    val_db = pd.DataFrame({
        "Ticker": ["AAPL"],
        "Date": [pd.Timestamp("2024-01-05")],
        "ret_7d": [0.05],  # outperforms Tech avg (0.01) by 0.04 > 0.02 threshold
    })
    probs = np.array([0.70])
    signals = ["BUY"]
    result, agreed = filt.apply(probs, signals, val_db)
    assert result[0] == "BUY"
    assert agreed[0]


def test_momentum_vetoes_disagreeing_buy():
    """BUY signal is downgraded when ticker underperforms sector."""
    from pattern_engine.momentum_signal import MomentumSignalFilter
    sector_map = {"AAPL": "Tech"}
    filt = MomentumSignalFilter(sector_map, lookback_col="ret_7d", min_outperformance=0.02)
    filt.sector_mean_returns_ = {"Tech": 0.04}

    val_db = pd.DataFrame({
        "Ticker": ["AAPL"],
        "Date": [pd.Timestamp("2024-01-05")],
        "ret_7d": [0.01],  # underperforms Tech avg (0.04): delta=-0.03
    })
    probs = np.array([0.68])
    signals = ["BUY"]
    result, agreed = filt.apply(probs, signals, val_db)
    assert result[0] == "HOLD"
    assert not agreed[0]
