"""
Shared fixtures for pattern_engine test suite.

Provides synthetic market data that mimics the structure of real
OHLCV + feature data without requiring yfinance downloads.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.RandomState(42)


def _make_ticker_data(ticker: str, n_days: int, start_date: str,
                      rng: np.random.RandomState, base_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV + feature data for one ticker."""
    dates = pd.bdate_range(start=start_date, periods=n_days)
    returns_daily = rng.randn(n_days) * 0.02
    prices = base_price * np.cumprod(1 + returns_daily)

    df = pd.DataFrame({
        "Date": dates,
        "Open": prices * (1 + rng.randn(n_days) * 0.005),
        "High": prices * (1 + np.abs(rng.randn(n_days) * 0.01)),
        "Low": prices * (1 - np.abs(rng.randn(n_days) * 0.01)),
        "Close": prices,
        "Volume": (rng.rand(n_days) * 1e6 + 5e5).astype(int),
        "Ticker": ticker,
    })

    # Trailing return features
    for w in [1, 3, 7, 14, 30, 45, 60, 90]:
        df[f"ret_{w}d"] = df["Close"].pct_change(w)

    # Supplementary features
    df["vol_10d"] = df["Close"].pct_change().rolling(10).std()
    df["vol_30d"] = df["Close"].pct_change().rolling(30).std()
    df["vol_ratio"] = df["vol_10d"] / (df["vol_30d"] + 1e-8)
    vol_mean = df["Volume"].rolling(20).mean()
    df["vol_abnormal"] = (df["Volume"] - vol_mean) / (vol_mean + 1)
    df["rsi_14"] = 50 + rng.randn(n_days) * 15
    df["atr_14"] = prices * 0.02 * (1 + rng.rand(n_days) * 0.5)
    df["price_vs_sma20"] = rng.randn(n_days) * 0.05
    df["price_vs_sma50"] = rng.randn(n_days) * 0.08

    # Forward returns and binary targets
    for w in [1, 3, 7, 14, 30]:
        df[f"fwd_{w}d"] = df["Close"].shift(-w) / df["Close"] - 1
        df[f"fwd_{w}d_up"] = (df[f"fwd_{w}d"] > 0).astype(int)

    # Candlestick features (simplified)
    for tf in ["1d", "3d", "5d"]:
        df[f"candle_{tf}_body_to_range"] = rng.rand(n_days) * 0.8 + 0.1
        df[f"candle_{tf}_upper_wick"] = rng.rand(n_days) * 0.3
        df[f"candle_{tf}_lower_wick"] = rng.rand(n_days) * 0.3
        df[f"candle_{tf}_body_pos"] = rng.rand(n_days) * 0.6 + 0.2
        df[f"candle_{tf}_direction"] = rng.choice([-1.0, 1.0], n_days)

    # Sector features
    df["sector_relative_return_7d"] = rng.randn(n_days) * 0.01
    df["spy_correlation_30d"] = rng.rand(n_days) * 0.4 + 0.5
    df["sector_rank_30d"] = rng.rand(n_days)

    # ADX for regime testing
    df["adx_14"] = rng.rand(n_days) * 40 + 10

    return df


@pytest.fixture
def synthetic_db(rng):
    """Full synthetic database with 4 tickers, 500 days each.

    Includes SPY (required for regime labeling), AAPL, MSFT, JPM.
    All feature columns populated. Rows with NaN from lookback dropped.
    """
    tickers = ["SPY", "AAPL", "MSFT", "JPM"]
    dfs = []
    for t in tickers:
        df = _make_ticker_data(t, 500, "2018-01-01", rng,
                               base_price=100 + rng.rand() * 200)
        dfs.append(df)

    db = pd.concat(dfs, ignore_index=True)
    # Drop rows with NaN from lookback windows
    db = db.dropna(subset=["ret_90d", "fwd_7d", "fwd_7d_up"]).reset_index(drop=True)
    return db


@pytest.fixture
def train_db(synthetic_db):
    """Training split: first 70% of dates per ticker."""
    dates = sorted(synthetic_db["Date"].unique())
    cutoff = dates[int(len(dates) * 0.7)]
    return synthetic_db[synthetic_db["Date"] <= cutoff].copy().reset_index(drop=True)


@pytest.fixture
def val_db(synthetic_db):
    """Validation split: last 30% of dates per ticker."""
    dates = sorted(synthetic_db["Date"].unique())
    cutoff = dates[int(len(dates) * 0.7)]
    return synthetic_db[synthetic_db["Date"] > cutoff].copy().reset_index(drop=True)


@pytest.fixture
def small_ohlcv(rng):
    """Small OHLCV DataFrame (20 rows) for candlestick testing."""
    n = 20
    prices = 100 + np.cumsum(rng.randn(n) * 2)
    return pd.DataFrame({
        "Date": pd.bdate_range("2024-01-01", periods=n),
        "Open": prices + rng.randn(n) * 0.5,
        "High": prices + np.abs(rng.randn(n)) * 2,
        "Low": prices - np.abs(rng.randn(n)) * 2,
        "Close": prices,
        "Volume": (rng.rand(n) * 1e6).astype(int),
    })
