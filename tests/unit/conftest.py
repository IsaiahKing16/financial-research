# tests/unit/conftest.py
"""Shared fixtures for walkforward / sweep / experiment_log tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


_RET_WINDOWS = [1, 3, 7, 14, 30, 45, 60, 90]


def _make_synthetic_db(
    n_tickers: int = 3,
    start: str = "2017-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal synthetic database with all columns walkforward needs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    tickers = [f"T{i}" for i in range(1, n_tickers + 1)]
    # Add SPY — required for H7 HOLD regime
    tickers = ["SPY"] + tickers

    rows = []
    for ticker in tickers:
        for dt in dates:
            row = {"Date": dt, "Ticker": ticker}
            row["Open"] = 100.0 + rng.normal(0, 5)
            row["High"] = row["Open"] + abs(rng.normal(1, 0.5))
            row["Low"] = row["Open"] - abs(rng.normal(1, 0.5))
            row["Close"] = row["Open"] + rng.normal(0, 2)
            for w in _RET_WINDOWS:
                row[f"ret_{w}d"] = rng.normal(0, 0.02 * np.sqrt(w))
            # ret_90d used by H7 HOLD regime — make SPY oscillate around threshold
            if ticker == "SPY":
                row["ret_90d"] = rng.choice([0.02, 0.08], p=[0.3, 0.7])
            row["fwd_7d_up"] = float(rng.random() > 0.5)
            rows.append(row)

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@pytest.fixture
def synthetic_full_db() -> pd.DataFrame:
    """Full synthetic DB spanning 2017-2024 with SPY + 3 tickers."""
    return _make_synthetic_db()


@pytest.fixture
def small_synthetic_db() -> pd.DataFrame:
    """Smaller DB (2 tickers, 2018-2020) for fast sweep tests."""
    return _make_synthetic_db(n_tickers=2, start="2018-01-01", end="2020-12-31", seed=99)
