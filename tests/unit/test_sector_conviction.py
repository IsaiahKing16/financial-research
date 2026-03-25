"""Tests for SectorConvictionLayer."""
import numpy as np
import pandas as pd
import pytest


def _make_train_db(n=100):
    import random
    rng = np.random.RandomState(42)
    sectors = ["Tech", "Finance", "Health"]
    tickers_by_sector = {
        "Tech": ["AAPL", "MSFT"],
        "Finance": ["JPM", "BAC"],
        "Health": ["JNJ", "PFE"],
    }
    rows = []
    for i in range(n):
        s = sectors[i % 3]
        t = tickers_by_sector[s][i % 2]
        rows.append({
            "Ticker": t,
            "Date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
            "fwd_7d_up": int(rng.rand() > 0.4),
        })
    return pd.DataFrame(rows)


def test_fit_computes_sector_base_rates():
    """SectorConvictionLayer.fit() computes per-sector base rates."""
    from pattern_engine.sector_conviction import SectorConvictionLayer
    sector_map = {
        "AAPL": "Tech", "MSFT": "Tech",
        "JPM": "Finance", "BAC": "Finance",
        "JNJ": "Health", "PFE": "Health",
    }
    layer = SectorConvictionLayer(sector_map)
    train_db = _make_train_db(120)
    layer.fit(train_db, target_col="fwd_7d_up")

    assert "Tech" in layer.sector_base_rates_
    assert "Finance" in layer.sector_base_rates_
    assert "Health" in layer.sector_base_rates_
    for sector, rate in layer.sector_base_rates_.items():
        assert 0.0 <= rate <= 1.0


def test_sector_scores_returns_per_sector_mean():
    """sector_scores() aggregates probs by sector correctly."""
    from pattern_engine.sector_conviction import SectorConvictionLayer
    sector_map = {"AAPL": "Tech", "MSFT": "Tech", "JPM": "Finance"}
    layer = SectorConvictionLayer(sector_map)
    layer.sector_base_rates_ = {"Tech": 0.50, "Finance": 0.50}
    probs = np.array([0.70, 0.80, 0.60])
    tickers = np.array(["AAPL", "MSFT", "JPM"], dtype=object)
    scores = layer.sector_scores(probs, tickers)
    assert abs(scores["Tech"] - 0.75) < 1e-9
    assert abs(scores["Finance"] - 0.60) < 1e-9


def test_conviction_filter_vetoes_weak_sector():
    """Signals in sectors below conviction threshold are downgraded to HOLD."""
    from pattern_engine.sector_conviction import SectorConvictionLayer
    sector_map = {"AAPL": "Tech", "JPM": "Finance"}
    layer = SectorConvictionLayer(sector_map, min_sector_lift=0.05)
    # Tech base_rate=0.55, Finance base_rate=0.55
    layer.sector_base_rates_ = {"Tech": 0.55, "Finance": 0.55}

    # AAPL (Tech): prob=0.70 → lift=0.15 → above threshold → keep BUY
    # JPM (Finance): prob=0.58 → lift=0.03 → below min_sector_lift → veto to HOLD
    probs = np.array([0.70, 0.58])
    signals = ["BUY", "BUY"]
    val_db = pd.DataFrame({"Ticker": ["AAPL", "JPM"]})

    filtered_signals, veto_mask = layer.apply(probs, signals, val_db)
    assert filtered_signals[0] == "BUY"
    assert filtered_signals[1] == "HOLD"
    assert veto_mask[1]
