"""Tests for pattern_engine.journal — JournalEntry and build helpers."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import date


def test_journal_entry_fields():
    """JournalEntry dataclass has required fields."""
    from pattern_engine.journal import JournalEntry, AnalogueRecord
    rec = AnalogueRecord(
        rank=1,
        ticker="AAPL",
        date=date(2020, 3, 15),
        distance=0.42,
        label=1,
        fwd_return=0.035,
    )
    entry = JournalEntry(
        query_date=date(2024, 1, 5),
        query_ticker="MSFT",
        raw_prob=0.62,
        calibrated_prob=0.67,
        signal="BUY",
        n_matches=42,
        top_analogues=[rec],
    )
    assert entry.query_ticker == "MSFT"
    assert entry.signal == "BUY"
    assert len(entry.top_analogues) == 1
    assert entry.top_analogues[0].rank == 1


def test_build_journal_entries_basic():
    """build_journal_entries returns one entry per BUY/SELL row."""
    from pattern_engine.journal import build_journal_entries
    train_dates = np.array([date(2020, 1, i+1) for i in range(10)])
    train_tickers = np.array(["AAPL"] * 10, dtype=object)
    train_targets = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1], dtype=np.float64)
    train_returns = np.array([0.01, 0.03, -0.02, 0.01, 0.04,
                               -0.01, 0.02, 0.01, -0.03, 0.05])

    top_masks = np.array([
        [True, True, True, False, False, False, False, False, False, False],
        [True, False, False, False, False, False, False, False, False, False],
    ])
    distances = np.zeros((2, 10))
    distances[0] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    distances[1] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    indices = np.tile(np.arange(10), (2, 1))

    val_tickers = np.array(["MSFT", "GOOG"], dtype=object)
    val_dates = np.array([date(2024, 1, 5), date(2024, 1, 6)])
    raw_probs = np.array([0.62, 0.51])
    cal_probs = np.array([0.67, 0.53])
    signals = ["BUY", "HOLD"]
    n_matches = [3, 1]
    top_n = 5

    entries = build_journal_entries(
        top_masks=top_masks,
        distances=distances,
        indices=indices,
        val_tickers=val_tickers,
        val_dates=val_dates,
        raw_probs=raw_probs,
        cal_probs=cal_probs,
        signals=signals,
        n_matches=n_matches,
        train_tickers=train_tickers,
        train_dates=train_dates,
        train_targets=train_targets,
        train_returns=train_returns,
        top_n=top_n,
    )
    assert len(entries) == 1
    assert entries[0].signal == "BUY"
    assert entries[0].query_ticker == "MSFT"
    assert len(entries[0].top_analogues) == 3
    assert entries[0].top_analogues[0].rank == 1
    assert entries[0].top_analogues[0].ticker == "AAPL"


def test_build_journal_entries_top_n_cap():
    """top_n caps the number of analogues stored per entry."""
    from pattern_engine.journal import build_journal_entries
    n = 50
    train_dates = np.array([date(2019, 1, 1)] * n)
    train_tickers = np.array(["SPY"] * n, dtype=object)
    train_targets = np.ones(n)
    train_returns = np.ones(n) * 0.01
    top_masks = np.ones((1, n), dtype=bool)
    distances = np.arange(n, dtype=float).reshape(1, n)
    indices = np.arange(n).reshape(1, n)

    entries = build_journal_entries(
        top_masks=top_masks,
        distances=distances,
        indices=indices,
        val_tickers=np.array(["AAPL"], dtype=object),
        val_dates=np.array([date(2024, 6, 1)]),
        raw_probs=np.array([0.70]),
        cal_probs=np.array([0.71]),
        signals=["BUY"],
        n_matches=[50],
        train_tickers=train_tickers,
        train_dates=train_dates,
        train_targets=train_targets,
        train_returns=train_returns,
        top_n=10,
    )
    assert len(entries) == 1
    assert len(entries[0].top_analogues) == 10


def test_write_and_read_journal_parquet(tmp_path):
    """Journal entries can be written to Parquet and read back."""
    from pattern_engine.journal import JournalEntry, AnalogueRecord, write_journal_parquet, read_journal_parquet
    rec = AnalogueRecord(rank=1, ticker="AAPL", date=date(2020, 3, 15),
                          distance=0.42, label=1, fwd_return=0.035)
    entry = JournalEntry(
        query_date=date(2024, 1, 5),
        query_ticker="MSFT",
        raw_prob=0.62,
        calibrated_prob=0.67,
        signal="BUY",
        n_matches=42,
        top_analogues=[rec],
    )
    out_path = tmp_path / "test_journal.parquet"
    write_journal_parquet([entry], out_path)
    assert out_path.exists()

    df = read_journal_parquet(out_path)
    assert len(df) == 1
    assert df["query_ticker"].iloc[0] == "MSFT"
    assert df["signal"].iloc[0] == "BUY"
    assert df["analogue_rank"].iloc[0] == 1
    assert df["analogue_ticker"].iloc[0] == "AAPL"

    from pattern_engine.journal import top_n_view
    view5 = top_n_view(df, n=5)
    assert len(view5) == 1
    view3 = top_n_view(df, n=3)
    assert len(view3) == 1
    with pytest.raises(ValueError):
        top_n_view(df, n=0)
