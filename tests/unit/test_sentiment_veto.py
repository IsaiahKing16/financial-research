"""Tests for SentimentVetoFilter — uses mock FMP data."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def test_veto_negative_sentiment_buy():
    """BUY is downgraded to HOLD when sentiment score < threshold."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    filt = SentimentVetoFilter(veto_threshold=-0.20)
    probs = np.array([0.70, 0.68])
    signals = ["BUY", "BUY"]
    tickers = ["AAPL", "META"]
    # AAPL sentiment is fine; META has bad news
    sentiment = {"AAPL": 0.10, "META": -0.35}
    filtered = filt.apply_with_sentiment(probs, signals, tickers, sentiment)
    assert filtered[0] == "BUY"
    assert filtered[1] == "HOLD"


def test_sell_not_vetoed_by_negative_sentiment():
    """SELL signals are NOT vetoed by negative sentiment (confirms bearish thesis)."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    filt = SentimentVetoFilter(veto_threshold=-0.20)
    probs = np.array([0.25])
    signals = ["SELL"]
    tickers = ["META"]
    sentiment = {"META": -0.50}
    filtered = filt.apply_with_sentiment(probs, signals, tickers, sentiment)
    assert filtered[0] == "SELL"  # Negative sentiment confirms SELL — don't veto


def test_hold_passes_through():
    """HOLD signals are unaffected by sentiment filter."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    filt = SentimentVetoFilter()
    filtered = filt.apply_with_sentiment(
        np.array([0.51]), ["HOLD"], ["AAPL"], {"AAPL": -1.0}
    )
    assert filtered[0] == "HOLD"


def test_missing_ticker_sentiment_neutral():
    """Tickers with no sentiment data are treated as neutral (not vetoed)."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    filt = SentimentVetoFilter(veto_threshold=-0.20)
    filtered = filt.apply_with_sentiment(
        np.array([0.70]), ["BUY"], ["UNKNOWN_TICKER"], {}
    )
    assert filtered[0] == "BUY"


def test_apply_unified_interface():
    """apply() satisfies SignalFilterBase interface — neutral sentiment means no veto."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    filt = SentimentVetoFilter(veto_threshold=-0.20)
    probs = np.array([0.70, 0.68])
    signals = ["BUY", "BUY"]
    val_db = pd.DataFrame({"Ticker": ["AAPL", "META"]})
    # No sentiment kwarg provided → empty dict → neutral → no vetos
    filtered, mask = filt.apply(probs, signals, val_db)
    assert filtered == ["BUY", "BUY"]
    assert not mask.any()
