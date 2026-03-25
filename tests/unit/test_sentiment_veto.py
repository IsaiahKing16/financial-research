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


def test_fetch_sentiment_stub_returns_neutral():
    """fetch_sentiment() delegates to _fetch_ticker() and returns 0.0 per ticker (stub)."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    from datetime import date
    filt = SentimentVetoFilter()
    scores = filt.fetch_sentiment(["AAPL", "MSFT"], query_date=date(2024, 1, 5))
    assert scores == {"AAPL": 0.0, "MSFT": 0.0}


def test_circuit_breaker_raises_when_error_rate_exceeds_threshold():
    """RuntimeError is raised when FMP failure rate exceeds circuit_breaker_threshold."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    from datetime import date
    from unittest.mock import patch

    filt = SentimentVetoFilter(circuit_breaker_threshold=0.30)

    def always_fail(ticker, since_date):
        raise ConnectionError("FMP MCP unreachable")

    with patch.object(filt, "_fetch_ticker", side_effect=always_fail):
        with pytest.raises(RuntimeError, match="Halting execution"):
            filt.fetch_sentiment(
                ["AAPL", "MSFT", "GOOG", "META"],
                query_date=date(2024, 1, 5),
            )


def test_circuit_breaker_passes_when_error_rate_below_threshold():
    """No exception raised when error rate is below threshold."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    from datetime import date
    from unittest.mock import patch

    filt = SentimentVetoFilter(circuit_breaker_threshold=0.30)

    call_count = {"n": 0}

    def fail_first_only(ticker, since_date):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("one transient error")
        return 0.0

    # 1 failure in 4 tickers = 25% < 30% threshold — should NOT raise
    with patch.object(filt, "_fetch_ticker", side_effect=fail_first_only):
        scores = filt.fetch_sentiment(
            ["AAPL", "MSFT", "GOOG", "META"],
            query_date=date(2024, 1, 5),
        )
    assert scores["AAPL"] == 0.0
    assert scores["MSFT"] == 0.0
    assert scores["GOOG"] == 0.0
    assert scores["META"] == 0.0
