"""
sentiment_veto.py — News sentiment veto filter for FPPE BUY signals.

Inspired by ARC Solomon's narrative analyst + risk officer architecture:
a BUY signal should be vetoed if recent news sentiment for the ticker
is strongly negative, since markets increasingly move on narrative.

Two usage modes:
    1. Backtesting: pass pre-fetched sentiment dict (ticker -> score)
    2. Live trading: call fetch_sentiment() which queries FMP MCP

Sentiment score range: [-1.0, +1.0] (FMP normalized sentiment).
Veto threshold: -0.20 by default (veto BUY if score < -0.20).

Usage:
    filt = SentimentVetoFilter(veto_threshold=-0.20, lookback_days=3)

    # Backtesting (pre-fetched):
    sentiment = {"AAPL": 0.15, "META": -0.35}
    filtered = filt.apply_with_sentiment(probs, signals, tickers, sentiment)

    # Live (FMP MCP call):
    sentiment = filt.fetch_sentiment(tickers, current_date)
    filtered = filt.apply_with_sentiment(probs, signals, tickers, sentiment)

Linear: M9 (Signal Intelligence Layer)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from pattern_engine.signal_filter_base import SignalFilterBase


class SentimentVetoFilter(SignalFilterBase):
    """Vetoes BUY signals when recent news sentiment is strongly negative.

    Negative sentiment confirms bearish thesis for SELL signals — those
    are NOT vetoed. Only BUY signals are affected.

    Sentiment score interpretation:
        > +0.20:  Positive news — supports BUY
        -0.20 to +0.20: Neutral — no veto
        < -0.20:  Negative news — veto BUY -> HOLD

    Args:
        veto_threshold: Sentiment score below which BUY is vetoed (default -0.20).
        lookback_days:  Days of news to average for sentiment score (default 3).
    """

    def __init__(
        self,
        veto_threshold: float = -0.20,
        lookback_days: int = 3,
    ):
        self.veto_threshold = veto_threshold
        self.lookback_days = lookback_days

    def fetch_sentiment(
        self,
        tickers: list[str],
        query_date: Optional[date] = None,
    ) -> Dict[str, float]:
        """Fetch news sentiment scores from FMP MCP for a list of tickers.

        Queries the FMP stock-news endpoint for each ticker, averaging
        sentiment over the past lookback_days. Returns a dict mapping
        ticker -> mean sentiment score [-1.0, +1.0].

        NOTE: This method requires FMP MCP to be configured. For backtesting,
        use apply_with_sentiment() directly with pre-fetched data instead.

        Args:
            tickers:    List of ticker symbols (uppercase).
            query_date: Date to fetch sentiment for (defaults to today).

        Returns:
            Dict[str, float]: ticker -> mean sentiment score.
            Tickers with no news return 0.0 (neutral).
        """
        if query_date is None:
            query_date = date.today()

        since_date = query_date - timedelta(days=self.lookback_days)
        scores: Dict[str, float] = {}

        for ticker in tickers:
            try:
                # FMP MCP call — stock-news endpoint
                # In live usage: call mcp__fmp__stock_news(symbol=ticker, limit=20)
                # and average the sentimentScore field over lookback_days
                # This is a placeholder — wire to FMP MCP in live runner
                scores[ticker] = 0.0  # neutral default until FMP wired
            except (ConnectionError, TimeoutError, OSError, ValueError) as exc:
                logging.warning("SentimentVetoFilter: failed to fetch %s: %s", ticker, exc)
                scores[ticker] = 0.0  # neutral on error

        return scores

    def apply_with_sentiment(
        self,
        probs: np.ndarray,
        signals: list[str],
        tickers: list[str],
        sentiment: Dict[str, float],
    ) -> list[str]:
        """Apply sentiment veto to signals using pre-fetched sentiment scores.

        BUY signals with sentiment < veto_threshold are downgraded to HOLD.
        SELL signals are never vetoed by negative sentiment (it confirms them).
        HOLD signals pass through unchanged.

        Args:
            probs:     (N,) calibrated probabilities (informational only).
            signals:   List of N signal strings.
            tickers:   List of N ticker symbols.
            sentiment: Dict[str, float] from fetch_sentiment() or pre-fetched.

        Returns:
            List[str]: Filtered signal strings.
        """
        filtered = list(signals)

        for i, sig in enumerate(signals):
            if sig != "BUY":
                continue  # SELL confirms negative news; HOLD unchanged

            ticker = tickers[i] if i < len(tickers) else ""
            score = sentiment.get(ticker, 0.0)  # missing = neutral

            if score < self.veto_threshold:
                filtered[i] = "HOLD"

        return filtered

    def apply(
        self,
        probs: np.ndarray,
        signals: list[str],
        val_db: pd.DataFrame,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """Unified SignalFilterBase interface.

        Wraps apply_with_sentiment() using sentiment from kwargs.
        Pass sentiment={ticker: score} for live mode; omit for neutral (no veto).

        Args:
            probs:    (N,) calibrated probabilities.
            signals:  List of N signal strings.
            val_db:   Validation DataFrame (must have Ticker).
            **kwargs: Optional ``sentiment`` dict override.

        Returns:
            (filtered_signals, veto_mask).
        """
        tickers = list(val_db["Ticker"].values)
        sentiment: Dict[str, float] = kwargs.get("sentiment", {})
        filtered = self.apply_with_sentiment(probs, signals, tickers, sentiment)
        veto_mask = np.array(
            [filtered[i] != signals[i] for i in range(len(signals))], dtype=bool
        )
        return filtered, veto_mask
