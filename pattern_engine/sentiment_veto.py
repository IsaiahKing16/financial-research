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
        circuit_breaker_threshold: float = 0.30,
    ):
        self.veto_threshold = veto_threshold
        self.lookback_days = lookback_days
        self.circuit_breaker_threshold = circuit_breaker_threshold

    def _fetch_ticker(self, ticker: str, since_date: date) -> float:
        """Fetch sentiment score for a single ticker.

        Stub implementation returns 0.0 (neutral) until FMP MCP is wired.
        In live usage, replace with:
            news = mcp__fmp__stock_news(symbol=ticker, limit=20)
            recent = [n for n in news if n["date"] >= str(since_date)]
            return mean([n["sentimentScore"] for n in recent]) if recent else 0.0

        Raises:
            ConnectionError, TimeoutError, OSError, ValueError: on fetch failure.
        """
        return 0.0  # neutral default until FMP wired

    def fetch_sentiment(
        self,
        tickers: list[str],
        query_date: date | None = None,
    ) -> dict[str, float]:
        """Fetch news sentiment scores for a list of tickers.

        Calls _fetch_ticker() for each ticker and aggregates results.
        If the failure rate exceeds circuit_breaker_threshold, raises
        RuntimeError rather than silently returning all-neutral scores.

        Args:
            tickers:    List of ticker symbols (uppercase).
            query_date: Date to fetch sentiment for (defaults to today).

        Returns:
            Dict[str, float]: ticker -> mean sentiment score [-1.0, +1.0].
            Tickers with no news or transient errors return 0.0 (neutral),
            provided the overall error rate stays below circuit_breaker_threshold.

        Raises:
            RuntimeError: If error_count / len(tickers) > circuit_breaker_threshold.
        """
        if query_date is None:
            query_date = date.today()

        since_date = query_date - timedelta(days=self.lookback_days)
        scores: dict[str, float] = {}
        error_count = 0

        for ticker in tickers:
            try:
                scores[ticker] = self._fetch_ticker(ticker, since_date)
            except (ConnectionError, TimeoutError, OSError, ValueError) as exc:
                logging.warning("SentimentVetoFilter: failed to fetch %s: %s", ticker, exc)
                scores[ticker] = 0.0  # neutral on individual error
                error_count += 1

        if tickers and error_count / len(tickers) > self.circuit_breaker_threshold:
            raise RuntimeError(
                f"SentimentVetoFilter circuit breaker: {error_count}/{len(tickers)} "
                f"fetch failures ({error_count / len(tickers):.0%}) exceed threshold "
                f"{self.circuit_breaker_threshold:.0%}. Halting execution pipeline."
            )

        return scores

    def apply_with_sentiment(
        self,
        probs: np.ndarray,
        signals: list[str],
        tickers: list[str],
        sentiment: dict[str, float],
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
        sentiment: dict[str, float] = kwargs.get("sentiment", {})
        filtered = self.apply_with_sentiment(probs, signals, tickers, sentiment)
        veto_mask = np.array(
            [filtered[i] != signals[i] for i in range(len(signals))], dtype=bool
        )
        return filtered, veto_mask
