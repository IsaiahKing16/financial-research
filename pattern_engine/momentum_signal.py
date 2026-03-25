"""
momentum_signal.py — Ticker-vs-sector momentum agreement filter.

Inspired by ARC Solomon's multi-agent agreement requirement: a BUY signal
should only be acted on when the ticker is outperforming its sector average
on the same return window. A SELL should only be acted on when underperforming.

Signals where K-NN and momentum disagree are downgraded to HOLD.

Usage:
    filt = MomentumSignalFilter(SECTOR_MAP, lookback_col="ret_7d",
                                 min_outperformance=0.015)
    filt.fit(train_db)
    filtered_signals, agreed = filt.apply(probs, signals, val_db)

Linear: M9 (Signal Intelligence Layer)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from pattern_engine.signal_filter_base import SignalFilterBase


class MomentumSignalFilter(SignalFilterBase):
    """Filters signals where ticker momentum disagrees with the K-NN direction.

    For BUY: require ticker's lookback_col return > sector avg + min_outperformance.
    For SELL: require ticker's lookback_col return < sector avg - min_outperformance.
    Disagreeing signals are downgraded to HOLD.

    Args:
        sector_map:           ticker -> sector name mapping.
        lookback_col:         Return column to use for momentum (e.g. "ret_7d").
        min_outperformance:   Minimum excess return vs sector to retain signal
                              (default 0.015 = 1.5 percentage points).
    """

    def __init__(
        self,
        sector_map: Dict[str, str],
        lookback_col: str = "ret_7d",
        min_outperformance: float = 0.015,
    ):
        self.sector_map = sector_map
        self.lookback_col = lookback_col
        self.min_outperformance = min_outperformance
        self.sector_mean_returns_: Dict[str, float] = {}

    def fit(self, train_db: pd.DataFrame) -> "MomentumSignalFilter":
        """Compute sector mean returns from training data.

        Args:
            train_db: Training DataFrame (must have Ticker and lookback_col).
        Returns:
            self.
        """
        if self.lookback_col not in train_db.columns:
            return self

        df = train_db.copy()
        df["_sector"] = df["Ticker"].map(lambda t: self.sector_map.get(str(t), "Unknown"))
        grouped = df.groupby("_sector")[self.lookback_col].mean()
        self.sector_mean_returns_ = grouped.to_dict()
        return self

    def apply(
        self,
        probs: np.ndarray,
        signals: list[str],
        val_db: pd.DataFrame,
        **kwargs,
    ) -> Tuple[list[str], np.ndarray]:
        """Downgrade signals where ticker momentum disagrees with K-NN direction.

        Args:
            probs:   (N,) calibrated probabilities.
            signals: List of N signal strings.
            val_db:  Validation DataFrame (must have Ticker, lookback_col).

        Returns:
            (filtered_signals, agreed):
              filtered_signals: list[str] with vetoed signals set to HOLD.
              agreed: (N,) bool array, True where signal was kept.
        """
        if not self.sector_mean_returns_ or self.lookback_col not in val_db.columns:
            return signals, np.ones(len(signals), dtype=bool)

        tickers = val_db["Ticker"].values
        returns = val_db[self.lookback_col].values

        filtered = list(signals)
        agreed = np.ones(len(signals), dtype=bool)

        for i, sig in enumerate(signals):
            if sig == "HOLD":
                continue
            ticker = str(tickers[i])
            sector = self.sector_map.get(ticker, "Unknown")
            sector_avg = self.sector_mean_returns_.get(sector, 0.0)
            ticker_ret = float(returns[i]) if not np.isnan(returns[i]) else sector_avg

            if sig == "BUY":
                if ticker_ret - sector_avg < self.min_outperformance:
                    filtered[i] = "HOLD"
                    agreed[i] = False
            elif sig == "SELL":
                if sector_avg - ticker_ret < self.min_outperformance:
                    filtered[i] = "HOLD"
                    agreed[i] = False

        return filtered, agreed
