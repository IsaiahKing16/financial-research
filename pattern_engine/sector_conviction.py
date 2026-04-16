"""
sector_conviction.py — Sector-level conviction scoring for FPPE signals.

Computes the mean K-NN predicted probability per sector and compares to
that sector's historical base rate. Signals in sectors where K-NN doesn't
beat the base rate by at least min_sector_lift are downgraded to HOLD.

Inspired by ARC Solomon's thematic conviction scoring: rather than acting
on individual ticker signals alone, require that the sector context supports
the direction.

Usage (post-query filter):
    layer = SectorConvictionLayer(SECTOR_MAP, min_sector_lift=0.005)
    layer.fit(train_db, target_col=cfg.projection_horizon)
    filtered_signals, veto_mask = layer.apply(probs, signals, val_tickers)

Linear: M9 (Signal Intelligence Layer)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pattern_engine.signal_filter_base import SignalFilterBase


class SectorConvictionLayer(SignalFilterBase):
    """Post-query filter that vetoes signals in low-conviction sectors.

    A sector has conviction when the K-NN aggregate predicted probability
    exceeds the sector's historical base rate by at least min_sector_lift.
    Signals in sectors below this bar are downgraded to HOLD.

    Args:
        sector_map:       ticker -> sector name mapping (SECTOR_MAP from sector.py).
        min_sector_lift:  Minimum excess probability over base rate required
                          to retain signals. Default 0.005 (0.5pp — calibrated for 585T universe).
    """

    def __init__(self, sector_map: dict[str, str], min_sector_lift: float = 0.005):
        self.sector_map = sector_map
        self.min_sector_lift = min_sector_lift
        self.sector_base_rates_: dict[str, float] = {}

    def fit(self, train_db: pd.DataFrame, target_col: str = "fwd_7d_up") -> SectorConvictionLayer:
        """Compute per-sector historical base rates from training data.

        Args:
            train_db:   Training DataFrame (must have Ticker and target_col).
            target_col: Binary target column name.
        Returns:
            self (for method chaining).
        """
        if target_col not in train_db.columns:
            return self  # no-op if target not present

        df = train_db.copy()
        df["_sector"] = df["Ticker"].map(lambda t: self.sector_map.get(t, "Unknown"))
        grouped = df.groupby("_sector")[target_col].mean()
        self.sector_base_rates_ = grouped.to_dict()
        return self

    def sector_scores(
        self,
        probs: np.ndarray,
        tickers: np.ndarray,
    ) -> dict[str, float]:
        """Compute mean predicted probability per sector.

        Args:
            probs:   (N,) calibrated probabilities from PatternMatcher.query().
            tickers: (N,) object array of ticker strings.

        Returns:
            Dict mapping sector name -> mean probability for tickers in that sector.
        """
        sectors = pd.Series([self.sector_map.get(str(t), "Unknown") for t in tickers])
        return pd.Series(probs).groupby(sectors).mean().to_dict()

    def apply(
        self,
        probs: np.ndarray,
        signals: list[str],
        val_db: pd.DataFrame,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """Veto BUY/SELL signals where sector conviction is insufficient.

        For each BUY/SELL signal, check whether the sector-level mean
        probability exceeds the sector's base rate by min_sector_lift.
        Signals that fail this check are downgraded to HOLD.

        Args:
            probs:   (N,) calibrated probabilities.
            signals: List of N signal strings ("BUY"/"SELL"/"HOLD").
            val_db:  Validation DataFrame (must have Ticker column).

        Returns:
            (filtered_signals, veto_mask):
              filtered_signals: list[str] with vetoed signals set to "HOLD".
              veto_mask: (N,) bool array, True where signal was vetoed.
        """
        if not self.sector_base_rates_:
            return signals, np.zeros(len(signals), dtype=bool)

        tickers = val_db["Ticker"].values
        sector_scores = self.sector_scores(probs, tickers)
        sectors = pd.Series([self.sector_map.get(str(t), "Unknown") for t in tickers]).values

        filtered = list(signals)
        veto_mask = np.zeros(len(signals), dtype=bool)

        for i, sig in enumerate(signals):
            if sig == "HOLD":
                continue  # already HOLD, skip
            sector = sectors[i]
            base_rate = self.sector_base_rates_.get(sector, 0.5)
            sector_mean_prob = sector_scores.get(sector, 0.5)
            lift = sector_mean_prob - base_rate
            if lift < self.min_sector_lift:
                filtered[i] = "HOLD"
                veto_mask[i] = True

        return filtered, veto_mask
