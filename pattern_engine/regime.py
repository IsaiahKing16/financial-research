"""
regime.py — Multi-factor market regime labeler for FPPE pattern matching.

Labels each observation as Bull (1) or Bear (0) using:
  Base signal:  SPY 90-day trailing return (positive → Bull, negative → Bear)
  Override 1:   VIX daily change z-score > vix_spike_zscore → Bear
                (detects volatility regime shifts independent of SPY trend)
  Override 2:   10Y-2Y yield spread < 0 → Bear
                (inverted yield curve as recession/risk-off signal)

Interface contract (required by PatternMatcher):
    labeler.fitted        bool: True after fit()
    labeler.mode          str:  "binary" or "multi_factor"
    labeler.label(db, reference_db=None) → np.ndarray[int]

Usage:
    labeler = RegimeLabeler(vix_series=vix_df["VIX"])
    labeler.fit(train_db)
    labels = labeler.label(val_db)
    # Pass to PatternMatcher.fit()/query() to enable regime filtering

Linear: M9 (Signal Intelligence Layer)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class RegimeLabeler:
    """Multi-factor market regime labeler.

    Classifies each row of a feature database as Bull (1) or Bear (0).

    Base classification uses SPY's 90-day trailing return extracted from the
    reference database at fit() time.  Two optional overrides can force Bear
    regardless of the SPY signal:

    * VIX spike override — when the daily VIX change z-score (rolling window)
      exceeds ``vix_spike_zscore``, the observation is forced to Bear.
    * Inverted yield-curve override — when the 10Y-2Y spread is negative, the
      observation is forced to Bear.

    Parameters
    ----------
    spy_ticker:
        Ticker symbol used to locate the SPY rows in the reference DataFrame.
        Default ``"SPY"``.
    lookback_days:
        Nominal lookback used for the SPY return column (informational; the
        actual column name is ``"ret_90d"``).  Default ``90``.
    vix_series:
        Optional ``pd.Series`` with a ``DatetimeIndex`` containing daily VIX
        closing levels.  When provided, activates the VIX-spike override and
        sets ``mode = "multi_factor"``.
    yield_spread_series:
        Optional ``pd.Series`` with a ``DatetimeIndex`` containing the 10Y-2Y
        Treasury yield spread (positive = normal, negative = inverted).  When
        provided, activates the yield-curve override and sets
        ``mode = "multi_factor"``.
    vix_spike_zscore:
        Z-score threshold for the VIX-spike override.  Daily VIX changes whose
        rolling z-score exceeds this value trigger a Bear override.
        Default ``1.0``.
    vix_spike_window:
        Rolling window (in trading days) used to compute the VIX-change
        z-score.  Default ``20``.
    """

    def __init__(
        self,
        spy_ticker: str = "SPY",
        lookback_days: int = 90,
        vix_series: Optional[pd.Series] = None,
        yield_spread_series: Optional[pd.Series] = None,
        vix_spike_zscore: float = 1.0,
        vix_spike_window: int = 20,
    ) -> None:
        self.spy_ticker = spy_ticker
        self.lookback_days = lookback_days
        self.vix_series = vix_series
        self.yield_spread_series = yield_spread_series
        self.vix_spike_zscore = vix_spike_zscore
        self.vix_spike_window = vix_spike_window

        self.fitted: bool = False
        self.mode: str = "binary"

        # Populated by fit()
        self._spy_ret90: Optional[pd.Series] = None
        self._vix_zscore: Optional[pd.Series] = None

    # ──────────────────────────────────────────────────────────────────────────
    # fit
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, reference_db: pd.DataFrame) -> "RegimeLabeler":
        """Fit the labeler on a reference (training) database.

        Extracts the SPY 90-day return series from ``reference_db`` and,
        when provided, pre-computes the rolling VIX z-score.

        Parameters
        ----------
        reference_db:
            DataFrame containing at minimum columns ``"Ticker"``, ``"Date"``,
            and ``"ret_90d"``.  SPY rows are extracted to build the base regime
            signal.

        Returns
        -------
        self
            Supports method chaining.
        """
        # Extract SPY ret_90d series indexed by Date
        spy_rows = reference_db[reference_db["Ticker"] == self.spy_ticker].copy()
        spy_rows["Date"] = pd.to_datetime(spy_rows["Date"])
        spy_rows = spy_rows.set_index("Date").sort_index()
        self._spy_ret90 = spy_rows["ret_90d"]

        # Pre-compute VIX z-score if vix_series provided
        if self.vix_series is not None:
            vix = self.vix_series.copy()
            vix.index = pd.to_datetime(vix.index)
            vix_change = vix.diff()
            window = self.vix_spike_window
            rolling_mean = vix_change.rolling(window).mean()
            rolling_std = vix_change.rolling(window).std()
            self._vix_zscore = (vix_change - rolling_mean) / (rolling_std + 1e-8)
        else:
            self._vix_zscore = None

        # Determine mode
        if self.vix_series is not None or self.yield_spread_series is not None:
            self.mode = "multi_factor"
        else:
            self.mode = "binary"

        self.fitted = True
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # label
    # ──────────────────────────────────────────────────────────────────────────

    def label(
        self,
        db: pd.DataFrame,
        reference_db: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Label each row of ``db`` as Bull (1) or Bear (0).

        Must call ``fit()`` before ``label()``.

        Parameters
        ----------
        db:
            DataFrame to label.  Must contain a ``"Date"`` column.
        reference_db:
            Unused; accepted for API parity with PatternMatcher's calling
            convention.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(len(db),)`` with values 1 (Bull) or
            0 (Bear).

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self.fitted:
            raise RuntimeError(
                "RegimeLabeler.label() called before fit(). "
                "Call fit(reference_db) first."
            )

        dates = pd.to_datetime(db["Date"])

        # ── Stage 1: Base signal — SPY ret_90d ────────────────────────────
        # Map ret_90d → 0 (Bear) if negative, 1 (Bull) if ≥ 0
        spy_regime = self._spy_ret90.map(lambda r: 0 if r < 0 else 1)

        # Reindex to the dates in db using nearest-neighbour fill (handles
        # weekends and gaps without introducing look-ahead bias at daily level)
        regime_by_date = spy_regime.reindex(dates.values, method="nearest")
        labels = pd.Series(regime_by_date.values, index=db.index)
        labels = labels.fillna(1).astype(int)

        # ── Stage 2: VIX spike override ────────────────────────────────────
        if self._vix_zscore is not None:
            vix_z = self._vix_zscore.copy()
            vix_z.index = pd.to_datetime(vix_z.index)
            # Reindex to query dates
            vix_z_reindexed = vix_z.reindex(dates.values, method="nearest")
            vix_z_series = pd.Series(vix_z_reindexed.values, index=db.index)
            spike_mask = vix_z_series > self.vix_spike_zscore
            labels[spike_mask] = 0

        # ── Stage 3: Inverted yield curve override ─────────────────────────
        if self.yield_spread_series is not None:
            spread = self.yield_spread_series.copy()
            spread.index = pd.to_datetime(spread.index)
            spread_reindexed = spread.reindex(dates.values, method="nearest")
            spread_series = pd.Series(spread_reindexed.values, index=db.index)
            inverted_mask = spread_series < 0
            labels[inverted_mask] = 0

        return labels.values.astype(int)
