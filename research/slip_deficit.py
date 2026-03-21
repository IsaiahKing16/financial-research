"""
slip_deficit.py — Seismic Slip-Deficit + Time-To-Failure (TTF) risk overlay.

Draws an analogy from seismology:
  - Slip-deficit: accumulated displacement of price above/below long-run mean (SMA200).
    Positive → price has run ahead of trend; negative → below trend.
  - TTF: volatility acceleration signal. High short-term vol vs baseline vol
    (measured as a Z-score) signals elevated failure risk.

Computation
-----------
slip_deficit = (current_close - SMA_N) / SMA_N

vol_series    = rolling(10-day) annualised std of log returns
                = log_returns.rolling(10).std() * sqrt(252)

baseline      = vol_series over last vol_lookback bars
vol_zscore    = (vol_series.iloc[-1] - baseline.mean()) / (baseline.std() + 1e-8)

ttf_probability = sigmoid(vol_zscore)          # ∈ [0, 1]
tighten_stops   = vol_zscore > ttf_threshold   # direct Z-score comparison
"""

import math

import numpy as np
import pandas as pd

from research import BaseRiskOverlay, RiskOverlayResult


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class SlipDeficit(BaseRiskOverlay):
    """Seismic Slip-Deficit + TTF risk overlay.

    Args:
        sma_window:    Number of bars for the slow moving average (default 200).
        vol_lookback:  Number of bars for vol Z-score baseline (default 60).
        ttf_threshold: Z-score threshold above which stops are tightened (default 2.0).
    """

    def __init__(
        self,
        sma_window: int = 200,
        vol_lookback: int = 60,
        ttf_threshold: float = 2.0,
    ) -> None:
        self.sma_window = sma_window
        self.vol_lookback = vol_lookback
        self.ttf_threshold = ttf_threshold

    def compute(
        self,
        prices_df: pd.DataFrame,
        positions=None,  # accepted, ignored — SlipDeficit is stateless w.r.t. positions
    ) -> RiskOverlayResult:
        """Compute slip-deficit and TTF signals from a price series.

        Args:
            prices_df: DataFrame with a 'close' column and DatetimeIndex.
                       Must have at least max(sma_window, vol_lookback + 10) rows.
            positions: Optional list of open positions (accepted, ignored).

        Returns:
            RiskOverlayResult with slip_deficit, ttf_probability, tighten_stops.

        Raises:
            ValueError: If prices_df has insufficient history.
        """
        # vol_series uses rolling(10).std(), which needs 10 bars to warm up.
        # So we need vol_lookback + 10 rows to get vol_lookback settled vol values.
        required = max(self.sma_window, self.vol_lookback + 10)
        if len(prices_df) < required:
            raise ValueError(
                f"SlipDeficit requires at least {required} rows of price history "
                f"(sma_window={self.sma_window}, vol_lookback={self.vol_lookback}, "
                f"rolling_vol_window=10), got {len(prices_df)}. insufficient history"
            )

        close = prices_df["close"].astype(float)

        # --- Slip-deficit ---
        sma = float(close.rolling(self.sma_window).mean().iloc[-1])
        current_price = float(close.iloc[-1])
        slip_deficit = (current_price - sma) / sma

        # --- Volatility Z-score (TTF) ---
        log_returns = np.log(close / close.shift(1)).dropna()
        # Rolling 10-day annualised volatility
        vol_series = log_returns.rolling(10).std() * math.sqrt(252)
        vol_series = vol_series.dropna()

        recent_vol = float(vol_series.iloc[-1])
        baseline = vol_series.iloc[-self.vol_lookback:]
        baseline_mean = float(baseline.mean())
        baseline_std = float(baseline.std()) + 1e-8   # guard against zero variance
        vol_zscore = (recent_vol - baseline_mean) / baseline_std

        ttf_probability = float(_sigmoid(vol_zscore))
        tighten_stops = bool(vol_zscore > self.ttf_threshold)

        return RiskOverlayResult(
            slip_deficit=float(slip_deficit),
            ttf_probability=ttf_probability,
            tighten_stops=tighten_stops,
        )
