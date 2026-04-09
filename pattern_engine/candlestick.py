"""
pattern_engine/candlestick.py — Continuous multi-timeframe candlestick features.

Computes 5 continuous proportional features × 3 timeframes (1d, 3d, 5d) = 15 columns.
Designed for the Phase 6 'returns_candle' feature set.

Feature definitions (per timeframe):
    body_to_range  = abs(close - open) / (high - low)         [0, 1]
    upper_wick     = (high - max(open, close)) / (high - low)  [0, 1]
    lower_wick     = (min(open, close) - low) / (high - low)   [0, 1]
    body_position  = (min(open, close) - low) / (high - low)   [0, 1]
    direction      = 1 if close >= open else 0                 {0, 1}

Identity constraint: upper_wick + body_to_range + lower_wick == 1.0 (enforced with
RuntimeError, tolerance 1e-9, on non-NaN rows).

Multi-timeframe composite OHLC (rolling windows per-ticker):
    1-day: raw daily OHLC
    3-day: O=open[t-2], H=max(high[t-2:t+1]), L=min(low[t-2:t+1]), C=close[t]
    5-day: O=open[t-4], H=max(high[t-4:t+1]), L=min(low[t-4:t+1]), C=close[t]

Edge cases:
    zero range (high==low) → all proportions 0.0, direction=1.0, no ZeroDivisionError
    negative range (high<low, data error) → clamp range to 0.0, emit UserWarning
    NaN OHLC → NaN for all 5 features at that timeframe
    window < N days (start of history) → NaN for that timeframe

NaN imputation at engine input boundary: proportions→0.0, direction→1 so hnswlib
does not crash. This is documented here; imputation is the caller's responsibility.

Phase 6 reference: HANDOFF_phase6-candlestick-features.md §3
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Public constants ─────────────────────────────────────────────────────────

TIMEFRAMES: list[int] = [1, 3, 5]
FEATURE_NAMES: list[str] = [
    "body_to_range",
    "upper_wick",
    "lower_wick",
    "body_position",
    "direction",
]

CANDLE_COLS: list[str] = [
    f"candle_{tf}d_{feat}"
    for tf in TIMEFRAMES
    for feat in FEATURE_NAMES
]
"""All 15 candlestick column names in deterministic order.

Deterministic order is required for HNSW index consistency — appending or
reordering columns would break any pre-built index.
"""


# ─── Core computation ─────────────────────────────────────────────────────────

def _compute_features_from_arrays(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute 5 candlestick proportion features from raw OHLC arrays.

    Vectorized, no iterrows. Handles NaN, zero-range, and negative-range rows.

    Args:
        o: Open prices, shape (N,).
        h: High prices, shape (N,).
        l: Low prices, shape (N,).
        c: Close prices, shape (N,).

    Returns:
        Dict mapping feature name → float64 array of shape (N,).
        NaN where any input is NaN; 0.0 proportions for zero-range rows.

    Raises:
        RuntimeError: If identity upper_wick + body_to_range + lower_wick != 1.0
                      on any non-NaN row (tolerance 1e-9).
    """
    n = len(o)

    # Detect data errors: high < low
    neg_range_mask = np.isfinite(h) & np.isfinite(l) & (h < l)
    if neg_range_mask.any():
        n_bad = int(neg_range_mask.sum())
        warnings.warn(
            f"Found {n_bad} row(s) where high < low (data error). "
            "Clamping range to 0.0 for affected rows.",
            UserWarning,
            stacklevel=4,
        )
    # Clamp bad rows: treat as zero-range doji
    h_eff = np.where(neg_range_mask, l, h)

    total_range = h_eff - l  # >= 0 by construction

    # NaN mask: any input is NaN → output NaN
    has_nan = ~(np.isfinite(o) & np.isfinite(h_eff) & np.isfinite(l) & np.isfinite(c))
    has_range = total_range > 0
    valid = ~has_nan

    # Pre-allocate NaN outputs
    body_to_range = np.full(n, np.nan, dtype=np.float64)
    upper_wick    = np.full(n, np.nan, dtype=np.float64)
    lower_wick    = np.full(n, np.nan, dtype=np.float64)
    body_position = np.full(n, np.nan, dtype=np.float64)
    direction     = np.full(n, np.nan, dtype=np.float64)

    # Zero-range valid rows → all proportions 0.0, direction=1
    zero_range_valid = valid & ~has_range
    body_to_range[zero_range_valid] = 0.0
    upper_wick[zero_range_valid]    = 0.0
    lower_wick[zero_range_valid]    = 0.0
    body_position[zero_range_valid] = 0.0
    direction[zero_range_valid]     = 1.0

    # Normal rows: valid OHLC with positive range
    normal = valid & has_range
    if normal.any():
        o_n = o[normal]
        h_n = h_eff[normal]
        l_n = l[normal]
        c_n = c[normal]
        r_n = total_range[normal]

        upper_body_n = np.maximum(o_n, c_n)
        lower_body_n = np.minimum(o_n, c_n)

        btr = np.abs(c_n - o_n) / r_n
        uw  = (h_n - upper_body_n) / r_n
        lw  = (lower_body_n - l_n) / r_n
        bp  = (lower_body_n - l_n) / r_n   # same geometry as lower_wick
        dir_ = np.where(c_n >= o_n, 1.0, 0.0)

        body_to_range[normal] = btr
        upper_wick[normal]    = uw
        lower_wick[normal]    = lw
        body_position[normal] = bp
        direction[normal]     = dir_

        # Identity constraint: upper_wick + body_to_range + lower_wick == 1.0
        identity = uw + btr + lw
        max_err = float(np.abs(identity - 1.0).max())
        if max_err >= 1e-9:
            raise RuntimeError(
                f"Candlestick identity constraint violated: "
                f"max(|upper_wick + body_to_range + lower_wick - 1.0|) = {max_err:.2e} "
                f"(tolerance 1e-9). Check for floating-point accumulation."
            )

    return {
        "body_to_range": body_to_range,
        "upper_wick":    upper_wick,
        "lower_wick":    lower_wick,
        "body_position": body_position,
        "direction":     direction,
    }


def _composite_ohlc(
    series_open: pd.Series,
    series_high: pd.Series,
    series_low: pd.Series,
    series_close: pd.Series,
    n_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct composite OHLC for an N-day rolling window.

    For row at position t:
        O = open[t - (n_days-1)]   — first bar's open
        H = max(high[t-(n_days-1) .. t])
        L = min(low[t-(n_days-1)  .. t])
        C = close[t]               — last bar's close

    The first (n_days - 1) rows are NaN (insufficient lookback).

    Args:
        series_open/high/low/close: Per-ticker Series (already date-sorted).
        n_days: Window length (1, 3, or 5).

    Returns:
        (o, h, l, c) arrays, each shape (N,).
    """
    if n_days == 1:
        return (
            series_open.values,
            series_high.values,
            series_low.values,
            series_close.values,
        )

    shift = n_days - 1

    # Composite open: open from (n_days-1) bars ago
    comp_o = series_open.shift(shift).values

    # Composite high/low: rolling max/min over n_days bars
    # min_periods=n_days ensures NaN for incomplete windows at start of history
    comp_h = series_high.rolling(n_days, min_periods=n_days).max().values
    comp_l = series_low.rolling(n_days, min_periods=n_days).min().values

    # Composite close: current bar's close
    comp_c = series_close.values

    return comp_o, comp_h, comp_l, comp_c


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_candlestick_features(
    df: pd.DataFrame,
    ticker_col: str = "Ticker",
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.DataFrame:
    """Compute 15 continuous candlestick features for all rows.

    Processes each ticker independently to prevent rolling windows from
    crossing ticker boundaries. Column names follow CANDLE_COLS ordering.

    Args:
        df: Input DataFrame. Must contain OHLC columns and a Ticker column.
            Column names default to capital-case ("Open", "High", etc.) to
            match prepare.py output. Pass custom names for lowercase data.
            Index is preserved unchanged in the output.
        ticker_col: Column for per-ticker grouping (default "Ticker").
        open_col:   Open price column (default "Open").
        high_col:   High price column (default "High").
        low_col:    Low price column (default "Low").
        close_col:  Close price column (default "Close").

    Returns:
        DataFrame with exactly 15 columns (CANDLE_COLS), same index as input.
        Contains only the new columns — caller joins to the source DataFrame.

    Raises:
        KeyError: If any required column is missing from df.
        RuntimeError: If identity constraint violated on non-NaN rows.
    """
    required = [ticker_col, open_col, high_col, low_col, close_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for candlestick computation: {missing}")

    # Pre-allocate full output with NaN
    result = pd.DataFrame(
        np.nan,
        index=df.index,
        columns=CANDLE_COLS,
        dtype=np.float64,
    )

    for _ticker, group in df.groupby(ticker_col, sort=False):
        idx = group.index  # original index positions for .loc assignment

        for tf in TIMEFRAMES:
            o_arr, h_arr, l_arr, c_arr = _composite_ohlc(
                group[open_col],
                group[high_col],
                group[low_col],
                group[close_col],
                n_days=tf,
            )

            feats = _compute_features_from_arrays(o_arr, h_arr, l_arr, c_arr)

            prefix = f"candle_{tf}d_"
            for feat_name, arr in feats.items():
                result.loc[idx, prefix + feat_name] = arr

    return result
