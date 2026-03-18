"""
candlestick.py — Continuous multi-timeframe candlestick encoding.

Translates OHLC data into continuous proportional features that
preserve Euclidean KNN integrity. Unlike discrete candlestick
categories (which create false ordinal distances), continuous
proportions maintain meaningful spatial relationships.

Multi-timeframe encoding:
  - 1-day: standard daily candles
  - 3-day: composite candles (Open=day1 open, Close=day3 close, etc.)
  - 5-day: composite candles (weekly-scale patterns)

Each timeframe produces 5 features:
  - body_to_range (RB/TR): 0=doji → 1=marubozu
  - upper_wick (US/TR): selling pressure rejection
  - lower_wick (LS/TR): demand absorption
  - body_pos: body position within range (high=bullish)
  - direction: +1 bullish, -1 bearish

Source: Gemini Doc 5 — continuous proportional encoding outperforms
13-class discrete labels for distance-based algorithms (Lin et al., 2021).
"""

import numpy as np
import pandas as pd


def compute_candle_features(open_: np.ndarray, high: np.ndarray,
                            low: np.ndarray, close: np.ndarray,
                            prefix: str = "candle_1d") -> dict[str, np.ndarray]:
    """Compute continuous candlestick features from OHLC arrays.

    All features are bounded [0,1] except direction [-1, +1].

    Args:
        open_: array of open prices
        high: array of high prices
        low: array of low prices
        close: array of close prices
        prefix: column name prefix (e.g. "candle_1d", "candle_3d")

    Returns:
        dict mapping column names to feature arrays
    """
    eps = 1e-8
    total_range = high - low + eps
    body = np.abs(close - open_)
    upper_body = np.maximum(open_, close)
    lower_body = np.minimum(open_, close)
    upper_wick = high - upper_body
    lower_wick = lower_body - low

    return {
        f"{prefix}_body_to_range": body / total_range,
        f"{prefix}_upper_wick": upper_wick / total_range,
        f"{prefix}_lower_wick": lower_wick / total_range,
        f"{prefix}_body_pos": (upper_body - low) / total_range,
        f"{prefix}_direction": np.where(close > open_, 1.0, -1.0),
    }


def compute_composite_ohlc(df: pd.DataFrame, window: int) -> tuple:
    """Compute composite OHLC for N-day windows.

    Composite candle: Open = first day's open, Close = last day's close,
    High = max high over window, Low = min low over window.

    Args:
        df: DataFrame with Open, High, Low, Close columns (sorted by date)
        window: number of days to composite

    Returns:
        (open_, high, low, close) arrays aligned to the last day of each window
    """
    open_ = df["Open"].shift(window - 1).values  # first day's open
    high = df["High"].rolling(window).max().values
    low = df["Low"].rolling(window).min().values
    close = df["Close"].values  # last day's close

    return open_, high, low, close


def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all multi-timeframe candlestick features to a DataFrame.

    Computes 1-day, 3-day, and 5-day composite candle features
    (15 features total = 5 per timeframe × 3 timeframes).

    Args:
        df: DataFrame with Open, High, Low, Close columns, sorted by Date

    Returns:
        DataFrame with candlestick feature columns added
    """
    df = df.copy()

    # 1-day candles (standard)
    features_1d = compute_candle_features(
        df["Open"].values, df["High"].values,
        df["Low"].values, df["Close"].values,
        prefix="candle_1d"
    )
    for col, vals in features_1d.items():
        df[col] = vals

    # 3-day composite candles
    o3, h3, l3, c3 = compute_composite_ohlc(df, 3)
    features_3d = compute_candle_features(o3, h3, l3, c3, prefix="candle_3d")
    for col, vals in features_3d.items():
        df[col] = vals

    # 5-day composite candles
    o5, h5, l5, c5 = compute_composite_ohlc(df, 5)
    features_5d = compute_candle_features(o5, h5, l5, c5, prefix="candle_5d")
    for col, vals in features_5d.items():
        df[col] = vals

    return df
