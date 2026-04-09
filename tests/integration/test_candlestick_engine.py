"""
tests/integration/test_candlestick_engine.py — Engine round-trip tests for Phase 6.

Tests:
    test_returns_candle_fit_predict_roundtrip — Full engine cycle with returns_candle.
    test_returns_candle_feature_count         — Feature matrix has exactly 23 columns.
    test_returns_only_unaffected              — Default config still produces 8 columns.

Synthetic dataset: ≥20 rows, 2 tickers, full OHLC + VOL_NORM_COLS + fwd_7d_up.
NaN at the start of composite windows (first 4 rows of each ticker) are imputed
(proportions→0.0, direction→1) before calling fit/query to prevent hnswlib crash.
This is the documented strategy per HANDOFF_phase6-candlestick-features.md §3.3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pattern_engine.candlestick import CANDLE_COLS, compute_candlestick_features
from pattern_engine.config import EngineConfig
from pattern_engine.features import FeatureRegistry, VOL_NORM_COLS
from pattern_engine.matcher import PatternMatcher


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_ohlcv_df(n_rows_per_ticker: int = 25, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic OHLCV + VOL_NORM_COLS + fwd_7d_up DataFrame.

    Two tickers, each with n_rows_per_ticker rows.  OHLC values are realistic
    (open/close near mid-price, high above, low below) so proportion formulas
    produce valid [0,1] outputs.
    """
    rng = np.random.RandomState(seed)
    tickers = ["SYN_A", "SYN_B"]
    rows = []

    for ticker in tickers:
        prices = 100.0 + np.cumsum(rng.randn(n_rows_per_ticker) * 0.5)
        for i in range(n_rows_per_ticker):
            mid = prices[i]
            half_spread = rng.uniform(0.2, 2.0)
            wick_up  = rng.uniform(0.0, 1.0)
            wick_dn  = rng.uniform(0.0, 1.0)
            c = mid + rng.uniform(-half_spread, half_spread)
            o = mid + rng.uniform(-half_spread, half_spread)
            h = max(o, c) + wick_up
            l = min(o, c) - wick_dn
            rows.append({
                "Date":    pd.Timestamp("2020-01-01") + pd.offsets.BDay(i),
                "Ticker":  ticker,
                "Open":  float(o),
                "High":  float(h),
                "Low":   float(l),
                "Close": float(c),
                "fwd_7d_up": float(rng.randint(0, 2)),
            })

    df = pd.DataFrame(rows)
    df.index = range(len(df))  # reset index

    # Add synthetic VOL_NORM_COLS (i.i.d. normal, matching parity test conventions)
    rng2 = np.random.RandomState(seed + 100)
    for col in VOL_NORM_COLS:
        df[col] = rng2.randn(len(df))

    return df


def _build_candle_db(n_rows_per_ticker: int = 25) -> pd.DataFrame:
    """Build full train/val DataFrame with 23-column feature set.

    Steps:
    1. Generate synthetic OHLCV + VOL_NORM_COLS
    2. Compute candlestick features
    3. Concat; impute NaN in candlestick columns (proportions→0.0, direction→1)
    """
    df = _make_ohlcv_df(n_rows_per_ticker=n_rows_per_ticker)
    candle = compute_candlestick_features(df)

    db = pd.concat([df, candle], axis=1)

    # Impute NaN in candlestick proportion columns → 0.0 (zero-range doji semantics)
    # Impute direction NaN → 1.0 (neutral assumption)
    # This prevents hnswlib from crashing on NaN inputs.
    proportion_cols = [c for c in CANDLE_COLS if not c.endswith("_direction")]
    direction_cols  = [c for c in CANDLE_COLS if c.endswith("_direction")]
    db[proportion_cols] = db[proportion_cols].fillna(0.0)
    db[direction_cols]  = db[direction_cols].fillna(1.0)

    return db


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestReturnsCandleRoundTrip:
    """Full PatternMatcher round-trip with the returns_candle feature set."""

    def test_returns_candle_fit_predict_roundtrip(self):
        """fit() + query() complete without error using returns_candle."""
        feature_cols = FeatureRegistry.get("returns_candle").columns
        assert len(feature_cols) == 23  # sanity

        db = _build_candle_db(n_rows_per_ticker=25)
        # Use first 30 rows for train, remaining for val
        train_db = db.iloc[:30].copy()
        val_db   = db.iloc[30:].copy()

        cfg = EngineConfig(
            feature_set="returns_candle",
            use_hnsw=False,  # BallTree for test stability (no hnswlib dep)
            nn_jobs=1,
            max_distance=999.0,  # wide open — synthetic data has large distances
            top_k=5,
        )

        matcher = PatternMatcher(cfg)
        matcher.fit(train_db, feature_cols)

        probs, signals, reasons, n_matches, *_ = matcher.query(val_db, verbose=0)

        assert len(probs) == len(val_db)
        assert len(signals) == len(val_db)
        assert all(isinstance(s, str) for s in signals)
        assert all(0.0 <= float(p) <= 1.0 for p in probs)

    def test_returns_candle_feature_count(self):
        """PatternMatcher._feature_cols has exactly 23 columns after fit."""
        feature_cols = FeatureRegistry.get("returns_candle").columns
        db = _build_candle_db(n_rows_per_ticker=25)
        train_db = db.iloc[:30].copy()

        cfg = EngineConfig(
            feature_set="returns_candle",
            use_hnsw=False,
            nn_jobs=1,
            max_distance=999.0,
            top_k=5,
        )
        matcher = PatternMatcher(cfg)
        matcher.fit(train_db, feature_cols)

        assert len(matcher._feature_cols) == 23
        assert matcher._feature_cols == list(feature_cols)


class TestReturnsOnlyUnaffected:
    """Regression guard: default feature set still produces 8 columns."""

    def test_returns_only_unaffected(self):
        """EngineConfig() with default feature_set produces 8-column fit."""
        feature_cols = FeatureRegistry.get("returns_only").columns
        assert len(feature_cols) == 8

        rng = np.random.RandomState(1)
        n = 60
        df = pd.DataFrame({
            "Date":      pd.date_range("2020-01-01", periods=n, freq="B"),
            "Ticker":    rng.choice(["X", "Y"], size=n),
            "fwd_7d_up": rng.randint(0, 2, size=n).astype(float),
        })
        for col in VOL_NORM_COLS:
            df[col] = rng.randn(n)
        train_db = df.iloc[:40].copy()
        val_db   = df.iloc[40:].copy()

        cfg = EngineConfig(
            use_hnsw=False, nn_jobs=1,
            max_distance=999.0, top_k=5,
        )
        matcher = PatternMatcher(cfg)
        matcher.fit(train_db, feature_cols)

        assert len(matcher._feature_cols) == 8

        probs, signals, *_ = matcher.query(val_db, verbose=0)
        assert len(probs) == len(val_db)
