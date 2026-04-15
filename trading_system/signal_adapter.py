"""
signal_adapter.py — Unified signal format for the Phase 3Z trading system.

Breaks the production signal_adapter.py's hard dependency on legacy strategy.py.
The rebuilt simulate_signals_from_val_db() uses PatternMatcher directly.

Signal contract:
    UnifiedSignal is upgraded from a plain @dataclass to a Pydantic BaseModel.
    This gives us:
      - Validation on construction (confidence in [0,1], uppercase ticker)
      - JSON serialization for caching and audit logs
      - Type safety at the trading layer boundary

Legacy dependency eliminated:
    Old: simulate_signals_from_val_db() imports from strategy._run_matching_loop,
         fit_platt_scaling, calibrate_probabilities — legacy root-level script.
    New: Uses pattern_engine.matcher.PatternMatcher
         with production EngineConfig (try/import with fallback for isolation).

Linear: SLE-69
"""

from __future__ import annotations

import logging
from datetime import date as Date
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# ─── Signal enums — canonical definitions live in contracts/signals.py ─────────
# Imported (not re-defined) so that isinstance checks are consistent across the
# pattern_engine ↔ trading_system boundary.  I7 (SLE review).
from pattern_engine.contracts.signals import (
    SignalDirection,
    SignalSource,
)


# ─── UnifiedSignal ─────────────────────────────────────────────────────────────

class UnifiedSignal(BaseModel):
    """
    Single normalized signal from FPPE.

    This is the only signal format the trading system sees.
    All model-specific details are preserved in raw_metadata for analysis
    but never used for trading decisions.

    Pydantic v2 (upgraded from plain @dataclass in the production module):
      - Validates confidence ∈ [0, 1] on construction
      - Validates ticker is uppercase
      - JSON-serializable via model_dump_json()

    Args:
        date: Signal generation date.
        ticker: Stock ticker (uppercase).
        signal: BUY / SELL / HOLD.
        confidence: Model confidence [0, 1], source-agnostic.
        signal_source: KNN / DL / ENSEMBLE.
        sector: Sector classification.
        raw_metadata: Model-specific outputs preserved for analysis.
    """
    model_config = {"frozen": True}

    date: Date
    ticker: str = Field(min_length=1, max_length=10)
    signal: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    signal_source: SignalSource
    sector: str = Field(min_length=1)
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        """Tickers must be uppercase (AAPL not aapl)."""
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase; got '{v}'")
        return v


# ─── KNN adapter ───────────────────────────────────────────────────────────────

def adapt_knn_signals(
    signals_list: List[Dict],
    sector_map: Dict[str, str],
) -> List[UnifiedSignal]:
    """
    Convert raw FPPE K-NN signal output to unified format.

    Args:
        signals_list: Output from PatternMatcher.query() or run_live_signals()["signals"].
            Each dict has: ticker, signal, calibrated_prob, raw_prob,
            n_matches, mean_7d_return, sector, regime, reason, top_analogues, date.
        sector_map: Ticker-to-sector mapping from config.

    Returns:
        List of UnifiedSignal objects, one per ticker.
    """
    unified: List[UnifiedSignal] = []
    for s in signals_list:
        raw_date = s.get("date", Date.today())
        if isinstance(raw_date, pd.Timestamp):
            signal_date = raw_date.date()
        elif isinstance(raw_date, Date):
            signal_date = raw_date
        else:
            signal_date = pd.Timestamp(raw_date).date()

        unified.append(UnifiedSignal(
            date=signal_date,
            ticker=s["ticker"],
            signal=SignalDirection(s["signal"]),
            confidence=float(s["calibrated_prob"]),
            signal_source=SignalSource.KNN,
            sector=sector_map.get(s["ticker"], s.get("sector", "Unknown")),
            raw_metadata={
                "raw_prob": s.get("raw_prob"),
                "n_matches": s.get("n_matches"),
                "mean_7d_return": s.get("mean_7d_return"),
                "regime": s.get("regime"),
                "reason": s.get("reason"),
                "top_analogues": s.get("top_analogues", []),
            },
        ))
    return unified


def adapt_dl_signals(
    predictions: Dict,
    sector_map: Dict[str, str],
) -> List[UnifiedSignal]:
    """
    Convert deep learning model output to unified format.

    Args:
        predictions: Dict with keys:
            date, tickers, mc_means, mc_stds, signals, confidence_threshold.
        sector_map: Ticker-to-sector mapping from config.

    Returns:
        List of UnifiedSignal objects.
    """
    unified: List[UnifiedSignal] = []
    raw_date = predictions["date"]
    if isinstance(raw_date, pd.Timestamp):
        signal_date = raw_date.date()
    elif isinstance(raw_date, Date):
        signal_date = raw_date
    else:
        signal_date = pd.Timestamp(raw_date).date()

    for i, ticker in enumerate(predictions["tickers"]):
        mc_mean = float(predictions["mc_means"][i])
        mc_std = float(predictions["mc_stds"][i])
        signal_str = predictions["signals"][i]

        # DL confidence: penalize by uncertainty (high std = lower effective confidence)
        confidence = mc_mean * (1.0 - min(mc_std * 2, 0.5))
        confidence = max(0.0, min(1.0, confidence))

        unified.append(UnifiedSignal(
            date=signal_date,
            ticker=ticker,
            signal=SignalDirection(signal_str),
            confidence=confidence,
            signal_source=SignalSource.DL,
            sector=sector_map.get(ticker, "Unknown"),
            raw_metadata={
                "mc_mean": mc_mean,
                "mc_std": mc_std,
                "raw_confidence": mc_mean,
            },
        ))
    return unified


# ─── Backtesting signal generation ────────────────────────────────────────────

def simulate_signals_from_val_db(
    val_db: pd.DataFrame,
    train_db: pd.DataFrame,
    sector_map: Dict[str, str],
    confidence_threshold: float = 0.65,
    agreement_spread: float = 0.10,
    min_matches: int = 10,
    engine_config=None,
) -> pd.DataFrame:
    """
    Generate a signal DataFrame from the validation database for backtesting.

    Replaces the legacy implementation (which required strategy.py) with a
    clean PatternMatcher call.

    This runs the FPPE analogue matching pipeline on each row of val_db
    and produces one signal row per ticker per day. For Phase 1 equal-weight
    backtesting, we need the full signal stream.

    NOTE: This is a SIMULATION for backtesting purposes. In production,
    signals come from run_live_signals(). This function reproduces
    what those signals would have been historically.

    CALIBRATION NOTE: PatternMatcher.fit() runs the standard calibration
    double-pass (train-as-query → PlattCalibrator fitted on raw frequencies).
    The `confidence` column returned here is Platt-calibrated, matching the
    production PatternEngine.predict() output.  Wired in M8 (SLE-89).

    Args:
        val_db: Validation DataFrame with OHLC + return features + forward columns.
        train_db: Training DataFrame (the analogue search database).
        sector_map: Ticker-to-sector mapping.
        confidence_threshold: Min confidence for BUY/SELL (PatternMatcher gate).
        agreement_spread: Min agreement for signal generation.
        min_matches: Min analogues required.
        engine_config: Optional EngineConfig instance. If None, uses production
                       EngineConfig defaults (override confidence_threshold,
                       agreement_spread, min_matches from parameters).

    Returns:
        DataFrame with unified signal columns:
        [date, ticker, signal, confidence, signal_source, sector, n_matches,
         raw_prob, mean_7d_return]

    Raises:
        ImportError: If neither production EngineConfig nor the rebuild can be
                     imported (should not happen in a correctly installed venv).
    """
    from pattern_engine.matcher import PatternMatcher
    from pattern_engine.features import RETURNS_ONLY_COLS

    # ── Resolve EngineConfig ───────────────────────────────────────────────────
    if engine_config is None:
        try:
            from pattern_engine.config import EngineConfig
        except ImportError:
            raise ImportError(
                "simulate_signals_from_val_db() requires pattern_engine.EngineConfig. "
                "Activate the project venv: venv\\Scripts\\activate"
            )

        engine_config = EngineConfig(
            confidence_threshold=confidence_threshold,
            agreement_spread=agreement_spread,
            min_matches=min_matches,
            use_hnsw=False,          # BallTree for determinism
            nn_jobs=1,               # nn_jobs=1 — Windows/Py3.12 joblib safety
            exclude_same_ticker=True,
            regime_filter=True,
        )

    # ── Determine feature columns ─────────────────────────────────────────────
    # Use RETURNS_ONLY_COLS that exist in both databases (parity with production)
    feature_cols = [c for c in RETURNS_ONLY_COLS if c in train_db.columns and c in val_db.columns]
    if not feature_cols:
        raise RuntimeError(
            f"No RETURNS_ONLY_COLS found in train_db or val_db. "
            f"Expected columns like {RETURNS_ONLY_COLS[:3]}…"
        )

    # ── Fit and query ─────────────────────────────────────────────────────────
    matcher = PatternMatcher(engine_config)
    _log.info("Fitting PatternMatcher on training set...")
    matcher.fit(train_db, feature_cols)

    _log.info("Running analogue matching on %d validation rows...", len(val_db))
    (
        probabilities,
        signals,
        reasons,
        n_matches,
        mean_returns,
        ensemble_list,
    ) = matcher.query(val_db, verbose=1)

    # ── Build output DataFrame ─────────────────────────────────────────────────
    rows = []
    for i in range(len(val_db)):
        raw_date = val_db.iloc[i]["Date"]
        if hasattr(raw_date, "date"):
            signal_date = raw_date.date()
        elif hasattr(raw_date, "astype"):
            signal_date = pd.Timestamp(raw_date).date()
        else:
            signal_date = raw_date

        rows.append({
            "date": signal_date,
            "ticker": val_db.iloc[i]["Ticker"],
            "signal": signals[i] if i < len(signals) else "HOLD",
            "confidence": float(probabilities[i]),
            "signal_source": "KNN",
            "sector": sector_map.get(val_db.iloc[i]["Ticker"], "Unknown"),
            "n_matches": n_matches[i] if i < len(n_matches) else 0,
            "raw_prob": float(probabilities[i]),
            "mean_7d_return": float(mean_returns[i]) if i < len(mean_returns) else 0.0,
        })

    signal_df = pd.DataFrame(rows)
    signal_df["date"] = pd.to_datetime(signal_df["date"])

    # Summary stats
    buy_count = (signal_df["signal"] == "BUY").sum()
    sell_count = (signal_df["signal"] == "SELL").sum()
    hold_count = (signal_df["signal"] == "HOLD").sum()
    _log.info("Signal summary: %d BUY, %d SELL, %d HOLD", buy_count, sell_count, hold_count)
    _log.info("Signal rate: %.1f%% actionable", (buy_count + sell_count) / len(signal_df) * 100)

    return signal_df


# ─── Caching helpers ───────────────────────────────────────────────────────────

def load_cached_signals(filepath: str) -> pd.DataFrame:
    """
    Load previously generated signals from CSV.

    For development iteration, generating signals is slow. This loads a
    cached signal file so the backtester can be developed independently.

    Args:
        filepath: Path to the cached signal CSV.

    Returns:
        DataFrame with at least [date, ticker, signal, confidence].

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(filepath, parse_dates=["date"])
    required_cols = ["date", "ticker", "signal", "confidence"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Signal file missing required columns: {missing}")
    return df


def save_signals(signal_df: pd.DataFrame, filepath: str) -> None:
    """
    Save generated signals to CSV for caching.

    Args:
        signal_df: Signal DataFrame to save.
        filepath: Destination path.
    """
    signal_df.to_csv(filepath, index=False)
    _log.info("Signals saved to %s", filepath)
