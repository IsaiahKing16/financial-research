"""
signal_adapter.py — Normalizes FPPE output into a unified signal format.

FPPE v2.1 is a hybrid system:
  - K-NN analogue matching: outputs n_matches, agreement_spread, calibrated_prob
  - Conv1D + LSTM (MC Dropout): outputs probability distributions, mc_mean, mc_std

This adapter decouples the trading system from FPPE internals. If FPPE changes
models, adds ensemble methods, or adjusts output format, only this file changes.
The four trading layers never see model-specific fields.

Design doc reference: FPPE_TRADING_SYSTEM_DESIGN.md v0.4, Section 5.1
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class SignalDirection(Enum):
    """Trade signal direction."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalSource(Enum):
    """Which FPPE model generated the signal."""
    KNN = "KNN"
    DL = "DL"
    ENSEMBLE = "ENSEMBLE"


@dataclass
class UnifiedSignal:
    """Single normalized signal from FPPE.

    This is the only signal format the trading system sees.
    All model-specific details are preserved in raw_metadata
    for analysis but never used for trading decisions.
    """
    date: pd.Timestamp
    ticker: str
    signal: SignalDirection
    confidence: float              # 0.0 to 1.0, source-agnostic
    signal_source: SignalSource
    sector: str
    raw_metadata: Dict[str, Any]   # Model-specific outputs preserved for analysis

    def __post_init__(self):
        """Validate signal fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be [0,1], got {self.confidence}")
        if not isinstance(self.signal, SignalDirection):
            self.signal = SignalDirection(self.signal)
        if not isinstance(self.signal_source, SignalSource):
            self.signal_source = SignalSource(self.signal_source)


def adapt_knn_signals(signals_list: List[Dict], sector_map: Dict[str, str]) -> List[UnifiedSignal]:
    """Convert raw FPPE K-NN signal output to unified format.

    Args:
        signals_list: Output from strategy.run_live_signals()["signals"].
            Each dict has: ticker, signal, calibrated_prob, raw_prob,
            n_matches, mean_7d_return, sector, regime, reason, top_analogues

        sector_map: Ticker-to-sector mapping from config.

    Returns:
        List of UnifiedSignal objects, one per ticker.
    """
    unified = []
    for s in signals_list:
        unified.append(UnifiedSignal(
            date=pd.Timestamp(s.get("date", pd.Timestamp.now())),
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


def adapt_dl_signals(predictions: Dict, sector_map: Dict[str, str]) -> List[UnifiedSignal]:
    """Convert deep learning model output to unified format.

    Args:
        predictions: Dict with keys:
            - date: signal date
            - tickers: list of ticker strings
            - mc_means: array of MC dropout mean probabilities
            - mc_stds: array of MC dropout standard deviations
            - signals: list of BUY/SELL/HOLD strings
            - confidence_threshold: threshold used for signal generation

        sector_map: Ticker-to-sector mapping from config.

    Returns:
        List of UnifiedSignal objects.
    """
    unified = []
    date = pd.Timestamp(predictions["date"])

    for i, ticker in enumerate(predictions["tickers"]):
        mc_mean = float(predictions["mc_means"][i])
        mc_std = float(predictions["mc_stds"][i])
        signal_str = predictions["signals"][i]

        # DL confidence: use MC dropout mean as probability,
        # penalized by uncertainty (high std = lower effective confidence)
        # This maps mc_mean to a [0,1] range that accounts for model uncertainty
        confidence = mc_mean * (1.0 - min(mc_std * 2, 0.5))
        confidence = max(0.0, min(1.0, confidence))

        unified.append(UnifiedSignal(
            date=date,
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


def simulate_signals_from_val_db(
    val_db: pd.DataFrame,
    train_db: pd.DataFrame,
    sector_map: Dict[str, str],
    confidence_threshold: float = 0.65,
    agreement_spread: float = 0.10,
    min_matches: int = 10,
) -> pd.DataFrame:
    """Generate a signal DataFrame from the validation database for backtesting.

    This runs the FPPE analogue matching pipeline on each row of val_db
    and produces one signal per ticker per day. For Phase 1 equal-weight
    backtesting, we need the full signal stream.

    NOTE: This is a SIMULATION for backtesting purposes. In production,
    signals come from run_live_signals(). This function reproduces
    what those signals would have been historically.

    Args:
        val_db: Validation DataFrame with OHLC + return features + forward columns.
        train_db: Training DataFrame (the analogue search database).
        sector_map: Ticker-to-sector mapping.
        confidence_threshold: Min confidence for BUY/SELL (FPPE's gate).
        agreement_spread: Min agreement for signal generation.
        min_matches: Min analogues required.

    Returns:
        DataFrame with unified signal columns:
        [date, ticker, signal, confidence, signal_source, sector, n_matches]
    """
    # Import FPPE internals for signal generation.
    #
    # DEPENDENCY NOTE (Issue #6): This function depends on root-level strategy.py
    # (the legacy FPPE entrypoint).  On the v2.2 branch, strategy.py was moved to
    # archive/.  If that branch merges before this function is ported to the new
    # pattern_engine API, this import will raise ImportError at call time.
    #
    # Migration path: replace this block with pattern_engine.engine.PatternEngine
    # calls once the trading_system↔pattern_engine integration is complete (Phase 2).
    import sys
    from pathlib import Path

    # Add parent dir to path so we can import strategy
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from strategy import (
            _run_matching_loop, fit_platt_scaling, calibrate_probabilities,
            generate_signal, FEATURE_WEIGHTS, PROJECTION_HORIZON,
            MAX_DISTANCE, TOP_K, DISTANCE_WEIGHTING, AGREEMENT_SPREAD,
            MIN_MATCHES, CONFIDENCE_THRESHOLD,
        )
        from prepare import FEATURE_COLS
    except ImportError as exc:
        raise ImportError(
            "simulate_signals_from_val_db() requires root-level strategy.py. "
            "If strategy.py has been moved to archive/, migrate this function to "
            "use pattern_engine.engine.PatternEngine instead. "
            f"Original error: {exc}"
        ) from exc
    from sklearn.preprocessing import StandardScaler

    RETURN_ONLY = [c for c in FEATURE_COLS if c.startswith("ret_")]

    # Fit scaler and calibrator on training data
    scaler = StandardScaler()
    scaler.fit(train_db[RETURN_ONLY].values)

    print("  Fitting Platt calibrator on training set...")
    train_probs, _, _, _, _, _ = _run_matching_loop(
        train_db, train_db, scaler, FEATURE_COLS, verbose=0,
        max_distance=MAX_DISTANCE,
        metric_override="euclidean",
        feature_cols_override=RETURN_ONLY,
        top_k=TOP_K, distance_weighting=DISTANCE_WEIGHTING,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        agreement_spread=AGREEMENT_SPREAD,
        exclude_same_ticker=True,
        regime_filter=True,
    )
    train_y_bin = train_db[PROJECTION_HORIZON].values
    calibrator = fit_platt_scaling(train_probs, train_y_bin)

    # Run matching on validation data
    print(f"  Running analogue matching on {len(val_db):,} validation rows...")
    raw_probs, signals, reasons, n_matches, mean_rets, ensembles = _run_matching_loop(
        train_db, val_db, scaler, FEATURE_COLS, verbose=1,
        max_distance=MAX_DISTANCE,
        metric_override="euclidean",
        feature_cols_override=RETURN_ONLY,
        top_k=TOP_K, distance_weighting=DISTANCE_WEIGHTING,
        confidence_threshold=confidence_threshold,
        agreement_spread=agreement_spread,
        regime_filter=True,
    )

    # Calibrate probabilities
    cal_probs = calibrate_probabilities(calibrator, raw_probs)

    # Re-generate signals with calibrated probabilities
    final_signals = []
    for i in range(len(val_db)):
        cal_p = float(cal_probs[i])
        nm = int(n_matches[i])

        proj_mock = {
            "probability_up": cal_p,
            "agreement": abs(cal_p - 0.5) * 2,
            "n_matches": nm,
            "mean_return": float(mean_rets[i]),
            "median_return": float(mean_rets[i]),
            "ensemble_returns": ensembles[i],
        }
        signal, reason = generate_signal(
            proj_mock,
            threshold=confidence_threshold,
            min_agreement=agreement_spread,
            min_matches=min_matches,
        )

        final_signals.append({
            "date": val_db.iloc[i]["Date"],
            "ticker": val_db.iloc[i]["Ticker"],
            "signal": signal,
            "confidence": cal_p,
            "signal_source": "KNN",
            "sector": sector_map.get(val_db.iloc[i]["Ticker"], "Unknown"),
            "n_matches": nm,
            "raw_prob": float(raw_probs[i]),
            "mean_7d_return": float(mean_rets[i]),
        })

    signal_df = pd.DataFrame(final_signals)
    signal_df["date"] = pd.to_datetime(signal_df["date"])

    # Summary stats
    buy_count = (signal_df["signal"] == "BUY").sum()
    sell_count = (signal_df["signal"] == "SELL").sum()
    hold_count = (signal_df["signal"] == "HOLD").sum()
    print(f"\n  Signal summary: {buy_count} BUY, {sell_count} SELL, {hold_count} HOLD")
    print(f"  Signal rate: {(buy_count + sell_count) / len(signal_df):.1%} actionable")

    return signal_df


def load_cached_signals(filepath: str) -> pd.DataFrame:
    """Load previously generated signals from CSV.

    For development iteration, generating signals is slow (requires full
    K-NN matching). This loads a cached signal file so the backtester
    can be developed and tested independently.
    """
    df = pd.read_csv(filepath, parse_dates=["date"])
    required_cols = ["date", "ticker", "signal", "confidence"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Signal file missing required columns: {missing}")
    return df


def save_signals(signal_df: pd.DataFrame, filepath: str) -> None:
    """Save generated signals to CSV for caching."""
    signal_df.to_csv(filepath, index=False)
    print(f"  Signals saved to {filepath}")
