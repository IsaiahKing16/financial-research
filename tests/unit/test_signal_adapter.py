"""
test_signal_adapter.py — Unit tests for the rebuilt signal adapter (SLE-69).

Tests:
  - UnifiedSignal Pydantic model: validation, immutability, JSON round-trip
  - adapt_knn_signals(): correct field mapping from raw K-NN output
  - adapt_dl_signals(): confidence penalization by MC dropout std
  - load_cached_signals() / save_signals(): CSV round-trip
  - simulate_signals_from_val_db(): skipped in unit context (requires full env)

Note on simulate_signals_from_val_db:
  This function requires production EngineConfig + PatternMatcher + yfinance data.
  Full integration is tested in the parity test suite. Here we test the
  adapter contract (UnifiedSignal) and lightweight helpers only.

Linear: SLE-69
"""

from __future__ import annotations

import csv
import json
import tempfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from trading_system.signal_adapter import (
    SignalDirection,
    SignalSource,
    UnifiedSignal,
    adapt_dl_signals,
    adapt_knn_signals,
    load_cached_signals,
    save_signals,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JPM": "Financials",
    "XOM": "Energy",
}


def _make_knn_signal_dict(
    ticker: str = "AAPL",
    signal: str = "BUY",
    calibrated_prob: float = 0.72,
    n_matches: int = 42,
    mean_7d_return: float = 0.025,
    signal_date: date = date(2026, 3, 21),
) -> dict:
    return {
        "ticker": ticker,
        "signal": signal,
        "calibrated_prob": calibrated_prob,
        "raw_prob": calibrated_prob - 0.02,
        "n_matches": n_matches,
        "mean_7d_return": mean_7d_return,
        "sector": SECTOR_MAP.get(ticker, "Unknown"),
        "regime": 1,
        "reason": "High confidence BUY",
        "top_analogues": [1, 2, 3],
        "date": signal_date,
    }


# ─── TestUnifiedSignal ────────────────────────────────────────────────────────

class TestUnifiedSignal:
    """UnifiedSignal Pydantic model tests."""

    def test_basic_construction(self):
        """Valid fields construct without error."""
        sig = UnifiedSignal(
            date=date(2026, 3, 21),
            ticker="AAPL",
            signal=SignalDirection.BUY,
            confidence=0.72,
            signal_source=SignalSource.KNN,
            sector="Technology",
        )
        assert sig.ticker == "AAPL"
        assert sig.confidence == 0.72
        assert sig.signal == SignalDirection.BUY

    def test_is_frozen(self):
        """UnifiedSignal is immutable after construction."""
        sig = UnifiedSignal(
            date=date(2026, 3, 21),
            ticker="AAPL",
            signal=SignalDirection.BUY,
            confidence=0.72,
            signal_source=SignalSource.KNN,
            sector="Technology",
        )
        with pytest.raises(Exception):  # frozen model raises ValidationError or TypeError
            sig.confidence = 0.99  # type: ignore[misc]

    def test_confidence_below_zero_rejected(self):
        """confidence < 0 raises ValueError."""
        with pytest.raises(Exception):
            UnifiedSignal(
                date=date(2026, 3, 21),
                ticker="AAPL",
                signal=SignalDirection.BUY,
                confidence=-0.1,
                signal_source=SignalSource.KNN,
                sector="Technology",
            )

    def test_confidence_above_one_rejected(self):
        """confidence > 1 raises ValueError."""
        with pytest.raises(Exception):
            UnifiedSignal(
                date=date(2026, 3, 21),
                ticker="AAPL",
                signal=SignalDirection.BUY,
                confidence=1.01,
                signal_source=SignalSource.KNN,
                sector="Technology",
            )

    def test_lowercase_ticker_rejected(self):
        """Lowercase ticker raises ValueError."""
        with pytest.raises(Exception):
            UnifiedSignal(
                date=date(2026, 3, 21),
                ticker="aapl",  # must be uppercase
                signal=SignalDirection.BUY,
                confidence=0.72,
                signal_source=SignalSource.KNN,
                sector="Technology",
            )

    def test_all_signal_directions(self):
        """BUY, SELL, HOLD all construct successfully."""
        for direction in SignalDirection:
            sig = UnifiedSignal(
                date=date(2026, 3, 21),
                ticker="AAPL",
                signal=direction,
                confidence=0.5,
                signal_source=SignalSource.KNN,
                sector="Technology",
            )
            assert sig.signal == direction

    def test_all_signal_sources(self):
        """KNN, DL, ENSEMBLE sources all construct successfully."""
        for source in SignalSource:
            sig = UnifiedSignal(
                date=date(2026, 3, 21),
                ticker="AAPL",
                signal=SignalDirection.HOLD,
                confidence=0.5,
                signal_source=source,
                sector="Technology",
            )
            assert sig.signal_source == source

    def test_json_round_trip(self):
        """UnifiedSignal can be serialized and deserialized via Pydantic JSON."""
        sig = UnifiedSignal(
            date=date(2026, 3, 21),
            ticker="AAPL",
            signal=SignalDirection.BUY,
            confidence=0.72,
            signal_source=SignalSource.KNN,
            sector="Technology",
            raw_metadata={"n_matches": 42, "raw_prob": 0.70},
        )
        json_str = sig.model_dump_json()
        restored = UnifiedSignal.model_validate_json(json_str)
        assert restored.ticker == sig.ticker
        assert restored.confidence == sig.confidence
        assert restored.date == sig.date
        assert restored.raw_metadata["n_matches"] == 42

    def test_raw_metadata_defaults_to_empty_dict(self):
        """raw_metadata defaults to {} when not provided."""
        sig = UnifiedSignal(
            date=date(2026, 3, 21),
            ticker="AAPL",
            signal=SignalDirection.HOLD,
            confidence=0.5,
            signal_source=SignalSource.KNN,
            sector="Technology",
        )
        assert sig.raw_metadata == {}


# ─── TestAdaptKnnSignals ──────────────────────────────────────────────────────

class TestAdaptKnnSignals:
    """adapt_knn_signals() unit tests."""

    def test_basic_conversion(self):
        """Single K-NN dict converts to UnifiedSignal correctly."""
        raw = [_make_knn_signal_dict()]
        signals = adapt_knn_signals(raw, SECTOR_MAP)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.ticker == "AAPL"
        assert sig.signal == SignalDirection.BUY
        assert sig.confidence == 0.72
        assert sig.signal_source == SignalSource.KNN
        assert sig.sector == "Technology"

    def test_multiple_signals(self):
        """Multiple dicts all convert."""
        raw = [
            _make_knn_signal_dict("AAPL", "BUY"),
            _make_knn_signal_dict("MSFT", "HOLD", calibrated_prob=0.55),
            _make_knn_signal_dict("JPM", "SELL", calibrated_prob=0.35),
        ]
        signals = adapt_knn_signals(raw, SECTOR_MAP)
        assert len(signals) == 3

    def test_sector_map_lookup(self):
        """Sector is taken from sector_map when available."""
        raw = [_make_knn_signal_dict("AAPL")]
        signals = adapt_knn_signals(raw, {"AAPL": "Consumer Electronics"})
        assert signals[0].sector == "Consumer Electronics"

    def test_sector_fallback_to_unknown(self):
        """Sector falls back to 'Unknown' for unmapped tickers."""
        raw = [_make_knn_signal_dict("ZZZ")]  # valid ticker, not in sector map
        signals = adapt_knn_signals(raw, {})
        assert signals[0].sector == "Unknown"

    def test_raw_metadata_preserved(self):
        """n_matches, mean_7d_return, etc. are stored in raw_metadata."""
        raw = [_make_knn_signal_dict(n_matches=37, mean_7d_return=0.033)]
        signals = adapt_knn_signals(raw, SECTOR_MAP)
        meta = signals[0].raw_metadata
        assert meta["n_matches"] == 37
        assert meta["mean_7d_return"] == 0.033

    def test_date_conversion_from_date_object(self):
        """date objects are handled correctly."""
        raw = [_make_knn_signal_dict(signal_date=date(2026, 3, 21))]
        signals = adapt_knn_signals(raw, SECTOR_MAP)
        assert isinstance(signals[0].date, date)
        assert signals[0].date == date(2026, 3, 21)

    def test_date_conversion_from_timestamp(self):
        """pd.Timestamp dates are converted to date."""
        raw = [_make_knn_signal_dict(signal_date=pd.Timestamp("2026-03-21"))]
        signals = adapt_knn_signals(raw, SECTOR_MAP)
        assert isinstance(signals[0].date, date)
        assert signals[0].date == date(2026, 3, 21)

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list."""
        assert adapt_knn_signals([], SECTOR_MAP) == []


# ─── TestAdaptDlSignals ───────────────────────────────────────────────────────

class TestAdaptDlSignals:
    """adapt_dl_signals() unit tests."""

    def _make_dl_predictions(
        self,
        tickers=("AAPL", "MSFT"),
        mc_means=(0.7, 0.45),
        mc_stds=(0.05, 0.20),
        signals=("BUY", "HOLD"),
        pred_date=date(2026, 3, 21),
    ):
        return {
            "date": pred_date,
            "tickers": list(tickers),
            "mc_means": list(mc_means),
            "mc_stds": list(mc_stds),
            "signals": list(signals),
            "confidence_threshold": 0.65,
        }

    def test_basic_conversion(self):
        """DL predictions convert to UnifiedSignal."""
        preds = self._make_dl_predictions()
        signals = adapt_dl_signals(preds, SECTOR_MAP)
        assert len(signals) == 2
        assert signals[0].signal_source == SignalSource.DL

    def test_confidence_penalized_by_std(self):
        """High MC std reduces confidence below raw mc_mean."""
        preds = self._make_dl_predictions(
            mc_means=(0.80,),
            mc_stds=(0.40,),  # high uncertainty
            tickers=("AAPL",),
            signals=("BUY",),
        )
        signals = adapt_dl_signals(preds, SECTOR_MAP)
        # mc_mean=0.80, penalty = 1 - min(0.40*2, 0.5) = 1 - 0.50 = 0.50
        # confidence = 0.80 * 0.50 = 0.40
        assert signals[0].confidence < 0.80

    def test_confidence_clamped_to_zero_one(self):
        """Confidence is always in [0, 1]."""
        preds = self._make_dl_predictions(
            mc_means=(0.01,),
            mc_stds=(1.0,),
            tickers=("AAPL",),
            signals=("HOLD",),
        )
        signals = adapt_dl_signals(preds, SECTOR_MAP)
        assert 0.0 <= signals[0].confidence <= 1.0

    def test_date_conversion_from_timestamp(self):
        """pd.Timestamp date is correctly converted."""
        preds = self._make_dl_predictions(pred_date=pd.Timestamp("2026-03-21"))
        signals = adapt_dl_signals(preds, SECTOR_MAP)
        assert isinstance(signals[0].date, date)


# ─── TestCachingHelpers ───────────────────────────────────────────────────────

class TestCachingHelpers:
    """load_cached_signals / save_signals CSV round-trip."""

    def _make_signal_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": pd.to_datetime(["2026-03-21", "2026-03-21"]),
            "ticker": ["AAPL", "MSFT"],
            "signal": ["BUY", "HOLD"],
            "confidence": [0.72, 0.55],
            "signal_source": ["KNN", "KNN"],
            "sector": ["Technology", "Technology"],
            "n_matches": [42, 38],
        })

    def test_round_trip(self):
        """save_signals / load_cached_signals preserves required columns."""
        df = self._make_signal_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "signals.csv")
            save_signals(df, filepath)
            loaded = load_cached_signals(filepath)
        assert set(["date", "ticker", "signal", "confidence"]).issubset(loaded.columns)
        assert len(loaded) == 2
        assert "AAPL" in loaded["ticker"].values

    def test_load_raises_on_missing_columns(self):
        """load_cached_signals raises ValueError if required columns are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "bad_signals.csv")
            # Write CSV without 'confidence' column
            pd.DataFrame({"date": ["2026-03-21"], "ticker": ["AAPL"], "signal": ["BUY"]}).to_csv(
                filepath, index=False
            )
            with pytest.raises(ValueError, match="missing required columns"):
                load_cached_signals(filepath)

    def test_date_column_parsed(self):
        """Date column is parsed to datetime after load."""
        df = self._make_signal_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "signals.csv")
            save_signals(df, filepath)
            loaded = load_cached_signals(filepath)
        assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
