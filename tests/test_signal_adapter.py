"""
tests/test_signal_adapter.py — Regression tests for trading_system/signal_adapter.py

Covers:
  - UnifiedSignal construction and __post_init__ validation
  - adapt_knn_signals(): field mapping and sector lookup
  - adapt_dl_signals(): confidence penalization, field mapping
  - load_cached_signals() / save_signals() round-trip
  - simulate_signals_from_val_db(): ImportError on missing strategy.py is
    clear and actionable (not a bare ImportError)
"""

import os
import tempfile

import pandas as pd
import pytest

from trading_system.signal_adapter import (
    UnifiedSignal,
    SignalDirection,
    SignalSource,
    adapt_knn_signals,
    adapt_dl_signals,
    load_cached_signals,
    save_signals,
    simulate_signals_from_val_db,
)
from trading_system.config import SECTOR_MAP


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def knn_signal_dicts():
    """Minimal dicts matching adapt_knn_signals() input contract."""
    return [
        {
            "ticker": "AAPL",
            "signal": "BUY",
            "calibrated_prob": 0.72,
            "raw_prob": 0.68,
            "n_matches": 15,
            "mean_7d_return": 0.023,
            "sector": "Tech",
            "regime": "bull",
            "reason": "strong_uptrend",
            "top_analogues": [],
            "date": "2024-01-15",
        },
        {
            "ticker": "JPM",
            "signal": "HOLD",
            "calibrated_prob": 0.55,
            "raw_prob": 0.52,
            "n_matches": 12,
            "mean_7d_return": 0.005,
            "sector": "Finance",
            "regime": "neutral",
            "reason": "insufficient_signal",
            "top_analogues": [],
            "date": "2024-01-15",
        },
    ]


@pytest.fixture
def dl_predictions():
    """Minimal dict matching adapt_dl_signals() input contract."""
    return {
        "date": "2024-01-15",
        "tickers": ["AAPL", "MSFT"],
        "mc_means": [0.80, 0.45],
        "mc_stds": [0.05, 0.20],
        "signals": ["BUY", "HOLD"],
        "confidence_threshold": 0.65,
    }


# ============================================================
# UnifiedSignal construction
# ============================================================

class TestUnifiedSignal:
    """UnifiedSignal must validate confidence and coerce enum types."""

    def test_basic_construction(self):
        sig = UnifiedSignal(
            date=pd.Timestamp("2024-01-15"),
            ticker="AAPL",
            signal=SignalDirection.BUY,
            confidence=0.72,
            signal_source=SignalSource.KNN,
            sector="Tech",
            raw_metadata={},
        )
        assert sig.ticker == "AAPL"
        assert sig.signal == SignalDirection.BUY
        assert sig.confidence == 0.72

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValueError, match="Confidence"):
            UnifiedSignal(
                date=pd.Timestamp("2024-01-15"),
                ticker="AAPL",
                signal=SignalDirection.BUY,
                confidence=1.01,
                signal_source=SignalSource.KNN,
                sector="Tech",
                raw_metadata={},
            )

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValueError, match="Confidence"):
            UnifiedSignal(
                date=pd.Timestamp("2024-01-15"),
                ticker="AAPL",
                signal=SignalDirection.BUY,
                confidence=-0.01,
                signal_source=SignalSource.KNN,
                sector="Tech",
                raw_metadata={},
            )

    def test_string_signal_coerced_to_enum(self):
        sig = UnifiedSignal(
            date=pd.Timestamp("2024-01-15"),
            ticker="AAPL",
            signal="BUY",  # string, not enum
            confidence=0.72,
            signal_source=SignalSource.KNN,
            sector="Tech",
            raw_metadata={},
        )
        assert sig.signal == SignalDirection.BUY

    def test_string_source_coerced_to_enum(self):
        sig = UnifiedSignal(
            date=pd.Timestamp("2024-01-15"),
            ticker="AAPL",
            signal=SignalDirection.BUY,
            confidence=0.72,
            signal_source="KNN",  # string, not enum
            sector="Tech",
            raw_metadata={},
        )
        assert sig.signal_source == SignalSource.KNN

    def test_boundary_confidence_zero_ok(self):
        sig = UnifiedSignal(
            date=pd.Timestamp("2024-01-15"),
            ticker="AAPL",
            signal=SignalDirection.HOLD,
            confidence=0.0,
            signal_source=SignalSource.KNN,
            sector="Tech",
            raw_metadata={},
        )
        assert sig.confidence == 0.0

    def test_boundary_confidence_one_ok(self):
        sig = UnifiedSignal(
            date=pd.Timestamp("2024-01-15"),
            ticker="AAPL",
            signal=SignalDirection.BUY,
            confidence=1.0,
            signal_source=SignalSource.KNN,
            sector="Tech",
            raw_metadata={},
        )
        assert sig.confidence == 1.0


# ============================================================
# adapt_knn_signals()
# ============================================================

class TestAdaptKnnSignals:
    """adapt_knn_signals must map FPPE dict fields to UnifiedSignal correctly."""

    def test_returns_correct_count(self, knn_signal_dicts):
        result = adapt_knn_signals(knn_signal_dicts, SECTOR_MAP)
        assert len(result) == 2

    def test_confidence_mapped_from_calibrated_prob(self, knn_signal_dicts):
        result = adapt_knn_signals(knn_signal_dicts, SECTOR_MAP)
        assert result[0].confidence == pytest.approx(0.72)

    def test_signal_source_is_knn(self, knn_signal_dicts):
        result = adapt_knn_signals(knn_signal_dicts, SECTOR_MAP)
        assert all(s.signal_source == SignalSource.KNN for s in result)

    def test_sector_from_sector_map_overrides_dict(self, knn_signal_dicts):
        """sector_map lookup takes priority over the dict's 'sector' field."""
        result = adapt_knn_signals(knn_signal_dicts, SECTOR_MAP)
        # AAPL is "Tech" in SECTOR_MAP
        assert result[0].sector == "Tech"

    def test_ticker_preserved(self, knn_signal_dicts):
        result = adapt_knn_signals(knn_signal_dicts, SECTOR_MAP)
        assert result[0].ticker == "AAPL"
        assert result[1].ticker == "JPM"

    def test_raw_metadata_populated(self, knn_signal_dicts):
        result = adapt_knn_signals(knn_signal_dicts, SECTOR_MAP)
        meta = result[0].raw_metadata
        assert "n_matches" in meta
        assert meta["n_matches"] == 15

    def test_unknown_ticker_falls_back_to_dict_sector(self):
        signals = [{
            "ticker": "UNKNOWN",
            "signal": "HOLD",
            "calibrated_prob": 0.50,
            "raw_prob": 0.50,
            "n_matches": 10,
            "mean_7d_return": 0.0,
            "sector": "Special",
            "regime": "neutral",
            "reason": "test",
            "top_analogues": [],
            "date": "2024-01-15",
        }]
        result = adapt_knn_signals(signals, SECTOR_MAP)
        assert result[0].sector == "Special"

    def test_empty_list_returns_empty(self):
        assert adapt_knn_signals([], SECTOR_MAP) == []


# ============================================================
# adapt_dl_signals()
# ============================================================

class TestAdaptDlSignals:
    """adapt_dl_signals must penalize confidence by MC dropout uncertainty."""

    def test_returns_correct_count(self, dl_predictions):
        result = adapt_dl_signals(dl_predictions, SECTOR_MAP)
        assert len(result) == 2

    def test_signal_source_is_dl(self, dl_predictions):
        result = adapt_dl_signals(dl_predictions, SECTOR_MAP)
        assert all(s.signal_source == SignalSource.DL for s in result)

    def test_high_uncertainty_reduces_confidence(self, dl_predictions):
        """MSFT: mc_mean=0.45, mc_std=0.20 → penalized confidence < 0.45."""
        result = adapt_dl_signals(dl_predictions, SECTOR_MAP)
        msft_sig = next(s for s in result if s.ticker == "MSFT")
        # raw mc_mean is 0.45; penalization must reduce it
        assert msft_sig.confidence <= 0.45

    def test_low_uncertainty_preserves_confidence(self, dl_predictions):
        """AAPL: mc_mean=0.80, mc_std=0.05 → slight penalty only."""
        result = adapt_dl_signals(dl_predictions, SECTOR_MAP)
        aapl_sig = next(s for s in result if s.ticker == "AAPL")
        # With 5% std, penalty is mc_mean * (1 - 0.1) = 0.72
        assert aapl_sig.confidence == pytest.approx(0.80 * (1.0 - 0.10), abs=1e-6)

    def test_confidence_always_in_0_1(self, dl_predictions):
        result = adapt_dl_signals(dl_predictions, SECTOR_MAP)
        assert all(0.0 <= s.confidence <= 1.0 for s in result)

    def test_raw_metadata_contains_mc_fields(self, dl_predictions):
        result = adapt_dl_signals(dl_predictions, SECTOR_MAP)
        meta = result[0].raw_metadata
        assert "mc_mean" in meta
        assert "mc_std" in meta

    def test_empty_tickers_returns_empty(self):
        predictions = {
            "date": "2024-01-15",
            "tickers": [],
            "mc_means": [],
            "mc_stds": [],
            "signals": [],
        }
        assert adapt_dl_signals(predictions, SECTOR_MAP) == []


# ============================================================
# load_cached_signals() / save_signals()
# ============================================================

class TestSignalCacheIO:
    """load/save round-trip must preserve all required columns."""

    @pytest.fixture
    def sample_signal_df(self):
        return pd.DataFrame([
            {
                "date": pd.Timestamp("2024-01-15"),
                "ticker": "AAPL",
                "signal": "BUY",
                "confidence": 0.72,
                "signal_source": "KNN",
                "sector": "Tech",
                "n_matches": 15,
            },
            {
                "date": pd.Timestamp("2024-01-15"),
                "ticker": "JPM",
                "signal": "HOLD",
                "confidence": 0.55,
                "signal_source": "KNN",
                "sector": "Finance",
                "n_matches": 12,
            },
        ])

    def test_round_trip(self, sample_signal_df):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_signals(sample_signal_df, path)
            loaded = load_cached_signals(path)
            assert list(loaded["ticker"]) == ["AAPL", "JPM"]
            assert list(loaded["signal"]) == ["BUY", "HOLD"]
            assert loaded["confidence"].tolist() == pytest.approx([0.72, 0.55])
        finally:
            os.unlink(path)

    def test_date_column_parsed_as_datetime(self, sample_signal_df):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_signals(sample_signal_df, path)
            loaded = load_cached_signals(path)
            assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
        finally:
            os.unlink(path)

    def test_missing_required_column_raises(self):
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            f.write("date,ticker\n2024-01-15,AAPL\n")
            path = f.name
        try:
            with pytest.raises(ValueError, match="missing required columns"):
                load_cached_signals(path)
        finally:
            os.unlink(path)


# ============================================================
# simulate_signals_from_val_db() — import guard
# ============================================================

class TestSimulateSignalsImportGuard:
    """When strategy.py is missing, must raise ImportError with clear guidance."""

    def test_missing_strategy_gives_actionable_error(self):
        """Setting sys.modules['strategy'] = None simulates the module being absent.

        Python raises ImportError on any 'from strategy import ...' when the module
        is None in sys.modules.  Our guard must catch this and re-raise with a clear
        migration message mentioning 'pattern_engine'.
        """
        import sys
        import unittest.mock

        empty_db = pd.DataFrame()
        # Block 'strategy' by registering it as None — standard import-mock idiom
        with unittest.mock.patch.dict(sys.modules, {"strategy": None}):
            with pytest.raises(ImportError, match="pattern_engine"):
                simulate_signals_from_val_db(empty_db, empty_db, SECTOR_MAP)
