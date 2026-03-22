"""
test_signals_extended.py — Unit tests for NeighborResult and CalibratedProbability.

Tests:
  - NeighborResult: construction, validators, raw_prob property, mean_distance
  - CalibratedProbability: construction, validators

Linear: SLE-57
"""

from datetime import date

import pytest

from pattern_engine.contracts.signals import (
    CalibratedProbability,
    NeighborResult,
    SignalDirection,
    SignalRecord,
    SignalSource,
)


# ─── NeighborResult ────────────────────────────────────────────────────────────

class TestNeighborResult:

    def _make_result(self, k: int = 5, labels=None) -> NeighborResult:
        if labels is None:
            labels = [1, 0, 1, 1, 0]
        return NeighborResult(
            query_ticker="AAPL",
            query_date=date(2024, 1, 2),
            neighbor_indices=list(range(k)),
            neighbor_distances=[0.1 * (i + 1) for i in range(k)],
            neighbor_labels=labels,
            n_neighbors_requested=k,
            n_neighbors_found=k,
        )

    def test_valid_construction(self):
        result = self._make_result()
        assert result.n_neighbors_found == 5
        assert len(result.neighbor_indices) == 5

    def test_frozen(self):
        result = self._make_result()
        with pytest.raises(Exception):
            result.n_neighbors_found = 10  # type: ignore

    def test_raw_prob_is_mean_of_labels(self):
        result = self._make_result(k=5, labels=[1, 0, 1, 1, 0])
        # 3 out of 5 = 0.6
        assert abs(result.raw_prob - 0.6) < 1e-10

    def test_raw_prob_zero_neighbors(self):
        result = NeighborResult(
            query_ticker="MSFT",
            query_date=date(2024, 1, 2),
            neighbor_indices=[],
            neighbor_distances=[],
            neighbor_labels=[],
            n_neighbors_requested=50,
            n_neighbors_found=0,
        )
        assert result.raw_prob == 0.5  # Neutral when no matches

    def test_mean_distance(self):
        result = self._make_result(k=4, labels=[1, 0, 1, 0])
        expected = (0.1 + 0.2 + 0.3 + 0.4) / 4
        assert abs(result.mean_distance - expected) < 1e-10

    def test_array_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="neighbor_indices"):
            NeighborResult(
                query_ticker="AAPL",
                query_date=date(2024, 1, 2),
                neighbor_indices=[0, 1, 2],   # 3 items
                neighbor_distances=[0.1, 0.2, 0.3],
                neighbor_labels=[1, 0, 1],
                n_neighbors_requested=50,
                n_neighbors_found=4,           # Says 4, but arrays have 3
            )

    def test_found_exceeds_requested_raises(self):
        with pytest.raises(ValueError, match="n_neighbors_found"):
            NeighborResult(
                query_ticker="AAPL",
                query_date=date(2024, 1, 2),
                neighbor_indices=[0, 1, 2],
                neighbor_distances=[0.1, 0.2, 0.3],
                neighbor_labels=[1, 0, 1],
                n_neighbors_requested=2,       # Asked for 2
                n_neighbors_found=3,           # Got 3 — impossible
            )

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError, match="negative"):
            NeighborResult(
                query_ticker="AAPL",
                query_date=date(2024, 1, 2),
                neighbor_indices=[0, 1],
                neighbor_distances=[-0.1, 0.2],  # Negative → sqrt() missing
                neighbor_labels=[1, 0],
                n_neighbors_requested=50,
                n_neighbors_found=2,
            )

    def test_non_binary_labels_raise(self):
        with pytest.raises(ValueError, match="0 and 1"):
            NeighborResult(
                query_ticker="AAPL",
                query_date=date(2024, 1, 2),
                neighbor_indices=[0, 1],
                neighbor_distances=[0.1, 0.2],
                neighbor_labels=[0, 2],  # 2 is not binary
                n_neighbors_requested=50,
                n_neighbors_found=2,
            )


# ─── CalibratedProbability ─────────────────────────────────────────────────────

class TestCalibratedProbability:

    def _make_cal(self, calibrated_prob: float = 0.72) -> CalibratedProbability:
        return CalibratedProbability(
            query_ticker="AAPL",
            query_date=date(2024, 1, 2),
            raw_prob=0.68,
            calibrated_prob=calibrated_prob,
            n_neighbors_found=50,
            mean_distance=0.85,
        )

    def test_valid_construction(self):
        cal = self._make_cal()
        assert cal.calibrated_prob == 0.72
        assert cal.calibration_method == "platt"

    def test_frozen(self):
        cal = self._make_cal()
        with pytest.raises(Exception):
            cal.calibrated_prob = 0.5  # type: ignore

    def test_calibrated_prob_bounds(self):
        with pytest.raises(ValueError):
            self._make_cal(calibrated_prob=1.1)
        with pytest.raises(ValueError):
            self._make_cal(calibrated_prob=-0.1)

    def test_ticker_must_be_uppercase(self):
        with pytest.raises(ValueError, match="uppercase"):
            CalibratedProbability(
                query_ticker="aapl",  # lowercase
                query_date=date(2024, 1, 2),
                raw_prob=0.6,
                calibrated_prob=0.65,
                n_neighbors_found=50,
                mean_distance=0.5,
            )

    def test_negative_mean_distance_raises(self):
        with pytest.raises(ValueError):
            CalibratedProbability(
                query_ticker="AAPL",
                query_date=date(2024, 1, 2),
                raw_prob=0.6,
                calibrated_prob=0.65,
                n_neighbors_found=50,
                mean_distance=-0.1,
            )


# ─── SignalRecord (existing — regression guard) ────────────────────────────────

class TestSignalRecordRegression:
    """Ensure existing SignalRecord tests still pass after signals.py extension."""

    def test_basic_construction(self):
        sr = SignalRecord(
            date=date(2024, 1, 2),
            ticker="AAPL",
            signal=SignalDirection.BUY,
            confidence=0.72,
            signal_source=SignalSource.KNN,
            sector="Technology",
            n_matches=50,
            raw_prob=0.68,
            mean_7d_return=1.23,
        )
        assert sr.ticker == "AAPL"
        assert sr.signal == SignalDirection.BUY

    def test_frozen(self):
        sr = SignalRecord(
            date=date(2024, 1, 2), ticker="MSFT", signal=SignalDirection.HOLD,
            confidence=0.55, signal_source=SignalSource.KNN, sector="Technology",
            n_matches=30, raw_prob=0.52, mean_7d_return=0.5,
        )
        with pytest.raises(Exception):
            sr.ticker = "AAPL"  # type: ignore

    def test_lowercase_ticker_raises(self):
        with pytest.raises(ValueError, match="uppercase"):
            SignalRecord(
                date=date(2024, 1, 2), ticker="aapl", signal=SignalDirection.BUY,
                confidence=0.72, signal_source=SignalSource.KNN, sector="Technology",
                n_matches=50, raw_prob=0.68, mean_7d_return=1.23,
            )
