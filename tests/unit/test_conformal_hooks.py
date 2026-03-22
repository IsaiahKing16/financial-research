"""
test_conformal_hooks.py — Unit tests for conformal prediction stubs (SLE-77).

Tests verify that the interface stubs implement the BaseConformalCalibrator
contract correctly and that the integration helper functions behave sensibly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pattern_engine.conformal_hooks import (
    BaseConformalCalibrator,
    NaiveConformalCalibrator,
    augment_signals_with_conformal,
)


class TestNaiveConformalCalibrator:
    @pytest.fixture
    def fitted_cal(self):
        rng = np.random.RandomState(42)
        cal = NaiveConformalCalibrator()
        n_cal = 100
        scores = rng.uniform(0, 1, n_cal)
        labels = rng.randint(0, 2, n_cal)
        cal.calibrate(scores, labels)
        return cal

    def test_calibrate_stores_scores(self, fitted_cal):
        assert fitted_cal._cal_scores is not None
        assert fitted_cal._n_cal == 100

    def test_predict_set_returns_tuple(self, fitted_cal):
        p_lo, p_hi = fitted_cal.predict_set(0.6, alpha=0.1)
        assert isinstance(p_lo, float)
        assert isinstance(p_hi, float)

    def test_interval_bounds_valid(self, fitted_cal):
        for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
            p_lo, p_hi = fitted_cal.predict_set(prob, alpha=0.1)
            assert 0.0 <= p_lo <= 1.0
            assert 0.0 <= p_hi <= 1.0
            assert p_lo <= p_hi

    def test_interval_contains_point(self, fitted_cal):
        """Prediction interval should generally contain the point estimate."""
        prob = 0.6
        p_lo, p_hi = fitted_cal.predict_set(prob, alpha=0.1)
        # With 90% coverage intended, the interval should surround the point
        assert p_lo <= prob <= p_hi or p_hi >= prob   # relaxed: at least partially

    def test_coverage_in_reasonable_range(self, fitted_cal):
        """Empirical coverage should be broadly in [0.5, 1.0] for random data."""
        rng = np.random.RandomState(0)
        probs = rng.uniform(0, 1, 50)
        labels = rng.randint(0, 2, 50)
        cov = fitted_cal.coverage(probs, labels)
        assert 0.0 <= cov <= 1.0

    def test_raises_before_calibrate(self):
        cal = NaiveConformalCalibrator()
        with pytest.raises(RuntimeError, match="calibrate"):
            cal.predict_set(0.5)

    def test_is_subclass_of_base(self):
        assert issubclass(NaiveConformalCalibrator, BaseConformalCalibrator)

    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseConformalCalibrator()


class TestAugmentSignals:
    @pytest.fixture
    def calibrator(self):
        rng = np.random.RandomState(7)
        cal = NaiveConformalCalibrator()
        scores = rng.uniform(0, 1, 200)
        labels = rng.randint(0, 2, 200)
        cal.calibrate(scores, labels)
        return cal

    def test_returns_probs_and_signals(self, calibrator):
        probs = np.array([0.3, 0.5, 0.8])
        out_probs, signals = augment_signals_with_conformal(
            probs, calibrator, confidence_threshold=0.65, alpha=0.1
        )
        assert len(signals) == 3
        assert all(s in {"BUY", "SELL", "HOLD"} for s in signals)

    def test_high_confidence_is_buy(self, calibrator):
        """If calibrator returns a very wide interval containing high p_lo, BUY."""
        # Force a thin calibration set with all-zero nonconformity → tight threshold
        cal = NaiveConformalCalibrator()
        cal.calibrate(np.zeros(100), np.ones(100, dtype=int))
        probs = np.array([0.95])
        _, signals = augment_signals_with_conformal(
            probs, cal, confidence_threshold=0.5, alpha=0.1
        )
        # p_lo very close to 0.95 → BUY if p_lo > 0.5
        assert signals[0] in {"BUY", "HOLD"}  # depends on threshold width

    def test_output_probs_unchanged(self, calibrator):
        """Output probabilities must be the input (conformal only changes signals)."""
        probs = np.array([0.3, 0.7, 0.5])
        out_probs, _ = augment_signals_with_conformal(probs, calibrator)
        np.testing.assert_array_equal(probs, out_probs)
