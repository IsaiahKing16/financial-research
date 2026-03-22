"""Tests for pattern_engine.calibration."""

import numpy as np
import pytest
from pattern_engine.calibration import (
    PlattCalibrator, IsotonicCalibrator, NoCalibrator, make_calibrator,
)


@pytest.fixture
def calibration_data():
    """Synthetic calibration data."""
    rng = np.random.RandomState(42)
    raw_probs = rng.rand(200)
    y_true = (raw_probs > 0.5 + rng.randn(200) * 0.1).astype(int)
    return raw_probs, y_true


class TestPlattCalibrator:
    def test_fit_transform(self, calibration_data):
        raw_probs, y_true = calibration_data
        cal = PlattCalibrator()
        assert not cal.fitted
        cal.fit(raw_probs, y_true)
        assert cal.fitted
        result = cal.transform(raw_probs)
        assert result.shape == raw_probs.shape
        assert np.all((result >= 0) & (result <= 1))

    def test_fit_returns_self(self, calibration_data):
        raw_probs, y_true = calibration_data
        cal = PlattCalibrator()
        result = cal.fit(raw_probs, y_true)
        assert result is cal


class TestIsotonicCalibrator:
    def test_fit_transform(self, calibration_data):
        raw_probs, y_true = calibration_data
        cal = IsotonicCalibrator()
        assert not cal.fitted
        cal.fit(raw_probs, y_true)
        assert cal.fitted
        result = cal.transform(raw_probs)
        assert result.shape == raw_probs.shape
        assert np.all((result >= 0) & (result <= 1))


class TestNoCalibrator:
    def test_identity(self):
        cal = NoCalibrator()
        assert cal.fitted  # Always fitted
        raw = np.array([0.1, 0.5, 0.9])
        result = cal.transform(raw)
        np.testing.assert_array_equal(result, raw)

    def test_fit_returns_self(self):
        cal = NoCalibrator()
        result = cal.fit(np.array([0.5]), np.array([1]))
        assert result is cal


class TestMakeCalibrator:
    def test_platt(self):
        cal = make_calibrator("platt")
        assert isinstance(cal, PlattCalibrator)

    def test_isotonic(self):
        cal = make_calibrator("isotonic")
        assert isinstance(cal, IsotonicCalibrator)

    def test_none(self):
        cal = make_calibrator("none")
        assert isinstance(cal, NoCalibrator)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown calibration method"):
            make_calibrator("unknown")
