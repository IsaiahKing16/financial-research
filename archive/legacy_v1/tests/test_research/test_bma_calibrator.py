"""Smoke tests for BMACalibrator — correctness and interface contract."""

import numpy as np
import pytest

from research.bma_calibrator import BMACalibrator


def _make_data(n_samples=200, n_analogs=5, seed=0):
    rng = np.random.default_rng(seed)
    raw_probs = rng.uniform(0.1, 0.9, (n_samples, n_analogs))
    y_true = rng.integers(0, 2, n_samples)
    return raw_probs, y_true


def test_init_and_fitted_false():
    """BMACalibrator starts unfitted."""
    cal = BMACalibrator()
    assert not cal.fitted


def test_fit_returns_self_and_fitted_true():
    """fit() returns self and sets fitted=True."""
    cal = BMACalibrator()
    raw_probs, y_true = _make_data()
    result = cal.fit(raw_probs, y_true)
    assert result is cal
    assert cal.fitted


def test_weights_sum_to_one():
    """After fit, mixture weights sum to 1.0."""
    cal = BMACalibrator()
    raw_probs, y_true = _make_data()
    cal.fit(raw_probs, y_true)
    assert cal.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_transform_output_in_unit_interval():
    """transform() returns a scalar in [0, 1]."""
    cal = BMACalibrator()
    raw_probs, y_true = _make_data()
    cal.fit(raw_probs, y_true)
    # Single query: K analogue probs
    single_query = np.array([0.2, 0.5, 0.7, 0.4, 0.6])
    out = cal.transform(single_query)
    assert 0.0 <= float(out) <= 1.0


def test_generate_pdf_returns_array():
    """generate_pdf() returns an array of the same length as the return grid."""
    cal = BMACalibrator()
    raw_probs, y_true = _make_data()
    cal.fit(raw_probs, y_true)
    analogue_probs = np.array([0.3, 0.5, 0.7, 0.4, 0.6])
    return_grid = np.linspace(0.0, 1.0, 50)
    pdf = cal.generate_pdf(analogue_probs, return_grid)
    assert pdf.shape == (50,)
    assert np.all(pdf >= 0.0)
