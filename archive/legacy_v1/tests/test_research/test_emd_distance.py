"""Smoke tests for EMDDistance — correctness and interface contract."""

import numpy as np
import pytest

from research.emd_distance import EMDDistance


def test_init_and_fit_no_error():
    """EMDDistance initialises and fits without raising."""
    metric = EMDDistance()
    X_train = np.random.default_rng(0).standard_normal((50, 8))
    result = metric.fit(X_train)
    assert result is metric  # fit() returns self


def test_compute_returns_array_of_distances():
    """compute() returns shape (N,) of non-negative floats."""
    rng = np.random.default_rng(1)
    metric = EMDDistance().fit(rng.standard_normal((10, 8)))
    query = rng.standard_normal(8)
    candidates = rng.standard_normal((5, 8))
    result = metric.compute(query, candidates)
    assert result.shape == (5,)
    assert np.all(result >= 0.0)


def test_identical_distributions_zero():
    """EMD between identical fingerprints is 0."""
    metric = EMDDistance().fit(np.zeros((1, 8)))
    fp = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    result = metric.compute(fp, fp.reshape(1, -1))
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_directional_ordering():
    """A closer fingerprint has smaller EMD than a far one."""
    rng = np.random.default_rng(2)
    metric = EMDDistance().fit(rng.standard_normal((20, 8)))
    base = np.zeros(8)
    close = np.full(8, 0.01)
    far = np.full(8, 1.00)
    dist_close = metric.compute(base, close.reshape(1, -1))[0]
    dist_far = metric.compute(base, far.reshape(1, -1))[0]
    assert dist_close < dist_far


def test_time_penalty_zero_collapses_time_axis():
    """With time_penalty=0, fingerprints with same return VALUES but reversed temporal
    ordering have distance ≈ 0 — the time axis is zeroed so only return magnitude matters.
    With time_penalty=1 (default), the same pair would have nonzero distance.
    """
    metric = EMDDistance(time_penalty=0.0, price_penalty=1.0).fit(np.zeros((1, 8)))
    # fp_a: returns increase with time. fp_b: same values, reversed order.
    # With time_penalty=0 → coords are (0, r_i) for both → same multiset → EMD = 0
    fp_a = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    fp_b = fp_a[::-1].copy()  # same values, different temporal assignment
    result = metric.compute(fp_a, fp_b.reshape(1, -1))
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_price_penalty_zero_collapses_price_axis():
    """With price_penalty=0, fingerprints with different returns have distance ≈ 0
    because the return axis is zeroed in the cost matrix."""
    metric = EMDDistance(time_penalty=1.0, price_penalty=0.0).fit(np.zeros((1, 8)))
    # Returns differ dramatically — but price axis is zeroed, so only time-lags matter
    # Both use the same time lags [1,3,7,...,90], so cost matrix rows are identical
    fp_a = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    fp_b = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20])
    result = metric.compute(fp_a, fp_b.reshape(1, -1))
    assert result[0] == pytest.approx(0.0, abs=1e-6)
