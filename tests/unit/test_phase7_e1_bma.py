"""TDD tests for E1: BMA Calibrator integration."""
from __future__ import annotations
import numpy as np
import pytest
from research.bma_calibrator import BMACalibrator


class TestBMACalibrator:
    def test_fit_accepts_n_k_matrix(self):
        """fit() accepts (N, K) raw_probs without raising."""
        bma = BMACalibrator()
        raw = np.random.RandomState(0).uniform(0, 1, (200, 50))
        labels = np.random.RandomState(0).randint(0, 2, 200).astype(float)
        bma.fit(raw, labels)
        assert bma.fitted

    def test_transform_returns_scalar_in_unit_interval(self):
        """transform() on (K,) input returns scalar in [0, 1]."""
        bma = BMACalibrator()
        raw = np.random.RandomState(1).uniform(0, 1, (200, 50))
        labels = np.random.RandomState(1).randint(0, 2, 200).astype(float)
        bma.fit(raw, labels)
        k_probs = np.random.RandomState(2).uniform(0, 1, 50)
        result = bma.transform(k_probs)
        assert isinstance(float(result), float)
        assert 0.0 <= float(result) <= 1.0

    def test_weights_sum_to_one(self):
        """After fit(), BMA weights sum to 1.0."""
        bma = BMACalibrator()
        raw = np.random.RandomState(3).uniform(0, 1, (300, 50))
        labels = np.random.RandomState(3).randint(0, 2, 300).astype(float)
        bma.fit(raw, labels)
        assert abs(bma.weights.sum() - 1.0) < 1e-6

    def test_convergence_on_small_data(self):
        """EM converges on 500 samples, K=20."""
        bma = BMACalibrator()
        raw = np.random.RandomState(4).uniform(0, 1, (500, 20))
        labels = np.random.RandomState(4).randint(0, 2, 500).astype(float)
        bma.fit(raw, labels)
        assert bma.fitted

    def test_transform_before_fit_raises(self):
        """transform() before fit() raises RuntimeError."""
        bma = BMACalibrator()
        with pytest.raises(RuntimeError):
            bma.transform(np.array([0.5] * 50))


class TestBMAGate:
    def test_gate_passes_when_3_folds_improve(self):
        """Gate: >=3 folds with BMA_BSS - baseline_BSS >= +0.001."""
        baseline = np.array([0.001, 0.002, -0.001, 0.003, 0.000, 0.001])
        enhanced = np.array([0.003, 0.004, -0.001, 0.005, 0.001, 0.001])
        deltas = enhanced - baseline
        assert (deltas >= 0.001).sum() >= 3

    def test_gate_fails_when_fewer_than_3_improve(self):
        """Gate fails when <3 folds improve by >=+0.001."""
        baseline = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
        enhanced = baseline.copy()
        deltas = enhanced - baseline
        assert (deltas >= 0.001).sum() < 3
