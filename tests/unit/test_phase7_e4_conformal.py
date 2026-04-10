# tests/unit/test_phase7_e4_conformal.py
"""TDD tests for E4: Adaptive Conformal Prediction."""
from __future__ import annotations
import numpy as np
import pytest


class TestAdaptiveConformalPredictor:
    def _make_predictor(self):
        from pattern_engine.conformal_hooks import AdaptiveConformalPredictor
        return AdaptiveConformalPredictor(nominal_alpha=0.10, gamma=0.05)

    def test_calibrate_and_predict_interval(self):
        """predict_interval returns (lower, upper) with lower < upper."""
        pred = self._make_predictor()
        rng = np.random.RandomState(0)
        cal_probs = rng.uniform(0.4, 0.7, 200)
        cal_labels = rng.randint(0, 2, 200).astype(float)
        pred.calibrate(cal_probs, cal_labels)
        lo, hi = pred.predict_interval(0.6)
        assert lo < hi
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_interval_width_positive(self):
        """Interval width > 0 for all test probs."""
        pred = self._make_predictor()
        rng = np.random.RandomState(1)
        pred.calibrate(rng.uniform(0, 1, 200), rng.randint(0, 2, 200).astype(float))
        for p in [0.4, 0.5, 0.6, 0.7]:
            lo, hi = pred.predict_interval(p)
            assert hi - lo > 0

    def test_aci_alpha_adjusts_after_update(self):
        """alpha_t changes after calling update()."""
        pred = self._make_predictor()
        rng = np.random.RandomState(2)
        pred.calibrate(rng.uniform(0, 1, 200), rng.randint(0, 2, 200).astype(float))
        alpha_before = pred.alpha_t
        pred.update(0.7, 1)
        assert pred.alpha_t != alpha_before

    def test_coverage_on_well_calibrated_data(self):
        """Coverage >= 88% on synthetic well-calibrated data at nominal 90%."""
        pred = self._make_predictor()
        rng = np.random.RandomState(0)
        # Well-calibrated: prob near true frequency
        probs = rng.uniform(0.3, 0.8, 500)
        labels = (rng.uniform(size=500) < probs).astype(float)
        # Use first 250 for calibration, last 250 for test
        pred.calibrate(probs[:250], labels[:250])
        covered = 0
        for p, y in zip(probs[250:], labels[250:]):
            lo, hi = pred.predict_interval(p)
            if lo <= y <= hi:
                covered += 1
        coverage = covered / 250
        assert coverage >= 0.88, f"Coverage {coverage:.3f} < 0.88"

    def test_mean_interval_width_computable(self):
        """mean_interval_width() returns a float."""
        pred = self._make_predictor()
        rng = np.random.RandomState(4)
        pred.calibrate(rng.uniform(0, 1, 200), rng.randint(0, 2, 200).astype(float))
        test_probs = rng.uniform(0.4, 0.7, 50)
        width = pred.mean_interval_width(test_probs)
        assert isinstance(width, float)
        assert width > 0


class TestConformalGate:
    def test_gate_requires_all_folds_above_88(self):
        """Gate fails if any single fold has coverage < 88%."""
        coverages = np.array([0.91, 0.89, 0.92, 0.87, 0.90, 0.93])
        gate = (coverages >= 0.88).all()
        assert not gate  # fold 4 has 0.87 < 0.88

    def test_gate_passes_when_all_folds_above_88(self):
        coverages = np.array([0.91, 0.89, 0.92, 0.90, 0.90, 0.93])
        widths = np.array([0.20, 0.22, 0.18, 0.25, 0.21, 0.19])
        gate = (coverages >= 0.88).all() and (widths < 0.30).all()
        assert gate
