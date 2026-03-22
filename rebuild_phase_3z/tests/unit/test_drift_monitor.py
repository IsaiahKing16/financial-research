"""
test_drift_monitor.py — Unit tests for SPC/CUSUM drift monitoring (SLE-76).

Tests cover:
  - CUSUMState: update mechanics, alert threshold, two-sided detection
  - EWMAState: smoothing formula, alert threshold
  - CalibrationBucket: empirical_rate, drift computation
  - DriftMonitor: set_baseline, update_features, update_calibration, update_bss
  - DriftMonitor.get_report(): structure, any_alert logic
  - Edge cases: empty baseline, zero-std features, no calibration data
"""

from __future__ import annotations

import numpy as np
import pytest

from rebuild_phase_3z.fppe.trading_system.drift_monitor import (
    CUSUMState,
    DriftMonitor,
    EWMAState,
    CalibrationBucket,
)


# ─── CUSUMState tests ─────────────────────────────────────────────────────────

class TestCUSUMState:
    def test_no_alert_on_in_control_data(self):
        """Small deviations from target should not trigger alert."""
        cusum = CUSUMState(target=0.0, k=0.5, h=4.0)
        for _ in range(50):
            alert = cusum.update(0.0)   # exactly on target
        assert not alert
        assert not cusum.is_alert

    def test_alert_on_sustained_positive_shift(self):
        """Sustained positive shift should trigger upper CUSUM alert."""
        cusum = CUSUMState(target=0.0, k=0.5, h=4.0)
        alerted = False
        for _ in range(30):
            if cusum.update(2.0, std=1.0):  # 2-sigma shift, above allowance
                alerted = True
        assert alerted

    def test_alert_on_sustained_negative_shift(self):
        """Sustained negative shift should trigger lower CUSUM alert."""
        cusum = CUSUMState(target=0.0, k=0.5, h=4.0)
        alerted = False
        for _ in range(30):
            if cusum.update(-2.0, std=1.0):
                alerted = True
        assert alerted

    def test_reset_clears_sums(self):
        cusum = CUSUMState(target=0.0, k=0.5, h=2.0)
        for _ in range(10):
            cusum.update(2.0)
        assert cusum.is_alert
        cusum.reset()
        assert not cusum.is_alert
        assert cusum.s_hi == pytest.approx(0.0)
        assert cusum.s_lo == pytest.approx(0.0)

    def test_s_hi_never_negative(self):
        cusum = CUSUMState(target=0.0)
        for _ in range(100):
            cusum.update(-5.0)
        assert cusum.s_hi >= 0.0

    def test_s_lo_never_negative(self):
        cusum = CUSUMState(target=0.0)
        for _ in range(100):
            cusum.update(5.0)
        assert cusum.s_lo >= 0.0

    def test_n_observations_increments(self):
        cusum = CUSUMState(target=0.0)
        for i in range(7):
            cusum.update(0.0)
        assert cusum.n_observations == 7

    def test_zero_std_handled(self):
        """std=0 should not raise (handled as std=1)."""
        cusum = CUSUMState(target=0.0)
        cusum.update(1.0, std=0.0)   # should not raise


# ─── EWMAState tests ──────────────────────────────────────────────────────────

class TestEWMAState:
    def test_initial_ewma_equals_baseline(self):
        ewma = EWMAState(lambda_=0.2, baseline=0.5)
        assert ewma.current == pytest.approx(0.5)

    def test_ewma_formula(self):
        """EWMA(t) = lambda * x + (1-lambda) * EWMA(t-1)."""
        ewma = EWMAState(lambda_=0.5, baseline=0.0)
        ewma.update(1.0)
        assert ewma.current == pytest.approx(0.5)
        ewma.update(1.0)
        assert ewma.current == pytest.approx(0.75)

    def test_no_alert_above_threshold(self):
        ewma = EWMAState(lambda_=0.2, baseline=0.0, alert_threshold=-0.05)
        alert = ewma.update(0.1)   # EWMA goes up → no alert
        assert not alert

    def test_alert_below_threshold(self):
        ewma = EWMAState(lambda_=0.5, baseline=0.0, alert_threshold=-0.01)
        for _ in range(10):
            ewma.update(-1.0)   # drive EWMA far negative
        assert ewma.is_alert

    def test_reset_returns_to_baseline(self):
        ewma = EWMAState(lambda_=0.5, baseline=0.1)
        ewma.update(-5.0)
        ewma.reset()
        assert ewma.current == pytest.approx(0.1)
        assert ewma.n_observations == 0


# ─── CalibrationBucket tests ──────────────────────────────────────────────────

class TestCalibrationBucket:
    def test_empirical_rate_none_when_empty(self):
        b = CalibrationBucket(prob_lo=0.4, prob_hi=0.6)
        assert b.empirical_rate is None

    def test_empirical_rate_correct(self):
        b = CalibrationBucket(prob_lo=0.4, prob_hi=0.6)
        b.update(0.5, 1)
        b.update(0.5, 1)
        b.update(0.5, 0)
        assert b.empirical_rate == pytest.approx(2 / 3)

    def test_midpoint(self):
        b = CalibrationBucket(prob_lo=0.2, prob_hi=0.4)
        assert b.midpoint == pytest.approx(0.3)

    def test_drift_none_when_empty(self):
        b = CalibrationBucket(prob_lo=0.4, prob_hi=0.6)
        assert b.drift is None

    def test_drift_positive_when_overconfident(self):
        """Bucket midpoint=0.9, empirical=0.5 → drift = 0.9 - 0.5 = 0.4 > 0."""
        b = CalibrationBucket(prob_lo=0.8, prob_hi=1.0)
        for _ in range(10):
            b.update(0.9, 0)   # predicting high but always wrong
        assert b.drift > 0.0

    def test_drift_zero_when_perfectly_calibrated(self):
        """Bucket midpoint=0.5, empirical=0.5 → drift ≈ 0."""
        b = CalibrationBucket(prob_lo=0.4, prob_hi=0.6)
        for _ in range(100):
            b.update(0.5, 1)
        for _ in range(100):
            b.update(0.5, 0)
        assert abs(b.drift) < 0.01


# ─── DriftMonitor integration tests ──────────────────────────────────────────

class TestDriftMonitor:
    FEATURE_NAMES = [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]

    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    @pytest.fixture
    def X_train(self, rng):
        return rng.randn(200, 8).astype(np.float64)

    @pytest.fixture
    def monitor(self, X_train):
        m = DriftMonitor(feature_names=self.FEATURE_NAMES)
        m.set_baseline(X_train, baseline_bss=0.001)
        return m

    def test_set_baseline_stores_statistics(self, monitor, X_train):
        assert monitor._baseline_means is not None
        assert monitor._baseline_stds is not None
        assert monitor._baseline_means.shape == (8,)

    def test_initial_report_no_alert(self, monitor):
        report = monitor.get_report()
        # No updates yet — CUSUMs are at zero → no alert
        assert not report["any_alert"]

    def test_update_features_returns_dict(self, monitor, rng):
        X_batch = rng.randn(20, 8)
        alerts = monitor.update_features(X_batch)
        assert set(alerts.keys()) == set(self.FEATURE_NAMES)
        assert all(isinstance(v, bool) for v in alerts.values())

    def test_sustained_mean_shift_triggers_alert(self):
        """Feeding data with 5-sigma mean shift should eventually alert."""
        rng = np.random.RandomState(0)
        X_train = rng.randn(200, 8)
        monitor = DriftMonitor(
            feature_names=self.FEATURE_NAMES,
            cusum_h=2.0,  # lower threshold for test speed
        )
        monitor.set_baseline(X_train, baseline_bss=0.0)

        # Feed batches with large mean shift on feature 0
        any_alert = False
        for _ in range(30):
            X_batch = rng.randn(10, 8)
            X_batch[:, 0] += 5.0   # 5-sigma shift on feature 0
            alerts = monitor.update_features(X_batch)
            if alerts.get("ret_1d"):
                any_alert = True
        assert any_alert

    def test_update_calibration_fills_buckets(self, monitor, rng):
        probs = rng.uniform(0, 1, size=100)
        y_true = rng.randint(0, 2, size=100)
        monitor.update_calibration(probs, y_true)
        report = monitor.get_report()
        total = sum(b["n_forecast"] for b in report["calibration"])
        assert total == 100

    def test_update_bss_returns_bool(self, monitor):
        result = monitor.update_bss(0.005)
        assert isinstance(result, bool)

    def test_bss_alert_when_ewma_drops(self, monitor):
        """Feeding many negative BSS values should eventually trigger alert."""
        alerted = False
        for _ in range(20):
            if monitor.update_bss(-0.1):
                alerted = True
        assert alerted

    def test_get_report_structure(self, monitor):
        report = monitor.get_report()
        assert "any_alert" in report
        assert "feature_alerts" in report
        assert "bss_ewma" in report
        assert "calibration" in report
        assert "n_updates" in report
        assert isinstance(report["calibration"], list)
        assert len(report["calibration"]) == 5  # default n_buckets

    def test_update_features_single_row_no_error(self, monitor):
        """Single-row batches must not raise (ddof falls back to 0)."""
        X_single = np.zeros((1, 8), dtype=np.float64)
        alerts = monitor.update_features(X_single)  # must not raise
        assert set(alerts.keys()) == set(self.FEATURE_NAMES)
        assert all(isinstance(v, bool) for v in alerts.values())

    def test_update_features_raises_before_baseline(self):
        m = DriftMonitor(feature_names=self.FEATURE_NAMES)
        with pytest.raises(RuntimeError, match="set_baseline"):
            m.update_features(np.zeros((5, 8)))

    def test_update_bss_raises_before_baseline(self):
        m = DriftMonitor(feature_names=self.FEATURE_NAMES)
        with pytest.raises(RuntimeError, match="set_baseline"):
            m.update_bss(0.0)

    def test_reset_cusum_all(self, monitor, rng):
        """After reset_cusum(), all S_hi and S_lo should be 0."""
        X_batch = rng.randn(20, 8) + 3.0   # shifted
        for _ in range(5):
            monitor.update_features(X_batch)
        monitor.reset_cusum()
        for name in self.FEATURE_NAMES:
            mc = monitor._mean_cusum[name]
            assert mc.s_hi == pytest.approx(0.0)
            assert mc.s_lo == pytest.approx(0.0)

    def test_reset_cusum_single_feature(self, monitor, rng):
        X_batch = rng.randn(20, 8) + 3.0
        for _ in range(5):
            monitor.update_features(X_batch)
        monitor.reset_cusum("ret_1d")
        mc = monitor._mean_cusum["ret_1d"]
        assert mc.s_hi == pytest.approx(0.0)
        # Other features unchanged (still accumulated)
        mc2 = monitor._mean_cusum["ret_3d"]
        assert mc2.s_hi > 0.0 or mc2.s_lo > 0.0

    def test_feature_names_empty_raises(self):
        with pytest.raises(ValueError, match="feature_names"):
            DriftMonitor(feature_names=[])

    def test_n_buckets_minimum_raises(self):
        with pytest.raises(ValueError, match="n_buckets"):
            DriftMonitor(feature_names=self.FEATURE_NAMES, n_buckets=1)

    def test_wrong_feature_count_raises(self, monitor):
        with pytest.raises(ValueError):
            monitor.set_baseline(np.zeros((10, 5)))  # wrong D
