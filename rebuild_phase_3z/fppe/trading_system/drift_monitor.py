"""
drift_monitor.py — SPC/CUSUM drift detection and EWMA alerting (SLE-76).

Monitors three types of signal quality degradation over time:

  1. Feature distribution shift — detects when the live feature distribution
     (mean/variance per feature) deviates from the training baseline via
     CUSUM control charts.

  2. Calibration bucket drift — tracks the empirical hit rate per probability
     bucket to detect calibration degradation (buckets should match forecast
     probability).

  3. BSS / signal quality EWMA — exponentially-weighted tracking of rolling
     Brier Skill Score; alerts when the EWMA falls below a threshold.

Integration:
    DriftMonitor integrates with StrategyEvaluator by contributing
    drift_alert flags to EvaluatorState, which StrategyEvaluator can use
    to transition to YELLOW status (BSS drift criterion in SLE-70).

All state is mutable but deliberately NOT a Pydantic model — it accumulates
continuously and would create excessive copying overhead if frozen.

Linear: SLE-76
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class CUSUMState:
    """State for a single CUSUM control chart (two-sided).

    Tracks the upper (S_hi) and lower (S_lo) cumulative sums.
    An alert is raised when either exceeds the decision threshold H.

    Args:
        target:  In-control mean (baseline, from training data).
        k:       Allowance (half the shift to detect, in std units). Default 0.5.
        h:       Decision threshold (std units). Default 4.0 (≈false-alarm rate
                 of 1/370 — matching Shewhart 3-sigma).
    """
    target: float
    k: float = 0.5
    h: float = 4.0
    s_hi: float = field(default=0.0, init=False)
    s_lo: float = field(default=0.0, init=False)
    n_observations: int = field(default=0, init=False)
    n_alerts: int = field(default=0, init=False)

    def update(self, value: float, std: float = 1.0) -> bool:
        """Update CUSUM with a new observation.

        Args:
            value: Observed statistic (e.g. sample mean of a feature).
            std:   Standard deviation for normalisation (training std).

        Returns:
            True if an alert triggered this observation.
        """
        if std <= 0:
            std = 1.0
        z = (value - self.target) / std  # standardised deviation
        self.s_hi = max(0.0, self.s_hi + z - self.k)
        self.s_lo = max(0.0, self.s_lo - z - self.k)
        self.n_observations += 1
        alert = (self.s_hi > self.h or self.s_lo > self.h)
        if alert:
            self.n_alerts += 1
        return alert

    def reset(self) -> None:
        """Reset cumulative sums (typically after an alert is actioned)."""
        self.s_hi = 0.0
        self.s_lo = 0.0

    @property
    def is_alert(self) -> bool:
        """True if CUSUM is currently above the decision threshold."""
        return self.s_hi > self.h or self.s_lo > self.h


@dataclass
class EWMAState:
    """State for an EWMA (Exponentially Weighted Moving Average) monitor.

    Args:
        lambda_:  Smoothing factor in (0, 1].  Lower = slower response.
                  Default 0.2 → responds to changes over ~5 periods.
        baseline: Initial/baseline value (also used as initial EWMA).
        alert_threshold: Alert when EWMA falls below this value.
    """
    lambda_: float = 0.2
    baseline: float = 0.0
    alert_threshold: float = -0.01
    current: float = field(init=False)
    n_observations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.current = self.baseline

    def update(self, value: float) -> bool:
        """Update EWMA with a new observation.

        Returns:
            True if the EWMA has fallen below alert_threshold.
        """
        self.current = self.lambda_ * value + (1 - self.lambda_) * self.current
        self.n_observations += 1
        return self.current < self.alert_threshold

    def reset(self) -> None:
        """Reset EWMA to baseline."""
        self.current = self.baseline
        self.n_observations = 0

    @property
    def is_alert(self) -> bool:
        return self.current < self.alert_threshold


@dataclass
class CalibrationBucket:
    """Tracks empirical hit rate for one probability bucket.

    Args:
        prob_lo: Lower bound of the bucket (inclusive).
        prob_hi: Upper bound of the bucket (exclusive, except last).
    """
    prob_lo: float
    prob_hi: float
    n_forecast: int = field(default=0, init=False)
    n_correct: int = field(default=0, init=False)

    def update(self, predicted_prob: float, actual_outcome: int) -> None:
        """Record a calibration observation.

        Args:
            predicted_prob: Forecast probability for this bucket.
            actual_outcome: 1 if the event occurred, 0 if not.
        """
        self.n_forecast += 1
        self.n_correct += int(actual_outcome)

    @property
    def empirical_rate(self) -> Optional[float]:
        """Empirical hit rate, or None if no observations yet."""
        if self.n_forecast == 0:
            return None
        return self.n_correct / self.n_forecast

    @property
    def midpoint(self) -> float:
        return (self.prob_lo + self.prob_hi) / 2.0

    @property
    def drift(self) -> Optional[float]:
        """Signed drift from the expected midpoint probability.

        Positive = overconfident (predicting high but hitting less).
        Negative = underconfident.
        """
        emp = self.empirical_rate
        if emp is None:
            return None
        return self.midpoint - emp


# ─── DriftMonitor ─────────────────────────────────────────────────────────────

class DriftMonitor:
    """Composite drift monitor: feature shift + calibration + EWMA BSS.

    Usage:
        monitor = DriftMonitor(feature_names=["ret_1d", ..., "ret_90d"])
        monitor.set_baseline(X_train, y_train)

        # During live operation, after each batch:
        monitor.update_features(X_live_batch)
        monitor.update_calibration(probs, y_true)
        monitor.update_bss(current_bss)

        report = monitor.get_report()
        if report["any_alert"]:
            # Feed to StrategyEvaluator...

    Args:
        feature_names:   List of feature column names (length D).
        n_buckets:       Number of probability buckets for calibration. Default 5.
        cusum_k:         CUSUM allowance. Default 0.5.
        cusum_h:         CUSUM threshold. Default 4.0.
        ewma_lambda:     EWMA smoothing. Default 0.2.
        bss_alert_threshold: EWMA BSS below this → alert. Default -0.01.
    """

    def __init__(
        self,
        feature_names: List[str],
        n_buckets: int = 5,
        cusum_k: float = 0.5,
        cusum_h: float = 4.0,
        ewma_lambda: float = 0.2,
        bss_alert_threshold: float = -0.01,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        if n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")

        self.feature_names = list(feature_names)
        self.n_buckets = n_buckets
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.ewma_lambda = ewma_lambda
        self.bss_alert_threshold = bss_alert_threshold

        # Baseline statistics (set by set_baseline)
        self._baseline_means: Optional[np.ndarray] = None
        self._baseline_stds: Optional[np.ndarray] = None
        self._baseline_bss: float = 0.0

        # Per-feature CUSUM charts (mean shift detection)
        self._mean_cusum: Dict[str, CUSUMState] = {}
        self._var_cusum: Dict[str, CUSUMState] = {}

        # BSS EWMA
        self._bss_ewma: Optional[EWMAState] = None

        # Calibration buckets
        edges = np.linspace(0.0, 1.0, n_buckets + 1)
        self._cal_buckets: List[CalibrationBucket] = [
            CalibrationBucket(prob_lo=edges[i], prob_hi=edges[i + 1])
            for i in range(n_buckets)
        ]

        # Alert history
        self._feature_alerts: Dict[str, int] = defaultdict(int)
        self._bss_alerts: int = 0
        self._n_updates: int = 0

    # ── Baseline ──────────────────────────────────────────────────────────────

    def set_baseline(
        self,
        X_train: np.ndarray,
        baseline_bss: float = 0.0,
    ) -> None:
        """Compute and store the in-control baseline statistics.

        Must be called once with the training feature matrix before any
        `update_*` calls.  Initialises all CUSUM charts and BSS EWMA.

        Args:
            X_train:      (N, D) training feature matrix (scaled).
            baseline_bss: Known-good BSS to use as the EWMA baseline.
        """
        D = len(self.feature_names)
        if X_train.shape[1] != D:
            raise ValueError(
                f"X_train has {X_train.shape[1]} columns, expected {D}"
            )

        self._baseline_means = X_train.mean(axis=0)
        stds = X_train.std(axis=0)
        self._baseline_stds = np.where(stds > 0, stds, 1.0)
        self._baseline_bss = baseline_bss

        # Initialise per-feature CUSUM charts
        self._mean_cusum = {
            name: CUSUMState(
                target=float(self._baseline_means[i]),
                k=self.cusum_k,
                h=self.cusum_h,
            )
            for i, name in enumerate(self.feature_names)
        }
        self._var_cusum = {
            name: CUSUMState(
                target=float(self._baseline_stds[i] ** 2),
                k=self.cusum_k,
                h=self.cusum_h,
            )
            for i, name in enumerate(self.feature_names)
        }

        # BSS EWMA
        self._bss_ewma = EWMAState(
            lambda_=self.ewma_lambda,
            baseline=baseline_bss,
            alert_threshold=self.bss_alert_threshold,
        )

    # ── Update methods ─────────────────────────────────────────────────────────

    def update_features(self, X_batch: np.ndarray) -> Dict[str, bool]:
        """Update feature distribution monitors with a new batch.

        Computes batch mean and variance per feature; feeds into CUSUM charts.

        Args:
            X_batch: (B, D) live feature matrix (same scaling as training).

        Returns:
            Dict mapping feature name → True if a CUSUM alert triggered.
        """
        if self._baseline_means is None:
            raise RuntimeError("Call set_baseline() before update_features()")

        alerts: Dict[str, bool] = {}
        batch_means = X_batch.mean(axis=0)
        # Use Bessel correction (ddof=1) for multi-row batches; fall back to
        # population variance (ddof=0) for single-row batches where ddof=1
        # is undefined.  Single-row variance is always 0, which is a valid
        # CUSUM input (no shift detected in a one-sample batch).
        _ddof = 1 if len(X_batch) >= 2 else 0
        batch_vars = X_batch.var(axis=0, ddof=_ddof)

        for i, name in enumerate(self.feature_names):
            mean_alert = self._mean_cusum[name].update(
                batch_means[i], self._baseline_stds[i]
            )
            var_alert = self._var_cusum[name].update(
                batch_vars[i], self._baseline_stds[i] ** 2
            )
            # Cast to Python bool: numpy.bool_ propagates through CUSUM
            # arithmetic when batch means/vars are numpy scalars.
            triggered = bool(mean_alert) or bool(var_alert)
            alerts[name] = triggered
            if triggered:
                self._feature_alerts[name] += 1

        self._n_updates += 1
        return alerts

    def update_calibration(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
    ) -> None:
        """Update calibration bucket hit rates.

        Args:
            probs:  (N,) predicted probabilities in [0, 1].
            y_true: (N,) binary outcomes (0 or 1).
        """
        for prob, outcome in zip(probs, y_true):
            bucket_idx = min(
                int(prob * self.n_buckets),
                self.n_buckets - 1,
            )
            self._cal_buckets[bucket_idx].update(float(prob), int(outcome))

    def update_bss(self, bss: float) -> bool:
        """Update the BSS EWMA monitor.

        Args:
            bss: Current Brier Skill Score observation.

        Returns:
            True if the EWMA BSS alert threshold was breached.
        """
        if self._bss_ewma is None:
            raise RuntimeError("Call set_baseline() before update_bss()")
        alert = self._bss_ewma.update(bss)
        if alert:
            self._bss_alerts += 1
        return alert

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_report(self) -> Dict:
        """Return a summary of current drift state.

        Returns:
            Dict with keys:
                any_alert (bool): True if any monitor is in alert state.
                feature_alerts (dict): feature → {mean_cusum, var_cusum} state.
                bss_ewma (dict): {current, baseline, is_alert}.
                calibration (list): Per-bucket {midpoint, empirical_rate, drift}.
                n_updates (int): Number of feature update calls.
        """
        feature_alerts = {}
        any_feature_alert = False
        for name in self.feature_names:
            mc = self._mean_cusum.get(name)
            vc = self._var_cusum.get(name)
            if mc is None or vc is None:
                feature_alerts[name] = {"initialised": False}
                continue
            feature_alerts[name] = {
                "mean_s_hi": round(mc.s_hi, 4),
                "mean_s_lo": round(mc.s_lo, 4),
                "var_s_hi": round(vc.s_hi, 4),
                "var_s_lo": round(vc.s_lo, 4),
                "alert": mc.is_alert or vc.is_alert,
            }
            if mc.is_alert or vc.is_alert:
                any_feature_alert = True

        bss_report = {}
        bss_alert = False
        if self._bss_ewma is not None:
            bss_alert = self._bss_ewma.is_alert
            bss_report = {
                "current": round(self._bss_ewma.current, 6),
                "baseline": round(self._baseline_bss, 6),
                "threshold": self.bss_alert_threshold,
                "is_alert": bss_alert,
                "n_alerts": self._bss_alerts,
            }

        cal_report = [
            {
                "bucket": f"[{b.prob_lo:.2f}, {b.prob_hi:.2f})",
                "midpoint": b.midpoint,
                "n_forecast": b.n_forecast,
                "empirical_rate": b.empirical_rate,
                "drift": b.drift,
            }
            for b in self._cal_buckets
        ]

        return {
            "any_alert": any_feature_alert or bss_alert,
            "feature_alerts": feature_alerts,
            "bss_ewma": bss_report,
            "calibration": cal_report,
            "n_updates": self._n_updates,
        }

    def reset_cusum(self, feature_name: Optional[str] = None) -> None:
        """Reset CUSUM charts after an alert has been actioned.

        Args:
            feature_name: If given, reset only that feature's CUSUM.
                          If None, reset all feature CUSUM charts.
        """
        if feature_name is not None:
            if feature_name in self._mean_cusum:
                self._mean_cusum[feature_name].reset()
                self._var_cusum[feature_name].reset()
        else:
            for cusum in self._mean_cusum.values():
                cusum.reset()
            for cusum in self._var_cusum.values():
                cusum.reset()
