"""
conformal_hooks.py — Conformal prediction interface stubs (SLE-77).

Design document: docs/rebuild/CONFORMAL_UQ_PLAN.md

Conformal prediction provides distribution-free coverage guarantees:
    P(y ∈ C(x)) >= 1 - alpha   for any alpha in (0, 1)

Unlike Platt calibration (which gives well-calibrated *point estimates*),
conformal prediction yields *set predictions* — guaranteed to contain the
true outcome with coverage probability 1 - alpha, regardless of the data
distribution, as long as calibration and test data are exchangeable.

For FPPE specifically:
    - Instead of a single probability estimate P(fwd_7d_up=1 | x), the
      system would output a prediction interval: [p_lo, p_hi].
    - A BUY signal fires only if p_lo > confidence_threshold (high
      confidence even at the lower bound).
    - A HOLD fires when the interval spans the threshold.

Feasibility for KNN:
    KNN naturally supports conformal prediction via the Mondrian
    (class-conditional) approach:
        nonconformity score = 1 - P(y | x)  for the KNN probability estimate.
    The calibration set nonconformity scores are sorted; the (1-alpha)
    quantile defines the threshold for future predictions.

    Coverage guarantee holds as long as:
        1. Calibration and test samples are exchangeable (i.i.d. assumption).
        2. Calibration set is large enough (n_cal > 50 recommended).

Limitations for FPPE:
    - Time-series data is NOT i.i.d.: temporal autocorrelation breaks
      exchangeability. Weighted conformal prediction (weights decay with time
      gap) partially addresses this.
    - The coverage guarantee is marginal, not conditional on regime.
      Mondrian conformal prediction (separate calibration per regime) gives
      conditional coverage guarantees.

This module contains interface stubs showing how conformal UQ would attach
to the existing calibration pipeline at Stage 5 of PatternMatcher.

Linear: SLE-77
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


# ─── Interface stubs ──────────────────────────────────────────────────────────

class BaseConformalCalibrator(ABC):
    """Abstract base for conformal calibrators.

    A conformal calibrator takes a set of nonconformity scores computed on
    a held-out calibration set and provides prediction sets for new queries.

    The concrete implementation (not built in this stub) would replace the
    current Platt scaler in Stage 5 with a conformal wrapper that augments
    the point estimate with coverage bounds.
    """

    @abstractmethod
    def calibrate(
        self,
        cal_scores: np.ndarray,
        cal_labels: np.ndarray,
    ) -> None:
        """Fit calibrator on held-out nonconformity scores.

        Args:
            cal_scores: (N_cal,) nonconformity scores on calibration data.
                        For KNN: score_i = 1 - P(y_i | x_i).
            cal_labels: (N_cal,) binary labels (0/1) for the calibration rows.
        """

    @abstractmethod
    def predict_set(
        self,
        point_prob: float,
        alpha: float = 0.1,
    ) -> Tuple[float, float]:
        """Return a (1-alpha)-coverage prediction interval.

        Args:
            point_prob: Platt-calibrated point probability estimate.
            alpha:      Miscoverage rate (default 0.1 → 90% coverage).

        Returns:
            (p_lo, p_hi): Prediction interval in [0, 1].
        """

    @abstractmethod
    def coverage(self, test_probs: np.ndarray, test_labels: np.ndarray) -> float:
        """Empirical coverage on a test set.

        Should be close to 1 - alpha if exchangeability holds.

        Args:
            test_probs:  (N,) point probability estimates.
            test_labels: (N,) binary outcomes.

        Returns:
            Empirical fraction of outcomes falling in the predicted set.
        """


class NaiveConformalCalibrator(BaseConformalCalibrator):
    """Marginal conformal calibrator (split conformal, naive version).

    Uses the (ceil((n_cal + 1) * (1 - alpha)) / n_cal)-th quantile of
    calibration nonconformity scores as the threshold.

    Coverage is marginal (not conditional on class or regime).
    This stub implements the full interface to validate the integration plan.

    For production use, replace with a Mondrian (class-conditional) variant
    or a weighted conformal calibrator for time-series data.
    """

    def __init__(self) -> None:
        self._cal_scores: Optional[np.ndarray] = None
        self._n_cal: int = 0

    def calibrate(
        self,
        cal_scores: np.ndarray,
        cal_labels: np.ndarray,
    ) -> None:
        """Store sorted calibration nonconformity scores."""
        self._cal_scores = np.sort(cal_scores)
        self._n_cal = len(cal_scores)

    def _threshold(self, alpha: float) -> float:
        """Conformal threshold for given alpha."""
        if self._cal_scores is None:
            raise RuntimeError("calibrate() must be called first")
        idx = int(np.ceil((self._n_cal + 1) * (1 - alpha))) - 1
        idx = max(0, min(idx, self._n_cal - 1))
        return float(self._cal_scores[idx])

    def predict_set(
        self,
        point_prob: float,
        alpha: float = 0.1,
    ) -> Tuple[float, float]:
        """Return conformal prediction interval centred on point_prob.

        STUB — NO FORMAL COVERAGE GUARANTEE.
        This implementation uses a symmetric half-interval of width tau/2
        around the point estimate.  It does NOT satisfy the conformal
        prediction coverage guarantee P(y ∈ C(x)) >= 1 - alpha because:
          1. The interval is not derived from nonconformity scores of the
             specific test point.
          2. The symmetry assumption is incorrect for asymmetric class
             distributions.
        For formal coverage, replace with split-conformal or Mondrian
        calibration.  See docs/rebuild/CONFORMAL_UQ_PLAN.md for the roadmap.
        """
        tau = self._threshold(alpha)
        half = tau / 2.0
        return (max(0.0, point_prob - half), min(1.0, point_prob + half))

    def coverage(self, test_probs: np.ndarray, test_labels: np.ndarray) -> float:
        """Empirical coverage fraction."""
        if self._cal_scores is None:
            raise RuntimeError("calibrate() must be called first")
        covered = 0
        for prob, label in zip(test_probs, test_labels):
            p_lo, p_hi = self.predict_set(prob)
            # Label 1 is covered if the prediction set includes values above 0.5
            # Label 0 is covered if the prediction set includes values below 0.5
            if label == 1 and p_hi > 0.5:
                covered += 1
            elif label == 0 and p_lo < 0.5:
                covered += 1
        return covered / len(test_labels) if test_labels.size > 0 else 0.0


# ─── Integration stub ─────────────────────────────────────────────────────────

def augment_signals_with_conformal(
    probs: np.ndarray,
    calibrator: BaseConformalCalibrator,
    confidence_threshold: float = 0.65,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, List[str]]:
    """Wrap point probabilities with conformal intervals to sharpen signals.

    A BUY signal requires: p_lo > confidence_threshold (high confidence
    even at the lower bound of the prediction set).
    A SELL signal requires: p_hi < (1 - confidence_threshold).
    Otherwise: HOLD.

    This stub shows how conformal UQ would attach to Stage 5 of PatternMatcher.
    PatternMatcher._package_results() would call this instead of the current
    threshold logic when use_conformal=True.

    Args:
        probs:                (N,) point probability estimates.
        calibrator:           Fitted BaseConformalCalibrator.
        confidence_threshold: Signal firing threshold.
        alpha:                Conformal miscoverage rate.

    Returns:
        (probs, signals): unchanged probs + updated signal list.
    """
    signals = []
    for p in probs:
        p_lo, p_hi = calibrator.predict_set(float(p), alpha=alpha)
        if p_lo > confidence_threshold:
            signals.append("BUY")
        elif p_hi < (1.0 - confidence_threshold):
            signals.append("SELL")
        else:
            signals.append("HOLD")
    return probs, signals


# ─── E4: Adaptive Conformal Inference ────────────────────────────────────────

class AdaptiveConformalPredictor:
    """Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021).

    Provides distribution-free prediction intervals for time-series data
    by dynamically adjusting the quantile level based on recent coverage errors.

    Unlike NaiveConformalCalibrator (which uses a fixed quantile), ACI
    updates alpha_t online after each prediction to track non-stationarity.

    Args:
        nominal_alpha: Target miscoverage rate (default 0.10 → 90% coverage).
        gamma: Learning rate for alpha_t adjustment (default 0.05).
    """

    def __init__(self, nominal_alpha: float = 0.10, gamma: float = 0.05) -> None:
        self.nominal_alpha = nominal_alpha
        self.gamma = gamma
        self.alpha_t = nominal_alpha
        self._scores: Optional[np.ndarray] = None

    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> None:
        """Compute non-conformity scores on calibration set.

        Score = |predicted_prob - actual_outcome|
        """
        self._scores = np.sort(np.abs(cal_probs - cal_labels))

    def predict_interval(self, prob: float) -> Tuple[float, float]:
        """Return (lower, upper) prediction interval for a single probability.

        Uses the (1 - alpha_t) quantile of calibration non-conformity scores
        as the half-width around the point estimate.
        """
        if self._scores is None:
            raise RuntimeError("calibrate() must be called before predict_interval().")
        n = len(self._scores)
        idx = int(np.ceil((1 - self.alpha_t) * (n + 1))) - 1
        idx = int(np.clip(idx, 0, n - 1))
        threshold = float(self._scores[idx])
        lower = max(0.0, prob - threshold)
        upper = min(1.0, prob + threshold)
        return lower, upper

    def update(self, prob: float, actual: int) -> None:
        """ACI online update: adjust alpha_t based on observed coverage error.

        If the last prediction MISSED coverage, decrease alpha_t (widen intervals).
        If it HIT coverage, increase alpha_t (narrow intervals).
        """
        lower, upper = self.predict_interval(prob)
        err = 1 if not (lower <= float(actual) <= upper) else 0
        self.alpha_t = self.alpha_t + self.gamma * (err - self.nominal_alpha)
        self.alpha_t = float(np.clip(self.alpha_t, 0.01, 0.50))

    def mean_interval_width(self, test_probs: np.ndarray) -> float:
        """Compute mean interval width over a set of test probabilities."""
        widths = [
            self.predict_interval(float(p))[1] - self.predict_interval(float(p))[0]
            for p in test_probs
        ]
        return float(np.mean(widths))
