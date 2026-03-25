# Conformal Uncertainty Quantification Integration Plan
# SLE-77 — Linear: https://linear.app/sleepern/issue/SLE-77

## Summary

This document describes how conformal prediction could be integrated into
the FPPE calibration pipeline to provide distribution-free uncertainty bounds
on PatternMatcher probability estimates.

---

## 1. Current State

The existing calibration pipeline (Stage 5 of PatternMatcher) applies a
Platt scaler (logistic regression) to convert raw KNN proportion estimates
into calibrated probabilities.  This gives *point estimates* with good
empirical calibration but no coverage guarantees.

## 2. Conformal Prediction Primer

Split conformal prediction (Papadopoulos et al. 2002, Venn-Abers 2015):
1. Hold out a calibration set from the training data (not used for fitting).
2. Compute a **nonconformity score** for each calibration point:
   `s_i = 1 - P_hat(y_i | x_i)`  (probability assigned to the true class).
3. Sort scores: `s_(1) <= s_(2) <= ... <= s_(n_cal)`.
4. At prediction time, the (1-alpha)-quantile of scores defines the threshold.
   Any class with predicted score below the threshold is included in the
   prediction set.

**Coverage guarantee** (finite-sample, distribution-free):
   P(y_test ∈ C(x_test)) >= 1 - alpha

---

## 3. KNN-Specific Implementation

### Nonconformity Score

For KNN with proportion estimate `P(y=1 | x) = k_1 / k_total`:

```
nonconformity_score(x, y=1) = 1 - P(y=1 | x) = 1 - k_1 / k_total
nonconformity_score(x, y=0) = 1 - P(y=0 | x) = k_1 / k_total
```

### Mondrian Variant (class-conditional coverage)

To guarantee `P(y=1 ∈ C(x) | y=1) >= 1 - alpha` separately from
`P(y=0 ∈ C(x) | y=0) >= 1 - alpha`, use Mondrian conformal prediction
with separate calibration quantiles per class.

### Integration Point

In `PatternMatcher._package_results()` (Stage 5):

```python
if getattr(cfg, 'use_conformal', False):
    intervals = [calibrator.predict_set(p, alpha=0.1) for p in prob_up]
    # Signal fires only if conformal lower bound exceeds threshold
    signals = [
        "BUY"  if lo > cfg.confidence_threshold else
        "SELL" if hi < (1 - cfg.confidence_threshold) else
        "HOLD"
        for lo, hi in intervals
    ]
```

The interface stub in `fppe/pattern_engine/conformal_hooks.py` implements
`BaseConformalCalibrator` and `augment_signals_with_conformal()`.

---

## 4. Feasibility Assessment for FPPE

### Exchangeability Concern

Conformal coverage guarantees require *exchangeability* — roughly, that
calibration and test points are drawn i.i.d. from the same distribution.
Financial time series violates this: returns have autocorrelation, regime
shifts, and non-stationarity.

**Mitigations:**
- Weighted conformal prediction (Tibshirani et al. 2019): downweight old
  calibration points using `w_i ∝ exp(-lambda * (t_test - t_i))`.
- Rolling calibration: use only the last N_cal trading days as the
  calibration set (sliding window).
- Mondrian by regime: separate calibration sets per regime label.

### Coverage Calibration vs. Conformal

For the current FPPE configuration, Platt calibration already achieves
95th-percentile calibration error < 2% on the 2024 fold.  Conformal
prediction would provide *guaranteed* coverage but at the cost of wider
prediction intervals (typically 10–30% wider than the Platt interval).

**Net effect on signal frequency:** expected reduction in BUY/SELL signals
by ~15–25% (more rows classified as HOLD due to wider intervals).

### Recommendation

1. **Near-term (M6):** Implement the interface stubs (DONE — conformal_hooks.py).
2. **Medium-term:** Add rolling weighted conformal calibrator (n_cal=252 bars).
3. **Long-term:** Evaluate Mondrian + regime conditioning on a held-out year.
   Gate: coverage >= 0.88 on 2024 fold (target alpha=0.10).

---

## 5. Files

| File | Purpose |
|------|---------|
| `rebuild_phase_3z/fppe/pattern_engine/conformal_hooks.py` | Interface stubs |
| `rebuild_phase_3z/tests/unit/test_conformal_hooks.py` | Unit tests |
| `docs/rebuild/CONFORMAL_UQ_PLAN.md` | This document |

## 6. References

- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World.*
- Tibshirani et al. (2019). *Conformal Prediction Under Covariate Shift.* NeurIPS.
- Angelopoulos & Bates (2022). *A Gentle Introduction to Conformal Prediction.*
