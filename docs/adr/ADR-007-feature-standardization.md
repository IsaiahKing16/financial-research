# ADR-007: Feature Standardization — Confirmed Present and Beneficial

**Date:** 2026-04-15
**Status:** ACCEPTED
**Experiment:** P8-PRE-4 (results/phase8_pre/standardization_experiment.tsv)

## Decision

Feature standardization (StandardScaler to unit variance) is CONFIRMED PRESENT and
CONFIRMED BENEFICIAL in FPPE's KNN pipeline.

## Context

The 12-domain survey (Domain 8) identified that raw candlestick features (variance ≈ 0.05–0.08)
contribute only ~9% of L2 distance variance compared to vol-normalized returns (variance ≈ 1.0).
This was interpreted as a standardization deficiency.

**Finding:** `matcher.py:_prepare_features()` already applies `StandardScaler.fit_transform()`
to all 23 returns_candle features during each fold's `fit()` call. The scaler is fitted on
training data only and applied to validation data without refitting (no leakage).

## Evidence

- Per-feature variance post-scaling: returns = 1.0000, candle = 1.0000 (test_per_feature_variance_post_scaling)
- Scaler leakage test: independent scaler per fold (confirmed by test_no_scaler_leakage_across_folds)
- Walk-forward BSS: scaled wins 4/6 folds vs no-scaler baseline; mean delta +0.000222

### Per-fold BSS comparison

| Fold       | Scaled     | Raw        | Delta      | Winner  |
|------------|------------|------------|------------|---------|
| 2019       | +0.000086  | -0.001020  | +0.001106  | Scaled  |
| 2020-COVID | +0.000016  | -0.000948  | +0.000964  | Scaled  |
| 2021       | -0.000063  | -0.001037  | +0.000974  | Scaled  |
| 2022-Bear  | -0.122284  | -0.120492  | -0.001792  | Raw     |
| 2023       | -0.018695  | -0.018163  | -0.000532  | Raw     |
| 2024-Val   | -0.000408  | -0.001018  | +0.000610  | Scaled  |

Mean BSS (scaled): -0.023558
Mean BSS (raw):    -0.023780
Mean delta:        +0.000222

Gate (≥4/6 wins OR mean delta ≥+0.005): **PASS** (4/6 wins)

## Consequences

- `standardize_features: bool = True` added to EngineConfig (default True)
- No change to locked settings required (standardization was already active)
- `max_distance=2.5` remains valid (was calibrated with standardization already active)
- The spec concern about "candlestick features at ~9% of distance variance" referred to
  the raw feature space, not the scaled space. Post-scaling, both feature groups
  contribute equally (variance ≈ 1.0 per feature).
- `standardize_features=False` is available as an experimental condition only; do not
  set in production without new experiment evidence.

## Implementation Notes

The scaler initialization pattern (`self._scaler: Optional[StandardScaler] = None`,
then `self._scaler = StandardScaler()` in `fit()`) acts as an "unfitted sentinel" —
calling `_prepare_features(fit_scaler=False)` before `fit()` raises a `RuntimeError`
with a clear message rather than a cryptic sklearn `AttributeError`.

When `standardize_features=False`, `_prepare_features` bypasses the scaler entirely
and passes raw features as `np.float64` directly to `apply_feature_weights`.
