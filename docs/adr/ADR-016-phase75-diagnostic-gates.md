# ADR-016: Phase 7.5 Diagnostic Gates — T7.5-2 through T7.5-5

**Date:** 2026-04-19  
**Status:** ACCEPTED  
**Author:** Claude (Sonnet 4.6) / Isaiah King (Sleep)  
**Relates to:** FPPE_MASTER_PLAN_v4.md §Phase 7.5, HANDOFF_T75-1-5_phase75-gates.md §2–5

---

## Context

Phase 7.5 requires 8 diagnostic gates before Phase 8 (paper trading) can start. T7.5-1
(z-score normalization verification) was completed 2026-04-18 with a CONDITIONAL PASS.
This ADR documents the implementation and verdicts for the remaining four tasks in this
session (2026-04-19): T7.5-2, T7.5-3, T7.5-4, T7.5-5.

---

## G7.5-2 — Braess Gate

### Decision

**IMPLEMENTED** — `braess_gate()` added to `pattern_engine/diagnostics.py`.

### What Was Built

`braess_gate(full_db, feature_cols_with, feature_cols_without, folds, cfg_overrides) -> dict`

The gate invokes `run_walkforward()` twice (once per feature set) and compares per-fold BSS.
PASS criterion: `feature_cols_with` wins ≥ 4/6 folds vs `feature_cols_without`.

Return dict keys: `verdict`, `wins_with`, `fold_deltas`, `mean_bss_with`, `mean_bss_without`, `n_folds`.

### Design Choices

- The fold comparison logic was extracted to `_compare_fold_bss()` to keep `braess_gate()`
  within the ≤60-line / ≤50-statement P10-R4 limit.
- NaN BSS in any fold raises `RuntimeError` (P10-R7 — no silent swallows).
- `@icontract.ensure` verifies the verdict is always "PASS" or "FAIL".
- Unit tests mock `run_walkforward` at its usage site in `diagnostics`
  (`pattern_engine.diagnostics.run_walkforward`) — not at definition site. No real 52T
  walk-forward runs in tests.

### Gate Verdict

GATE IMPLEMENTED. Sanity-check baseline (23D returns_candle vs 8D returns_only) must pass
when the real 52T walk-forward is run as part of P8-PRE-1. The gate infrastructure is
verified by 9 unit tests.

---

## G7.5-3 — Identifiability Gate

### Decision

**IMPLEMENTED AND VERIFIED PASS** — `identifiability_gate()` added to `pattern_engine/diagnostics.py`.

### What Was Built

`identifiability_gate(training_n, k, min_ratio=20.0) -> dict`

Uses Hastie et al. (2009) effective degrees of freedom for local methods: `df ≈ N/k`.
PASS criterion: `training_n / k ≥ min_ratio` (threshold 20:1, conservative for KNN).

Return dict keys: `verdict`, `ratio`, `training_n`, `effective_params`, `min_ratio`.

### Verified Pass on Current Production Config

| Parameter | Value |
|-----------|-------|
| `top_k` (k) | 50 |
| Typical fold training_n | ~50,000–200,000 rows |
| Resulting ratio | 1,000–4,000 |
| min_ratio | 20.0 |
| Verdict | **PASS** (ratio >> 20:1) |

The gate is vacuously true on current config (as anticipated in the handoff), but provides
documented proof that the system is not overparameterized. Required before any tuning.

### Gate Verdict

**G7.5-3 PASS** — Current production config (top_k=50) passes at all fold sizes.
Verified by 13 unit tests covering PASS, FAIL, edge cases, and contract guards.

---

## G7.5-4 — HMM Look-Ahead Audit

### Decision

**CLEAN AUDIT — NO MIGRATION REQUIRED**

### Audit Commands and Results

```
grep -rn "hmmlearn" pattern_engine/ trading_system/        → 0 hits
grep -rn "smoothed_marginal" pattern_engine/ trading_system/ → 0 hits
grep -rn "predict_proba" pattern_engine/ trading_system/    → 1 hit
```

The single `predict_proba` hit is `pattern_engine/matcher.py:80`:
`return self._lr.predict_proba(raw.reshape(-1, 1))[:, 1]`

This is `sklearn.linear_model.LogisticRegression.predict_proba()` — the Platt scalar
calibrator. It is a deterministic, stateless forward-only transform. There is no HMM
look-ahead contamination.

`pattern_engine/regime.py` uses VIX thresholds and yield curve spreads only. No HMM.

### Regression Test

`tests/unit/test_regime_lookahead.py` contains 3 tests:
1. `test_hmmlearn_not_in_production_source` — scans all `.py` files under `pattern_engine/`
   and `trading_system/` for the string `"hmmlearn"`.
2. `test_smoothed_marginal_not_in_production_source` — same for `"smoothed_marginal"`.
3. `test_predict_proba_only_in_matcher` — asserts `predict_proba` appears only in `matcher.py`.

These tests will catch any future accidental introduction of `hmmlearn`.

### Gate Verdict

**G7.5-4 PASS** — Codebase is clean of HMM look-ahead contamination.
No code changes to `pattern_engine/` were required.

---

## G7.5-5 — Control-Variate BSS Estimator

### Decision

**IMPLEMENTED** — `cv_bss_estimator()` added to new module `pattern_engine/scoring.py`.

### What Was Built

`cv_bss_estimator(predictions, actuals, n_bootstrap=1000, ci_level=0.95, rng_seed=42) -> dict`

**Formula:**
```
BS_CV(model) = BS(model) - beta * (BS(clim) - E[BS(clim)])
beta = cov(BS_model, BS_clim) / var(BS_clim)
Variance reduction = (1 - rho^2)  where rho = corr(BS_model, BS_clim)
95% CI via bootstrap (n_bootstrap=1000)
```

Return dict keys: `bss_point`, `bss_cv`, `ci_lower`, `ci_upper`, `variance_reduction`, `beta`, `n`.

**Decision criterion:** `ci_lower > 0` → edge is statistically significant.

### Design Notes

- `_murphy_decomposition()` at `walkforward.py:91–129` was NOT duplicated.
  `cv_bss_estimator()` operates at the Brier Score layer directly, separate from the
  Murphy decomposition.
- The bootstrap samples BS_model and BS_clim independently per resample, then computes
  the CV adjustment using the aggregate beta from those samples. This correctly propagates
  the correlation structure through the CI.
- `np.random.default_rng(rng_seed)` is used (modern Generator API) for reproducibility.
- `@icontract.require(n_bootstrap >= 100)` enforces statistical validity.

### Synthetic Data Verification

| Scenario | Expected | Actual |
|----------|----------|--------|
| Strong predictions (p=0.9 for +, p=0.1 for -) | `ci_lower > 0` | PASS |
| Random predictions (p ~ Uniform[0.4, 0.6]) | `ci_lower < 0` | PASS |
| Base-rate predictions (p = base_rate constant) | `bss_point ≈ 0` | PASS |
| Any correlated scenario | `variance_reduction ∈ (0, 1]` | PASS |
| Same seed | Identical results | PASS |

### Gate Verdict

GATE IMPLEMENTED. Real-world verdict (whether the FPPE edge is statistically significant)
requires running `cv_bss_estimator()` on all 6 walk-forward fold results. This is a
P8-PRE-1 deliverable. The infrastructure is verified by 14 unit tests.

---

## Summary

| Gate | Status | Verdict | New Files |
|------|--------|---------|-----------|
| G7.5-2 Braess | Implemented | Infrastructure PASS; sanity run deferred to P8-PRE-1 | `diagnostics.py`, `test_braess_gate.py` |
| G7.5-3 Identifiability | Implemented + Verified | **PASS** on production config | `diagnostics.py`, `test_identifiability_gate.py` |
| G7.5-4 HMM Audit | Audit Complete | **PASS** — clean codebase | `test_regime_lookahead.py` |
| G7.5-5 CV-BSS | Implemented | Infrastructure PASS; statistical verdict deferred to P8-PRE-1 | `scoring.py`, `test_cv_bss_estimator.py` |

### Test Count

| Session start | After T7.5-2 through T7.5-5 |
|---------------|------------------------------|
| 948 tests     | 988 tests (+40)              |

### Ruff Baseline

276 findings (unchanged — zero new findings from this session's code).

---

## Consequences

All Phase 7.5 diagnostic gate infrastructure is now in place. Phase 7.5 tasks T7.5-1
through T7.5-5 are complete. The remaining gates (G7.5-6 Murphy B3, G7.5-7 MI ceiling,
G7.5-8 multi-horizon) require compute-intensive runs against the full 52T dataset and are
addressed in subsequent sessions.

The next session should proceed to **P8-PRE-1** (585T full-stack walk-forward) and apply
`cv_bss_estimator()` to its fold results to get the G7.5-5 statistical verdict.
