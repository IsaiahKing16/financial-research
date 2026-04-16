# Session Log: P8-PRE-7 max_distance Re-Sweep
**Date:** 2026-04-15
**Branch:** main (committed directly, consistent with project pattern)
**Handoff source:** HANDOFF_P8-PRE-7_max-distance-resweep.md

---

## What Was Done

Executed the full P8-PRE-7 plan — re-validated `max_distance=2.5` after ADR-007
confirmed StandardScaler was applied to all 23 `returns_candle` features. Value
unchanged; provenance formally documented.

### Tasks Completed

| Task | Status | Notes |
|------|--------|-------|
| Pre-flight: 945 tests pass | DONE | 945 passed, 1 skipped, 15 deselected |
| Pre-flight: ADR-007 standardization check | DONE | Circuit breaker triggered (different condition) — see findings |
| Create sweep script | DONE | `scripts/sweep_max_distance_23d_standardized.py` |
| Run sweep [1.0–5.0] | DONE | Winner: 2.5. Runtime: 178s (6 fits × ~28s) |
| Select winner | DONE | 2.5 (2.0 fails: AvgK=19.8 on fold 2019) |
| Create BSS confirmation script | DONE | `scripts/phase8_pre_bss_confirmation.py` |
| Run BSS confirmation | DONE | 6/6 PASS, delta=0.000000 on all folds. Runtime: 141s |
| Lock provenance (LOCKED_SETTINGS_PROVENANCE.md) | DONE | Re-validation block added (winner=2.5, unchanged) |
| P8-PRE-2: commit Phase 6 tests | DONE | Already committed — no action needed |
| P8-PRE-3: confidence_threshold provenance | DONE | Entry expanded with full P8-PRE-3 clarification |
| Final verification | DONE | 945 pass, ruff 275 (frozen baseline) |
| Commit + push | DONE | commit 460da25 pushed to origin/main |

---

## Key Findings and Decisions

### Pre-flight circuit breaker (different condition than expected)

The handoff's pre-flight check said: "If only 8 columns are standardized, STOP —
re-sweep unnecessary." What was actually found: ADR-007 (written this same session,
2026-04-15) explicitly states:

> **Finding:** `_prepare_features()` already applies `StandardScaler.fit_transform()`
> to all 23 returns_candle features during each fold's `fit()` call.
> **max_distance=2.5 remains valid** (was calibrated with standardization already active).

This means the re-sweep premise (P8-PRE-4 changed the distance geometry) was false.
The conservative decision was to run the sweep anyway to produce formal dated provenance.
This turned out to be correct — the sweep proved the value rigorously rather than relying
on a documentation assertion.

### Sweep results — 2.0 borderline fail

At `max_distance=2.0`, fold 2019 returns AvgK=19.8 (gate: ≥20). This is consistent
with the Phase 6 Task 6.1 result which found the same 2019 fold as the binding
constraint. The 0.2 margin is thin but reproducible — standardization did not
compress or expand distances relative to the Phase 6 run.

### BSS confirmation: delta=0.000000

The BSS confirmation script produced identical values to the Phase 6 baseline on
all 6 folds (delta=0.000000). This is the strongest possible confirmation: the
geometry was identical throughout Phase 6 and P8-PRE-7. ADR-007's conclusion is
correct and now has direct numerical evidence.

### P8-PRE-2 already done

The handoff instructed committing untracked `tests/unit/test_phase6_*.py` files.
Both files (`test_phase6_bss_comparison.py`, `test_phase6_redundancy_test.py`) were
already tracked and committed. No action required.

### LOCKED_SETTINGS_PROVENANCE.md was untracked

The provenance doc existed on disk but had never been committed to git. It was added
in this commit along with the P8-PRE-7 artifacts.

---

## Sweep Results

**File:** `results/phase8_pre/sweep_max_distance_23d_standardized.tsv`

| max_dist | mean_AvgK | folds≥20 | mean_BSS | note |
|----------|-----------|----------|----------|------|
| 1.0 | 0.1 | 0 | -0.01131 | |
| 1.5 | 3.2 | 0 | -0.08317 | |
| 2.0 | 22.4 | 5 | -0.03791 | (2019 fold: AvgK=19.8, fails gate) |
| **2.5** | **42.8** | **6** | **-0.00499** | **★ WINNER** |
| 3.0 | 48.9 | 6 | -0.00157 | |
| 3.5 | 49.8 | 6 | -0.00074 | |
| 4.0 | 49.9 | 6 | -0.00063 | |
| 5.0 | 50.0 | 6 | -0.00055 | |

## BSS Confirmation Results

**File:** `results/phase8_pre/bss_confirmation_standardized.tsv`

| Fold | BSS (new) | BSS (baseline) | Delta | Gate |
|------|-----------|----------------|-------|------|
| 2019 | +0.006241 | +0.006241 | 0.000000 | WITHIN |
| 2020-COVID | +0.000388 | +0.000388 | 0.000000 | WITHIN |
| 2021 | +0.000791 | +0.000791 | 0.000000 | WITHIN |
| 2022-Bear | -0.002802 | -0.002802 | 0.000000 | WITHIN |
| 2023 | -0.001102 | -0.001102 | 0.000000 | WITHIN |
| 2024-Val | -0.000013 | -0.000013 | 0.000000 | WITHIN |

Gate: ≥4/6 folds within ±0.001 → **PASS (6/6)**

---

## Adversarial Self-Review

1. **The zero delta is suspicious.** Delta=0.000000 on all folds means the BSS
   confirmation script reproduced the Phase 6 result exactly. This is only possible
   if the data pipeline, feature computation, fold boundaries, random state, and
   calibrator are all identical to Phase 6. This is EXPECTED (same code, same data)
   and confirms ADR-007's finding rather than being a bug — but it means the
   confirmation didn't provide new information beyond what Phase 6 already showed.

2. **The sweep BSS at 2.5 differs from Phase 6 BSS.** The sweep reports BSS≈-0.005
   at d=2.5 while Phase 6 shows BSS≈+0.003 for the same fold. This is because the
   sweep calibrator is fitted at d=5.0 (maximum pool), producing miscalibrated probs
   at d=2.5. This is by design (sweep script comment documents this) and is not a bug.

3. **BSS tolerance gate (±0.001) was trivially easy to pass** given the geometry
   didn't change. The gate was designed for cross-version comparison; within-version
   comparison naturally produces delta≈0. The gate is still useful as a regression
   check but provides little discriminative power here.

---

## Files Changed

| File | Type | Purpose |
|------|------|---------|
| `scripts/sweep_max_distance_23d_standardized.py` | new | P8-PRE-7 sweep script |
| `scripts/phase8_pre_bss_confirmation.py` | new | BSS confirmation script |
| `results/phase8_pre/sweep_max_distance_23d_standardized.tsv` | new | Sweep results |
| `results/phase8_pre/bss_confirmation_standardized.tsv` | new | BSS confirmation results |
| `docs/LOCKED_SETTINGS_PROVENANCE.md` | new (was untracked) | Full provenance trail + P8-PRE-7 + P8-PRE-3 blocks |

---

## What Comes Next

**P8-PRE-7 COMPLETE.** All pre-conditions for P8-PRE-1 are now met.

**NEXT: P8-PRE-1 — 585T End-to-End Revalidation (UNBLOCKED)**
- Run the full trading stack on 585T universe with confirmed locked settings
- Use `max_distance=2.5`, `returns_candle(23)`, `beta_abm`, all other locked settings
- Gate: 585T validation must reproduce Phase 6 signal counts and BSS within tolerance

**In parallel: T8.1 — EOD Pipeline Automation**
- Build `scripts/eod_pipeline.py` — 60-day autonomous signal-to-order pipeline
- Use structlog from day one (ADR-010). All new code must pass Ruff/mypy/Bandit.
- Can proceed without waiting for P8-PRE-1.

**Locked settings confirmed:**
```
max_distance=2.5  (re-validated P8-PRE-7, 2026-04-15)
All other settings unchanged.
```
