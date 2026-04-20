# HANDOFF: Phase 7.5 Gates T7.5-6, T7.5-7, T7.5-8
**Created:** 2026-04-19  
**Branch:** main (`22bdfd9`)  
**Task type:** SR (statistical methods + diagnostic scripts)

---

## §0 — Session Context

T7.5-1 through T7.5-5 are COMPLETE as of 2026-04-19. This handoff covers the three
remaining Phase 7.5 gates. All infrastructure from prior sessions is available.

### Completed Gates Summary

| Gate | File | Verdict |
|------|------|---------|
| G7.5-1 Z-score | ADR-015 | CONDITIONAL PASS (standardize_features=True locked) |
| G7.5-2 Braess | `diagnostics.py` | Infrastructure PASS; real run deferred to P8-PRE-1 |
| G7.5-3 Identifiability | `diagnostics.py` | **PASS** — ratio ~1000 >> 20:1 |
| G7.5-4 HMM Audit | `test_regime_lookahead.py` | **PASS** — clean codebase |
| G7.5-5 CV-BSS | `scoring.py` | Infrastructure PASS; stat verdict deferred to P8-PRE-1 |

**Test count:** 988. **Ruff:** 276 (unchanged). **Branch:** main.

### What Exists That Matters

These files exist and must NOT be duplicated or overwritten:

| File | Relevant contents |
|------|-------------------|
| `pattern_engine/walkforward.py:58` | `MURPHY_BINS = 10` constant |
| `pattern_engine/walkforward.py:91–129` | `_murphy_decomposition()` — private, returns `(reliability, resolution, uncertainty)` |
| `pattern_engine/scoring.py` | `cv_bss_estimator()` — already here, T7.5-6 adds to this file |
| `pattern_engine/config.py:50` | `projection_horizon: str = "fwd_7d_up"` — EngineConfig default |
| `pattern_engine/data.py:498` | `df["fwd_7d_up"] = (df["fwd_7d"] > 0).astype(float)` — only 7d currently computed |
| `pattern_engine/schema.py:127–209` | Handles `projection_horizon != "fwd_7d_up"` already — partial multi-horizon support |
| `results/phase7_5/zscore_bss_comparison.tsv` | T7.5-1 output — 18 rows, 6 folds × {on, off} × 3 metrics |

---

## §1 — Pre-Flight Checklist

Before starting:

1. `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"` → must show **988 passed**
2. `py -3.12 -m ruff check pattern_engine/ trading_system/` → must show ≤ 276 findings
3. Confirm `pattern_engine/scoring.py` exists and contains `cv_bss_estimator`
4. Confirm `pattern_engine/diagnostics.py` exists and contains `braess_gate`, `identifiability_gate`
5. `py -3.12 -c "from pattern_engine.walkforward import _murphy_decomposition; print('ok')"` → must print ok

---

## §2 — T7.5-6: Murphy B3 Decomposition

**Goal:** Add `murphy_b3_decomposition()` to `pattern_engine/scoring.py`. Diagnose whether
the system's thin BSS is calibration-dominated (REL >> RES) or resolution-dominated (RES ≈ 0).
This directly determines R1 hypothesis ordering.

**Hard block gate.** Must pass before RIA 7.5 and Phase 8.

### Critical: Do NOT Duplicate the Private Function

`_murphy_decomposition()` at `walkforward.py:91–129` is private and already does the
per-fold computation. The new **public** `murphy_b3_decomposition()` should call or mirror
this logic (the private function is not importable by name from outside walkforward.py —
it starts with `_`). Write a clean public implementation in `scoring.py` that duplicates
the *algorithm* but not the function. Keep the same `MURPHY_BINS = 10` default.

The private function signature for reference:
```python
def _murphy_decomposition(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = MURPHY_BINS,  # 10
) -> tuple[float, float, float]:
    # returns (reliability, resolution, uncertainty)
    # reliability:  calibration term (lower = better)
    # resolution:   discrimination term (higher = better)
    # uncertainty:  base_rate * (1 - base_rate)
```

### Implementation Contract

```python
# pattern_engine/scoring.py — ADD to existing file

_MURPHY_BINS = 10  # match walkforward.py constant


@icontract.require(lambda predictions: len(predictions) > 0)
@icontract.require(lambda actuals: len(actuals) > 0)
@icontract.require(
    lambda predictions, actuals: len(predictions) == len(actuals),
    "predictions and actuals must be same length",
)
@icontract.require(lambda n_bins: n_bins >= 2)
@icontract.ensure(lambda result: result["diagnosis"] in (
    "RELIABILITY_DOMINATED", "RESOLUTION_DOMINATED", "BALANCED", "DEGENERATE"
))
def murphy_b3_decomposition(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = _MURPHY_BINS,
) -> dict:
    """Murphy (1973) Brier Score decomposition: BS = REL - RES + UNC.

    BSS = (RES - REL) / UNC when UNC > 0.

    Diagnosis categories:
      RELIABILITY_DOMINATED: REL > 2 * RES → calibration is the limiting factor
      RESOLUTION_DOMINATED:  RES < 0.001 * UNC → model lacks discriminative power
      BALANCED:              both present
      DEGENERATE:            UNC < 1e-10 (all-same-class target)

    Returns:
        {
          "reliability": float,
          "resolution": float,
          "uncertainty": float,
          "bss_decomposed": float,   # (RES - REL) / UNC
          "diagnosis": str,          # one of four categories above
          "n_bins": int,
          "n": int,
        }
    """
```

### Diagnosis Logic

```python
# After computing rel, res, unc:
if unc < 1e-10:
    diagnosis = "DEGENERATE"
elif res < 0.001 * unc:
    diagnosis = "RESOLUTION_DOMINATED"
elif rel > 2.0 * res:
    diagnosis = "RELIABILITY_DOMINATED"
else:
    diagnosis = "BALANCED"
```

### Analysis Script

After implementing the function, write a script to run it on the actual 52T walk-forward
fold results and produce the TSV:

```python
# scripts/run_murphy_decomposition.py
# Load walk-forward fold results from results/phase7_5/zscore_bss_comparison.tsv
# OR re-run walkforward and extract probs + actuals per fold
# Call murphy_b3_decomposition() on each fold
# Write results/phase7_5/murphy_b3_decomposition.tsv
```

**Output schema** (`results/phase7_5/murphy_b3_decomposition.tsv`):
```
fold    n_scored  reliability  resolution  uncertainty  bss_decomposed  diagnosis
2018    ...
2019    ...
2020    ...
2021    ...
2022    ...
2023    ...
```

### Files

| File | Action |
|------|--------|
| `pattern_engine/scoring.py` | MODIFY — add `murphy_b3_decomposition()` |
| `tests/unit/test_murphy_decomposition.py` | CREATE |
| `scripts/run_murphy_decomposition.py` | CREATE — analysis script |
| `results/phase7_5/murphy_b3_decomposition.tsv` | OUTPUT |

### Acceptance Criteria

1. `murphy_b3_decomposition()` in `scoring.py`, all icontract guards present
2. Returns dict with `reliability`, `resolution`, `uncertainty`, `bss_decomposed`, `diagnosis`, `n_bins`, `n`
3. Unit tests: RELIABILITY_DOMINATED case, RESOLUTION_DOMINATED case, BALANCED case, DEGENERATE case
4. Unit test: `abs(reliability - resolution + uncertainty - brier_score) < 1e-10` (BSS identity)
5. Script produces `results/phase7_5/murphy_b3_decomposition.tsv`
6. All 988+ tests still pass. Zero new ruff findings.

### Decision Rules After G7.5-6

```
IF diagnosis is RESOLUTION_DOMINATED on most folds:
    → Feature space or universe is the problem
    → Activate R1 multi-retriever ensemble (priority 1)
    
IF diagnosis is RELIABILITY_DOMINATED on most folds:
    → Calibration is wrong; model has signal but outputs wrong probabilities
    → Activate Venn-ABERS or isotonic calibration experiment
    
IF BALANCED:
    → Both calibration and resolution improvements needed
    → Multi-retriever ensemble (R1) still preferred first
```

---

## §3 — T7.5-7: MI Ceiling Diagnostic

**Goal:** Run `sklearn.feature_selection.mutual_info_classif` on the 23D feature set against
`fwd_7d_up`. Establishes the theoretical upper bound on BSS. If joint MI < 0.001, the
feature space fundamentally cannot support a tradeable edge.

**Hard block gate.** If MI < 0.001 → halt and activate R1 immediately.

### Implementation

This is a **script**, not a module function. Write `scripts/mi_ceiling_diagnostic.py`:

```python
# scripts/mi_ceiling_diagnostic.py

from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np

# Load the augmented database (same as walk-forward uses)
# Use the FULL dataset (not fold-split) for MI estimation — more data = better estimate
# Feature cols: the locked 23D returns_candle set
# Target: fwd_7d_up

# Step 1: Per-feature MI (most accurate — lower dimensional)
mi_per_feature = mutual_info_classif(
    X[feature_cols], y, random_state=42, n_neighbors=3
)

# Step 2: Joint MI (full 23D vector — may underestimate in high dims)
mi_joint = mutual_info_classif(
    X[feature_cols], y, random_state=42, n_neighbors=3
).sum()  # sum of per-feature MI as conservative lower bound

# Step 3: Approximate BSS ceiling
# BSS_max ≈ 2 * MI / UNC  (approximate relationship from information theory)
unc = base_rate * (1 - base_rate)
bss_ceiling = 2.0 * mi_joint / unc if unc > 1e-10 else 0.0
```

### Output Schema (`results/phase7_5/mi_ceiling_diagnostic.tsv`)

```
feature              mutual_information  rank
fwd_1d               0.00234             1
ret_5d               0.00198             2
...                  ...                 ...
open_ratio           0.00012             23
JOINT_23D            0.02341             -
BSS_CEILING_APPROX   0.04521             -
```

### Acceptance Criteria

1. Script runs end-to-end without error on the 52T dataset
2. Per-feature MI table produced (23 rows ranked by MI descending)
3. Joint MI computed as sum of per-feature MI (conservative lower bound)
4. `BSS_CEILING_APPROX = 2 * joint_MI / UNC` computed and reported
5. Verdict: if `BSS_CEILING_APPROX < 2 * 0.00033` (i.e. ceiling is below our current BSS) → log as anomaly
6. Results committed to `results/phase7_5/mi_ceiling_diagnostic.tsv`

**No unit tests required for the script.** Verify by inspecting the TSV output manually.

### Decision Rules After G7.5-7

```
IF joint_MI < 0.001:
    → HARD STOP: feature space insufficient
    → Activate R1 immediately (rough vol, H-statistics, multi-retriever)
    → Document in ADR and halt Phase 8 gate process

IF BSS_CEILING_APPROX < 0.005 (barely above current 0.00033):
    → Thin theoretical margin — note in ADR, proceed cautiously

IF BSS_CEILING_APPROX > 0.01:
    → Meaningful ceiling exists; current BSS is leaving significant value on the table
    → Strengthen the case for R1 multi-retriever ensemble
```

---

## §4 — T7.5-8: Multi-Horizon BSS Curve

**Goal:** Compute BSS at 1d, 3d, 5d, 7d, 10d, 14d horizons across all 6 walk-forward folds.
Identifies the optimal prediction horizon. **INFORMATIONAL — does not block Phase 8.**

### Pre-Work: Verify What's Already Computed

The data pipeline currently only computes `fwd_7d_up` (see `data.py:498`). Multi-horizon
requires adding equivalent columns. Check `data.py` carefully before modifying.

```python
# data.py — currently:
df["fwd_7d_up"] = (df["fwd_7d"] > 0).astype(float)

# Need to add (check if fwd_1d, fwd_3d etc are already computed):
for days in [1, 3, 5, 10, 14]:
    col = f"fwd_{days}d"
    if col in df.columns:
        df[f"fwd_{days}d_up"] = (df[col] > 0).astype(float)
```

Check `data.py` first — some `fwd_Nd` columns may already exist as raw returns even if
`fwd_Nd_up` binary targets don't.

### Implementation

Write `scripts/multi_horizon_bss.py`:

```python
# For each horizon in [1, 3, 5, 7, 10, 14]:
#   cfg_overrides = {"projection_horizon": f"fwd_{h}d_up"}
#   result = run_walkforward(full_db, feature_cols, cfg_overrides=cfg_overrides)
#   fold_results = result["fold_results"]
#   For each fold: call cv_bss_estimator(probs, actuals) to get BSS + CI
#   Aggregate per-horizon mean BSS and CI

# Use cv_bss_estimator from pattern_engine.scoring (T7.5-5)
from pattern_engine.scoring import cv_bss_estimator
from pattern_engine.walkforward import run_walkforward
```

### Schema validation: `projection_horizon != "fwd_7d_up"` is already partially handled

`schema.py:127–209` has logic for non-standard projection horizons. Read that section
carefully before passing `cfg_overrides={"projection_horizon": "fwd_1d_up"}` — you may
need to ensure the target column is present in `full_db` before calling `run_walkforward`.

### Output Schema (`results/phase7_5/multi_horizon_bss.tsv`)

```
horizon_days  fold  n_scored  bss_point  bss_cv  ci_lower  ci_upper
1             2018  ...
1             2019  ...
...
14            2023  ...
```

Also write a summary row per horizon: mean BSS across folds + CI.

### Acceptance Criteria

1. Script runs for all 6 horizons (may require overnight compute — start it, don't wait)
2. TSV output with 36 rows (6 horizons × 6 folds)
3. Per-horizon mean BSS + CI from `cv_bss_estimator` reported
4. Optimal horizon identified (highest mean BSS with `ci_lower > 0`)
5. **No code changes to locked files** (`matcher.py`, `config.py` default values)

**COMPUTE WARNING:** 6 horizons × full 52T walk-forward ≈ 90–180 minutes total.
Start the script, let it run. Do not block session progress on this completing.

---

## §5 — Sequencing and Dependencies

```
T7.5-6 (Murphy B3)  ──────────────────────────────────────► ADR-017
T7.5-7 (MI Ceiling) ──────────────────────────────────────► ADR-017
                                                              │
T7.5-8 (Multi-Horizon) ──────────────────────► (background) ┘
```

**Recommended order:**
1. **T7.5-6** first — adds to existing `scoring.py`, fast to implement and test (~1h)
2. **T7.5-7** second — script only, run against 52T data (~30min compute, 1h implementation)
3. **T7.5-8** last — start the script and let it run in the background (~2–3h compute)
4. **ADR-017** after T7.5-6 and T7.5-7 verdict known (G7.5-8 can be noted as in-progress)

Do not wait for T7.5-8 to finish before writing ADR-017. Document its in-progress state
and update when results arrive.

---

## §6 — New Files This Session Creates

| File | Task | Notes |
|------|------|-------|
| `pattern_engine/scoring.py` | T7.5-6 | MODIFY — already exists, add `murphy_b3_decomposition()` |
| `tests/unit/test_murphy_decomposition.py` | T7.5-6 | Pure synthetic data, no I/O |
| `scripts/run_murphy_decomposition.py` | T7.5-6 | Analysis script — runs on 52T data |
| `results/phase7_5/murphy_b3_decomposition.tsv` | T7.5-6 | Script output |
| `scripts/mi_ceiling_diagnostic.py` | T7.5-7 | Script only |
| `results/phase7_5/mi_ceiling_diagnostic.tsv` | T7.5-7 | Script output |
| `scripts/multi_horizon_bss.py` | T7.5-8 | Long-running — start and background |
| `results/phase7_5/multi_horizon_bss.tsv` | T7.5-8 | Script output |
| `docs/adr/ADR-017-phase75-murphy-mi-horizon.md` | T7.5-6+7+8 | Single ADR for remaining gates |

**Do NOT modify:**
- `pattern_engine/walkforward.py` — `_murphy_decomposition()` stays private
- `pattern_engine/matcher.py` — locked
- `pattern_engine/config.py` defaults — `standardize_features=True`, `projection_horizon="fwd_7d_up"` stay locked
- Any file in `results/phase8_pre/`

---

## §7 — Key Facts the Next Session Must Not Re-Derive

1. **`_murphy_decomposition()` at `walkforward.py:91–129` is private.** Do NOT import it from outside. Write a new public implementation in `scoring.py` with the same algorithm.
2. **`MURPHY_BINS = 10` is at `walkforward.py:58`.** Use `_MURPHY_BINS = 10` as a local constant in `scoring.py`.
3. **BSS identity to assert in tests:** `abs(reliability - resolution + uncertainty - brier_score) < 1e-10`
4. **`scoring.py` already exists** — it contains `cv_bss_estimator()`. Import from there in tests.
5. **`fwd_7d_up` is the only binary target currently computed** in `data.py:498`. Multi-horizon requires adding `fwd_1d_up`, `fwd_3d_up`, `fwd_5d_up`, `fwd_10d_up`, `fwd_14d_up` — check what raw `fwd_Nd` columns exist first.
6. **`schema.py` already handles `projection_horizon != "fwd_7d_up"`** — see lines 127–209. The validation path exists but needs the target column to be present in `full_db`.
7. **T7.5-8 is INFORMATIONAL** — it does not block Phase 8. If compute is slow, document "in progress" in ADR-017 and proceed.
8. **Ruff baseline is 276.** Zero new findings allowed. Run after every new file.
9. **All new public functions need `@require`/`@ensure`** (R10). `murphy_b3_decomposition()` must have both.

---

## §8 — Critical Rules Reminder

From CLAUDE.md (all apply):

| Rule | Applies to T7.5-6/7/8 |
|------|------------------------|
| R4: ≤60 lines / ≤50 statements | `murphy_b3_decomposition()` is borderline — extract binning loop to helper if needed |
| R5/R10: icontract on public APIs | `murphy_b3_decomposition()` requires `@require`/`@ensure` |
| R7: No silent swallows | NaN from empty bins must raise RuntimeError, not return 0.0 |
| R9: Zero new ruff findings | Run `py -3.12 -m ruff check pattern_engine/ trading_system/` after every new file |
| R2: No unbounded loops | `for b in range(1, n_bins + 1)` — bounded by n_bins parameter |

---

## §9 — Success Criteria (All Phase 7.5 Complete)

- [x] T7.5-1: COMPLETE — ADR-015, standardize_features=True confirmed
- [x] T7.5-2: braess_gate() implemented, tests pass
- [x] T7.5-3: identifiability_gate() PASS on production config
- [x] T7.5-4: HMM audit clean — test_regime_lookahead.py passes
- [x] T7.5-5: cv_bss_estimator() implemented, tests pass
- [ ] T7.5-6: murphy_b3_decomposition() + diagnosis verdict on 52T folds
- [ ] T7.5-7: MI ceiling computed, BSS_CEILING_APPROX documented
- [ ] T7.5-8: Multi-horizon BSS curve (informational, non-blocking)
- [ ] ADR-017 written covering G7.5-6, G7.5-7, G7.5-8 verdicts
- [ ] Full test suite passes (expect ≥ 995 after new test file)
- [ ] Ruff baseline unchanged (≤ 276)
- [ ] CLAUDE.md "Current Phase" updated to reflect Phase 7.5 completion

---

## §10 — Reference Files

| File | Why |
|------|-----|
| `pattern_engine/walkforward.py:91–129` | `_murphy_decomposition()` — algorithm reference |
| `pattern_engine/scoring.py` | Add `murphy_b3_decomposition()` here |
| `pattern_engine/data.py:488–500` | Where `fwd_7d_up` is computed — extend for multi-horizon |
| `pattern_engine/schema.py:127–209` | Non-standard horizon validation already present |
| `pattern_engine/config.py:50` | `projection_horizon` field on EngineConfig |
| `results/phase7_5/zscore_bss_comparison.tsv` | Probs + actuals per fold are needed for T7.5-6 script |
| `docs/adr/ADR-015-g75-zscore-revalidation.md` | T7.5-1 CONDITIONAL PASS context |
| `docs/adr/ADR-016-phase75-diagnostic-gates.md` | T7.5-2 through T7.5-5 verdicts |

---

*Handoff created 2026-04-19 — covers T7.5-6 through T7.5-8 implementation briefs*
