# HANDOFF: Phase 7.5 Gates T7.5-1 through T7.5-5
**Created:** 2026-04-18 | **Updated:** 2026-04-19
**Task type:** SR (multi-task — statistical methods + diagnostics infrastructure)

---

## §0 — Session Context

This handoff covers **Phase 7.5 diagnostic gates T7.5-1 through T7.5-5**. These are the
structural prerequisites before Phase 8 (paper trading) can restart.

**T7.5-1 is COMPLETE** (completed 2026-04-18, this session). The remaining work is
T7.5-2 through T7.5-5, all of which build new infrastructure in files that do not yet exist.

### What Was Completed This Session (T7.5-1)

| Commit | Description |
|--------|-------------|
| `87fa912` | fix(baseline): icontract guards on features.py, cal_frac field, slow pytest mark |
| `b0bbdf7` | test(t75-1): 3 failing tests for group_balanced_weights (RED) |
| `7c803eb` | feat(t75-1): group_balanced_weights() in features.py (GREEN) |
| `8759569` | feat(t75-1): Phase 7.5 zscore validation script |
| `6cb04ee` | docs(t75-1): ADR-015 G7.5-1 zscore revalidation gate result |
| `818ae16` | docs(t75-1): all T7.5-1 plan tasks marked complete |

**G7.5-1 result — CONDITIONAL PASS:**
- Condition A: z_score_on wins **4/6 folds** vs z_score_off → PASS
- Condition B: mean BSS **-0.023558** < +0.00033 → FAIL (2022-Bear pool dilution artefact)
- `standardize_features=True` **remains locked**
- `group_balanced_weights()` added to `pattern_engine/features.py` but NOT adopted
- See [ADR-015](docs/adr/ADR-015-g75-zscore-revalidation.md) for full reasoning

**Test count:** 948 (945 baseline + 3 new T7.5-1 unit tests). All pass.

### Additional Work Completed 2026-04-19 (Documentation Closure)

Master plan review against T7.5-1 results identified two gaps. Both fixed in commit `6bde221`:

**ADR-015 updated** — added "Key Findings" section with:
- `z_score_off` has `resolution = 0.0` on **all 6 folds** without StandardScaler. Normalization
  is not incrementally better — it is the *only* source of discriminative power in the system.
- 2022-Bear fold: `n_scored = 416 / 13,052 (3.2%)`. The catastrophic -0.122 BSS and the
  Condition B failure both trace to coverage collapse in bear regimes at 52T, not normalization.
  This is the mechanistic justification for the CONDITIONAL PASS verdict.

**FPPE_MASTER_PLAN_v4.md line 25 corrected** — the claim *"~91% of L2 distance variance from
returns / +0.005–0.02 lift expected"* was a false premise: StandardScaler was already in place
per ADR-007. The paragraph now correctly states that T7.5-1 was a verification task, that the
lift was already baked in to all prior metrics, and that the real finding is zero resolution
without normalization.

T7.5-1 is now fully and properly closed. No code changes in this session.

---

## §1 — Pre-Flight Checklist

Before starting work:

1. `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"` → must show **948 passed**
2. `py -3.12 -m ruff check pattern_engine/` → must show baseline (≤ 275 findings, no new findings)
3. Confirm `results/phase7_5/zscore_bss_comparison.tsv` exists (18 rows)
4. Read `docs/adr/ADR-015-g75-zscore-revalidation.md` — T7.5-1 context affects T7.5-2 baseline
5. Read `CLAUDE.md` (critical rules section) — especially R4 (≤60 lines/function), R9 (zero new ruff/mypy)

---

## §2 — T7.5-2: Braess Gate Implementation

**Goal:** `braess_gate()` — verifies that adding a feature group doesn't degrade BSS. Named after Braess's paradox (adding a road can slow a network; adding a feature can hurt KNN distance geometry).

### Files

| File | Action |
|------|--------|
| `pattern_engine/diagnostics.py` | **CREATE** — new module, starts empty |
| `tests/unit/test_braess_gate.py` | **CREATE** |

`diagnostics.py` does NOT exist yet. Create it.

### Implementation Contract

```python
# pattern_engine/diagnostics.py

@icontract.require(lambda feature_cols_with: len(feature_cols_with) > 0)
@icontract.require(lambda feature_cols_without: len(feature_cols_without) > 0)
def braess_gate(
    full_db: pd.DataFrame,
    feature_cols_with: list[str],
    feature_cols_without: list[str],
    folds: list[dict] | None = None,
    cfg_overrides: dict | None = None,
) -> dict:
    """Returns: {"verdict": "PASS"|"FAIL", "wins_with": int, "fold_deltas": list[float], ...}

    PASS = feature_cols_with wins >= 4/6 folds vs feature_cols_without.
    """
```

The gate invokes `run_walkforward()` twice (once per condition) and compares per-fold BSS.

### Baseline Sanity Check

The current 23D `returns_candle` set must PASS vs a 8D `returns_only` baseline:
```python
# In tests/unit/test_braess_gate.py — sanity test using mocked run_walkforward
# Do NOT run a real 52T walk-forward in unit tests — mock the fold results
```

**CRITICAL:** Do NOT run the real 52T walk-forward in unit tests. Mock `run_walkforward`
to return canned fold results. The real walk-forward takes 15-30 minutes and is a script, not a test.

### Acceptance Criteria

1. `braess_gate()` exists in `pattern_engine/diagnostics.py`
2. Returns dict with `verdict`, `wins_with`, `fold_deltas` keys
3. Unit tests pass with mocked walk-forward (no real compute in tests)
4. All existing 948 tests still pass
5. Zero new ruff findings

---

## §3 — T7.5-3: Identifiability Gate

**Goal:** `identifiability_gate()` — confirms effective parameters << training samples. Goes
in the same `diagnostics.py` as T7.5-2.

### Implementation Contract

```python
@icontract.require(lambda training_n: training_n > 0)
@icontract.require(lambda k: k > 0)
def identifiability_gate(
    training_n: int,
    k: int,
    min_ratio: float = 20.0,
) -> dict:
    """Returns: {"verdict": "PASS"|"FAIL", "ratio": float, "training_n": int, "effective_params": int}

    PASS = training_n / k >= min_ratio.
    Rationale: Hastie et al. (2009) effective df for local methods ≈ N/k.
    Threshold 20:1 is conservative for nonparametric classification.
    """
```

### Important Note

This gate is almost certainly vacuously PASS on current config (50K+ training rows,
k=50 → ratio 1000+). Still required as documentation that the system is not overparameterized.
Implement and prove it passes — do not skip.

### Acceptance Criteria

1. `identifiability_gate()` exists in `pattern_engine/diagnostics.py`
2. Unit test confirms it PASSES for current production config (top_k=50, training_n from any fold)
3. Unit test confirms it FAILS when k > training_n/20

**Sequencing:** T7.5-3 should be implemented in the same commit block as T7.5-2
(both live in `diagnostics.py`).

---

## §4 — T7.5-4: HMM Look-Ahead Audit

**Goal:** Audit the codebase for `hmmlearn.predict_proba()` on sequences including test data.
This is a look-ahead trap — smoothed posteriors (Kim smoother) use future observations.

### Pre-Audit Finding (Already Done in This Session)

A quick grep was already run. **Result: `hmmlearn` is NOT used anywhere in `pattern_engine/`.**

The only `predict_proba` call in `pattern_engine/` is at `matcher.py:80` — this is
`sklearn.linear_model.LogisticRegression.predict_proba()` (Platt calibrator), which is
forward-only. This is correct.

`pattern_engine/regime.py` uses VIX thresholds and yield curve spreads — no HMM.

### What The Task Actually Requires

1. **Formal audit:** run the grep commands below and document the results in a test
2. **Regression test:** write `tests/unit/test_regime_lookahead.py` — a test that asserts
   `hmmlearn` is NOT imported anywhere in the production path (import guard)
3. **No migration required** — there is nothing to migrate

```bash
# Audit commands to run and document
grep -rn "hmmlearn" pattern_engine/ trading_system/
grep -rn "smoothed_marginal" pattern_engine/ trading_system/
grep -rn "predict_proba" pattern_engine/ trading_system/
```

Expected output: only `matcher.py:80` for `predict_proba`; empty for the others.

### Files

| File | Action |
|------|--------|
| `tests/unit/test_regime_lookahead.py` | **CREATE** — import guard tests |

```python
# tests/unit/test_regime_lookahead.py
def test_hmmlearn_not_imported_in_pattern_engine():
    """Regression guard: hmmlearn must never appear in production pattern_engine path."""
    import importlib.util
    assert importlib.util.find_spec("hmmlearn") is None or True  # not used
    # The real check: grep the source files
    import pathlib
    src = pathlib.Path("pattern_engine")
    for py in src.rglob("*.py"):
        assert "hmmlearn" not in py.read_text(), f"hmmlearn found in {py}"
```

### Acceptance Criteria

1. Grep results documented (all three commands run with zero hits on hmmlearn/smoothed)
2. `tests/unit/test_regime_lookahead.py` exists and passes
3. No code changed in `pattern_engine/` (audit is clean)

---

## §5 — T7.5-5: Control-Variate BSS Estimator

**Goal:** Add `cv_bss_estimator()` to a new `pattern_engine/scoring.py` module.
Produces confidence intervals for BSS using control-variate variance reduction.

**Formula:**
```
BS_CV(model) = BS(model) − β · (BS(clim) − E[BS(clim)])
β = cov(BS_model, BS_clim) / var(BS_clim)
Variance reduction factor ≈ (1 − ρ²),  ρ = corr(BS_model, BS_clim)
95% CI via bootstrap (n_bootstrap=1000)
```

Where `BS(clim)` is the climatological (base-rate) Brier Score = `base_rate × (1 - base_rate)`.

### Files

| File | Action |
|------|--------|
| `pattern_engine/scoring.py` | **CREATE** — new module |
| `tests/unit/test_cv_bss_estimator.py` | **CREATE** |

`scoring.py` does NOT exist yet. Create it.

### Implementation Contract

```python
# pattern_engine/scoring.py

import numpy as np
import icontract

@icontract.require(lambda predictions: len(predictions) > 0)
@icontract.require(lambda actuals: len(actuals) > 0)
@icontract.require(
    lambda predictions, actuals: len(predictions) == len(actuals),
    "predictions and actuals must be same length"
)
def cv_bss_estimator(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng_seed: int = 42,
) -> dict:
    """Control-variate BSS estimator with bootstrap confidence interval.

    Returns:
        {
          "bss_point": float,           # standard point estimate
          "bss_cv": float,              # control-variate adjusted estimate
          "ci_lower": float,            # 95% CI lower bound
          "ci_upper": float,            # 95% CI upper bound
          "variance_reduction": float,  # 1 - rho^2
          "beta": float,                # CV regression coefficient
          "n": int,                     # sample size
        }

    Decision criterion: if ci_lower > 0 → edge is statistically significant.
    """
```

### Important Note on Existing Murphy Code

`_murphy_decomposition()` already exists as a private function in `pattern_engine/walkforward.py`
(lines 91–129). Do NOT duplicate it in `scoring.py`. T7.5-5 is a separate estimator —
it uses the Brier Score directly, not the Murphy decomposition.

### Acceptance Criteria

1. `cv_bss_estimator()` exists in `pattern_engine/scoring.py`
2. Returns dict with `bss_point`, `bss_cv`, `ci_lower`, `ci_upper`, `variance_reduction`, `beta`, `n`
3. Unit test: on perfect predictions (all correct), `ci_lower > 0`
4. Unit test: on random predictions, `ci_lower < 0`
5. Unit test: `variance_reduction` is in (0, 1] when there is correlation
6. All 948+ tests still pass. Zero new ruff findings.

---

## §6 — Sequencing and Dependencies

```
T7.5-1   COMPLETE ──────────────────────────────────────────────┐
                                                                  │
T7.5-2 (braess_gate)  ──┐  SEQUENTIAL after T7.5-1             │
T7.5-3 (identif gate) ──┘  same diagnostics.py, one commit      │
                                                                  │
T7.5-4 (HMM audit)    ──┐  PARALLEL with T7.5-2/3              │
T7.5-5 (CV-BSS)       ──┘  no dependencies on each other       │
                                                                  │
All five COMPLETE ─────────────────────────────────────────────► Phase 8 R1
```

**Recommended order:**
1. T7.5-4 first (fast audit — likely 1 commit). Gets it out of the way.
2. T7.5-2 + T7.5-3 together (both in `diagnostics.py`).
3. T7.5-5 last (most mathematical, own module).

---

## §7 — New Files to Create

| File | Task | Notes |
|------|------|-------|
| `pattern_engine/diagnostics.py` | T7.5-2 + T7.5-3 | Does NOT exist. Create from scratch. |
| `pattern_engine/scoring.py` | T7.5-5 | Does NOT exist. Create from scratch. |
| `tests/unit/test_braess_gate.py` | T7.5-2 | Mock run_walkforward — no real compute |
| `tests/unit/test_identifiability_gate.py` | T7.5-3 | Pure math, no I/O |
| `tests/unit/test_regime_lookahead.py` | T7.5-4 | Import guard + source grep |
| `tests/unit/test_cv_bss_estimator.py` | T7.5-5 | Use synthetic data |
| `docs/adr/ADR-016-phase75-diagnostic-gates.md` | All | Single ADR covering T7.5-2 through T7.5-5 verdict |

**Do NOT modify:**
- `pattern_engine/matcher.py` — locked
- `pattern_engine/config.py` — `standardize_features=True` default must stay
- `pattern_engine/walkforward.py` — do not extract or move `_murphy_decomposition`
- Any file in `results/phase8_pre/`

---

## §8 — Critical Rules Reminder

From CLAUDE.md (all apply to new code):

| Rule | Applies to T7.5-2/3/4/5 |
|------|--------------------------|
| R4: ≤60 lines / ≤50 statements per function | `braess_gate()` calls `run_walkforward()` twice — keep the wrapper thin; logic in helpers |
| R5/R10: icontract on all new public APIs | All of `diagnostics.py` and `scoring.py` public functions need `@require`/`@ensure` |
| R7: No silent swallows | Any NaN result from fold must be handled explicitly, not ignored |
| R9: Zero new ruff/mypy findings | Run `py -3.12 -m ruff check pattern_engine/` after each file; baseline is ≤275 |
| R4 (tests): No magic values | Use named constants for thresholds in tests |

---

## §9 — Success Criteria (All T7.5-1 through T7.5-5 Complete)

- [ ] T7.5-1: COMPLETE (2026-04-18) — ADR-015 written, standardize_features=True confirmed
- [ ] T7.5-2: `braess_gate()` in `diagnostics.py`; unit tests pass with mocked walk-forward
- [ ] T7.5-3: `identifiability_gate()` in `diagnostics.py`; PASS confirmed on current config
- [ ] T7.5-4: `test_regime_lookahead.py` passes; audit results documented (no hmmlearn found)
- [ ] T7.5-5: `cv_bss_estimator()` in `scoring.py`; CI test passes on synthetic data
- [ ] ADR-016 written covering T7.5-2 through T7.5-5 verdicts
- [ ] Full test suite passes (expect ≥960 tests after all new test files)
- [ ] Ruff baseline unchanged (≤275)
- [ ] CLAUDE.md "Current Phase" updated to reflect Phase 7.5 progress

---

## §10 — Reference Files

| File | Why |
|------|-----|
| `docs/adr/ADR-015-g75-zscore-revalidation.md` | T7.5-1 result; baseline context for braess gate |
| `pattern_engine/walkforward.py:91–129` | `_murphy_decomposition()` — exists, private, do not duplicate |
| `pattern_engine/matcher.py:80` | The only `predict_proba` in codebase — confirmed safe (Platt calibrator) |
| `pattern_engine/features.py` | `group_balanced_weights()` now at bottom — not used in production |
| `results/phase7_5/zscore_bss_comparison.tsv` | T7.5-1 results — reference for braess gate baseline |
| `docs/campaigns/P8_RECOVERY_CAMPAIGN.md` | Track B/C context — why Phase 7.5 gates matter |
| `FPPE_MASTER_PLAN_v4.md` (T7.5-2 through T7.5-5 sections) | Full spec per task |

---

*Handoff created 2026-04-18 — covers T7.5-1 completion + T7.5-2 through T7.5-5 work brief*
