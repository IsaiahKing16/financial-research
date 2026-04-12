# H8 Variant Plan: HMM Regime Upgrade

**Created:** 2026-04-10  
**Status:** SHELVED — H7 passed (2026-04-06), original activation condition not met. Research recommends HMM upgrade regardless; reassess in next roadmap.  
**Prerequisite:** H7 result + lookahead audit completion  
**Source:** Regime Detection Beyond Thresholds research paper synthesis  

---

## Context

H7 tests the existing threshold-based `RegimeLabeler` (`regime_filter=True` flag in `pattern_engine/regime.py`). If H7 fails the BSS gate (BSS > 0 on ≥3 of 6 folds), H8 replaces the threshold labeler with a 2-state HMM using filtered probabilities.

**Evidence basis:** The regime detection research paper surveyed five methods (HMMs, BOCPD, Turbulence Index, Absorption Ratio, FLUSS) and ranked threshold-based detection as the weakest approach. Hard categorical labels discard uncertainty at regime transition boundaries — exactly where the 2022 Bear fold (the dominant structural outlier per Murphy B3 decomposition) fails. The paper's recommended architecture for KNN integration is continuous filtered probabilities from a 2-state HMM, used as features or model-selection weights rather than a binary gate.

**If H7 passes:** The HMM upgrade moves to Phase 7 (E4/E5 timeframe). This document becomes a design reference, not an immediate action plan.

> **UPDATE (2026-04-10):** H7 passed (mean_BSS=+0.00033, 3/6 folds). The HMM upgrade was shelved. Phase 7 subsequently completed (2026-04-10) without implementing HMM -- E1 (BMA) FAIL, E2 (distance weighting) FAIL, E3 (feature selection) FAIL, E4 (Conformal/ACI) FAIL, E5-E6 DEFERRED. The research paper finding that threshold-based regime detection is the weakest approach remains relevant, and H7's thin margin supports this.

---

## Hypothesis

Replacing binary threshold regime labels with continuous HMM filtered probabilities improves BSS by reducing reliability error in regime-transition periods (specifically the 2022 Bear fold).

## Method Change from H7

| Aspect | H7 (Current) | H8 (Proposed) |
|--------|--------------|---------------|
| Regime source | `RegimeLabeler` — SPY 90-day return threshold | `statsmodels.MarkovRegression` — 2-state HMM |
| Output type | Binary bull/bear label | Continuous P(bull), P(bear) filtered probabilities |
| Integration method | Prediction gate (suppress in bear) | Feature dimension (H8a) or model-blending weight (H8b) |
| Lookahead risk | High if SPY return is forward-looking | Low with filtered-only protocol (forward algorithm) |

## H8 Sub-Variants (Test Sequentially)

**H8a — Regime probability as KNN feature:**  
Add P(bull) as a 24th feature dimension to the existing 23D returns_candle feature vector. This changes the analogue matching space directly. Simpler implementation, lower risk.

**H8b — Regime-weighted blending (only if H8a fails gate):**  
Train separate KNN models per regime, blend predictions weighted by filtered probabilities:  
`P(up) = w_bull × P_bull(up) + w_bear × P_bear(up)`  
More complex, higher potential, higher implementation cost.

## Gate Condition

Same as all Phase 1 experiments: **BSS > 0 on ≥3 of 6 walk-forward folds.**  
Murphy B3 decomposition required on each fold to verify reliability improvement vs H5-H7 baselines.

---

## Locked Configuration Entering H8

All H7 locked parameters carry forward, plus HMM-specific settings:

### Inherited Locks (from H5-H7, updated for Phase 6)
- Universe: 52T returns_candle
- `max_distance`: 2.5
- Calibrator: `beta_abm` (do NOT change calibrator during Phase 1)
- Weighting: uniform
- Features: 23-dimensional returns_candle (H8a extends to 24D)
- `nn_jobs`: 1 (Windows/Python 3.12 deadlock prevention)

### New Locks for H8
- **HMM library:** `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`
- **States:** `k_regimes=2`
- **Variance switching:** `switching_variance=True`
- **Inference:** `filtered_marginal_probabilities` ONLY
  - NEVER use `smoothed_marginal_probabilities`
  - NEVER use Viterbi decoding
  - NEVER use `hmmlearn.predict_proba()` (returns smoothed posteriors despite the name)
- **Refit schedule:** expanding-window, monthly refit (first trading day of each month)
- **Minimum training window:** 500 daily observations (~2 years) before first signal
- **Label-switching resolution:** sort states by emission variance after each refit
- **Input:** SPY daily returns (single univariate series)
- **Burn-in:** first 500 observations produce no regime signal (NaN/neutral)

---

## Implementation Spec (Claude Code Handoff)

### Goal

Replace threshold-based RegimeLabeler with 2-state HMM using statsmodels MarkovRegression. Output filtered probabilities only. Wire into walk-forward pipeline as either a 24th KNN feature (H8a) or a model-blending weight (H8b).

### Success Criteria

1. HMM fits on expanding window with monthly refit
2. Only `filtered_marginal_probabilities` used — zero smoothed/Viterbi calls in entire codebase
3. Label-switching resolved by sorting states by variance after each refit
4. Walk-forward BSS computed identically to H5-H7 experiments
5. Lookahead audit passes: no future data in any HMM input
6. All tests pass: `pytest tests/test_regime_hmm.py tests/test_lookahead_regime.py -v`

### Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `pattern_engine/regime.py` | Extend | Add `HMMRegimeDetector` class alongside existing `RegimeLabeler` |
| `pattern_engine/config.py` | Modify | Add HMM config parameters (k_regimes, min_train_obs, refit_freq) |
| `scripts/run_walkforward.py` | Modify | Wire HMM probabilities into feature pipeline |
| `tests/test_regime_hmm.py` | Create | HMM-specific unit and integration tests |
| `tests/test_lookahead_regime.py` | Create | Lookahead audit test suite (shared by H7 and H8) |

### Step-by-Step Implementation Plan

#### Step 1: Lookahead Audit (PREREQUISITE — blocks all subsequent steps)

This step is shared with H7. Must pass before ANY regime experiment runs.

**Checks:**
- [ ] SPY 90-day return uses trailing computation: `(price[t] - price[t-90]) / price[t-90]`
- [ ] Regime labels are shifted by ≥1 day before application to predictions (`.shift(1)`)
- [ ] VIX values use prior-day settlement, not intraday
- [ ] Yield curve data accounts for FRED publication lag (monthly indicators delayed ~3 weeks)

**Deliverable:** `tests/test_lookahead_regime.py` with assertions for each check.

**If any check fails:** Fix the existing `RegimeLabeler` FIRST, then re-run H7 before proceeding to H8.

#### Step 2: Implement HMMRegimeDetector Class

Location: `pattern_engine/regime.py`

```
class HMMRegimeDetector:
    __init__(k_regimes=2, min_train_obs=500, refit_freq='monthly')
    fit(spy_returns: pd.Series) -> self
        - Fits MarkovRegression with switching_variance=True
        - Calls _resolve_label_switching() after fit
        - Raises RuntimeError if len(spy_returns) < min_train_obs
    filtered_probs(spy_returns: pd.Series) -> np.ndarray
        - Returns P(state_k | data_1:t) for each state
        - Uses ONLY filtered_marginal_probabilities
        - Shape: (T, k_regimes)
    _resolve_label_switching() -> None
        - Sorts states by emission variance (lowest variance = bull)
        - Ensures consistent labeling across refits
    _check_no_smoothed() -> None
        - Static analysis guard: grep codebase for smoothed_marginal_probabilities
        - Raises RuntimeError if found outside of comments/tests
```

**Critical implementation note:** `statsmodels` exposes both `filtered_marginal_probabilities` and `smoothed_marginal_probabilities` as attributes. The class must NEVER access the smoothed attribute. Add a defensive check in `fit()` that the smoothed attribute is not inadvertently used downstream.

#### Step 3: Expanding-Window Walk-Forward Protocol

- Initial training window: first 500 observations of SPY returns
- Monthly refit: on first trading day of each month, refit HMM on all data from start to t
- Between refits: run forward algorithm with cached parameters for daily P(bull)
- Cache fitted model parameters to avoid redundant refits
- Log refit dates, parameter changes, and convergence diagnostics for auditability
- Handle non-convergence: if `MarkovRegression.fit()` fails to converge, fall back to previous month's parameters and log a warning

#### Step 4: H8a Integration — Regime Probability as 24th Feature

- At each prediction date t, look up P(bull | data_1:t) from HMM
- Append as 24th dimension to existing 23D returns_candle feature vector
- Normalize consistently with existing feature scaling scheme
- Run full walk-forward sweep with same 6-fold structure
- Output: `results/h8a/` with identical schema to H5-H7 results

#### Step 5: H8b Integration — Regime-Weighted Blending (only if H8a fails gate)

- Partition training data by dominant regime (P(bull) > 0.5 vs P(bull) ≤ 0.5)
- Train separate KNN models on each partition
- At prediction time: `P(up) = P(bull|t) × P_bull_model(up) + P(bear|t) × P_bear_model(up)`
- Verify each partition has sufficient training data (≥200 samples per regime per fold)
- Run full walk-forward sweep
- Output: `results/h8b/`

#### Step 6: Degenerate Fold Detection

- Apply existing >90% suppression exclusion rule
- Verify HMM doesn't create new degenerate folds via extreme P(bear) values
- Log per-fold diagnostics:
  - % of rows with P(bull) < 0.3
  - Mean P(bull) across fold
  - Number of regime transitions detected
  - Regime duration statistics (mean, min, max days in each state)

### Verification Commands

```bash
# Unit tests
pytest tests/test_regime_hmm.py -v
pytest tests/test_lookahead_regime.py -v

# Walk-forward experiments
python scripts/run_walkforward.py --experiment=H8a --output=results/h8a/
python scripts/run_walkforward.py --experiment=H8b --output=results/h8b/

# Gate check
python scripts/check_gate.py results/h8a/  # BSS > 0 on >= 3 of 6 folds
python scripts/check_gate.py results/h8b/  # Only if H8a fails

# Murphy B3 decomposition (compare reliability vs H5-H7)
python scripts/murphy_b3.py results/h8a/ --compare=results/h7/
```

### Task Type

**MIX** — Steps 1-2 are **SR** (architectural decisions, lookahead-critical, security-sensitive), Steps 3-6 are **JR** (mechanical wiring with clear specs and schemas).

---

## Carry-Forward Items (Active Regardless of H7/H8 Outcome)

| Item | Status | Blocks | Phase |
|------|--------|--------|-------|
| Lookahead audit of `RegimeLabeler` SPY 90-day return | **MANDATORY PREREQUISITE** | H7, H8 | Phase 1 |
| `.shift(1)` on regime labels before decision application | Part of lookahead audit | H7, H8 | Phase 1 |
| FRED yield curve publication lag accounting | Part of lookahead audit | H7, H8 | Phase 1 |
| Library lock: `statsmodels` for HMM (not `hmmlearn`) | **LOCKED** | H8 | Phase 1 |
| Continuous features > binary gate for regime integration | Documented | ~~Phase 7 E4/E5~~ Phase 7 COMPLETE (2026-04-10), not implemented | Future |
| 2-state HMM default (3-state only with economic justification) | **LOCKED** | H8 | Phase 1 |
| Measure per-fold calibration set size (Venn-ABERS viability) | **ACTION NEEDED** | ~~Phase 7 E1~~ Phase 7 E1 tested BMA instead; Venn-ABERS remains untested | Future |
| Library lock: `venn-abers` PyPI for calibration upgrade | **LOCKED** | ~~Phase 7 E1~~ Not yet used | Future |
| Beta calibration remains locked through Phase 1 | **LOCKED** | H7, H8 | Phase 1 |

---

## Branching Logic

> **STATUS (2026-04-10):** H7 passed --> Phase 1 complete --> HMM upgrade shelved. Phase 7 completed without implementing HMM. This branching logic is historical context only.

```
H7 runs with threshold RegimeLabeler
    ├── H7 PASSES gate (BSS > 0 on ≥3/6 folds)
    │   ├── Phase 1 complete — advance to Phase 2
    │   └── HMM upgrade shelved to Phase 7 (E4/E5)
    │
    └── H7 FAILS gate
        ├── Check: Is 52T universe viable at all?
        │   ├── YES → Proceed to H8
        │   └── NO → Fundamental architecture review before H8
        │
        └── H8a runs (regime prob as 9th feature)
            ├── H8a PASSES → Phase 1 complete
            └── H8a FAILS → H8b runs (regime-weighted blending)
                ├── H8b PASSES → Phase 1 complete
                └── H8b FAILS → Escalation:
                    ├── Option 1: Expand universe beyond 52T
                    ├── Option 2: Challenge KNN architecture (Paper #14)
                    └── Option 3: Fundamental rethink of prediction approach
```

---

## Post-Phase-7 Context (2026-04-10)

- **H7 passed** (mean_BSS=+0.00033, 3/6 folds positive). Phase 7 complete (2026-04-10). All experiments: E1 (BMA) FAIL, E2 (distance weighting) FAIL, E3 (feature selection) FAIL, E4 (Conformal/ACI) FAIL, E5 (Venn-ABERS) DEFERRED, E6 (ensemble) DEFERRED.
- **52T probability constraint identified:** 52T KNN produces probabilities only in [0.50-0.59], structurally below the 0.65 confidence threshold. This limits calibration improvements and caused E4 Conformal to produce trivial width-1.0 intervals.
- **Current locked settings:** Features=returns_candle(23), max_distance=2.5, calibration=beta_abm, cal_frac=0.76. Test count: 858.
- **If HMM is revisited:** it should use the 23D returns_candle feature space (not the original 8D VOL_NORM) and account for the 52T probability constraint. H8a would extend to 24D (not 9D as originally specified).
- **Research paper recommendation remains valid:** filtered HMM > threshold-based detection. H7's thin margin (mean_BSS=+0.00033) supports this -- the threshold approach works but barely clears the gate, consistent with the paper ranking it as the weakest method.

---

## Key References

- Hamilton (1989) — Original regime-switching framework
- Nystrup et al. (2015, 2018) — HMM walk-forward protocols, sticky variants
- Ang & Bekaert (2004) — OOS outperformance in global equity allocation
- Adams & MacKay (2007) — BOCPD for changepoint detection
- Kritzman & Li (2010) — Turbulence index
- Barber, Candès, Ramdas & Tibshirani (2023) — Weighted conformal beyond exchangeability
- Kull, Silva Filho & Flach (2017) — Beta calibration (current FPPE lock justification)
- Shu et al. (2024) — Statistical Jump Models outperform HMMs with trading delays

---

## Appendix: Why Not These Alternatives

| Alternative | Reason for Rejection |
|-------------|---------------------|
| `hmmlearn` | Limited maintenance, `predict_proba()` returns smoothed posteriors (lookahead trap), no filtered probability API |
| BOCPD standalone | Detects changepoints but doesn't label regimes — would need HMM anyway |
| 3-state HMM | Requires ~500 daily observations minimum, higher overfitting risk, no evidence it improves BSS for FPPE's universe |
| 4+ state HMM | Papers showing exceptional returns from 4-state models exhibit overfitting artifacts (BIC theoretically questionable for HMM state selection per Gassiat & Rousseau 2014) |
| Platt scaling on regime probs | Sigmoid family excludes identity function — can uncalibrate an already-calibrated input |
| FLUSS/Matrix Profile | Identifies where regimes change but provides no characterization of what changed — complementary tool, not a replacement |
| Full Bayesian HMM (PyMC) | Sampling takes minutes to hours vs seconds for EM — impractical for frequent refitting in walk-forward |
