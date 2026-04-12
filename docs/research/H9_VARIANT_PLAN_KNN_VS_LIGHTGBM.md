# H9 Variant Plan: KNN vs. LightGBM Head-to-Head Comparison

**Created:** 2026-04-10  
**Status:** READY — Phase 7 complete (2026-04-10). Recommended as next strategic experiment. Research paper evidence (34d1c6c2) strongly supports running this comparison.  
**Prerequisite:** Phase 1 BSS gate passed (or architectural pivot triggered by H7+H8 failure)  
**Source:** "KNN Analogue Matching Versus Modern Alternatives for Equity Return Prediction" research paper synthesis  

---

## Context

The evidence review found that gradient boosted trees and logistic regression consistently match or outperform KNN for equity return prediction across every published comparison. The gold-standard benchmark (Gu, Kelly, Xiu — Review of Financial Studies, 2020) excluded KNN entirely, as did all major replications (Leippold et al. 2022, Fieberg et al. ~2022, Hanauer & Kalsbach 2023). Khan et al. (PLOS ONE, 2023) measured KNN at 80.5% vs Random Forest at 91.3% on Tesla daily data.

However, FPPE's task — temporal analogue matching with calibrated probabilities on 23D returns_candle features across 52 tickers — is architecturally distinct from the broad cross-sectional prediction problem where trees dominate. No published study tests this exact configuration. The chinuy/stock-price-prediction finding that KNN's optimal feature window is 99 days (vs 3–4 days for tree-based methods) provides independent support for FPPE's extended fingerprint hypothesis.

**This experiment exists to answer one question before real capital is deployed:** Does KNN beat LightGBM on FPPE's specific task, measured by BSS on identical walk-forward folds?

---

## Hypothesis

KNN analogue matching provides equal or superior BSS to LightGBM on FPPE's 52T universe with 23D returns_candle features and beta_abm calibration, specifically in regime-transition periods where parametric model assumptions break down.

**Null hypothesis:** LightGBM achieves higher mean BSS across walk-forward folds than KNN, indicating FPPE should swap its prediction engine.

## Activation Conditions

This experiment activates under **either** of two paths:

**Path 1 — Post-Phase-7 strategic experiment (nominal):**  
Phase 7 complete (2026-04-10). E1-E4 all FAIL, E5-E6 DEFERRED. H9 is the next recommended experiment to validate the algorithm choice before capital scales up in Phase 10.

**Path 2 — Architectural pivot (escalation):**  
H7 AND H8a AND H8b all fail the BSS gate. **Update (2026-04-10):** H7 passed, H8 was never needed. This path was not triggered. Retained for documentation only.

## Gate Condition

**Primary gate:** LightGBM mean BSS across 6 walk-forward folds vs KNN mean BSS across the same 6 folds. The algorithm with higher mean BSS becomes the production prediction engine.

**Secondary gates (tiebreakers if mean BSS difference < 0.01):**
1. Number of folds with BSS > 0: higher count wins
2. Murphy B3 reliability component: lower reliability error wins
3. Worst-fold BSS: higher floor wins (robustness check)
4. Computational cost: faster training/inference wins (operational advantage)

**Ensemble gate (triggered if both algorithms have BSS > 0 on ≥3 folds):**  
If both pass individually, test a stacked ensemble (KNN + LightGBM with logistic regression meta-learner). If the ensemble BSS exceeds both individual algorithms, the ensemble becomes the production architecture.

---

## Experiment Design

### Critical Constraint: Identical Evaluation Conditions

The comparison is meaningless unless both algorithms are evaluated on exactly the same:
- Walk-forward fold boundaries (same 6 folds as H5–H8)
- Feature set (same 23D returns_candle features, same normalization)
- Calibration method (same beta_abm calibration, applied identically; Venn-ABERS not yet tested)
- Regime conditioning (same regime labels/probabilities, if active)
- HOLD exclusion rules (same suppression logic, same denominator)
- BSS computation (same Murphy B3 decomposition code)
- Confidence threshold (same operating threshold applied to both)

### LightGBM Configuration

LightGBM chosen over XGBoost for:
- Native categorical feature support (relevant for future feature expansion)
- Faster training on datasets with >10K samples (histogram-based splitting)
- Lower memory footprint (leaf-wise growth vs level-wise)
- Established dominance in tabular ML benchmarks (Grinsztajn et al., NeurIPS 2022)

**Initial hyperparameters (conservative defaults, not optimized):**
- `objective`: `binary` (direction classification: UP vs NOT-UP)
- `metric`: `binary_logloss`
- `n_estimators`: 500
- `learning_rate`: 0.05
- `max_depth`: 6
- `num_leaves`: 31 (default, ≤ 2^max_depth)
- `min_child_samples`: 50 (conservative for noisy financial data)
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `reg_alpha`: 0.1 (L1 regularization)
- `reg_lambda`: 1.0 (L2 regularization)
- `early_stopping_rounds`: 50
- `verbosity`: -1

**Hyperparameter tuning protocol:**
- Do NOT tune hyperparameters on test folds
- Use nested cross-validation: inner loop (3-fold time-series split) for tuning, outer loop (6-fold walk-forward) for evaluation
- Tuning budget: 50 Optuna trials maximum per outer fold
- Tuning objective: log-loss on inner validation set (NOT BSS, to avoid overfitting the gate metric)

### KNN Configuration

Identical to the configuration that passed (or last attempted) the Phase 1 gate:
- Current locked settings: returns_candle(23), max_distance=2.5, beta_abm calibration, top_k=50, confidence_threshold=0.65
- No hyperparameter changes — this is the existing FPPE pipeline

### Calibration Parity

Both algorithms must use the same post-hoc calibration:
- Currently both algorithms should use `beta_abm` (production calibrator)
- Phase 7 E1 tested BMA (FAIL); Venn-ABERS remains untested
- Calibration set is the same subset of each fold for both algorithms

### Ensemble Architecture (Conditional)

If both KNN and LightGBM individually pass BSS > 0 on ≥3 folds:

**Stacking design:**
- Level 0: KNN predicted probability P_knn(up), LightGBM predicted probability P_lgbm(up)
- Level 1: Logistic regression meta-learner trained on out-of-fold Level 0 predictions
- Calibration: Applied to Level 1 output (not Level 0 individually)
- Evaluation: Same 6-fold walk-forward BSS

**Blending design (simpler alternative):**
- P_ensemble(up) = α × P_knn(up) + (1 − α) × P_lgbm(up)
- α optimized on inner validation folds (grid search: 0.1 to 0.9 in steps of 0.1)
- Fixed α applied to outer test fold

---

## Implementation Spec (Claude Code Handoff)

### Goal

Implement a head-to-head comparison of KNN vs LightGBM on FPPE's exact task, using identical walk-forward folds, features, calibration, and BSS evaluation. Optionally test a stacked ensemble if both algorithms pass individually.

### Success Criteria

1. LightGBM pipeline produces BSS results on the same 6-fold structure as KNN
2. Identical features, calibration, regime conditioning, and HOLD exclusion applied to both
3. Murphy B3 decomposition available for both algorithms on each fold
4. Comparison report generated with primary and secondary gate metrics
5. If ensemble tested: Level 1 meta-learner trained only on out-of-fold predictions (no leakage)
6. All results traceable to walk-forward experiment outputs with provenance

### Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `pattern_engine/lightgbm_predictor.py` | Create | LightGBM prediction class with same interface as KNN predictor |
| `pattern_engine/config.py` | Modify | Add LightGBM config parameters |
| `pattern_engine/ensemble.py` | Create | Stacking and blending ensemble implementations |
| `scripts/run_walkforward.py` | Modify | Support `--algorithm=knn\|lightgbm\|ensemble` flag |
| `scripts/compare_algorithms.py` | Create | Head-to-head comparison report generator |
| `tests/test_lightgbm_predictor.py` | Create | Unit tests for LightGBM predictor |
| `tests/test_comparison_parity.py` | Create | Tests verifying identical evaluation conditions |

### Step-by-Step Implementation Plan

#### Step 1: LightGBM Predictor Class

Location: `pattern_engine/lightgbm_predictor.py`

```
class LightGBMPredictor:
    """Drop-in alternative to KNN predictor with identical interface."""
    
    __init__(config: LightGBMConfig)
    fit(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> self
        - Fits LightGBM with early stopping on validation set
        - Stores feature importances for diagnostics
        - Raises RuntimeError if training set < 500 samples
    predict_proba(X: np.ndarray) -> np.ndarray
        - Returns P(UP) for each sample
        - Shape: (n_samples,) — same as KNN predictor output
    get_diagnostics() -> dict
        - Returns feature importances, best iteration, training log-loss curve
```

**Interface parity requirement:** `LightGBMPredictor.predict_proba()` must return the same shape and semantics as the KNN predictor's output — raw uncalibrated P(UP) scores that feed into the shared calibration pipeline.

#### Step 2: Evaluation Parity Tests

Location: `tests/test_comparison_parity.py`

Assertions:
- [ ] Both algorithms receive identical X_train, y_train, X_test, y_test per fold
- [ ] Both algorithms' raw scores pass through the same calibrator instance
- [ ] Both algorithms' calibrated outputs are evaluated by the same BSS function
- [ ] HOLD exclusion applies identically (same rows excluded from both denominators)
- [ ] Fold boundaries are byte-identical between KNN and LightGBM runs
- [ ] Regime labels/probabilities are identical inputs to both pipelines

#### Step 3: Walk-Forward Integration

Modify `scripts/run_walkforward.py` to accept `--algorithm` flag:

```bash
python scripts/run_walkforward.py --algorithm=knn --output=results/h9_knn/
python scripts/run_walkforward.py --algorithm=lightgbm --output=results/h9_lgbm/
python scripts/run_walkforward.py --algorithm=ensemble --output=results/h9_ensemble/
```

The walk-forward loop must be factored so that fold splitting, feature preparation, calibration, and evaluation are shared code — only the prediction step differs.

#### Step 4: Nested Hyperparameter Tuning for LightGBM

- Inner loop: 3-fold time-series split within each training set
- Optuna sampler: TPESampler with 50 trials
- Search space: learning_rate [0.01, 0.1], max_depth [3, 8], num_leaves [15, 63], min_child_samples [20, 100], subsample [0.6, 1.0], colsample_bytree [0.6, 1.0]
- Objective: validation log-loss (NOT BSS)
- Best parameters applied to full training set, evaluated on test fold
- Log all 50 trials per fold for auditability

#### Step 5: Comparison Report Generator

Location: `scripts/compare_algorithms.py`

```bash
python scripts/compare_algorithms.py results/h9_knn/ results/h9_lgbm/ --output=results/h9_comparison.tsv
```

Output columns per fold:
- fold_id, knn_bss, lgbm_bss, bss_delta
- knn_reliability, lgbm_reliability, reliability_delta
- knn_resolution, lgbm_resolution, resolution_delta
- knn_brier, lgbm_brier, brier_delta
- knn_n_positive_folds, lgbm_n_positive_folds
- knn_mean_bss, lgbm_mean_bss
- winner (primary gate), tiebreaker_used (if applicable)

Summary row:
- mean_bss_knn, mean_bss_lgbm, recommendation (KNN / LightGBM / ENSEMBLE / INCONCLUSIVE)

#### Step 6: Ensemble (Conditional on Both Passing Individually)

Only execute if both KNN and LightGBM achieve BSS > 0 on ≥3 of 6 folds.

- Collect out-of-fold predictions from both algorithms
- Train logistic regression meta-learner on (P_knn, P_lgbm) → y
- Apply calibration to meta-learner output
- Evaluate on test folds with same BSS pipeline
- Compare ensemble BSS to individual algorithm BSS

### Verification Commands

```bash
# Unit tests
pytest tests/test_lightgbm_predictor.py -v
pytest tests/test_comparison_parity.py -v

# Walk-forward experiments
python scripts/run_walkforward.py --algorithm=knn --output=results/h9_knn/
python scripts/run_walkforward.py --algorithm=lightgbm --output=results/h9_lgbm/

# Comparison report
python scripts/compare_algorithms.py results/h9_knn/ results/h9_lgbm/ --output=results/h9_comparison.tsv

# Ensemble (conditional)
python scripts/run_walkforward.py --algorithm=ensemble --output=results/h9_ensemble/
python scripts/compare_algorithms.py results/h9_knn/ results/h9_lgbm/ results/h9_ensemble/ --output=results/h9_comparison_full.tsv

# Murphy B3 decomposition for both
python scripts/murphy_b3.py results/h9_knn/ --label=KNN
python scripts/murphy_b3.py results/h9_lgbm/ --label=LightGBM
```

### Task Type

**MIX** — Steps 1-2 are **SR** (interface design, parity guarantees), Steps 3-6 are **JR** (mechanical wiring with clear specs).

---

## Carry-Forward Items

| Item | Status | Blocks | Phase |
|------|--------|--------|-------|
| 23D feature space is above the commonly cited KNN viability range (8-15D); supports testing LightGBM which handles higher dimensions naturally | **Architecture awareness** — dimensionality increase from 8D to 23D (Phase 6) strengthens the case for LightGBM comparison | Phase 6 complete | Phase 7+ |
| No published BSS benchmark for financial KNN exists | FPPE would be the first — document rigorously for potential publication | None | Ongoing |
| chinuy validation: KNN optimal delta = 99 days supports extended fingerprints | Documented as supporting evidence for FPPE's return window design. Note: validated at 8D, not yet validated at 23D | None | Reference |
| Logistic regression baseline: surprisingly competitive with sophisticated ML (Wolff 2024, BenchStock ICLR 2025) | Consider adding as a third comparator in H9 if resources allow | None | H9 |
| KNN has regime-conditional advantage in turbulent markets (IJF 2016) | Validates FPPE's regime integration — test whether KNN advantage appears specifically in bear-regime folds | H9 analysis | H9 |
| Ensemble diversity: KNN + tree-based models complement each other (JBD 2025) | Ensemble gate in H9 design addresses this directly | H9 Step 6 | H9 |

---

## Decision Tree

```
H9 runs (KNN vs LightGBM head-to-head on identical folds)
    │
    ├── KNN mean BSS > LightGBM mean BSS (delta > 0.01)
    │   └── KNN validated. Keep current architecture. 
    │       Document result as first published financial KNN BSS benchmark.
    │
    ├── LightGBM mean BSS > KNN mean BSS (delta > 0.01)
    │   └── Swap prediction engine to LightGBM.
    │       Keep all other FPPE infrastructure (calibration, regime, sizing, execution).
    │       Rerun Phase 1 gate with LightGBM to confirm BSS > 0 on ≥3 folds.
    │
    ├── Delta < 0.01 (effective tie)
    │   ├── Apply tiebreakers (positive fold count → reliability → worst fold → speed)
    │   └── If still tied: test ensemble
    │       ├── Ensemble BSS > both individuals → adopt ensemble architecture
    │       └── Ensemble BSS ≤ both → choose whichever has lower implementation complexity (KNN)
    │
    └── Both have BSS > 0 on ≥3 folds (regardless of winner)
        └── Test ensemble (Step 6)
            ├── Ensemble improves mean BSS by > 0.005 over best individual → adopt ensemble
            └── Ensemble does not improve → keep individual winner
```

---

## Architecture Impact Assessment

**If LightGBM wins and replaces KNN:**

| FPPE Component | Impact | Migration Effort |
|----------------|--------|-----------------|
| `pattern_engine/knn.py` + hnswlib | Replaced by `lightgbm_predictor.py` | Medium — new class, remove hnswlib dependency |
| Feature pipeline (23D returns_candle) | **No change** — same features feed LightGBM | Zero |
| Calibration pipeline (beta_abm) | **No change** — calibrator receives raw scores from either algorithm | Zero |
| Regime detection (threshold or HMM) | **No change** — regime labels/features are algorithm-agnostic | Zero |
| Walk-forward validation | **No change** — fold structure is algorithm-agnostic | Zero |
| Kelly position sizing | **No change** — receives calibrated P(up) from either algorithm | Zero |
| `max_distance` parameter (currently 2.5) | **Removed** — LightGBM has no distance threshold; replaced by prediction confidence threshold | Low |
| `top_k` parameter | **Removed** — LightGBM has no neighbor count | Low |
| hnswlib index maintenance | **Removed** — LightGBM trains from scratch or incrementally | Low (simplification) |
| Online ANN index updates (Phase 8/11) | **Removed** — LightGBM supports incremental training natively | Low (simplification) |

**Key insight:** ~80% of FPPE's architecture is algorithm-agnostic. The prediction engine is a pluggable component. This is a feature of the modular design, not an accident.

---

## Post-Phase-7 Context (2026-04-10)

- **Phase 7 complete.** E1 (BMA calibration), E2 (conformal prediction), E3 (distance weighting), E4 (adaptive top_k) all FAIL. E5 (HNSW re-tuning), E6 (feature selection) DEFERRED. No enhancement improved BSS over the beta_abm baseline.
- **52T probability constraint identified.** KNN on 52T produces calibrated probabilities only in [0.50-0.59], well below the 0.65 confidence threshold. Production signals (585T Platt, results/cached_signals_2024.csv) reach [0.65-0.75].
- **Current features: 23D returns_candle** (8 return days + 15 continuous candlestick proportions), not the original 8D VOL_NORM features. max_distance scaled from 0.90 to 2.5 accordingly.
- **Dimensionality increase strengthens the case for LightGBM.** The move from 8D to 23D puts FPPE well above the commonly cited KNN viability range (8-15D). Gradient boosted trees handle high-dimensional feature spaces naturally without suffering from the curse of dimensionality.
- **Research paper 34d1c6c2 confirms:** "No published BSS benchmark for financial KNN exists" and "GBTs dominate KNN at every scale" in published comparisons. This makes H9 the single most important validation experiment before live capital deployment.
- **Test count: 858** (up from 846 at Phase 6 completion).

---

## Key References

- Gu, Kelly, Xiu (Review of Financial Studies, 2020) — Gold-standard ML horse-race for equity returns; excluded KNN
- Grinsztajn, Oyallon, Varoquaux (NeurIPS 2022) — Tree-based models SOTA on medium-sized tabular data
- Khan et al. (PLOS ONE, 2023) — KNN 80.5% vs RF 91.3% on Tesla daily data
- Wolff (Journal of Forecasting, 2024) — Logistic regression performs comparably to gradient boosting for S&P 500
- Niculescu-Mizil & Caruana (ICML 2005) — KNN calibration is coarse-grained; logistic regression naturally calibrated
- chinuy/stock-price-prediction — KNN optimal delta = 99 days (vs 3–4 for tree-based)
- IJF 2016 — KNN outperforms parametric models specifically in turbulent markets
- Krauss, Do, Huck (2017) — Random forests 0.43% daily returns vs deep learning 0.33% on S&P 500
- BenchStock (ICLR 2025 submission) — Prediction accuracy not correlated with portfolio return

---

## Appendix: Why LightGBM Over XGBoost

| Factor | LightGBM | XGBoost |
|--------|----------|---------|
| Training speed | Faster (histogram-based) | Slower on >10K samples |
| Memory | Lower (leaf-wise growth) | Higher (level-wise growth) |
| Categorical support | Native | Requires encoding |
| Tabular benchmarks | Dominant (Grinsztajn et al. 2022) | Competitive but slightly behind |
| Incremental training | Supported (`init_model`) | Supported (`xgb_model`) |
| Python API maturity | Excellent | Excellent |
| Community size | Large | Larger |

Both are viable. LightGBM chosen for speed and native categorical support, which matters if future feature expansion adds categorical regime labels.
