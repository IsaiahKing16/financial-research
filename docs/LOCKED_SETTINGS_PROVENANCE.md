# Locked Settings Provenance — FPPE

**Purpose:** Full provenance trail for every locked hyperparameter. Extracted from
CLAUDE.md to keep the root config file concise. Reference this doc before proposing
any locked setting change.

**Rule:** No locked setting may change without new walk-forward experiment evidence.

---

## Current Locked Values

Distance=Euclidean, Weighting=uniform, Features=returns_candle(23), Calibration=beta_abm,
cal_frac=0.76, max_distance=2.5, top_k=50, confidence_threshold=0.65,
regime=hold_spy_threshold+0.05, horizon=fwd_7d_up, stop_loss_atr_multiple=3.0

---

## Provenance Trail

### Features=returns_candle(23), max_distance=2.5 — Phase 6 (2026-04-09)
- Task 6.1 sweep: winner=2.5 (smallest with AvgK>=20 all 6 folds).
  Provenance: results/phase6/sweep_max_distance_23d.tsv
- Task 6.2 BSS comparison: returns_candle wins 5/6 folds vs returns_only. GATE PASS.
  Provenance: results/phase6/bss_comparison_candle_vs_baseline.tsv
- Task 6.3 body_position: gate triggered DROP (3/6) but KEPT by judgment — 2023/2024-Val
  deltas favor 23D by x10-36 vs the gains in earlier folds.
  Provenance: results/phase6/redundancy_body_position.tsv

### max_distance=2.5 — Re-validated P8-PRE-7 (2026-04-15)
- Re-swept after ADR-007 VOL_NORM standardization confirmation. Value unchanged.
- Sweep: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]. 2.5 remains smallest
  with AvgK≥20 all 6 folds. (2.0 fails: AvgK=19.8 on fold 2019.)
- BSS confirmation: 6/6 folds within ±0.001 of Phase 6 baseline. Delta=0.000000
  on all folds — geometry was identical to Phase 6 (standardization already active).
- ADR-007 finding: StandardScaler was applied to ALL 23 features before P8-PRE-4.
  P8-PRE-4 confirmed existing behavior; did not change the distance landscape.
  max_distance=2.5 was calibrated with all-23-feature standardization already active.
- Provenance: results/phase8_pre/sweep_max_distance_23d_standardized.tsv
- BSS confirmation: results/phase8_pre/bss_confirmation_standardized.tsv
- Supersedes: nothing (re-validation of Phase 6 setting)

### max_distance=0.90, beta_abm (8D baseline, superseded)
- Swept H5 (2026-04-02). Provenance: results/bss_fix_sweep_h5.tsv

### regime=hold_spy_threshold+0.05 — H7 (2026-04-06)
- GATE MET: 3/6 positive folds, mean_BSS=+0.00033.
- mode=hold: Bear rows (SPY ret_90d < +0.05) -> base_rate prob (HOLD signal).
- Bull mode only: KNN signal used when SPY 90d return > +5% (confirmed trend).
- Provenance: results/bss_fix_sweep_h7.tsv. Caution: thin margin, aggressive threshold.

### stop_loss_atr_multiple=3.0 — Phase 3.5 (2026-03-21)
- Swept 2.0-4.0 on 2024 fold. 3.0x won: Sharpe=1.53 (+32% vs 2.0x),
  MaxDD=5.7%, stops=28/171. Provenance: results/atr_sweep_results.tsv

### confidence_threshold=0.65 — Provenance Clarification (P8-PRE-3, 2026-04-15)
- 0.65 is the LOCKED production threshold. Signals below this are HOLD.
- On 52T beta_abm: max prob ≈ 0.58, so NO signals pass. This is expected —
  52T is the experimental/validation universe only.
- On 585T Platt: prob range [0.65, 0.75], generating 159 BUY signals in 2024 fold.
- For paper trading: use whatever the 585T pipeline naturally produces. The
  threshold is applied inside the pipeline, not manually adjusted.
- The 0.55 value referenced in some experiment scripts was a diagnostic-only
  override to observe 52T behavior. It is NOT a production parameter.

### min_sector_lift=0.005
- Recalibrated 2026-03-26; old 0.03 vetoed 100% on 585T.

### BSS Root Cause (2026-03-26)
- Analogue pool dilution at 585T scale, NOT miscalibration.
  Platt is helping (+0.023 BSS vs raw). Fix: tighten max_distance or same_sector_only.
  Locked settings, require new experiment evidence before changing.

---

## Failed Enhancement Experiments (flags remain False)

### E1 BMA — FAIL (0/6 folds improved by >=+0.001)
- use_bma stays False. BMA EM-fitted Student's t mixture degrades BSS
  (delta ~-0.09 to -0.14 all folds).
- Provenance: results/phase7/e1_bma_vs_beta_abm.tsv (2026-04-09)

### E2 OWA — FAIL (0/6 folds improved by >=+0.001)
- use_owa stays False. MI-ranked OWA weighting (best alpha=4.0) shows no BSS
  improvement on 23D returns_candle.
- Deltas: [-0.00026, -0.00044, -0.00015, +0.00025, +0.00015, -0.000079] all < +0.001.
- Provenance: results/phase7/e2_owa_vs_baseline.tsv (2026-04-09)

### E3 DTW Reranker — FAIL (Spearman fast-fail: mean rho=1.0000)
- use_dtw_reranker stays False. DTW on 8 return scalars is redundant with
  Euclidean distance (same values, no warping benefit).
- Provenance: results/phase7/e3_dtw_vs_baseline.tsv (2026-04-09)

### E4 Conformal — FAIL (coverage 0.814 mean; width 1.000 mean)
- use_conformal stays False. Root cause: 52T probs cluster in [0.50, 0.59];
  |prob-label| scores always >=0.41; threshold ~0.57 yields near-trivial [0,1]
  intervals (width ~1.0). 2020-COVID fold coverage 0% (no gamma achieves 88% --
  ACI over-tightens on COVID volatility).
- Provenance: results/phase7/e4_conformal_coverage.tsv (2026-04-09)

---

## Research Corpus Corrections (Master Plan v4.0, 2026-04-18)

These are corrections to locked settings or architectural choices identified by 57-paper research
synthesis in three documents. Source: FPPE_MASTER_PLAN_v4.0, §Corrections Applied from Research Corpus.

### Kelly Formula — Sortino-Adjusted (P8-PRE-4 correction)
- **Old:** `f* = mu/sigma²` (standard Kelly)
- **New:** `f* = mu/sigma_downside²` (Sortino-adjusted Kelly)
- **Applied in:** P8-PRE-4 position sizer. Downside standard deviation uses only negative returns.
- **Rationale:** Standard Kelly penalizes upside volatility equally to downside. Sortino-adjusted
  Kelly correctly ignores upside volatility, producing larger position sizes in trending markets.
- **Status:** Adopted. Verify `trading_system/position_sizer.py` uses downside deviation.

### scoringrules — Mandatory (P8-PRE-2 correction)
- **Old:** "should-have" / optional dependency
- **New:** "must-have" / mandatory for Phase 8 monitoring and CRPS evaluation
- **Applied in:** P8-PRE-2. `pip install scoringrules` required before Phase 8.
- **Rationale:** Double-confirmed by research corpus. CRPS is the primary scoring metric for
  probabilistic forecast evaluation alongside BSS.

### HMM Look-Ahead Risk (G7.5-4 diagnostic)
- **Old:** `hmmlearn.predict_proba()` (uses Kim smoother — future observations)
- **New (if hmmlearn used):** `statsmodels.MarkovRegression` with `filtered_marginal_probabilities`
- **Applied in:** G7.5-4 audit gate. Audit `pattern_engine/regime_labeler.py` first.
- **Rationale:** Kim smoother uses future observations to smooth state posteriors — a look-ahead
  trap in walk-forward validation. Filtered (forward-only) probabilities are correct.
- **Status:** AUDIT REQUIRED (T7.5-4). If hmmlearn not found, correction is vacuous.

### sklearn Deprecation — multi_class Parameter (R2-H9)
- **Old:** `LogisticRegression(multi_class='multinomial')`
- **New:** `LogisticRegression(multi_class='auto')` or omit the parameter
- **Applied in:** R2-H9 LightGBM head-to-head. Removed in sklearn 1.7.
- **Status:** Not yet implemented. Apply when building R2-H9 LightGBM matcher.

### EOQ Rebalancing Exponent (Phase 11)
- **Old:** `√λ` (square-root law for rebalancing frequency)
- **New:** `λ^(1/3)` (cube-root law, corrected formula)
- **Applied in:** Phase 11 hyper-scale rebalancing. Not relevant until 5,200T universe.
- **Status:** Not yet implemented. Apply when building Phase 11 rebalancing logic.

### KNN+LightGBM Architecture — Delta Model Pattern (R2-H9)
- **Old:** Simple probability averaging of KNN and LightGBM outputs
- **New:** Delta Model (residual learning): KNN predicts first; LightGBM predicts KNN's residual
  using out-of-fold KNN predictions. Final: `p_ensemble = p_knn + lgbm.predict([features, p_knn])`
- **Applied in:** R2-H9 spec. Out-of-fold constraint is critical — prevents LightGBM from
  learning KNN's training-set overfitting.
- **Status:** Not yet implemented. Apply when building R2-H9.
