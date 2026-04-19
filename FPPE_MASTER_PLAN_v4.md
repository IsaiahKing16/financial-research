# FPPE Master Implementation Plan v4.0

**Version:** 4.0  
**Date:** 2026-04-18  
**Author:** Strategic Advisor (Claude)  
**Owner:** Isaiah King (Sleep)  
**Status:** ACTIVE — Phases 1–7, P8-PRE-4/5/6, R3 complete. Phase 7.5 is next.  
**Research Corpus:** 57 papers across 3 synthesis documents + 7 strategic architecture notes  
**Goal:** FPPE live with $10,000 capital by January 2027 → 5,200+ tickers by Q2 2027

---

## Executive Assessment

### Where We Are

Seven phases and three pre-production hardening sprints are complete. The production pipeline is built end-to-end: KNN analogue matching → beta_abm calibration → regime HOLD filter → Half-Kelly sizing → ATR risk engine → portfolio manager → order manager → MockBroker. 945 tests pass. 167+ commits on main. Zero known bugs.

### The Elephant in the Room: BSS = +0.00033

This number is the central fact governing every decision in this plan. Mean BSS of +0.00033 with 3/6 positive folds means the system has a statistically marginal edge — positive, but barely distinguishable from noise with the current validation framework. Three things follow from this:

**First, we do not know with confidence whether the edge is real.** The Nuclear Safety synthesis (B1) established that FPPE's current validation framework has ~12% statistical power and a 65.6% false positive rate. A control-variate BSS estimator (G7.5-5) is mandatory before we can claim the system works. Without it, we are paper-trading on hope. Confidence: 95% that the CV estimator will either confirm the edge (in which case we proceed) or reveal it as noise (in which case we stop and fix the model, saving months of wasted paper trading).

**Second, z-score normalization was already in place — and it is the only source of discriminative power.** ~~91% of L2 distance variance comes from the 8 return features~~ — this premise was invalidated by ADR-007 (2026-04-15), which confirmed StandardScaler was already fully implemented in `_prepare_features()`. The expected +0.005 to +0.02 BSS lift from "adding" normalization was already baked in to all prior results; T7.5-1 verified rather than implemented it. The more important finding from T7.5-1 (2026-04-18, ADR-015) is that raw features produce **zero resolution on all 6 folds** — StandardScaler is the sole mechanism producing any discriminative power. The ~91% claim applied to raw features and is no longer relevant; the current system operates on StandardScaler-normalized features. The remaining BSS constraint is coverage collapse in bear regimes (2022-Bear: 3.2% of queries scored), tracked under Track B/C.

**Third, the thin margin means every downstream decision carries existential risk.** Adding features that fail the Braess gate could push BSS negative. Skipping the MI ceiling diagnostic means we might invest months in a system whose theoretical maximum BSS is below the actionable threshold. The plan below is structured to answer the existential questions first (Phase 7.5), then validate operationally (Phase 8), then deploy (Phase 9).

### Critical Path

```
Phase 7.5 (Research Integration Gate) → Phase 8 (Paper Trading 3mo) → Phase 9 (Live $10k)
    ↑                                        ↑
    |                                        |
    R1 (Signal Quality) ────────────────────►│ (ships during Phase 8 window)
    R2 (Architecture Validation) ───────────►│
```

Phase 7.5 is a hard blocker. It contains 8 gate criteria that collectively determine whether the system's edge is real, what its theoretical ceiling is, and whether the feature space is correctly configured. If Phase 7.5 fails — specifically, if the CV BSS estimator shows the edge is not statistically significant or if the MI ceiling is below the actionable threshold — the plan pivots to R2-H9 (KNN vs LightGBM head-to-head) before attempting paper trading.

### Highest-Risk Decisions

| Decision | Risk | Mitigation |
|----------|------|------------|
| Z-score normalization invalidates all historical metrics | ALL prior BSS numbers become non-comparable | Full 6-fold re-validation is mandatory; feature-flag the normalization |
| Thin BSS might be noise, not signal | Months of paper trading wasted on a non-edge | CV BSS estimator (G7.5-5) answers this in ~2 days of work |
| Yahoo Finance as primary data source | Survivorship bias may contaminate BSS | Data integrity audit (deferred to Phase 9 pre-work); Norgate subscription if confirmed |
| Pool dilution at 585T | Root failure mode identified but not fixed | Multi-retriever ensemble (R1) and tiered pool expansion address this directly |
| Multi-retriever ensemble is untested architecture | Could introduce complexity without BSS lift | Feature-flagged, tested against single-retriever baseline on all 6 folds |

### What This Plan Does NOT Do

- Does not restructure Phases 1–7. They are complete and locked.
- Does not propose synthetic data augmentation. Conclusively shown to poison the analogue engine.
- Does not route through the 52T universe. 585T Platt pipeline is the only production path.
- Does not produce handoff documents. Those are Level 3 deliverables for subsequent sessions.
- Does not change any locked setting without an ADR and walk-forward evidence.

---

## Phase 7.5 — Research Integration Gate

### Strategic Purpose

Phase 7.5 absorbs the Tier 0–4 foundation fixes from the research corpus that must precede paper trading. These are not enhancements — they are diagnostic and structural corrections that determine whether the system's edge is real and whether the feature space is correctly configured. Every Phase 8 item depends on Phase 7.5 gates passing.

**Duration:** 3–4 weeks (target: May 12–16, 2026)  
**Task type:** MIX (SR for diagnostics, JR for implementation)  
**Dependencies:** Phases 1–7 complete, P8-PRE-4/5/6 complete, R3 complete

### Gate Criteria

| Gate | Criterion | Source | Verdict |
|------|-----------|--------|---------|
| G7.5-1 | Z-score normalization on all 23 features verified, full 6-fold re-validation complete, BSS ≥ +0.00033 maintained or improved | Synthesis #2 A1.1, Candlestick Finding 1, ADR-007 VOL_NORM | HARD BLOCK |
| G7.5-2 | Braess gate (`braess_gate()`) implemented, passing on current 23D fingerprint. Every future feature addition must pass this gate before inclusion. | Domain Y (Y4) | HARD BLOCK |
| G7.5-3 | Identifiability gate (3h) installed. Confirms model parameters are determinable from data before any tuning. | Epidemiology B7 | HARD BLOCK |
| G7.5-4 | `hmmlearn.predict_proba()` look-ahead audit completed. If current codebase uses `hmmlearn`, migrate to `statsmodels.MarkovRegression` with `filtered_marginal_probabilities`. | Domain 4, Synthesis #2 C1 | HARD BLOCK |
| G7.5-5 | Control-variate BSS estimator producing confidence intervals on all 6 folds. Point estimate + 95% CI reported. | Nuclear Safety B1 | HARD BLOCK |
| G7.5-6 | Murphy B3 decomposition (REL, RES, UNC) computed on current pipeline. Diagnosis: reliability-dominated or resolution-dominated failure? | Synthesis #1 Cat A | HARD BLOCK |
| G7.5-7 | MI ceiling diagnostic (`mutual_info_classif`) establishing theoretical BSS upper bound. If MI ceiling < 0.01, the feature space cannot support a tradeable edge and architectural pivot is required. | Reliability Engineering, 12-Domain Survey | HARD BLOCK |
| G7.5-8 | Multi-horizon BSS curve (1d/3d/5d/7d/10d/14d) computed on current folds. Identifies optimal prediction horizon. | Strategic Note 4 | INFORMATIONAL (does not block Phase 8, but informs R1 priority) |

### Decision Logic After G7.5 Gates

```
IF G7.5-5 CI includes 0:
    → BSS edge is not statistically significant
    → HALT Phase 8. Activate R2-H9 (KNN vs LightGBM). 
    → If ensemble improves BSS significantly, re-gate and proceed.

IF G7.5-6 shows REL >> RES:
    → Reliability-dominated failure (calibration is wrong)
    → Hypothesis: beta_abm calibration needs replacement
    → Activate Venn-ABERS or isotonic calibration experiment

IF G7.5-6 shows RES >> REL:
    → Resolution-dominated failure (model lacks discriminative power)
    → Hypothesis: feature space or universe is insufficient
    → Activate multi-retriever ensemble (R1) as priority

IF G7.5-7 MI ceiling < 0.01:
    → Feature space cannot support tradeable edge
    → HALT. Evaluate: add features (rough vol, H-statistics) or pivot model class

IF G7.5-1 BSS degrades after z-score normalization:
    → Revert normalization (feature-flagged)
    → Investigate: which feature groups are degrading?
    → Possible fix: group-wise normalization (returns separate from candles)
```

### Sub-Tasks (Level 2 Detail)

#### T7.5-1: Z-Score Normalization + Full Re-Validation

**Description:** Apply z-score normalization to all 23 features before HNSW indexing. This equalizes variance so the 15 candlestick proportions participate meaningfully in L2 distance computation. Requires full 6-fold walk-forward re-validation because every neighbor changes.

**Acceptance Criteria:**
1. Z-score normalization applied per feature, per training fold (no test leakage)
2. BSS computed on all 6 folds with normalization ON
3. BSS on all 6 folds with normalization OFF (current baseline, for comparison)
4. Comparison table: per-fold BSS delta
5. If mean BSS improves or holds: normalization becomes the new baseline
6. If mean BSS degrades: investigate group-wise normalization before rejecting

**Files to Modify:**
- `pattern_engine/features.py` (MODIFY — add normalization step)
- `pattern_engine/matcher.py` (MODIFY — normalize query vectors at query time)
- `scripts/run_zscore_validation.py` (NEW)
- `results/phase7_5/zscore_bss_comparison.tsv` (output)
- `tests/unit/test_zscore_normalization.py` (NEW)
- `docs/adr/ADR-013-zscore-normalization.md` (NEW — if normalization is adopted)

**Estimated Effort:** 2–3 days (implementation: 4h, validation run: 8–16h compute, analysis: 4h)  
**Task Type:** MIX  
**Dependencies:** ADR-007 VOL_NORM must be compatible. Verify z-score doesn't conflict with existing VOL_NORM standardization.  
**Sequential/Parallel:** SEQUENTIAL — must complete before T7.5-2 (Braess gate needs the normalized feature space)

**Risks:**
1. **Z-score leakage.** If normalization parameters (mean, std) are computed on the full dataset including test fold, BSS is contaminated. Mitigation: compute per training fold only; store and apply to test fold.
2. **VOL_NORM conflict.** ADR-007 already specifies a standardization method. If VOL_NORM and z-score are different transforms, they may compound or conflict. Mitigation: audit ADR-007 implementation first; z-score may already be what VOL_NORM does.
3. **BSS regression.** If normalization makes BSS worse, the 15 candlestick features may be adding noise, not signal. This would invalidate the Phase 6 result. Mitigation: group-wise normalization experiment (normalize returns and candles independently with different scales).

---

#### T7.5-2: Braess Gate Implementation

**Description:** Implement `braess_gate()` — a statistical test that verifies adding a feature (or feature group) does not degrade system performance. Based on Braess's paradox from network theory: adding a road to a traffic network can increase total travel time. Similarly, adding a feature to KNN can worsen BSS if the feature introduces noise that corrupts the distance geometry.

The gate computes BSS with and without the candidate feature(s) on all 6 walk-forward folds. If BSS degrades on ≥ 3/6 folds, the feature fails the gate and is rejected.

**Acceptance Criteria:**
1. `braess_gate(feature_set_with, feature_set_without, folds)` returns PASS/FAIL + per-fold BSS deltas
2. Current 23D `returns_candle` passes the gate (baseline sanity check)
3. Gate is callable from any future feature-addition experiment
4. All future T7.5 feature experiments (H-statistics, rough vol) must pass this gate

**Files to Modify:**
- `pattern_engine/diagnostics.py` (NEW or MODIFY — add `braess_gate()`)
- `tests/unit/test_braess_gate.py` (NEW)

**Estimated Effort:** 1 day  
**Task Type:** JR (clear spec, straightforward implementation)  
**Dependencies:** T7.5-1 (needs the normalized baseline to gate against)  
**Sequential/Parallel:** SEQUENTIAL after T7.5-1

**Risks:**
1. **Computational cost.** Each gate invocation requires 6-fold walk-forward. At 585T, this may take hours. Mitigation: implement with a `--fast` mode that tests on 3 folds (even-numbered) for rapid screening, full 6 for final verdict.
2. **Threshold sensitivity.** "≥ 3/6 folds degrade" is a simple majority rule. A feature that massively improves 3 folds and slightly degrades 3 would fail. Mitigation: report mean BSS delta alongside the vote count; allow override with ADR + documented rationale.
3. **Correlation with other features.** Braess gate tests features in isolation. A feature that fails alone might pass in combination with another. Mitigation: support `braess_gate()` with feature groups, not just individual features.

---

#### T7.5-3: Identifiability Gate

**Description:** Install a structural identifiability check that confirms model parameters are determinable from data. In FPPE's context: confirm that the number of neighbors (k), distance threshold, and calibration parameters are identifiable given the training data volume per fold.

Concretely: for each walk-forward fold, compute the effective number of parameters vs. the number of training samples. If the ratio exceeds a threshold (typically < 1:20 for nonparametric methods), the model is overparameterized for that fold.

**Acceptance Criteria:**
1. `identifiability_gate(fold_config)` returns PASS/FAIL per fold
2. Reports: training samples, effective parameters, ratio
3. All 6 folds pass on current configuration

**Files to Modify:**
- `pattern_engine/diagnostics.py` (MODIFY — add `identifiability_gate()`)
- `tests/unit/test_identifiability_gate.py` (NEW)

**Estimated Effort:** 0.5 days  
**Task Type:** SR (requires understanding of effective parameters in nonparametric methods)  
**Dependencies:** None  
**Sequential/Parallel:** PARALLEL with T7.5-1

**Risks:**
1. **"Effective parameters" is undefined for KNN.** Unlike parametric models, KNN doesn't have a fixed parameter count. Mitigation: use Hastie et al. (2009) effective degrees of freedom for local methods: `df(x) = tr(S)` where S is the smoother matrix. Computationally expensive at 585T but feasible on a subsample.
2. **May be vacuously true.** KNN with k=50 neighbors on 500K training points is almost certainly identifiable. The gate may always pass. Mitigation: still valuable as documentation; confirms the system is not overparameterized.
3. **False sense of security.** Identifiability does not imply correctness. Mitigation: this gate is necessary but not sufficient; it complements, not replaces, BSS validation.

---

#### T7.5-4: HMM Look-Ahead Audit

**Description:** Audit the current codebase for any usage of `hmmlearn.predict_proba()` on sequences that include test-fold data. This method returns smoothed posteriors (Kim smoother), which use future observations — a look-ahead trap. If found, migrate to `statsmodels.MarkovRegression` with `filtered_marginal_probabilities`.

**Acceptance Criteria:**
1. Grep-level audit of all `.py` files for `hmmlearn`, `predict_proba`, `smoothed`
2. If `hmmlearn.predict_proba()` is used on any sequence including test data: flag as CRITICAL
3. If flagged: implement migration to `statsmodels.MarkovRegression`
4. Verify `filtered_marginal_probabilities` (forward-only) is used, not `smoothed_marginal_probabilities`
5. Re-run BSS on all 6 folds post-migration if any code was changed

**Files to Modify:**
- `pattern_engine/regime_labeler.py` (AUDIT, possibly MODIFY)
- `tests/unit/test_regime_lookahead.py` (NEW — regression test ensuring filtered-only)

**Estimated Effort:** 0.5–2 days (0.5 if no `hmmlearn` found; 2 if migration required)  
**Task Type:** SR (requires understanding of filtering vs smoothing in state-space models)  
**Dependencies:** None  
**Sequential/Parallel:** PARALLEL with T7.5-1

**Risks:**
1. **False positive.** FPPE may use `hmmlearn` correctly (fitting only on training data, predicting only on test data without joining sequences). The audit must verify the actual code path, not just grep for the import. Mitigation: trace the exact data flow from training fold through prediction.
2. **`statsmodels.MarkovRegression` API differences.** The `statsmodels` HMM API is less ergonomic than `hmmlearn`. State ordering may differ, causing label-switching. Mitigation: sort states by emission variance after each fit (canonical fix per H8 Variant Plan).
3. **BSS change from migration.** Even a correct migration may change BSS because filtered probabilities are noisier than smoothed. Mitigation: this is the right thing to do — smoothed BSS was artificially inflated.

---

#### T7.5-5: Control-Variate BSS Estimator

**Description:** Implement a control-variate variance reduction estimator for BSS. The current BSS = +0.00033 is a point estimate. Without a confidence interval, we cannot distinguish signal from noise.

The correct formulation (per Nuclear Safety B1, corrected formula): operate at the Brier Score layer. `BS_CV(model) = BS(model) − β·(BS(clim) − E[BS(clim)])` where β is the regression coefficient from `cov(BS_model, BS_clim) / var(BS_clim)`. Variance reduces by factor `(1 − ρ²)` where ρ = corr(BS_model, BS_clim), typically 0.6–0.9, yielding 2–10× variance reduction.

**Acceptance Criteria:**
1. `cv_bss_estimator(predictions, actuals, climatology)` returns: point estimate, 95% CI, variance reduction factor
2. Computed on all 6 walk-forward folds
3. Report: per-fold BSS point estimate, CI lower bound, CI upper bound
4. Decision criterion: if CI lower bound > 0 on ≥ 3/6 folds → edge is statistically significant

**Files to Modify:**
- `pattern_engine/scoring.py` (MODIFY — add `cv_bss_estimator()`)
- `tests/unit/test_cv_bss_estimator.py` (NEW)
- `results/phase7_5/cv_bss_confidence_intervals.tsv` (output)

**Estimated Effort:** 1–2 days (~350 LOC per Nuclear Safety estimate)  
**Task Type:** SR (statistical method implementation)  
**Dependencies:** None  
**Sequential/Parallel:** PARALLEL with T7.5-1

**Risks:**
1. **CI includes 0.** This is the existential risk. If the 95% CI includes 0 on 4+ folds, the edge may not be real. Mitigation: this is the point of the exercise — better to know now than after 3 months of paper trading.
2. **Correlation assumption.** The CV estimator assumes a linear relationship between `BS_model` and `BS_clim`. If the relationship is nonlinear, variance reduction is smaller. Mitigation: check the scatter plot; if nonlinear, use a kernel regression variant.
3. **Small sample size.** Each fold has a limited number of predictions. CIs may be wide even with variance reduction. Mitigation: report the width alongside the estimate; if CIs are wider than ±0.01, the current fold structure is insufficient and CPCV (≥10 groups) should be prioritized.

---

#### T7.5-6: Murphy B3 Decomposition

**Description:** Compute the Murphy (1973) Brier Score decomposition: BS = REL − RES + UNC. Convert to BSS terms: BSS = (RES − REL) / UNC. Diagnose whether the current system's thin BSS is reliability-dominated (calibration error) or resolution-dominated (lack of discriminative power).

This directly determines hypothesis ordering for all subsequent work:
- If REL >> RES: fix calibration first (try Venn-ABERS, isotonic)
- If RES ≈ 0: the model has no discriminative power; features or architecture must change
- If REL ≈ 0 and RES is small but positive: the system is well-calibrated but weakly discriminative; signal quality improvements (R1) are the correct next step

**Acceptance Criteria:**
1. `murphy_b3_decomposition(predictions, actuals)` returns REL, RES, UNC per fold
2. Computed on all 6 walk-forward folds
3. Diagnosis categorized as: RELIABILITY_DOMINATED / RESOLUTION_DOMINATED / BALANCED
4. Results table committed to `results/phase7_5/murphy_b3_decomposition.tsv`

**Files to Modify:**
- `pattern_engine/scoring.py` (MODIFY — add `murphy_b3_decomposition()`)
- `tests/unit/test_murphy_decomposition.py` (NEW)

**Estimated Effort:** 1 day  
**Task Type:** SR  
**Dependencies:** None  
**Sequential/Parallel:** PARALLEL with T7.5-1

**Risks:**
1. **Binning sensitivity.** Murphy B3 requires binning probabilities. With narrow probability ranges [0.65, 0.75], the number of bins matters. Too many → empty bins. Too few → coarse resolution. Mitigation: use 10 bins as default; sensitivity analysis with 5, 15, 20 bins.
2. **UNC dominance.** If UNC >> (REL + RES), the decomposition is uninformative — the signal is buried in base-rate uncertainty. Mitigation: this is itself diagnostic; if true, the system needs much higher resolution to be actionable.
3. **Interaction with z-score normalization.** The decomposition should be run AFTER T7.5-1 (on the normalized baseline), not before. Running before would diagnose a system we know is misconfigured. Mitigation: run both pre-normalization (for comparison) and post-normalization (for decision-making).

---

#### T7.5-7: MI Ceiling Diagnostic

**Description:** Compute the mutual information (MI) between the 23D feature vector and the binary target (up/down) using `sklearn.feature_selection.mutual_info_classif`. This establishes the theoretical upper bound on BSS achievable from the current feature space. If MI is near zero, no amount of model tuning or calibration can produce a tradeable edge from these features.

**Acceptance Criteria:**
1. MI computed per feature and for the joint 23D vector
2. Per-feature MI ranking table (which features carry the most information?)
3. Joint MI establishes ceiling: `BSS_max ≈ 2 × MI / UNC` (approximate relationship)
4. If joint MI < 0.001: feature space is insufficient → activate R1 feature engineering
5. Results committed to `results/phase7_5/mi_ceiling_diagnostic.tsv`

**Files to Modify:**
- `scripts/mi_ceiling_diagnostic.py` (NEW)
- `results/phase7_5/mi_ceiling_diagnostic.tsv` (output)

**Estimated Effort:** 0.5 days  
**Task Type:** MIX  
**Dependencies:** T7.5-1 (run on normalized features)  
**Sequential/Parallel:** SEQUENTIAL after T7.5-1

**Risks:**
1. **MI underestimation.** `mutual_info_classif` uses a KNN-based MI estimator that can underestimate MI in high dimensions with small samples. Mitigation: also compute per-feature MI (lower-dimensional, more accurate) and sum as a lower bound.
2. **MI ≈ 0 is actually informative.** If MI confirms the feature space carries near-zero information about the target, that is the most valuable finding in the entire project — it means the current architecture cannot work regardless of calibration fixes. Mitigation: this triggers the R1 research track (new features, multi-retriever ensemble) immediately.
3. **Feature interaction effects.** Individual feature MI may be low, but joint MI could be higher due to interactions. Mitigation: report both individual and joint MI; the gap indicates interaction effects.

---

#### T7.5-8: Multi-Horizon BSS Curve

**Description:** Compute BSS at prediction horizons of 1, 3, 5, 7, 10, and 14 trading days on the current 6-fold walk-forward setup. This answers: is 1-day-ahead the optimal prediction horizon, or would FPPE perform better at 3-day or 5-day horizons?

**Acceptance Criteria:**
1. BSS computed at all 6 horizons across all 6 folds (36 data points)
2. Per-horizon mean BSS with 95% CI (using T7.5-5's CV estimator)
3. Curve plotted: BSS vs horizon
4. Optimal horizon identified (highest mean BSS with CI not including 0)
5. Results committed to `results/phase7_5/multi_horizon_bss.tsv`

**Files to Modify:**
- `scripts/multi_horizon_bss.py` (NEW)
- `pattern_engine/features.py` (MODIFY — parameterize target horizon)
- `results/phase7_5/multi_horizon_bss.tsv` (output)

**Estimated Effort:** 2–3 days (6 horizons × full validation = significant compute)  
**Task Type:** SR  
**Dependencies:** T7.5-1 (normalized features), T7.5-5 (CV estimator for CIs)  
**Sequential/Parallel:** SEQUENTIAL after T7.5-1 and T7.5-5. Can overlap with T7.5-6 and T7.5-7.

**Risks:**
1. **Computational cost.** 6 horizons × 6 folds × 585T = 36 full validation runs. At ~30 min each, this is ~18 hours. Mitigation: parallelize across horizons (each is independent). Or run overnight.
2. **Target definition ambiguity.** "3-day return" could mean close-to-close over 3 days, or max intra-period return. Mitigation: use close-to-close (consistent with current 1-day target). Document the definition.
3. **Diminishing returns at longer horizons.** KNN analogues matched on daily patterns may lose relevance at 10–14 day horizons due to intervening events. Mitigation: this is exactly what the curve is designed to reveal. If BSS drops monotonically with horizon, the 1-day choice is confirmed.

---

### Phase 7.5 Task Dependency Graph

```
T7.5-3 (Identifiability) ─────────────────────────────────┐
T7.5-4 (HMM Audit) ───────────────────────────────────────┤
T7.5-5 (CV BSS) ──────────────────────────────────────────┤
                                                           ├──► RIA 7.5 → Phase 8
T7.5-1 (Z-Score) ──► T7.5-2 (Braess Gate) ──► T7.5-7 ───┤
                  └──► T7.5-6 (Murphy B3) ────────────────┤
                  └──► T7.5-8 (Multi-Horizon) ────────────┘
```

**Parallel group A (days 1–3):** T7.5-3, T7.5-4, T7.5-5, T7.5-6 (pre-normalization run)  
**Sequential chain (days 1–7):** T7.5-1 → T7.5-2 → T7.5-7  
**Parallel group B (days 7–14):** T7.5-6 (post-normalization), T7.5-8  
**RIA checkpoint (day 14–15):** Research Impact Assessment using Phase 7.5 results

### Top 3 Risks for Phase 7.5

1. **Z-score normalization degrades BSS.** Probability: 25%. Impact: HIGH — invalidates the Phase 6 finding that candlestick features help. Mitigation: group-wise normalization fallback; revert via feature flag.
2. **CV BSS CI includes 0 on most folds.** Probability: 35%. Impact: CRITICAL — means the edge may not be real. Mitigation: activate R2-H9 (KNN vs LightGBM) immediately; if ensemble BSS is significant, proceed.
3. **MI ceiling near zero.** Probability: 15%. Impact: CRITICAL — means the feature space is fundamentally insufficient. Mitigation: R1 research track (rough vol, H-statistics, multi-retriever ensemble) becomes the priority.

### Iteration Instructions for Level 3

In a subsequent session, produce one handoff document per sub-task (T7.5-1 through T7.5-8) using the format in §4.2 of the master planning prompt. Each handoff document must:
1. Translate the acceptance criteria above into exact `assert` statements or expected output patterns
2. Include the exact test commands (starting with `PYTHONUTF8=1 py -3.12 -m pytest`)
3. Include the 3-risk adversarial review from above, expanded with mitigation code patterns
4. Tag with SR/JR/MIX for routing
5. Reference the specific files/functions from the current codebase that need modification

---

## Phase 8 — Paper Trading + Autonomous EOD

### Strategic Purpose

Phase 8 validates the full trading stack in live market conditions with simulated capital. The 3-month trading window serves dual purposes: (1) operational validation (does the pipeline run autonomously without crashes, data gaps, or reconciliation errors?) and (2) performance validation (does the system produce positive risk-adjusted returns in real-time, not just in backtests?).

Phase 8 also serves as the integration window for the three highest-value R1 research items (multi-retriever ensemble, tiered pool expansion, vol-normalized fingerprint), which ship during the 3-month trading period as feature-flagged enhancements.

**Duration:** 14 weeks (2 weeks setup + 12 weeks trading)  
**Target Start:** May 19, 2026 (assuming Phase 7.5 completes by May 16)  
**Target Complete:** August 21, 2026  
**Task type:** JR (pipeline infrastructure) + SR (performance analysis, research integrations)  
**Dependencies:** All G7.5 gates passed

### Gate Criteria (Phase 8 → Phase 9)

| Gate | Criterion | Threshold |
|------|-----------|-----------|
| G8-1 | Pipeline autonomous operation | ≥ 20 consecutive trading days without manual intervention |
| G8-2 | Trading days completed | ≥ 60 |
| G8-3 | Round-trip trades | ≥ 50 |
| G8-4 | Win rate | ≥ 50% |
| G8-5 | Annualized Sharpe ratio | ≥ 1.0 |
| G8-6 | Maximum drawdown | ≤ 15% |
| G8-7 | Reconciliation error rate | < 0.1% |
| G8-8 | EOD pipeline latency | ≤ 30 minutes end-to-end |
| G8-9 | MTBF (Mean Time Between Failures) | ≥ 168 hours (1 week) |
| G8-10 | Deflated Sharpe Ratio (DSR) | > 0 (guards against multiple-testing bias) |

### Pre-Phase 8 Tasks (P8-PRE)

**P8-PRE-1 (585T End-to-End Validation)** — HARD BLOCKER. Run the full trading stack (signal generation → sizing → risk → PM → MockBroker) on all 6 walk-forward folds using the 585T Platt pipeline with post-7.5 normalized features. Produce gate metrics: BSS, Sharpe, MaxDD, trade count, win rate. This is the ground-truth validation that the system works at production scale.

**P8-PRE-2 (scoringrules Integration)** — Install `scoringrules` package (double-confirmed mandatory) for CRPS evaluation and proper scoring rule infrastructure. This is the scoring backbone for Phase 8 monitoring.

**P8-PRE-3 (Confidence Threshold Resolution)** — Resolve the 0.55 vs 0.65 confidence threshold provenance gap. Lock the paper-trading threshold with ADR.

**P8-PRE-4 (Sortino-Adjusted Kelly)** — Replace standard Kelly `f* = mu/sigma²` with Sortino-adjusted Kelly `f* = mu/sigma_downside²`. Zero additional complexity. Direct improvement to position sizing. Must pass walk-forward validation (Sharpe must not degrade).

### Sub-Tasks (Level 2 Detail)

#### T8.1 — EOD Pipeline Automation

**Description:** Build the autonomous end-of-day pipeline that fetches data, rebuilds features, generates signals, and submits orders without human intervention.

| Sub-task | Description | Acceptance Criteria | Task Type |
|----------|-------------|---------------------|-----------|
| T8.1a | `scripts/eod_pipeline.py` orchestrator | Calls data fetch → features → index → match → calibrate → filter → risk → PM → MockBroker in sequence | JR |
| T8.1b | `config/eod_schedule.json` step definitions | Timeout, required flag, retry count per step | JR |
| T8.1c | `scripts/health_check.py` pre-execution validation | 7-point check: index freshness, data currency, broker reachable, disk > 5GB, RAM > 8GB, no stale locks, feature data complete | JR |
| T8.1d | Windows Task Scheduler integration | `eod_runner.bat` wrapper; fires at 4:00 PM ET daily; runs whether user logged in or not | JR |
| T8.1e | Idempotent step design | Each step safe to re-run; logs start/end/duration/status to `results/execution_log.jsonl` | JR |
| T8.1f | Alert system | Email or local notification on: pipeline failure, drawdown brake, reconciliation error, timing overrun | JR |

**Estimated Effort:** 3–4 days  
**Files:** `scripts/eod_pipeline.py` (NEW), `scripts/health_check.py` (NEW), `config/eod_schedule.json` (NEW), `eod_runner.bat` (NEW), `tests/unit/test_eod_pipeline.py` (NEW)

---

#### T8.2 — MockBroker Paper Trading Mode

**Description:** Extend the Phase 5 MockBroker to simulate realistic paper trading conditions.

| Sub-task | Description | Acceptance Criteria | Task Type |
|----------|-------------|---------------------|-----------|
| T8.2a | Realistic slippage model | 10bps slippage + $0.005/share commission on all fills | JR |
| T8.2b | Daily mark-to-market | Equity curve updates at close; positions valued at close price | JR |
| T8.2c | Stop-loss execution simulation | Intraday check using daily OHLCV Low for longs (conservative) | JR |
| T8.2d | Order rejection simulation | Random 2% rejection rate to test error handling | JR |

**Estimated Effort:** 2 days  
**Files:** `trading_system/mock_broker.py` (MODIFY), `tests/unit/test_mock_broker_paper.py` (NEW)

---

#### T8.3 — Monitoring Dashboard + Scoring Infrastructure

**Description:** Build daily monitoring for paper trading performance using `scoringrules` for proper scoring rule evaluation.

| Sub-task | Description | Acceptance Criteria | Task Type |
|----------|-------------|---------------------|-----------|
| T8.3a | `scripts/daily_report.py` | Automated: P&L, equity curve, rolling BSS, calibration reliability diagram, sector attribution | JR |
| T8.3b | Rolling Sharpe ratio (252-day window) | Available after 20 trading days | JR |
| T8.3c | Drawdown chart | Current DD, max DD, DD duration | JR |
| T8.3d | Trade log | Entry/exit/P&L per position, queryable by date, ticker, sector, confidence | JR |
| T8.3e | Forward BSS tracking (rolling 30-day) | Measures live signal accuracy (not walk-forward) | MIX |
| T8.3f | `scoringrules` CRPS integration | CRPS computed alongside BSS for probabilistic forecast evaluation | SR |
| T8.3g | Murphy B3 decomposition (rolling) | Track REL/RES/UNC over time to detect calibration drift | SR |

**Estimated Effort:** 3–4 days  
**Files:** `scripts/daily_report.py` (NEW), `scripts/weekly_review.py` (NEW), `pattern_engine/scoring.py` (MODIFY — add CRPS via scoringrules)

---

#### T8.4 — Reconciliation Infrastructure

**Description:** Daily position reconciliation to catch drift between expected and actual state.

| Sub-task | Description | Acceptance Criteria | Task Type |
|----------|-------------|---------------------|-----------|
| T8.4a | Daily recon script | Compare expected positions vs MockBroker state | JR |
| T8.4b | Recon error alerting | Alert within 5 minutes of mismatch detection | JR |
| T8.4c | Recovery protocol document | Manual intervention steps for each error type | JR |

**Estimated Effort:** 1 day  
**Files:** `scripts/reconciliation.py` (MODIFY — extend for daily auto-run)

---

#### T8.5 — Performance Validation Checkpoints

Formal gate checks at predetermined intervals during paper trading.

| Checkpoint | Trading Day | Gate Criteria | Action if FAIL |
|-----------|-------------|---------------|----------------|
| **Week 2** | Day 10 | Pipeline runs autonomously ≥ 5 consecutive days, ≥ 5 trades executed | Fix pipeline reliability; do not proceed to Month 1 |
| **Month 1** | Day 22 | Win rate ≥ 45%, zero reconciliation errors, MTBF ≥ 168h | Investigate signal quality; check regime filter |
| **Month 2** | Day 44 | Sharpe ≥ 0.5 (lower bar for shorter window), MaxDD ≤ 20% | Review risk parameters; consider activating R1 improvements |
| **Month 3** | Day 60+ | ALL G8-1 through G8-10 | If PASS → Phase 9. If FAIL → diagnose, extend, or pivot. |

**Task Type:** SR (analysis at each checkpoint)

---

#### T8.6 — Autonomous Operation Validation

| Sub-task | Description | Acceptance Criteria | Task Type |
|----------|-------------|---------------------|-----------|
| T8.6a | 20-day hands-off test | Pipeline runs and logs correctly every day without intervention | JR |
| T8.6b | Weekend/holiday handling | Skip non-trading days, handle Monday data gaps | JR |
| T8.6c | Data source failure handling | Retry logic + fallback on yfinance outage | JR |
| T8.6d | Crash recovery | Pipeline resumes from last successful step after OS restart | JR |

**Estimated Effort:** 2 days  
**Files:** `scripts/eod_pipeline.py` (MODIFY — add recovery logic), `config/market_calendar.json` (NEW)

---

#### T8.7 — FMEA (Failure Mode and Effects Analysis)

**Description:** Conduct AIAG-VDA 7-step FMEA on every pipeline node before paper trading begins. This is institutional-grade risk assessment, not optional overhead.

| Pipeline Node | Potential Failure Mode | Severity | Detection | RPN Target |
|---------------|----------------------|----------|-----------|------------|
| Data fetch | Stale/missing data | High | Health check | < 100 |
| Feature rebuild | NaN/Inf propagation | Critical | FiniteFloat validation | < 50 |
| HNSW query | Recall degradation | Medium | Monthly recall audit | < 150 |
| Calibration | Probability drift | High | CUSUM drift monitor | < 100 |
| Signal filter | Regime misclassification | Medium | Look-ahead audit (G7.5-4) | < 150 |
| Position sizing | Oversized positions | Critical | Max position cap | < 50 |
| Order submission | Rejected/duplicate orders | Medium | Idempotency check | < 100 |

**Estimated Effort:** 1 day (document, not code)  
**Task Type:** SR  
**Files:** `docs/fmea/FMEA_PHASE8.md` (NEW)

---

### Research Integrations During Paper Trading Window

These three R1 items ship during the 3-month trading window as feature-flagged enhancements. They run in parallel with paper trading monitoring (they modify the signal generation path, not the execution path).

#### R1-INT-1: Multi-Retriever Ensemble (Weeks 3–8)

**This is the single most architecturally important intervention in the entire research corpus.**

The current system uses one HNSW index on the full 23D feature space. The multi-retriever ensemble builds 5 independent HNSW indices on different feature subspaces:

| Retriever | Feature Subspace | Rationale |
|-----------|-----------------|-----------|
| R1 | 8 return features | Raw price dynamics |
| R2 | 15 candlestick proportions | Bar shape geometry |
| R3 | 8 returns + vol-normalized | Volatility-adjusted dynamics |
| R4 | Top-k MI-ranked features | Data-driven optimal subset |
| R5 | PCA-reduced to 5D | Decorrelated summary |

Each retriever produces its own neighbor set and probability estimate. Results are fused via **reciprocal rank fusion** (Cormack et al. 2009): `RRF(d) = Σ 1/(k + rank_i(d))` where k=60 is the standard smoothing constant.

**Why this works:** Pool dilution is the root BSS failure mode. At 585T, the single HNSW index retrieves neighbors that may be geometrically close in the full 23D space but irrelevant in the subspace that matters for the current query. Multiple retrievers on different subspaces diversify the neighbor set, directly addressing dilution.

**Expected impact:** +5–15% relative BSS from ensemble diversity (Krogh-Vedelsby theorem: ensemble error ≤ mean-member-error minus diversity term). Confidence: 70%.

**Gate:** Must pass Braess gate (G7.5-2) — ensemble BSS must equal or exceed single-retriever BSS on ≥ 4/6 folds.

**Task Type:** SR  
**Estimated Effort:** 2 weeks  
**Files:** `pattern_engine/multi_retriever.py` (NEW), `pattern_engine/reciprocal_rank_fusion.py` (NEW), `tests/unit/test_multi_retriever.py` (NEW)

---

#### R1-INT-2: Value-Stream Map the EOD Pipeline (Week 2)

**Description:** Profile every step of the EOD pipeline (data fetch → features → index → match → calibrate → filter → risk → PM → orders) and produce a value-stream map showing: cycle time, wait time, value-add ratio per step. Identify bottlenecks. This is a one-time diagnostic that informs optimization throughout Phase 8.

**Expected finding:** HNSW index rebuild is likely the bottleneck (60 min at 585T). If confirmed, incremental index update (insert new vectors without full rebuild) becomes priority.

**Task Type:** JR  
**Estimated Effort:** 0.5 days  
**Files:** `docs/value_stream_map.md` (NEW)

---

#### R1-INT-3: Defense-in-Depth Audit (Week 4)

**Description:** The Nuclear Safety synthesis identified that FPPE's five "independent" safety layers (signal filter, position sizer, risk engine, portfolio manager, drawdown brake) share a single data bus (price feed + timestamp). A single corrupted price feed would bypass all five layers simultaneously — a common-cause failure. This audit documents the dependency graph and proposes at least one independent safety channel.

**Proposed fix:** Multi-vendor price feed with BFT-style agreement (deferred to Phase 9, but the audit and architecture proposal happen in Phase 8).

**Task Type:** SR  
**Estimated Effort:** 1 day  
**Files:** `docs/defense_in_depth_audit.md` (NEW)

---

### Phase 8 Diagnostic Protocol

| Symptom | Action |
|---------|--------|
| System crash | Root cause analysis. Check memory (32GB tight at 585T HNSW), broker timeout, uncaught exceptions. |
| BSS turns negative (30-day rolling) | Check regime shift. Compare to walk-forward fold matching current regime. If transient (< 2 weeks), continue. If sustained, activate R1. |
| Calibration drift (bucket diverges > 15pp) | Trigger CUSUM drift monitor. If sustained > 2 weeks, re-run calibration. |
| Slippage > 10bps consistently | Adjust slippage model upward. If > 25bps, consider VWAP execution window for Phase 9. |
| Idle cash > 50% for > 10 days | Lower confidence threshold (requires experiment + ADR) or activate multi-retriever ensemble. |
| Pipeline failure 3 consecutive days | HALT pipeline. Root cause diagnosis. 3-strike rule applies. |

### Phase 8 Deliverables

| Deliverable | Location |
|------------|----------|
| EOD pipeline | `scripts/eod_pipeline.py` |
| Health check | `scripts/health_check.py` |
| Daily report | `scripts/daily_report.py` |
| Weekly review | `scripts/weekly_review.py` |
| Trade log | `results/paper_trading/trades.csv` |
| Equity curve | `results/paper_trading/equity_curve.csv` |
| Daily JSON logs | `results/paper_trading/daily/YYYY-MM-DD.json` |
| FMEA document | `docs/fmea/FMEA_PHASE8.md` |
| Value-stream map | `docs/value_stream_map.md` |
| Defense-in-depth audit | `docs/defense_in_depth_audit.md` |
| Gate check document | `results/paper_trading/gate_check.txt` |
| RIA-8 document | `docs/ria/RIA_PHASE_8.md` |

### Top 3 Risks for Phase 8

1. **Signal quality insufficient for live trading.** Probability: 30%. The thin BSS margin (even if statistically significant per G7.5-5) may not translate to profitable paper trading when hit with real market noise, slippage, and timing effects. Mitigation: R1-INT-1 (multi-retriever ensemble) ships during the window; performance checkpoints detect problems at Week 2 and Month 1.

2. **Data source instability.** Probability: 20%. Yahoo Finance is flagged as a poisoning risk. Stale data, missing tickers, or corporate action errors could corrupt signals. Mitigation: health check validates data freshness; alert on missing tickers; Phase 9 pre-work includes Norgate evaluation.

3. **Pipeline reliability at 585T scale.** Probability: 25%. The pipeline has only been tested on 52T. At 585T, memory pressure (32GB), HNSW build times, and feature computation may exceed the 30-minute window. Mitigation: value-stream map (R1-INT-2) identifies bottlenecks early; incremental index update as fallback.

### Iteration Instructions for Level 3

In a subsequent session, produce one handoff document per sub-task (T8.1 through T8.7, plus P8-PRE-1 through P8-PRE-4, plus R1-INT-1 through R1-INT-3). Each handoff must:
1. Include the exact Windows Task Scheduler `schtasks` command for T8.1d
2. Include the exact `scoringrules` API calls for T8.3f
3. Include the reciprocal rank fusion algorithm specification for R1-INT-1
4. Reference the specific MockBroker methods that need modification for T8.2
5. Include the FMEA template (AIAG-VDA 7-step) for T8.7

---

## Phase 9 — Live Deployment ($10k)

### Strategic Purpose

Deploy FPPE with $10,000 real capital through Interactive Brokers. The system operates autonomously with weekly manual 2FA maintenance. Performance is monitored daily with automatic halt conditions. Capital scales through a staged ladder contingent on performance.

**Duration:** 4 weeks setup + ongoing operation  
**Target Start:** September 2026  
**Target Complete:** October 2026 (initial deployment)  
**Task type:** MIX  
**Dependencies:** All G8 gates passed, R2-H9 (KNN vs LightGBM) completed or deferred with ADR

### Gate Criteria (Phase 9 → Phase 10)

| Gate | Criterion |
|------|-----------|
| G9-1 | 30 live trading days completed |
| G9-2 | Sharpe ≥ 1.0 (live, annualized) |
| G9-3 | MaxDD ≤ 15% of current capital |
| G9-4 | Zero reconciliation failures in live |
| G9-5 | MTBF ≥ 336 hours (2 weeks) |
| G9-6 | Paper-to-live performance correlation ≥ 0.7 |
| G9-7 | SR 11-7 model risk documentation complete |
| G9-8 | Merkle audit trail operational |

### Capital Scaling Ladder

| Tier | Capital | Trigger | Drawdown Halt |
|------|---------|---------|---------------|
| T1 | $10,000 | Phase 8 gate pass | DD > 15% → halt, diagnose |
| T2 | $25,000 | 30 days at T1 + Sharpe ≥ 1.5 | DD > 12% → revert to T1 |
| T3 | $50,000 | 60 days at T2 + Sharpe ≥ 1.5 | DD > 10% → revert to T2 |
| T4 | $100,000+ | 90 days at T3 + external audit | DD > 8% → revert to T3 |

### Pre-Phase 9 Mandatory Tasks

**P9-PRE-1: IBKR Account Setup + API Configuration.** Open/verify IBKR account, enable TWS API, configure `ib_async` connection. Test with minimum lot orders. Task Type: JR.

**P9-PRE-2: R2-H9 KNN vs LightGBM Decision.** If this head-to-head comparison has completed during Phase 8, integrate the winner (or ensemble if both pass). If not completed, defer with ADR documenting the rationale. Task Type: SR.

**P9-PRE-3: Data Source Upgrade Evaluation.** Evaluate Norgate Data vs. Polygon.io vs. current Yahoo Finance. The Rigorous ML Trading Stack paper flags Yahoo Finance as "unacceptable for serious backtests." If survivorship bias is confirmed, migrate before live deployment. Task Type: SR.

**P9-PRE-4: SR 11-7 Model Risk Documentation.** Create the model card per Federal Reserve SR 11-7 guidance: assumptions, limitations, testing results, ongoing monitoring plan, model owner, model validator. Task Type: SR.

**P9-PRE-5: Merkle Audit Trail.** Implement tamper-evident hash chain of every order, HP config, model version, BSS gate decision. Task Type: MIX.

**P9-PRE-6: Multi-Vendor Price Feed Architecture.** Design (not implement) the multi-vendor BFT price feed: ≥3 vendors (Polygon, IEX, Alpaca), PBFT-style agreement. Implementation in Phase 10. Task Type: SR.

### Level 1 Phase Summary

- **Duration:** 4 weeks setup, then continuous
- **Critical task:** IBKR integration with `ib_async` (replacing MockBroker)
- **Key risk:** Slippage model mismatch between paper and live. MockBroker uses 10bps; real IBKR fills may differ.
- **Mitigation:** Start with 50% of Kelly fraction for first 10 trading days (half of half-Kelly = quarter-Kelly). Ramp to full half-Kelly after confirming fill quality.
- **Top 3 Risks:**
  1. IBKR API instability (weekly 2FA reauth, connection drops). Mitigation: automated reconnection, alert on disconnect.
  2. Regime change during initial capital deployment. Mitigation: drawdown halt at 15%.
  3. Tax/compliance surprises from high-frequency trading classification. Mitigation: consult CPA before deploying; document trade frequency.

### Iteration Instructions for Level 3

Produce handoff documents for P9-PRE-1 through P9-PRE-6. Pay special attention to `ib_async` API specifics (not `ib_insync`, which is deprecated). Include the IBKR paper trading account configuration for initial smoke testing.

---

## Phase 10 — NautilusTrader Evaluation

### Strategic Purpose

Build-vs-buy decision for production execution infrastructure. NautilusTrader is assessed as the "gold standard" by the research corpus. The evaluation determines whether migrating from the custom execution stack to NautilusTrader provides sufficient reliability, performance, and maintainability benefits to justify the migration cost.

**Duration:** 4–6 weeks  
**Target Start:** December 2026  
**Target Complete:** January 2027  
**Task type:** SR  
**Dependencies:** Phase 9 deployed and stable for ≥ 30 days

### Gate Criteria (Phase 10 → Phase 11)

| Gate | Criterion |
|------|-----------|
| G10-1 | NautilusTrader evaluation complete with scored decision matrix |
| G10-2 | If BUILD: custom stack hardened to MTBF ≥ 720 hours |
| G10-3 | If BUY: migration plan with rollback strategy documented |
| G10-4 | RIA-10 completed (research re-prioritization) |

### Evaluation Framework

Score each criterion 1–5, weight by importance:

| Criterion | Weight | Custom Stack Score | NautilusTrader Score |
|-----------|--------|-------------------|---------------------|
| IBKR integration quality | 25% | ? | ? |
| Backtest-to-live parity | 20% | ? | ? |
| Order management features | 15% | ? | ? |
| Python 3.12 compatibility | 15% | ? | ? |
| Community/maintenance trajectory | 10% | ? | ? |
| Migration effort (weeks) | 10% | 0 (baseline) | ? |
| Windows support | 5% | 5 (native) | ? |

**Decision rule:** If NautilusTrader scores > 20% higher weighted total AND migration effort ≤ 4 weeks → BUY. Otherwise → BUILD (harden custom stack).

### Top 3 Risks

1. **NautilusTrader may not support Windows well.** It's primarily developed on Linux/macOS. Mitigation: evaluate on WSL2 as fallback.
2. **Migration breaks the validated pipeline.** Mitigation: run parallel (custom + Nautilus) for 2 weeks before cutting over.
3. **NautilusTrader API changes.** Active development means API instability. Mitigation: pin version, test upgrade path.

---

## Phase 11 — Hyper-Scale (5200T+)

### Strategic Purpose

Expand from 585 tickers to 5,200+ tickers. This requires replacing HNSW with FAISS IVF for memory efficiency, implementing point-in-time universe construction (survivorship-bias-free), and potentially restructuring the index to support sector-aware retrieval.

**Duration:** 6–8 weeks  
**Target Start:** February 2027  
**Target Complete:** April 2027  
**Task type:** SR  
**Dependencies:** Phase 9 stable, Phase 10 decision made

### Gate Criteria

| Gate | Criterion |
|------|-----------|
| G11-1 | FAISS IVF index operational at 5200T with recall ≥ 0.99 |
| G11-2 | Point-in-time universe construction verified (no survivorship bias) |
| G11-3 | EOD pipeline completes within 30 minutes at 5200T |
| G11-4 | BSS does not degrade vs 585T baseline (pool dilution managed) |
| G11-5 | Memory usage ≤ 24GB at 5200T |

### Key Architecture Decisions

1. **HNSW → FAISS IVF.** At ~10M vectors (5200T × 2000 days), HNSW memory exceeds 32GB. FAISS IVFFlat with nprobe=128–256 provides good recall at lower memory. GPU acceleration optional.
2. **Tiered pool expansion.** Matchmaking-style cascade: 10-ticker sub-industry pool → sector pool → full universe. Directly addresses pool dilution at scale.
3. **Competence-model pruning.** Remove analogues from the HNSW/FAISS index that consistently produce bad predictions. Reduces index size and improves signal quality.
4. **Bloom filter deduplication.** O(1) membership check for candidate analogues and corruption-date blacklist.
5. **Intra-bar temporal features.** High_time_ratio, low_time_ratio from intraday Polygon.io data. Requires data infrastructure upgrade.

### Top 3 Risks

1. **Pool dilution worsens at 5200T.** Probability: 60%. More tickers = more irrelevant neighbors. Mitigation: tiered pool expansion (mandatory), competence pruning.
2. **Memory ceiling.** 32GB may be insufficient even with FAISS IVF. Mitigation: profile memory at 2000T, 3000T, 4000T before attempting 5200T. Consider 64GB upgrade.
3. **Point-in-time universe construction is hard.** Delisted tickers, name changes, spin-offs require clean data. Mitigation: Norgate Data subscription (recommended by multiple papers).

---

## Research Track R1 — Signal Quality (Parallel)

### Strategic Purpose

R1 contains the highest-value interventions for improving BSS. These run in parallel with Phases 8–9, shipping as feature-flagged enhancements. Every R1 item must pass the Braess gate (G7.5-2) before integration.

### Priority-Ranked Items

| Rank | Item | Expected BSS Impact | Confidence | Effort | Phase Window |
|------|------|---------------------|------------|--------|-------------|
| 1 | **Multi-retriever ensemble** (5 HNSW indices + RRF) | +5–15% relative BSS | 70% | 2 weeks | Phase 8 (weeks 3–8) |
| 2 | **Tiered pool expansion** (matchmaking cascade) | +3–10% relative BSS | 65% | 2 weeks | Phase 8–11 |
| 3 | **Per-ticker regime independence** (H-statistic per ticker + SPY hybrid) | +2–8% relative BSS | 55% | 1 week | Phase 8 (weeks 6–10) |
| 4 | **H-statistics regime detector** (Renko/Kagi construction, ~100 LOC) | +1–5% relative BSS | 50% | 2 days | Phase 7.5 or 8 |
| 5 | **Rough volatility feature** (multi-scale vol slope, 1 feature) | +1–3% relative BSS | 60% | 1 day | Phase 8 |
| 6 | **MLKR** (Metric Learning for KNN Regression) | +3–10% relative BSS | 45% | 2 weeks | Phase 8–9 |
| 7 | **Gravity graph features** (weighted correlation adjacency, Y3 Tier A) | +1–5% relative BSS | 40% | 1 week | Phase 8 |
| 8 | **Transfer entropy features** (directional causal information flow) | +1–5% relative BSS | 35% | 1 week | Phase 9 |
| 9 | **Optimal transport domain adaptation** (`ot.da` from POT library) | +2–8% relative BSS | 30% | 2 weeks | Phase 9 (research bet) |

### Gate Protocol for R1 Items

Every R1 item follows this protocol:
1. Implement behind feature flag (default OFF)
2. Run Braess gate: compute BSS with and without the enhancement on all 6 folds
3. If Braess gate PASSES (BSS improves or holds on ≥ 4/6 folds): flag ON, commit
4. If Braess gate FAILS: flag OFF, document results, move to next item
5. After each integration: re-run Murphy B3 to track REL/RES changes
6. After each integration: update CV BSS confidence intervals

---

## Research Track R2 — Architecture Validation (Parallel)

### Strategic Purpose

R2 contains architectural experiments that could fundamentally change the prediction engine. These are higher-risk, higher-effort items that run during Phases 8–9.

### Priority-Ranked Items

| Rank | Item | Description | Phase Window | Task Type |
|------|------|-------------|-------------|-----------|
| 1 | **R2-H9: KNN vs LightGBM** | Head-to-head comparison. Implement as Delta Model (residual learning): KNN predicts first, LightGBM predicts KNN's residual. Out-of-fold KNN predictions mandatory. If ensemble wins, it becomes the production architecture. | Phase 8 (weeks 4–10) | SR |
| 2 | **R2-E2: Conformal Prediction** | Dual-gating with BSS and adaptive coverage (AgACI). Replaces 3 arbitrary heuristic thresholds with one distribution-free mechanism. Mondrian partitions for per-regime validity. | Phase 8–9 | SR |
| 3 | **R2-BT: Behavior Tree Pipeline** | `py_trees` Sequence/Parallel/Fallback nodes for pipeline orchestration. Formalizes the multi-retriever + tiered pool + regime detection architecture. | Phase 8 (after R1-INT-1) | MIX |
| 4 | **R2-PBO: Probability of Backtest Overfitting** | CPCV-based PBO estimate. If PBO > 0.5, the entire backtest is suspect. Uses `skfolio.CombinatorialPurgedCV`. | Phase 8 | SR |

### R2-H9 Specification (Delta Model Pattern)

The Delta Model (per Gemini Finding 9) is the correct architecture for KNN + LightGBM:

1. KNN produces `p_knn(up)` as the base prediction
2. Compute residual: `r = y_actual − p_knn` (on training data only)
3. Train LightGBM on `[features, p_knn]` to predict `r`
4. Final: `p_ensemble = p_knn + lgbm.predict([features, p_knn])`
5. Calibrate the ensemble output (not individual components)

**Critical constraint:** Step 3 must use **out-of-fold** KNN predictions. If LightGBM trains on in-fold KNN predictions, it learns to mimic KNN's training-set overfitting. Use nested CV: inner fold for KNN, outer fold for LightGBM.

**Corrections to apply:**
- sklearn `multi_class='multinomial'` is deprecated since 1.5, removed in 1.7. Use `'auto'` or omit.
- When constructing the LightGBM target, use binary classification (up/down) with the residual as sample weight, not regression on the residual directly.

---

## RIA Protocol (Research Impact Assessment)

### When

Execute at every phase boundary: after Phase 7.5, after Phase 8 Month 1, after Phase 8 Month 3, after Phase 9 deployment, and after Phase 10 decision.

### How (30-Minute Exercise)

1. **Re-rank the top 10 remaining unimplemented research findings** by expected BSS lift given the *current* system state (not the state when the research was written).

2. **Apply the scoring formula:**
   ```
   priority = (expected_BSS_lift × confidence) / (engineering_hours × risk_factor)
   ```
   where:
   - `expected_BSS_lift` = estimated absolute BSS improvement (e.g., 0.005)
   - `confidence` = 0.0 to 1.0
   - `engineering_hours` = estimated implementation + validation time
   - `risk_factor` = 1.0 (low risk) to 3.0 (high risk of regression)

3. **If any finding's priority changes by ≥ 2 ranks**, update the implementation tier assignment in this master plan.

4. **Document in `docs/ria/RIA_PHASE_N.md`** with: date, current BSS, current Sharpe, ranked list, any re-assignments.

### Example RIA Entry

```
# RIA — Post Phase 7.5
Date: 2026-05-16
Current BSS: +0.0085 (post z-score normalization)
Current Sharpe: 2.7 (walk-forward)

| Rank | Item | Expected Lift | Conf | Hours | Risk | Priority |
|------|------|--------------|------|-------|------|----------|
| 1 | Multi-retriever ensemble | 0.010 | 0.70 | 40 | 1.5 | 0.000117 |
| 2 | Tiered pool expansion | 0.008 | 0.65 | 40 | 1.5 | 0.000087 |
| ...  | ... | ... | ... | ... | ... | ... |

Changes: None (rankings stable).
```

---

## Risk Register

| ID | Risk | Probability | Impact | Phase | Mitigation | Owner |
|----|------|-------------|--------|-------|------------|-------|
| R1 | BSS edge is noise, not signal | 35% | CRITICAL | 7.5 | CV BSS estimator (G7.5-5) | Sleep |
| R2 | Z-score normalization degrades BSS | 25% | HIGH | 7.5 | Group-wise normalization fallback; feature flag revert | Sleep |
| R3 | MI ceiling near zero | 15% | CRITICAL | 7.5 | Activate R1 feature engineering immediately | Sleep |
| R4 | Pipeline unreliable at 585T | 25% | HIGH | 8 | Value-stream map; incremental index update | Sleep |
| R5 | Yahoo Finance data poisoning | 20% | MEDIUM | 8–9 | Norgate evaluation in P9-PRE-3 | Sleep |
| R6 | Slippage model mismatch (paper→live) | 30% | MEDIUM | 9 | Quarter-Kelly for first 10 days; monitor fill quality | Sleep |
| R7 | IBKR API instability | 20% | MEDIUM | 9 | Automated reconnection; alert on disconnect | Sleep |
| R8 | Pool dilution worsens at 5200T | 60% | HIGH | 11 | Tiered pool expansion (R1 rank 2); competence pruning | Sleep |
| R9 | Memory ceiling at 5200T | 40% | MEDIUM | 11 | Staged profiling at 2K/3K/4K; 64GB upgrade | Sleep |
| R10 | Regime change during capital deployment | 30% | HIGH | 9 | Drawdown halt; capital scaling ladder | Sleep |

---

## Timeline Summary

```
2026
Apr 18 ─── NOW
Apr 21 ─── Phase 7.5 START
           ├─ Week 1: T7.5-1 (z-score), T7.5-3, T7.5-4, T7.5-5 (parallel)
           ├─ Week 2: T7.5-2 (Braess gate), T7.5-6 (Murphy B3 post-norm)
           ├─ Week 3: T7.5-7 (MI ceiling), T7.5-8 (multi-horizon, compute-heavy)
May 12 ─── Phase 7.5 GATES + RIA-7.5
May 14 ─── P8-PRE (585T validation, scoringrules, Sortino Kelly) — 5 days
May 19 ─── Phase 8 START (paper trading)
           ├─ Week 1–2: T8.1–T8.7 (pipeline setup, FMEA)
           ├─ Week 2: R1-INT-2 (value-stream map), Week 2 checkpoint
           ├─ Weeks 3–8: R1-INT-1 (multi-retriever ensemble)
           ├─ Week 4: R1-INT-3 (defense-in-depth audit), Month 1 checkpoint
           ├─ Weeks 4–10: R2-H9 (KNN vs LightGBM)
           ├─ Week 8: Month 2 checkpoint
           ├─ Weeks 6–10: R1 rank 3 (per-ticker regime)
           ├─ Week 12: Month 3 checkpoint + G8 gates
Aug 11 ─── Phase 8 GATES + RIA-8
           ├─ 1 week buffer for gate fixes
Aug 18 ─── Phase 9 SETUP (4 weeks)
           ├─ IBKR setup, Merkle trail, SR 11-7 docs
           ├─ P9-PRE-1 through P9-PRE-6
Sep 15 ─── Phase 9 LIVE ($10k deployed)
           ├─ 30-day T1 validation
Oct 15 ─── T1 gate check → T2 ($25k) if Sharpe ≥ 1.5
Dec 15 ─── T2 gate check → T3 ($50k) if Sharpe ≥ 1.5
           ├─ Phase 10 (NautilusTrader evaluation) runs in parallel

2027
Jan 2027 ── TARGET: $10k+ live (T1 or T2 achieved)
Feb 2027 ── Phase 11 START (hyper-scale)
Apr 2027 ── Phase 11 COMPLETE → 5200T+ operational
```

**Critical path:** Phase 7.5 (3 weeks) → P8-PRE (1 week) → Phase 8 (12 weeks) → Phase 9 setup (4 weeks) = 20 weeks from today. That places initial live deployment at **September 15, 2026** — well ahead of the January 2027 target.

**Schedule risk:** The 12-week Phase 8 trading window is the bottleneck and cannot be compressed. If Phase 7.5 slips by 2 weeks, Phase 9 deployment slips to October, which is still ahead of target.

---

## Dependency Graph (Full)

```
Phases 1–7 (COMPLETE) ─────────────────────────────────────────────┐
P8-PRE-4/5/6, R3 (COMPLETE) ──────────────────────────────────────┤
                                                                   ▼
                                                          Phase 7.5 ──────────┐
                                                           (3 weeks)          │
                                                               │              │
                                                               ▼              │
                                                          P8-PRE-1..4 ────────┤
                                                           (1 week)           │
                                                               │              │
                                                               ▼              │
                                                          Phase 8 ────────────┤
                                                           (12 weeks)         │
                                                               │              │
                                    R1 (parallel) ────────────►│              │
                                    R2-H9 (parallel) ─────────►│              │
                                                               │              │
                                                               ▼              │
                                                          Phase 9 ────────────┤
                                                           (4w + ongoing)     │
                                                               │              │
                                                    ┌──────────┤              │
                                                    ▼          ▼              │
                                              Phase 10    Capital Ladder      │
                                              (4–6 weeks)  (T1→T2→T3→T4)    │
                                                    │                         │
                                                    ▼                         │
                                              Phase 11                        │
                                              (6–8 weeks)                     │
                                                                              │
                                    RIA checkpoints at every ─────────────────┘
                                    phase boundary (ongoing)
```

### Corrections Applied from Research Corpus

This plan incorporates the following corrections identified in Synthesis #2 and the Addendum:

| Correction | Old | New | Applied In |
|-----------|-----|-----|-----------|
| HMM look-ahead | `hmmlearn.predict_proba()` | `statsmodels.MarkovRegression` with `filtered_marginal_probabilities` | G7.5-4 |
| sklearn deprecation | `multi_class='multinomial'` | `'auto'` or omit | R2-H9 spec |
| EOQ rebalancing exponent | √λ | λ^(1/3) (cube-root) | Phase 11 rebalancing |
| Kelly formula | `f* = mu/sigma²` | `f* = mu/sigma_downside²` (Sortino-adjusted) | P8-PRE-4 |
| `scoringrules` status | "should-have" | "must-have" (double-confirmed) | P8-PRE-2 |
| KNN+LightGBM architecture | Simple averaging | Delta Model (residual learning) | R2-H9 spec |

---

*End of FPPE Master Implementation Plan v4.0. This document is the Level 1–2 deliverable. Level 3 handoff documents are produced in subsequent sessions using the iteration instructions embedded in each phase section.*
