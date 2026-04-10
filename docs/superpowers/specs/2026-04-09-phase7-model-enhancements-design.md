# Phase 7 — Model Enhancements Design Spec
# Date: 2026-04-09
# Status: Approved by user

---

## Overview

Phase 7 adds six optional model enhancements to the FPPE system. Phases 1–6 produce a
deployable baseline; Phase 7 makes it better. If every enhancement fails its gate, the
Phase 1–6 baseline is deployed unchanged. No enhancement is mandatory.

Authoritative source: `HANDOFF_Phase7_Model_Enhancements.md` (supersedes fppe-roadmap-v2A.md
Phase 7 section). This design doc records the approved implementation approach.

---

## §1 — Pre-Flight (before T7.0)

Two corrections to `pattern_engine/config.py` required before any experiment runs.
Permission granted by user to edit locked settings.

### 1.1 Default value corrections

| Field | Old value | New (locked) value | Provenance |
|-------|-----------|--------------------|------------|
| `feature_set` | `"returns_only"` | `"returns_candle"` | Phase 6 Task 6.2 |
| `max_distance` | `1.1019` | `2.5` | Phase 6 Task 6.1 |
| `calibration_method` | `"platt"` | `"beta_abm"` | Phase 1 H5–H6 |

### 1.2 New Phase 7 flags (all default False)

```python
# Enhancement flags — all False in production until gate is passed
use_bma: bool = False
use_owa: bool = False
use_dtw_reranker: bool = False
use_conformal: bool = False
use_anomaly_filter: bool = False
use_stumpy: bool = False

# Enhancement parameters
owa_alpha: float = 1.0
dtw_rerank_k: int = 20
conformal_alpha: float = 0.10
conformal_gamma: float = 0.05
anomaly_contamination: float = 0.05
anomaly_penalty: float = 0.50
stumpy_weight: float = 0.20
stumpy_subsequence_length: int = 50
```

---

## §2 — Locked Configuration (entering Phase 7)

```
Distance          = Euclidean
Weighting         = uniform
Features          = returns_candle (23D: 8 VOL_NORM + 15 candlestick proportions)
Calibration       = beta_abm
cal_frac          = 0.76
max_distance      = 2.5
top_k             = 50
confidence_thresh = 0.65
regime            = hold_spy_threshold+0.05
horizon           = fwd_7d_up
stop_loss_atr     = 3.0
```

Key metrics entering Phase 7:
- BSS mean (52T, 6 folds): +0.00033 (results/bss_fix_sweep_h7.tsv)
- n_scored (52T): 0 — all probs in [0.50, 0.58], below 0.65 threshold
- 585T Platt signals: 159 BUY, conf [0.65–0.75] (results/cached_signals_2024.csv)
- Tests passing: 846

---

## §3 — Universal Protocol

Applies to all enhancements E1–E6 without exception.

### 3.1 TDD requirement
Write failing tests BEFORE implementing each enhancement.
Test files: `tests/unit/test_phase7_eN_<name>.py`

### 3.2 Feature flags
Every enhancement behind its config flag. Default False. Production is unaffected.

### 3.3 Walk-forward protocol (BSS-gated enhancements)
- 6-fold expanding window (WALKFORWARD_FOLDS from config.py)
- beta_abm injected via monkey-patch pattern (swap `_matcher_module._PlattCalibrator`,
  restore in `finally`)
- H7 HOLD regime applied (SPY ret_90d < +0.05 → base_rate prob)
- Template: `scripts/phase6_bss_comparison.py` (Config A pattern)
- Baseline loaded from TSV — do NOT re-run T7.0 for each enhancement

### 3.4 Gate evaluation
- PASS: flag=True in config.py, CLAUDE.md updated with provenance, becomes cumulative
  baseline for subsequent enhancements
- FAIL: flag=False, one-line CLAUDE.md comment, proceed to next enhancement

### 3.5 Cumulative baseline
Each subsequent enhancement compares against T7.0 baseline PLUS all prior passing
enhancements. Enhancements stack — passing E1 makes E2's job harder (correct behavior).

### 3.6 Per-enhancement tracking
Append one row to `results/phase7/enhancement_summary.tsv` after each enhancement.
Columns: `enhancement, flag_value, baseline_bss_per_fold, enhanced_bss_per_fold,
delta_per_fold, folds_improved, gate_metric, gate_threshold, gate_result,
provenance_file, runtime_seconds`

### 3.7 Test suite
Run `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"` before and after
every enhancement touching `pattern_engine/`.

### 3.8 Three-strike rule
Three consecutive implementation failures on one enhancement → STOP, flag=False,
log in session log, proceed to next.

---

## §4 — T7.0: Baseline Re-establishment (MANDATORY FIRST)

**Files:**
- `scripts/phase7_baseline.py` — single-config 6-fold walk-forward
- `results/phase7/baseline_23d.tsv` — THE comparison target for all Phase 7 deltas

**Pattern:** Clone Config A logic from `scripts/phase6_bss_comparison.py`. Remove
Config B entirely. Single config: returns_candle(23D), max_distance=2.5, beta_abm,
regime_filter=True.

**Output columns:** `fold, bss, n_scored, n_total, base_rate, mean_prob, reliability,
resolution, uncertainty` (Murphy decomposition — diagnoses WHERE improvements originate)

**Success criteria:** 6 rows, no NaN/Inf, BSS values plausible (~+0.00033 mean).

---

## §5 — E1: BMA Calibrator (T7.1)

**Goal:** Replace beta_abm with BMA (Student's t mixture EM) and measure BSS improvement.

**Gate:** BSS improvement ≥ +0.001 on ≥ 3/6 folds vs T7.0 baseline.

**Critical compatibility issue:** `BMACalibrator.fit()` takes `raw_probs: (N, K)` —
K per-analogue probabilities per training sample. The existing beta_abm/Platt
calibrators take 1D aggregated probabilities. Wiring BMA requires extracting
per-analogue probs at Stage 3 (filter) → Stage 4 (aggregate) boundary in the
5-stage pipeline (scale → search → filter → aggregate → calibrate), specifically
from the scored-neighbour list after distance filtering but before the uniform
weighted mean collapses them to a scalar. The monkey-patch approach used for beta_abm
cannot be reused — BMA integration requires a new code path in `_aggregate()` that
preserves the per-neighbour probability vector for BMA's EM. This is the primary
implementation risk for E1.

**Files:**
- `research/bma_calibrator.py` — EXISTS, audit for compatibility first
- `scripts/phase7_e1_bma_calibrator.py` — 6-fold walk-forward
- `tests/unit/test_phase7_e1_bma.py` — TDD tests
- `results/phase7/e1_bma_vs_beta_abm.tsv` — output
- Modify: `pattern_engine/config.py`, `pattern_engine/matcher.py`

**Task type:** SR (statistical depth — EM convergence, posterior interpretation)

---

## §6 — E2: OWA Feature Weighting (T7.2)

**Goal:** Concentrate KNN distance computation on high-MI features via OWA.

**Gate:** BSS ≥ +0.001 on ≥ 3/6 folds vs cumulative baseline. Additional: worst-fold
BSS must not degrade by more than -0.0005 vs baseline worst-fold.

**Design decisions:**
- GLOBAL weights (not per-regime) — regime filter already handles bear/bull split
- MI computed per fold from training data only (no leakage)
- Alpha sweep: [0.5, 1.0, 2.0, 4.0] via leave-one-fold-out CV
- AvgK sanity check post-weighting: if AvgK < 20 on any fold, mini-sweep
  max_distance in [2.5, 3.0, 3.5, 4.0]

**OWA formula:**
```python
w_i = ((n_features - rank_i + 1) / n_features) ** alpha
# normalize: weights sum to n_features (mean = 1.0)
```

**Files:**
- `pattern_engine/owa_weights.py` — NEW
- `scripts/phase7_e2_owa_weighting.py` — 6-fold walk-forward
- `tests/unit/test_phase7_e2_owa.py` — TDD tests
- `results/phase7/e2_owa_vs_baseline.tsv` — output
- Modify: `pattern_engine/config.py`, `pattern_engine/matcher.py`

**Task type:** SR (feature selection, CV design, information theory)

---

## §7 — E3: DTW Reranker (T7.3)

**Goal:** Rerank top-50 Euclidean neighbors by DTW on return columns only.

**Gate:** BSS ≥ +0.001 on ≥ 3/6 folds vs cumulative baseline. Latency < 60s added
to full walk-forward runtime.

**Design decisions:**
- DTW on 8 return columns ONLY (indices 0:8). Candlestick columns excluded — ratios
  have no meaningful temporal warping.
- `research/wfa_reranker.py` does NOT exist — must be written from scratch
- Library: `dtw-python` (install via pip if missing)
- Fast-fail: check Spearman ρ (Euclidean vs DTW ordering) on 100 random queries from
  fold 1. If ρ > 0.95 → FAIL immediately without full walk-forward.
- Also produces Competitive Benchmark B5 (DTW without cal → DTW+beta_abm → DTW+beta_abm+regime)

**Files:**
- `research/wfa_reranker.py` — NEW (from scratch)
- `scripts/phase7_e3_dtw_reranker.py` — 6-fold walk-forward
- `scripts/benchmarks/b5_dtw_vs_euclidean.py` — Benchmark B5
- `tests/unit/test_phase7_e3_dtw.py` — TDD tests
- `results/phase7/e3_dtw_vs_baseline.tsv` — output
- `results/benchmarks/b5_dtw_vs_euclidean.tsv` — B5 output
- Modify: `pattern_engine/config.py`, `pattern_engine/matcher.py`

**Task type:** SR (DTW implementation, Sakoe-Chiba constraints, library selection)

---

## §8 — E4: Conformal Prediction Intervals (T7.4)

**Goal:** Add statistically guaranteed prediction intervals. Infrastructure for Phase 8
Kelly sizing, not a BSS improvement.

**Gate:** Empirical coverage ≥ 88% at nominal 90% level across ALL 6 folds.
Mean interval width < 0.30.

**Design decisions:**
- Adaptive Conformal Inference (ACI — Gibbs & Candès, NeurIPS 2021) to handle
  time-series non-exchangeability
- `conformal_hooks.py` exists as a well-documented stub — promote `NaiveConformalCalibrator`
  to full `AdaptiveConformalPredictor`
- Gamma sweep: [0.01, 0.05, 0.10]
- Phase 8 integration point (interval width → position sizing) documented but NOT
  implemented in this phase

**ACI online update:**
```python
self.alpha_t = self.alpha_t + self.gamma * (err - self.nominal_alpha)
self.alpha_t = np.clip(self.alpha_t, 0.01, 0.50)
```

**Files:**
- `scripts/phase7_e4_conformal.py` — 6-fold coverage evaluation
- `tests/unit/test_phase7_e4_conformal.py` — TDD tests
- `results/phase7/e4_conformal_coverage.tsv` — output
- Modify: `pattern_engine/conformal_hooks.py`, `pattern_engine/config.py`

**Task type:** SR (conformal theory, ACI, coverage guarantees)

---

## §9 — E5: CPOD/LOF Anomaly Filter (T7.5)

**Goal:** Detect outlier query patterns and suppress their confidence to reduce FPR
on production signals.

**Gate:** FPR reduction ≥ 5%, TPR loss ≤ 2%, anomaly flag rate < 30%.

**Critical design note:**
- 52T pipeline produces ZERO signals (probs [0.50, 0.58], below 0.65 threshold)
- E5 is evaluated on 585T Platt signals (`results/cached_signals_2024.csv`, 159 BUY)
- NOT a 6-fold BSS walk-forward — FPR/TPR metric only

**Implementation:** `sklearn.LocalOutlierFactor` with `novelty=True` (not CPOD,
which has no maintained Python package).

**Parameter sweep:** 4×4 grid:
- contamination: [0.01, 0.05, 0.10, 0.20]
- penalty: [0.00, 0.25, 0.50, 0.75]

**Files:**
- `pattern_engine/anomaly_filter.py` — NEW (LOF-based SignalFilterBase subclass)
- `scripts/phase7_e5_cpod_filter.py` — FPR/TPR evaluation
- `tests/unit/test_phase7_e5_cpod.py` — TDD tests
- `results/phase7/e5_cpod_fpr_tpr.tsv` — output
- Modify: `pattern_engine/signal_pipeline.py`, `pattern_engine/config.py`

**Cumulative baseline note:** If E5 passes, its anomaly filter flag (`use_anomaly_filter=True`)
applies ONLY to the 585T production evaluation path. The 52T walk-forward used for E6's
BSS comparison produces zero signals — the LOF filter has nothing to suppress at 52T scale.
E6's BSS baseline therefore does NOT include E5's effect, even if E5 passes. The
enhancement_summary.tsv should reflect this: E6's baseline_bss_per_fold column loads from
T7.0 + passing flags among E1–E3 only (not E5).

**Task type:** MIX (LOF is JR; FPR/TPR analysis is SR)

---

## §10 — E6: STUMPY Matrix Profile (T7.6)

**Goal:** Cross-ticker pattern matching via STUMPY AB-join as secondary signal.

**Gate:** BSS ≥ +0.001 on ≥ 3/6 folds vs cumulative baseline. KNN-STUMPY Pearson
correlation < 0.50. Compute ≤ 30s/fold.

**Primary risk:** Compute budget. At 52T with same-sector filtering (~5 candidate
tickers per query), naive implementation exceeds 30s/fold by orders of magnitude.
Mitigation: pre-compute matrix profiles for all training pairs once per fold.
If fold 1 > 60s after optimization → FAIL immediately.

**Signal blend:** `p_combined = (1 - stumpy_weight) * p_knn + stumpy_weight * p_stumpy`
Applied AFTER KNN calibration (STUMPY signal is uncalibrated raw fraction).

**Also produces:** Competitive Benchmark B6 (KNN-only vs STUMPY-only vs blended)

**Files:**
- `research/stumpy_matcher.py` — NEW
- `scripts/phase7_e6_stumpy.py` — 6-fold walk-forward
- `scripts/benchmarks/b6_stumpy_vs_knn.py` — Benchmark B6
- `tests/unit/test_phase7_e6_stumpy.py` — TDD tests
- `results/phase7/e6_stumpy_vs_baseline.tsv` — output
- `results/benchmarks/b6_stumpy_vs_knn.tsv` — B6 output
- Modify: `pattern_engine/config.py`

**Task type:** SR (matrix profiles, AB-join interpretation, z-normalization)

---

## §11 — Phase 7 Completion Gate

```
CHECK 1: Each enhancement has a row in results/phase7/enhancement_summary.tsv
CHECK 2: Each row has gate_result = PASS or FAIL
CHECK 3: Provenance file exists for each enhancement
CHECK 4: CLAUDE.md updated with final locked settings (all passing flags = True)
CHECK 5: Full test suite passing (~900+ tests expected)
CHECK 6: Final cumulative BSS computed on locked config with all passing enhancements
ALL YES → Phase 7 COMPLETE. Proceed to Phase 8 (Paper Trading).
```

---

## §12 — Risk Register

| # | Risk | Probability | Mitigation |
|---|------|------------|------------|
| R1 | All 6 enhancements fail | 30% | Deploy P1–P6 baseline. Acceptable. |
| R2 | BMA EM fails to converge | 40% | Cap iterations at 500. beta_abm fallback. |
| R3 | OWA overfits on 6-fold CV | 50% | Conservative alpha. 4/5 inner folds positive. |
| R4 | DTW ≈ Euclidean at 8 points (ρ > 0.95) | 60% | Spearman fast-fail before full WF. |
| R5 | OWA changes AvgK below 20 | 30% | AvgK check in E2 script. Mini-sweep. |
| R6 | STUMPY exceeds compute budget | 50% | Fold 1 timing check before full run. |
| R7 | Conformal intervals trivially wide at 52T | 70% | Document. Becomes useful at 585T/1500T. |
| R8 | LOF FPR/TPR estimates noisy at n=159 | 30% | Report confidence intervals. |

---

## §13 — Execution Order

```
Pre-flight   Fix config.py defaults + add Phase 7 flags
T7.0         Baseline → results/phase7/baseline_23d.tsv
T7.1  E1     BMA Calibrator → results/phase7/e1_bma_vs_beta_abm.tsv
T7.2  E2     OWA Feature Weighting → results/phase7/e2_owa_vs_baseline.tsv
T7.3  E3     DTW Reranker → results/phase7/e3_dtw_vs_baseline.tsv
             → results/benchmarks/b5_dtw_vs_euclidean.tsv
T7.4  E4     Conformal Prediction → results/phase7/e4_conformal_coverage.tsv
T7.5  E5     CPOD/LOF Filter → results/phase7/e5_cpod_fpr_tpr.tsv
T7.6  E6     STUMPY Matrix Profile → results/phase7/e6_stumpy_vs_baseline.tsv
             → results/benchmarks/b6_stumpy_vs_knn.tsv
Final        results/phase7/enhancement_summary.tsv
```
