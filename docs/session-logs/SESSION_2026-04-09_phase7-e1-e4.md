# Session Log — Phase 7 E1–E4 (2026-04-09)

## Branch
`phase7-enhancements` (worktree `.worktrees/phase7-enhancements`)

## Objective
Implement 6 optional model enhancements (E1–E6) behind feature flags, each evaluated by a 6-fold walk-forward gate. Gate: BSS delta ≥ +0.001 on ≥3/6 folds (E1–E3), or coverage ≥88% all folds (E4), or FPR reduction ≥5pp (E5), or BSS + correlation + latency (E6).

## Work Completed This Session

### Task 0: Pre-flight (DONE)
- Fixed 3 stale defaults in `pattern_engine/config.py`: `feature_set="returns_candle"`, `max_distance=2.5`, `calibration_method="beta_abm"`
- Added 6 Phase 7 boolean flags (all False) and 8 parameters
- Commit: `14716ef`

### Task 1: T7.0 Baseline (DONE)
- Created `scripts/phase7_baseline.py` with `run_fold_with_config(fold, cfg_overrides=None)`
- Key design: `scored_mask = ~bear_mask` (all non-bear rows, not just BUY signals — 52T never fires BUY at 52T scale)
- Results: `results/phase7/baseline_23d.tsv`
  - Mean BSS = −0.02356 (2/6 positive folds)
  - 2022-Bear fold only 416 scored rows (H7 HOLD quarantines most of 2022)
- Commits: `0e51ecf`, `27431cf` (bug fix: n_scored=0 initially, fixed to use non-bear rows)

### Task 2: E1 BMA Calibrator (DONE, FAIL)
- Created `research/bma_calibrator.py` already existed; wrote 7 TDD tests
- Wired `_extract_neighbour_labels_for_bma()` + `_bma_calibrator` slot in `matcher.py`
- Result: 0/6 folds improved. Deltas −0.09 to −0.14
- Root cause: Binary 0/1 labels interact poorly with Student's t location model in EM
- `use_bma` stays False. Commit: `4d61392`

### Task 3: E2 OWA Feature Weighting (DONE, FAIL)
- Created `pattern_engine/owa_weights.py` (MI ranking + OWA weight computation)
- 10 TDD tests. Alpha CV sweep: 4 alphas × 6 folds = 24 evaluations; selected alpha=4.0
- Result: 0/6 folds improved. Best delta +0.00025 (2022-Bear). Gate required +0.001.
- Root cause: 23D feature space may not have strong MI signal differentiation; uniform KNN already implicitly captures structure
- `use_owa` stays False. Commit: `cf21600`

### Task 4: E3 DTW Reranker (DONE, FAIL — Spearman fast-fail)
- Created `research/wfa_reranker.py` with `dtw_rerank()` function
- 5 TDD tests. Fast-fail check ran on fold 2019 (100 queries).
- Result: Mean Spearman ρ = 1.0000 — DTW on equal-length 8-point sequences is mathematically identical to Euclidean distance (identity warping path, no temporal misalignment to exploit).
- Benchmark B5: KNN+beta_abm BSS=−0.00041, KNN+beta_abm+DTW BSS=−0.00044 (slight degradation)
- `use_dtw_reranker` stays False. Commit: `f231d31`

### Task 5: E4 Adaptive Conformal Prediction (DONE, FAIL)
- Appended `AdaptiveConformalPredictor` (ACI — Gibbs & Candès 2021) to `conformal_hooks.py`
- 7 TDD tests. Sweep: gamma=[0.01, 0.05, 0.10] per fold.
- Results: `results/phase7/e4_conformal_coverage.tsv`
  | Fold | Coverage | Width |
  |------|----------|-------|
  | 2019 | 0.986 | 1.000 |
  | 2020-COVID | 0.000 | 1.000 |
  | 2021 | 0.990 | 1.000 |
  | 2022-Bear | 0.933 | 0.999 |
  | 2023 | 0.989 | 1.000 |
  | 2024-Val | 0.989 | 1.000 |
- Root cause: 52T probs cluster in [0.50, 0.59]; |prob−label| nonconformity scores ≥ 0.41; threshold ~0.57 → near-trivial [0,1] intervals. Width gate (< 0.30) structurally impossible at 52T scale. 2020-COVID fold 0% coverage (ACI over-tightens on high-volatility regime).
- `use_conformal` stays False. Commit: `01d89d4`

## Current State
- Tests: 858 passing (851 before E4 + 7 new)
- All 6 feature flags remain False
- Latest commit: `01d89d4`

## Pending Tasks
- **E5 LOF Anomaly Filter** (`scripts/phase7_e5_cpod_filter.py`)
  - Evaluated on 585T signals (`results/cached_signals_2024.csv`, 159 BUY)
  - Gate: FPR reduction ≥5pp AND TPR loss ≤2pp AND anomaly rate <30%
  - Training data: likely `data/52t_features/` (52T) since 585T may not be in `data/`
  - File to create: `pattern_engine/anomaly_filter.py`, `tests/unit/test_phase7_e5_cpod.py`
  - Note: E5 does NOT use 6-fold BSS — it uses the cached 585T production signals

- **E6 STUMPY Matrix Profile** (`scripts/phase7_e6_stumpy.py`)
  - Install: `py -3.12 -m pip install stumpy`
  - Mini-benchmark on fold 1 FIRST — if fold 1 > 60s, FAIL immediately
  - File to create: `research/stumpy_matcher.py`, `scripts/benchmarks/b6_stumpy_vs_knn.py`
  - Gate: BSS delta ≥+0.001 on ≥3/6 folds + Pearson |corr(KNN, STUMPY)| < 0.50 + latency

- **Task 8: Phase 7 Completion Gate**
  - Verify `enhancement_summary.tsv` has 6 rows (E1–E6)
  - Compute cumulative BSS with all PASSING flags enabled
  - Merge worktree back to main
  - Update `CLAUDE.md` with final Phase 7 summary

## Structural Finding (Important for E5/E6)

The consistent FAIL pattern for E1–E4 has a single root cause: **the 52T KNN pipeline produces probabilities only in [0.50, 0.59]**, which is below the 0.65 confidence threshold and structurally constrains any calibration improvement. This is documented in CLAUDE.md: "52T beta_abm probability range: [0.50, 0.58] — below 0.65 threshold."

E5 (evaluated on 585T production signals) and E6 (STUMPY matrix profile) operate on a different signal path and may not have this constraint.

## Key Files

| File | Purpose |
|------|---------|
| `scripts/phase7_baseline.py` | T7.0 baseline; imports `run_fold_with_config` used by all E scripts |
| `results/phase7/baseline_23d.tsv` | DO NOT RE-RUN — load this for all comparisons |
| `results/phase7/enhancement_summary.tsv` | 4 rows so far: E1, E2, E3, E4 all FAIL |
| `pattern_engine/owa_weights.py` | OWA weight computation (flag=False but code present) |
| `research/wfa_reranker.py` | DTW reranker (flag=False but code present) |
| `pattern_engine/conformal_hooks.py` | AdaptiveConformalPredictor appended (flag=False) |
| `pattern_engine/matcher.py` | E1 BMA + E2 OWA + E3 DTW wired behind flags |
