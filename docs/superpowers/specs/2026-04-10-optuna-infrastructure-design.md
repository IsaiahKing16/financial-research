# P3 Optuna Infrastructure — Design Spec

**Created:** 2026-04-10
**Status:** APPROVED — ready for implementation
**Scope:** General-purpose Bayesian + Grid sweep framework, replacing all manual sweep infrastructure
**Approach:** Full rewrite (Approach C) — port worktree prototype ideas into production architecture

---

## Context

All FPPE hyperparameter sweeps (Phase 1 H1–H7, Phase 6 max_distance, Phase 7 E2 OWA alpha) use manual grid search with hardcoded parameter lists. The research paper (f4e7ad04) recommends Optuna with TPE sampling, reducing tuning from weeks to hours. A worktree prototype (`admiring-lichterman/pattern_engine/sweep.py`) exists but has stale assumptions (8D features, no H7 HOLD, missing beta_abm). This spec designs production infrastructure that incorporates the prototype's best ideas while supporting current locked settings and future experiments (H9 LightGBM, H8 HMM).

### Design Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | General-purpose framework | Reusable for KNN, LightGBM (H9), HMM (H8), any future model |
| Pruning | No pruning | 6 folds at ~2 min/trial — pruning saves little, risks discarding good configs on noisy early folds |
| Objective | Trimmed mean BSS (Optuna target) + positive-folds gate (TPE steering only) + Wilcoxon p-value (significance) | Gate steers TPE away from bad configs; Wilcoxon provides proper statistical test. Gate alone has 65.6% FPR under null. |
| Trial budget | 80 trials / 16 hours default | Research paper upper recommendation; thorough exploration |
| Location | `pattern_engine/` modules | Production infrastructure, not scripts |

---

## Architecture

Three new production modules:

```
pattern_engine/
├── walkforward.py    — Fold runner + data loading (extracted from phase7_baseline.py)
├── sweep.py          — OptunaSweep (Bayesian) + GridSweep + search space definitions
└── experiment_log.py — TSV provenance logger with trial metadata
```

### Data Flow

```
Caller (script or CLI)
  │
  ├─ load_and_augment_db()          ← walkforward.py (done ONCE)
  │     returns full_db DataFrame
  │
  └─ OptunaSweep(objective_fn, search_space, ...)
        │
        ├─ Creates Optuna study (TPE sampler, direction="maximize", SQLite persistence)
        ├─ For each trial (up to 80):
        │     ├─ Sample config from search space
        │     ├─ Call objective_fn(trial_config, full_db)
        │     │     └─ objective_fn runs 6 folds via run_fold()
        │     ├─ Store fold-level BSS + Murphy decomposition as trial attrs
        │     └─ Apply gate constraint (positive_folds >= 3)
        │
        ├─ ExperimentLogger writes per-trial TSV rows incrementally
        └─ Returns: best config, results DataFrame, study object
```

### Key Contracts

- **Objective function signature:** `(config: dict, full_db: DataFrame) -> dict` returning `{"mean_bss": float, "trimmed_mean_bss": float, "fold_results": list[dict], "positive_folds": int}`
- **Optimization target:** `trimmed_mean_bss` (drops worst fold) is the value returned to Optuna. `mean_bss` is logged for provenance but not used for TPE steering.
- Algorithm-agnostic — KNN, LightGBM, HMM, or any future model provides its own objective function
- **Search space:** dict mapping param names to Optuna-style ranges `{"max_distance": (0.8, 3.0), "top_k": (20, 100)}`

---

## Module 1: `pattern_engine/walkforward.py`

Extracts and consolidates proven logic from `scripts/phase7_baseline.py`.

### Public Functions

**`load_and_augment_db(data_dir: str = "data/52t_features") -> pd.DataFrame`**
- Loads `train_db.parquet` + `val_db.parquet` from data_dir, concatenates with `ignore_index=True`
- Converts `Date` column to datetime
- Augments candlestick features via `_augment_with_candlestick()` (5 proportions x 3 timeframes = 15 columns)
- Imputes NaNs with neutral values
- Returns full database ready for fold splitting
- Called ONCE per sweep, not per trial

**`run_fold(fold: dict, full_db: pd.DataFrame, feature_cols: list[str], cfg_overrides: dict | None = None) -> dict`**
- Identical contract to current `run_fold_with_config()`
- `cfg_overrides` dict maps to `EngineConfig` fields + extra params. Non-EngineConfig keys (e.g., `cal_frac`) are popped from the dict before calling `_build_cfg()`, then injected via `setattr(cfg, key, value)` after config construction. This allows `matcher.py` to access them via `getattr(cfg, 'cal_frac', 0.76)`. Stripping mechanism:
  ```python
  _NON_CONFIG_KEYS = {"cal_frac"}  # keys consumed by matcher via getattr(), not EngineConfig fields
  extra = {k: overrides.pop(k) for k in _NON_CONFIG_KEYS if k in overrides}
  cfg = _build_cfg(overrides)
  for k, v in extra.items():
      setattr(cfg, k, v)
  ```
- Splits train/val, applies beta_abm monkey-patch (`_PlattCalibrator → _BetaCalibrator`), fits PatternMatcher, queries, restores original calibrator, applies H7 HOLD regime, computes BSS + Murphy decomposition
- **Beta_abm monkey-patch:** Temporarily replaces `matcher._PlattCalibrator` with `_BetaCalibrator` (`betacal.BetaCalibration` with `parameters='abm'`) during fit+query, restoring original in a `try/finally` block. This pattern is extracted from `scripts/phase7_baseline.py:339-349`. **Note:** Single-threaded trials only — monkey-patch is not thread-safe. Do not use Optuna's `n_jobs > 1`.
- Returns:
  ```python
  {
      "fold": str,           # e.g., "2019"
      "bss": float,          # Brier Skill Score
      "n_scored": int,       # rows where signal fired (not bear regime)
      "n_total": int,        # total validation rows
      "base_rate": float,    # proportion of positive labels
      "mean_prob": float,    # average predicted probability
      "reliability": float,  # calibration term (Murphy)
      "resolution": float,   # discrimination term (Murphy)
      "uncertainty": float,  # climatological variance
  }
  ```

**`run_walkforward(full_db: pd.DataFrame, feature_cols: list[str], cfg_overrides: dict | None = None, folds: list[dict] | None = None) -> dict`**
- Convenience wrapper: runs all 6 folds, computes aggregates
- Returns:
  ```python
  {
      "mean_bss": float,
      "positive_folds": int,
      "fold_results": list[dict],
      "trimmed_mean_bss": float,  # drops worst fold
      "wilcoxon_p": float | None, # one-sided Wilcoxon signed-rank p-value (H₀: median BSS ≤ 0)
                                  # None if fewer than 6 non-zero BSS values (Wilcoxon requires ≥6)
  }
  ```
- **Statistical note:** The `wilcoxon_p` field provides a proper p-value for testing whether BSS is significantly positive across folds. This replaces the `positive_folds >= 3` majority-vote as the statistical significance measure. The majority-vote gate has a **65.6% false positive rate** under the null (P(≥3/6 | p=0.5) = 65.6%) — see research note on PBO/CSCV. The gate is retained for TPE steering only (penalizing clearly bad configs), NOT for claiming statistical significance.
- Implementation: `scipy.stats.wilcoxon(bss_values, alternative='greater')`. Returns the p-value from a one-sided test.

### Private Helpers

- `_bss(probs, y_true) -> float` — Brier Skill Score computation
- `_murphy_decomposition(probs, y_true) -> tuple[float, float, float]` — reliability, resolution, uncertainty
- `_apply_h7_hold_regime(val_db, train_db, base_rate, probs, threshold) -> tuple[ndarray, ndarray]` — bear mask + base_rate substitution
- `_impute_candle_nans(df) -> DataFrame` — candlestick NaN handling
- `_BetaCalibrator` — Drop-in replacement for `_PlattCalibrator` using `betacal.BetaCalibration(parameters='abm')`, extracted from `phase7_baseline.py:85-100`
- `_augment_with_candlestick(df) -> DataFrame` — Adds 15 candlestick proportion columns

### Constants

- `HORIZON = "fwd_7d_up"` — target column for BSS evaluation
- `SPY_THRESHOLD = 0.05` — H7 HOLD bear threshold
- `DATA_DIR = Path("data/52t_features")` — default data directory
- `FEATURE_COLS` — 23-column returns_candle feature list
- `CANDLE_COLS` — 15-column candlestick subset
- Imported: `WALKFORWARD_FOLDS` from `pattern_engine.config`

---

## Module 2: `pattern_engine/sweep.py`

### `OptunaSweep`

```python
OptunaSweep(
    study_name: str,                    # Identifies study for persistence
    objective_fn: Callable,             # (config_dict, full_db) -> dict
    search_space: dict,                 # {"param": (lo, hi) | ["cat1", "cat2"]}
    n_trials: int = 80,                 # Default per design decision
    max_hours: float = 16.0,            # Wall-clock budget; enforced via study.optimize(timeout=max_hours*3600)
    storage_path: str | None = None,    # SQLite path; None = in-memory
    gate_fn: Callable | None = None,    # (result_dict) -> bool; default: positive_folds >= 3
    seed: int = 42,                     # TPE sampler seed
)
```

**Methods:**
- `run(full_db: DataFrame, verbose: int = 1) -> SweepResult` — executes the full study
- `resume(full_db: DataFrame, verbose: int = 1) -> SweepResult` — loads existing SQLite study and continues until `n_trials` total are complete. Requires `storage_path` to be non-None (raises `RuntimeError` if in-memory). Internally calls `optuna.load_study()` then `study.optimize()` with `n_trials` set to remaining count (`n_trials - len(study.trials)`).
- `best() -> dict` — returns best trial's config + metrics
- `to_tsv(path: str)` — exports all completed trials to provenance TSV

**Objective wrapping internals:**
1. Sample config from search space via `trial.suggest_*` (type inference: both bounds `int` → `suggest_int`, else `suggest_float`; list → `suggest_categorical`)
2. Call `objective_fn(sampled_config, full_db)` inside `try/except Exception`
3. If exception or NaN `trimmed_mean_bss`: log warning, record trial as gate-failed with `trimmed_mean_bss = -0.10`
4. Apply `gate_fn` — trials failing the gate get penalized: `max(trimmed_mean_bss - 0.05, -0.10)` so TPE steers away without creating extreme outliers, but trials are still recorded
5. Return `trimmed_mean_bss` (or penalized value) to Optuna as the optimization target
6. Store fold-level results as trial user attributes
7. Log to `ExperimentLogger` incrementally

**Study creation:** `direction="maximize"` (BSS is a score to maximize). No pruning — all 6 folds run for every trial.

### `GridSweep`

Same objective function contract, exhaustive enumeration:

```python
GridSweep(
    objective_fn: Callable,
    param_grid: dict,                   # {"max_distance": [0.8, 1.0, 2.5], "top_k": [30, 50]}
    gate_fn: Callable | None = None,    # Same contract as OptunaSweep; default: positive_folds >= 3
)
```

- `run(full_db, verbose) -> SweepResult` — runs all combinations
- Useful for small focused sweeps

### `SweepResult` (dataclass)

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import optuna

@dataclass
class SweepResult:
    best_config: dict
    best_bss: float
    best_positive_folds: int
    best_wilcoxon_p: float | None  # Wilcoxon p-value for best trial (None if < 6 valid folds)
    results_df: pd.DataFrame       # All trials: trial_id, params, mean_bss, fold_bss, gate_pass, wilcoxon_p
    elapsed_minutes: float
    study: optuna.Study | None     # None for GridSweep; guarded by TYPE_CHECKING
```

### Pre-built Search Spaces

Module-level constants for common experiments:

```python
KNN_SEARCH_SPACE = {
    "max_distance": (1.0, 4.0),         # float
    "top_k": (20, 100),                  # int (both bounds int → suggest_int)
    "cal_frac": (0.5, 0.95),             # float — NOT an EngineConfig field; passed via cfg_overrides dict
    "confidence_threshold": (0.55, 0.80), # float
}
# Type inference rule: if both bounds are int → suggest_int; otherwise suggest_float.
# For categorical: use a list ["cat1", "cat2"] → suggest_categorical.
```

Future experiments add their own (e.g., `LIGHTGBM_SEARCH_SPACE`, `HMM_SEARCH_SPACE`).

---

## Module 3: `pattern_engine/experiment_log.py`

### `ExperimentLogger`

```python
ExperimentLogger(
    output_dir: str = "results",
    experiment_name: str = "",
)
```

**Methods:**
- `log_trial(trial_id: int, config: dict, result: dict) -> None` — appends one TSV row
- `log_header(columns: list[str]) -> None` — writes header + metadata comment block
- `finalize(best_config: dict, sweep_result: SweepResult) -> None` — writes summary footer

### TSV Format

```
# experiment: knn_optuna_sweep
# started: 2026-04-10T14:30:00
# search_space: {"max_distance": [1.0, 4.0], "top_k": [20, 100], ...}
# locked: returns_candle(23), beta_abm, regime=hold_spy_threshold+0.05
trial	max_distance	top_k	cal_frac	confidence_threshold	mean_bss	trimmed_mean_bss	positive_folds	wilcoxon_p	gate_pass	bss_2019	bss_2020	bss_2021	bss_2022	bss_2023	bss_2024	elapsed_s
0	2.31	42	0.72	0.63	+0.00041	+0.00058	3	0.156	True	...	12.3
1	1.85	67	0.81	0.70	-0.00102	-0.00054	1	0.844	False	...	11.8
```

**Design choices:**
- Per-fold BSS columns flattened (not nested arrays) — pandas/Excel friendly
- Rows appended incrementally (`mode="a"`) — valid even if sweep killed mid-run
- Comment block captures full experimental context for provenance

---

## Migration Plan

### Scripts That Change (import source moves)

| Script | Current | After |
|--------|---------|-------|
| `scripts/phase7_baseline.py` | Defines `run_fold_with_config()` inline | Imports from `pattern_engine.walkforward`, keeps `run_fold_with_config()` as thin re-export wrapper |
| `scripts/phase7_e1_*.py` through `e4_*.py` | Import from `phase7_baseline` | No change — `phase7_baseline` re-exports from `walkforward` |

### Backwards Compatibility

`phase7_baseline.py` keeps `run_fold_with_config()` as a thin wrapper calling `walkforward.run_fold()`. E1–E4 scripts import from `phase7_baseline` which re-exports. Zero risk to existing provenance.

**Full re-export list** (all symbols E1–E4 scripts import from `phase7_baseline`):
- `run_fold_with_config` — thin wrapper → `walkforward.run_fold()`
- `_BetaCalibrator` — re-exported from `walkforward`
- `_augment_with_candlestick` — re-exported from `walkforward`
- `_apply_h7_hold_regime` — re-exported from `walkforward`
- `_build_cfg` — re-exported from `walkforward`
- `DATA_DIR`, `FEATURE_COLS`, `HORIZON`, `SPY_THRESHOLD` — re-exported constants

### Manual Sweep Scripts

Kept for reference. Not deleted. New experiments use `OptunaSweep` or `GridSweep`.

### Worktree Prototype

Not ported directly. Best ideas incorporated:
- TPE sampler config, SQLite persistence, time budgeting, trial user attributes, results DataFrame

Stale parts discarded:
- 8D features, missing H7 HOLD, `WalkForwardRunner` class, `EngineConfig.replace()`

---

## Testing Strategy

### New Test Files

| File | ~Tests | Coverage |
|------|--------|----------|
| `tests/unit/test_walkforward.py` | 12 | `run_fold()` dict shape, BSS matches manual calc, H7 HOLD applies, Murphy decomposition sums, `load_and_augment_db()` columns |
| `tests/unit/test_sweep.py` | 10 | `OptunaSweep` runs 3 trials on synthetic data, gate penalizes failures, SQLite persist/resume, `GridSweep` exhaustive, `SweepResult` fields, `to_tsv()` output |
| `tests/unit/test_experiment_log.py` | 5 | Header metadata, incremental append, partial-write validity, finalize summary |

### Approach

- Sweep tests use **mock objective function** returning synthetic BSS — no real PatternMatcher. Tests run in <1s each.
- `test_walkforward.py` needs a synthetic database fixture. **Note:** Neither `tests/conftest.py` nor `tests/unit/conftest.py` currently exist. Create `tests/unit/conftest.py` with synthetic database fixtures (`synthetic_db`, `train_db`, `val_db`). Ensure the fixture includes the columns walkforward needs: `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `ret_90d`, `fwd_7d_up`, plus the 8 return features.
- One `@pytest.mark.slow` integration test: 3-trial OptunaSweep on real 52T data with `run_fold`. Excluded from fast suite.

### Parity Check (Critical)

Dedicated test verifying `walkforward.run_fold()` produces **identical BSS** to old `phase7_baseline.run_fold_with_config()` on the same fold + config. This is the regression gate.

### Expected Count

~27 new tests. Suite: 858 -> ~885.

---

## Success Criteria

1. `pattern_engine/walkforward.py` passes parity test against `phase7_baseline.run_fold_with_config()`
2. `OptunaSweep` completes 80 trials with SQLite persistence and produces valid `SweepResult`
3. `GridSweep` reproduces identical results to existing manual sweep scripts
4. All TSV output follows provenance format with metadata headers
5. E1–E4 Phase 7 scripts still work via re-export wrapper
6. 885+ tests pass (`pytest tests/ -q -m "not slow"`)
7. One `@pytest.mark.slow` integration test validates end-to-end on real data

---

## Statistical Validity Notes

### Gate Function Role

The default `gate_fn` (`positive_folds >= 3`) is used **only for TPE steering** — penalizing clearly poor configs so the Bayesian sampler avoids them. It is NOT a statistical significance test. Under the null (no skill), BSS > 0 occurs ~50% per fold, so P(≥3/6) = 65.6% — a gate this permissive cannot make significance claims.

**For claiming a config is statistically significant**, use the `wilcoxon_p` field from `run_walkforward()`. A Wilcoxon signed-rank p-value < 0.05 (one-sided, H₀: median BSS ≤ 0) is the minimum standard. When comparing multiple configs from a sweep, apply **Holm-Bonferroni correction** across the top-N candidates via `statsmodels.stats.multitest.multipletests(p_values, method='holm')`.

### Multiple Testing Across Experiments

When FPPE runs multiple sweep experiments (e.g., KNN sweep, LightGBM sweep, HMM sweep), the best config from each experiment should be collected and their `wilcoxon_p` values corrected jointly. This is outside P3's scope but the infrastructure supports it — each `SweepResult` carries `best_wilcoxon_p`.

### CPCV Future Path

The current 6-fold walk-forward produces 5 out-of-sample test periods — insufficient for robust inference. Combinatorially Purged Cross-Validation (CPCV) with 10 groups and k=8 test groups produces 36 backtest paths from 45 combinations. The `objective_fn` abstraction supports this: a CPCV-based objective function can replace the walk-forward one without changing `OptunaSweep` or `GridSweep`. Recommended library: `skfolio.model_selection.CombinatorialPurgedCV`. This is a separate research item (not P3 scope).

### DSR and MinBTL

Deflated Sharpe Ratio (DSR) and Minimum Backtest Length (MinBTL) computations are complementary validations that can be added to `experiment_log.py` as post-sweep analysis utilities. Not required for P3 MVP but noted as natural extensions.

---

## Dependencies

- `optuna>=3.4.0` — already installed in venv
- `betacal` — already installed in venv (used by `_BetaCalibrator`)
- `scipy` — already installed in venv (used by `scipy.stats.wilcoxon` for p-value computation)
- No new dependencies required

## Locked Settings Context

```
Features=returns_candle(23), max_distance=2.5, Calibration=beta_abm,
top_k=50, confidence_threshold=0.65, regime=hold_spy_threshold+0.05,
horizon=fwd_7d_up, nn_jobs=1
```
