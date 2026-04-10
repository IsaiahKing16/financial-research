# Phase 7 — Model Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add six optional model enhancements to the FPPE KNN system, each behind a feature flag, with TDD and walk-forward gate evaluation for each.

**Architecture:** Each enhancement is implemented behind a config flag (default False), tested via TDD, evaluated by a 6-fold walk-forward script, and either promoted (flag=True in config.py + CLAUDE.md update) or discarded (flag=False, one-line CLAUDE.md failure note). Enhancements are cumulative — each subsequent one runs with all prior passing enhancements enabled.

**Tech Stack:** Python 3.12, sklearn, betacal, dtw-python, stumpy, scipy, pytest, pandas/numpy. Windows 11. Always use `PYTHONUTF8=1 py -3.12` prefix.

**Spec:** `docs/superpowers/specs/2026-04-09-phase7-model-enhancements-design.md`
**Handoff:** `HANDOFF_Phase7_Model_Enhancements.md` (in Downloads — copy to repo root if needed)

---

## File Map

**Create:**
- `scripts/phase7_baseline.py` — T7.0 single-config 6-fold walk-forward
- `scripts/phase7_e1_bma_calibrator.py` — E1 BMA walk-forward
- `scripts/phase7_e2_owa_weighting.py` — E2 OWA walk-forward
- `scripts/phase7_e3_dtw_reranker.py` — E3 DTW walk-forward
- `scripts/benchmarks/b5_dtw_vs_euclidean.py` — E3 Competitive Benchmark B5
- `scripts/phase7_e4_conformal.py` — E4 conformal coverage eval
- `scripts/phase7_e5_cpod_filter.py` — E5 LOF FPR/TPR eval
- `scripts/phase7_e6_stumpy.py` — E6 STUMPY walk-forward
- `scripts/benchmarks/b6_stumpy_vs_knn.py` — E6 Competitive Benchmark B6
- `pattern_engine/owa_weights.py` — OWA weight computation
- `pattern_engine/anomaly_filter.py` — LOF-based SignalFilterBase
- `research/wfa_reranker.py` — DTW reranker (from scratch)
- `research/stumpy_matcher.py` — STUMPY AB-join signal generator
- `tests/unit/test_phase7_e1_bma.py`
- `tests/unit/test_phase7_e2_owa.py`
- `tests/unit/test_phase7_e3_dtw.py`
- `tests/unit/test_phase7_e4_conformal.py`
- `tests/unit/test_phase7_e5_cpod.py`
- `tests/unit/test_phase7_e6_stumpy.py`
- `results/phase7/` directory (created by scripts via `Path.mkdir(parents=True, exist_ok=True)`)
- `results/benchmarks/` directory (same)

**Modify:**
- `pattern_engine/config.py` — fix 3 defaults, add 6 flags + 8 params
- `pattern_engine/matcher.py` — wire BMA (new `_calibrate_bma()` path), OWA (in `_prepare_features`), DTW (between `_post_filter` and `_package_results`)
- `pattern_engine/conformal_hooks.py` — promote `NaiveConformalCalibrator` to full ACI
- `pattern_engine/signal_pipeline.py` — insert `AnomalyFilter` before existing filters

---

## Task 0: Pre-flight — Fix config.py defaults and add Phase 7 flags

**Files:**
- Modify: `pattern_engine/config.py`

- [ ] **Step 1: Read `pattern_engine/config.py` in full to understand current state**

- [ ] **Step 2: Apply corrections and additions**

Make these exact changes:

```python
# Fix 1: feature_set default
feature_set: str = "returns_candle"  # was "returns_only"

# Fix 2: max_distance default
max_distance: float = 2.5  # was 1.1019

# Fix 3: calibration_method default
calibration_method: str = "beta_abm"  # was "platt"

# Add Phase 7 flags after the existing research pilot section:

# ── Phase 7 enhancements (all False in production) ───────────────────────────
use_bma: bool = False               # E1: Bayesian Model Averaging calibrator
use_owa: bool = False               # E2: OWA feature weighting
use_dtw_reranker: bool = False      # E3: DTW post-retrieval reranker
use_conformal: bool = False         # E4: Adaptive Conformal Prediction
use_anomaly_filter: bool = False    # E5: LOF anomaly filter
use_stumpy: bool = False            # E6: STUMPY matrix profile signal

# Phase 7 enhancement parameters
owa_alpha: float = 1.0              # E2: concentration exponent
dtw_rerank_k: int = 20             # E3: neighbours to return after reranking
conformal_alpha: float = 0.10      # E4: nominal miscoverage rate
conformal_gamma: float = 0.05      # E4: ACI online learning rate
anomaly_contamination: float = 0.05  # E5: expected outlier fraction
anomaly_penalty: float = 0.50      # E5: confidence multiplier for outliers
stumpy_weight: float = 0.20        # E6: blend weight for STUMPY signal
stumpy_subsequence_length: int = 50  # E6: matrix profile subsequence length
```

- [ ] **Step 3: Run full test suite — must still pass with changed defaults**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"
```

Expected: 846 passed. If failures occur, check which tests hard-code `"returns_only"` or `1.1019` and update them to match the new locked values.

- [ ] **Step 4: Commit**

```
git add pattern_engine/config.py
git commit -m "feat(phase7): fix config.py defaults + add Phase 7 feature flags"
```

---

## Task 1: T7.0 — Baseline Re-establishment Script

**Files:**
- Create: `scripts/phase7_baseline.py`
- Create: `results/phase7/baseline_23d.tsv` (generated by script)

**Context:** This script produces the single comparison target for all Phase 7 delta measurements. Pattern: clone Config A from `scripts/phase6_bss_comparison.py`. Read that file first.

- [ ] **Step 1: Read `scripts/phase6_bss_comparison.py` in full to understand the walk-forward pattern**

Pay attention to:
- How `_BetaCalibrator` is monkey-patched
- How H7 HOLD regime is applied (bear rows → base_rate prob)
- How BSS is computed (Brier Skill Score formula)
- How Murphy decomposition columns are computed

- [ ] **Step 2: Create `scripts/phase7_baseline.py`**

Note: name the primary fold-running function `run_fold_with_config(fold, cfg_overrides=None)` (not `run_fold`) so that E1–E6 scripts can import it directly with `from phase7_baseline import run_fold_with_config, compute_bss, _BetaCalibrator`.

```python
"""
scripts/phase7_baseline.py — T7.0: 23D baseline BSS for all 6 folds.

Locked config: returns_candle(23D), max_distance=2.5, top_k=50,
beta_abm calibration, H7 HOLD regime (SPY ret_90d < +0.05 → base_rate).

Output: results/phase7/baseline_23d.tsv
  Columns: fold, bss, n_scored, n_total, base_rate, mean_prob,
           reliability, resolution, uncertainty

This TSV is THE comparison target for all Phase 7 enhancement deltas.
Do NOT re-run this script for each enhancement — load the TSV instead.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/phase7_baseline.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pattern_engine.matcher as _matcher_module
from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS
from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import FeatureRegistry

try:
    from betacal import BetaCalibration
except ImportError:
    raise RuntimeError("betacal not installed. Run: py -3.12 -m pip install betacal")

DATA_DIR   = REPO_ROOT / "data" / "52t_features"
OUTPUT_DIR = REPO_ROOT / "results" / "phase7"
OUTPUT_TSV = OUTPUT_DIR / "baseline_23d.tsv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = FeatureRegistry.get("returns_candle").columns  # 23 cols
SPY_THRESHOLD = 0.05   # H7 HOLD: bear = SPY ret_90d < +0.05
HORIZON       = "fwd_7d_up"
MAX_DISTANCE  = 2.5
TOP_K         = 50


class _BetaCalibrator:
    """Drop-in replacement for _PlattCalibrator using beta_abm."""
    def __init__(self):
        self._cal = BetaCalibration(parameters="abm")
    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_BetaCalibrator":
        self._cal.fit(raw.reshape(-1, 1), y)
        return self
    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self._cal.predict(raw.reshape(-1, 1))


def compute_bss(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute BSS with Murphy decomposition.

    BSS = 1 - Brier / Brier_clim
    Murphy decomposition: Brier = reliability - resolution + uncertainty
    """
    if len(probs) == 0:
        return dict(bss=float("nan"), n_scored=0, n_total=0,
                    base_rate=float("nan"), mean_prob=float("nan"),
                    reliability=float("nan"), resolution=float("nan"),
                    uncertainty=float("nan"))
    base_rate = labels.mean()
    brier_clim = base_rate * (1 - base_rate)
    brier = np.mean((probs - labels) ** 2)
    bss = 1 - brier / brier_clim if brier_clim > 0 else float("nan")

    # Murphy decomposition (binned reliability)
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    reliability = resolution = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        p_bar = probs[mask].mean()
        o_bar = labels[mask].mean()
        n_k = mask.sum()
        reliability += n_k * (p_bar - o_bar) ** 2
        resolution  += n_k * (o_bar - base_rate) ** 2
    n = len(probs)
    reliability /= n
    resolution  /= n
    uncertainty = base_rate * (1 - base_rate)

    return dict(bss=bss, n_scored=len(probs), n_total=len(probs),
                base_rate=base_rate, mean_prob=probs.mean(),
                reliability=reliability, resolution=resolution,
                uncertainty=uncertainty)


def run_fold_with_config(fold: dict, cfg_overrides: dict | None = None) -> dict:
    """Run one walk-forward fold. Returns BSS dict with fold label.

    Args:
        fold: One entry from WALKFORWARD_FOLDS.
        cfg_overrides: Optional dict of EngineConfig field overrides for
                       enhancement scripts (e.g. {'use_bma': True}).
    """
    # Load data
    all_files = sorted(DATA_DIR.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in all_files]
    db = pd.concat(dfs, ignore_index=True)
    db["Date"] = pd.to_datetime(db["Date"])

    train_db = db[db["Date"] <= fold["train_end"]].copy()
    val_db   = db[(db["Date"] >= fold["val_start"]) &
                  (db["Date"] <= fold["val_end"])].copy()

    # H7 HOLD regime: mark bear rows
    bear_mask_val = val_db.get("ret_90d_spy", pd.Series(dtype=float)) < SPY_THRESHOLD

    # Monkey-patch beta_abm calibrator
    _orig = _matcher_module._PlattCalibrator
    _matcher_module._PlattCalibrator = _BetaCalibrator
    try:
        base_kwargs = dict(
            feature_set="returns_candle",
            max_distance=MAX_DISTANCE,
            top_k=TOP_K,
            calibration_method="beta_abm",
            regime_filter=False,  # manual HOLD applied below
        )
        if cfg_overrides:
            base_kwargs.update(cfg_overrides)
        cfg = EngineConfig(**base_kwargs)
        matcher = PatternMatcher(cfg)
        matcher.fit(train_db, list(FEATURE_COLS))
        probs, _, _, _, _, _ = matcher.query(val_db, verbose=0)
    finally:
        _matcher_module._PlattCalibrator = _orig

    # Apply H7 HOLD: bear rows → base_rate (not scored)
    labels = val_db[HORIZON].values.astype(np.float64)
    scored_mask = ~bear_mask_val.values

    scored_probs  = probs[scored_mask]
    scored_labels = labels[scored_mask]

    result = compute_bss(scored_probs, scored_labels)
    result["fold"] = fold["label"]
    result["n_total"] = len(probs)
    return result


def main():
    rows = []
    for fold in WALKFORWARD_FOLDS:
        print(f"  Running fold {fold['label']}...", end=" ", flush=True)
        t0 = time.time()
        row = run_fold_with_config(fold)
        elapsed = time.time() - t0
        print(f"BSS={row['bss']:.5f}  n_scored={row['n_scored']}  ({elapsed:.1f}s)")
        rows.append(row)

    df = pd.DataFrame(rows)
    col_order = ["fold", "bss", "n_scored", "n_total", "base_rate",
                 "mean_prob", "reliability", "resolution", "uncertainty"]
    df = df[col_order]
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"\nSaved: {OUTPUT_TSV}")
    print(f"Mean BSS: {df['bss'].mean():.5f}")
    print(f"Positive folds: {(df['bss'] > 0).sum()}/6")
    print(df[["fold", "bss", "n_scored"]].to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify data exists before running**

```
PYTHONUTF8=1 py -3.12 -c "from pathlib import Path; print(list(Path('data/52t_features').glob('*.parquet'))[:3])"
```

Check that parquet files are listed. If the column `ret_90d_spy` doesn't exist, use `ret_90d` or whatever the SPY 90-day return column is named in the data.

- [ ] **Step 4: Run the baseline script**

```
PYTHONUTF8=1 py -3.12 scripts/phase7_baseline.py
```

Expected: 6 lines of output, one per fold, then summary. BSS values should be near +0.00033 mean. Verify `results/phase7/baseline_23d.tsv` has 6 rows with no NaN.

- [ ] **Step 5: Commit**

```
git add scripts/phase7_baseline.py results/phase7/baseline_23d.tsv
git commit -m "feat(phase7/t7.0): baseline 23D BSS script + results"
```

---

## Task 2: T7.1 — E1: BMA Calibrator

**Files:**
- Create: `tests/unit/test_phase7_e1_bma.py`
- Create: `scripts/phase7_e1_bma_calibrator.py`
- Modify: `pattern_engine/matcher.py`
- Read first: `research/bma_calibrator.py`, `pattern_engine/matcher.py` (fit() method lines 467–580)

**Key design:** BMA's `fit()` takes `raw_probs: (N, K)` — K per-neighbour labels per calibration sample. This is fundamentally different from PlattCalibrator. BMA needs a new `_calibrate_bma()` code path in `PatternMatcher.fit()` that extracts the per-neighbour label matrix before aggregation.

- [ ] **Step 1: Audit `research/bma_calibrator.py`**

Confirm:
- `fit(raw_probs, y_true)` where `raw_probs.shape = (N, K)`, `y_true.shape = (N,)` ✓
- `transform(raw_probs)` where `raw_probs.shape = (K,)` → scalar ✓
- No dependency on feature dimensionality (works on 1D labels of K neighbours) ✓
- Dependencies: `numpy`, `scipy.stats.t` (both available) ✓

- [ ] **Step 2: Write failing tests**

```python
# tests/unit/test_phase7_e1_bma.py
"""TDD tests for E1: BMA Calibrator integration."""
from __future__ import annotations
import numpy as np
import pytest
from research.bma_calibrator import BMACalibrator


class TestBMACalibrator:
    def test_fit_accepts_n_k_matrix(self):
        """fit() accepts (N, K) raw_probs without raising."""
        bma = BMACalibrator()
        raw = np.random.RandomState(0).uniform(0, 1, (200, 50))
        labels = np.random.RandomState(0).randint(0, 2, 200).astype(float)
        bma.fit(raw, labels)  # should not raise
        assert bma.fitted

    def test_transform_returns_scalar_in_unit_interval(self):
        """transform() on (K,) input returns scalar in [0, 1]."""
        bma = BMACalibrator()
        raw = np.random.RandomState(1).uniform(0, 1, (200, 50))
        labels = np.random.RandomState(1).randint(0, 2, 200).astype(float)
        bma.fit(raw, labels)
        k_probs = np.random.RandomState(2).uniform(0, 1, 50)
        result = bma.transform(k_probs)
        assert isinstance(float(result), float)
        assert 0.0 <= float(result) <= 1.0

    def test_weights_sum_to_one(self):
        """After fit(), BMA weights sum to 1.0."""
        bma = BMACalibrator()
        raw = np.random.RandomState(3).uniform(0, 1, (300, 50))
        labels = np.random.RandomState(3).randint(0, 2, 300).astype(float)
        bma.fit(raw, labels)
        assert abs(bma.weights.sum() - 1.0) < 1e-6

    def test_convergence_on_small_data(self):
        """EM converges (no exception) on 500 samples, K=20."""
        bma = BMACalibrator()
        raw = np.random.RandomState(4).uniform(0, 1, (500, 20))
        labels = np.random.RandomState(4).randint(0, 2, 500).astype(float)
        bma.fit(raw, labels)
        assert bma.fitted

    def test_transform_before_fit_raises(self):
        """transform() before fit() raises RuntimeError."""
        bma = BMACalibrator()
        with pytest.raises(RuntimeError):
            bma.transform(np.array([0.5] * 50))


class TestBMAGate:
    def test_gate_passes_when_3_folds_improve(self):
        """Gate passes when ≥3 folds show BMA_BSS - baseline_BSS ≥ +0.001."""
        baseline = np.array([0.001, 0.002, -0.001, 0.003, 0.000, 0.001])
        enhanced = np.array([0.003, 0.004, -0.001, 0.005, 0.001, 0.001])
        deltas = enhanced - baseline
        folds_improved = (deltas >= 0.001).sum()
        assert folds_improved >= 3

    def test_gate_fails_when_fewer_than_3_improve(self):
        """Gate fails when <3 folds improve by ≥+0.001."""
        baseline = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
        enhanced = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
        deltas = enhanced - baseline
        folds_improved = (deltas >= 0.001).sum()
        assert folds_improved < 3
```

- [ ] **Step 3: Run failing tests**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_phase7_e1_bma.py -v
```

Expected: `test_transform_before_fit_raises` may already pass. The others should pass or fail cleanly (no import errors). If `research.bma_calibrator` import fails, check `research/__init__.py`.

- [ ] **Step 4: Add `_calibrate_bma()` to `PatternMatcher` in `matcher.py`**

Read `matcher.py` `fit()` method (lines 467–580) carefully. Then add:

```python
def _extract_neighbour_labels_for_bma(
    self,
    cal_db: pd.DataFrame,
    regime_labeler=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-neighbour label matrix for BMA calibration.

    Returns:
        raw_matrix: (N_cal, K) float64 — K=top_k neighbour labels per query.
                    Rows with < top_k matches are padded with base_rate.
        y_true:     (N_cal,) float64 — true binary labels.
    """
    cfg = self.config
    horizon = cfg.projection_horizon
    base_rate = (
        self._train_target_arr.mean()
        if self._train_target_arr is not None
        else 0.5
    )
    K = cfg.top_k

    X_raw = cal_db[self._feature_cols].values
    X_cal = self._prepare_features(X_raw, fit_scaler=False)

    all_rows = []
    for start in range(0, len(X_cal), cfg.batch_size):
        end = min(start + cfg.batch_size, len(X_cal))
        X_batch = X_cal[start:end]
        val_slice = cal_db.iloc[start:end]

        distances_b, indices_b = self._query_batch(X_batch)

        val_tickers_b = np.asarray(val_slice["Ticker"], dtype=object)
        val_sectors_b = np.array(
            [_SECTOR_MAP.get(t, "") for t in val_tickers_b], dtype=object
        )
        val_regime_b = None

        top_mask = self._post_filter(
            distances_b, indices_b,
            val_tickers_b, val_sectors_b, val_regime_b,
            X_batch=None,
        )

        B = X_batch.shape[0]
        for i in range(B):
            accepted = indices_b[i][top_mask[i]]
            labels_i = self._train_target_arr[accepted]  # (n_accepted,)
            # Pad/truncate to exactly K slots
            row = np.full(K, base_rate, dtype=np.float64)
            n = min(len(labels_i), K)
            row[:n] = labels_i[:n]
            all_rows.append(row)

    raw_matrix = np.vstack(all_rows)  # (N_cal, K)
    y_true = (
        cal_db[horizon].values.astype(np.float64)
        if horizon in cal_db.columns
        else np.zeros(len(cal_db), dtype=np.float64)
    )
    return raw_matrix, y_true
```

Then modify the calibration double-pass section of `fit()` to wire BMA:

```python
# After: self._calibrator = None
_cal_method = getattr(cfg, 'calibration_method', 'platt')
self._calibrator = None
self._bma_calibrator = None  # NEW: BMA calibrator slot

if _cal_method not in ('none', None):
    # ... existing cal_frac/cal_max_samples logic to get _cal_db ...

    if getattr(cfg, 'use_bma', False):
        # BMA path: extract (N, K) per-neighbour label matrix
        from research.bma_calibrator import BMACalibrator
        _raw_matrix, _y_true = self._extract_neighbour_labels_for_bma(
            _cal_db, regime_labeler=regime_labeler
        )
        self._bma_calibrator = BMACalibrator()
        self._bma_calibrator.fit(_raw_matrix, _y_true)
        # self._calibrator stays None — BMA applied in _package_results
    else:
        # Existing Platt/beta_abm path (unchanged)
        _cal_raw_probs, _, _, _, _, _ = self.query(
            _cal_db, regime_labeler=regime_labeler, verbose=0
        )
        self._calibrator = _PlattCalibrator().fit(_cal_raw_probs, _y_true_existing)
```

Also modify `__init__` to add `self._bma_calibrator = None`.

In `_package_results()`, after computing `prob_up` (the uniform-mean prob), add:

```python
# BMA override: replace uniform-mean prob with BMA posterior-weighted prob
if getattr(cfg, 'use_bma', False) and self._bma_calibrator is not None:
    bma_probs = np.zeros(B, dtype=np.float64)
    for i in range(B):
        accepted = indices_b[i][top_mask[i]]
        labels_i = self._train_target_arr[accepted]
        K = cfg.top_k
        base_rate = self._train_target_arr.mean()
        row = np.full(K, base_rate, dtype=np.float64)
        n = min(len(labels_i), K)
        row[:n] = labels_i[:n]
        bma_probs[i] = float(self._bma_calibrator.transform(row))
    prob_up = bma_probs
```

Note: `_package_results` currently doesn't take `indices_b` — you'll need to pass it through from `query()`. Trace the call chain in `query()` and thread `indices_b` to `_package_results`. **Batch loop awareness:** `query()` processes `val_db` in batches of `batch_size`. The BMA accumulation (`_extract_neighbour_labels_for_bma`) and inference (building the per-query `(K,)` vector and calling `bma.transform()`) both operate on per-batch slices of `indices_b` — shape `(B, n_probe)` within each iteration. The BMA calibration in `fit()` loops over batches internally in `_extract_neighbour_labels_for_bma`. At inference time in `_package_results`, the loop `for i in range(B)` already processes one batch — `indices_b[i]` is the correct per-query slice.

- [ ] **Step 5: Run the BMA tests**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_phase7_e1_bma.py -v
```

Expected: all pass.

- [ ] **Step 6: Run full suite to confirm no regressions**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"
```

Expected: 846+ passed, 0 failed.

- [ ] **Step 7: Write the E1 walk-forward script `scripts/phase7_e1_bma_calibrator.py`**

Pattern: clone `phase7_baseline.py`. Changes:
- Load `results/phase7/baseline_23d.tsv` for Config A (do NOT re-run baseline)
- Config B: `use_bma=True`, all other settings locked
- Output: `results/phase7/e1_bma_vs_beta_abm.tsv`
- Columns: `fold, baseline_bss, bma_bss, delta, improved`
- Gate evaluation: count folds where `delta >= 0.001`. Print PASS/FAIL.
- Append to `results/phase7/enhancement_summary.tsv`

```python
"""
scripts/phase7_e1_bma_calibrator.py — E1: BMA vs beta_abm BSS comparison.
Gate: BMA_BSS - baseline_BSS >= +0.001 on >= 3/6 folds.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS
from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import FeatureRegistry
# Import compute_bss from phase7_baseline (or copy the function)
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from phase7_baseline import run_fold_with_config, compute_bss, _BetaCalibrator

DATA_DIR    = REPO_ROOT / "data" / "52t_features"
RESULTS_DIR = REPO_ROOT / "results" / "phase7"
BASELINE_TSV = RESULTS_DIR / "baseline_23d.tsv"
OUTPUT_TSV  = RESULTS_DIR / "e1_bma_vs_beta_abm.tsv"
SUMMARY_TSV = RESULTS_DIR / "enhancement_summary.tsv"

GATE_DELTA     = 0.001
GATE_MIN_FOLDS = 3

# ... [implement run_bma_fold() analogous to run_fold() but with use_bma=True]
# ... [load baseline from TSV, compare, evaluate gate, append to summary]
```

Note: Factor the fold-running logic in `phase7_baseline.py` into a reusable `run_fold_with_config(fold, cfg_overrides)` function when writing that script, so E1–E6 scripts can import it.

- [ ] **Step 8: Run E1 walk-forward**

```
PYTHONUTF8=1 py -3.12 scripts/phase7_e1_bma_calibrator.py
```

Expected: 6 fold results printed, gate verdict (PASS or FAIL), TSV written.

- [ ] **Step 9: Apply gate result**

**If PASS:** Edit `pattern_engine/config.py`: `use_bma: bool = True`. Update `CLAUDE.md` locked settings section: `Calibration=bma` and provenance line.

**If FAIL:** Keep `use_bma: bool = False`. Add one-line comment in CLAUDE.md: `# E1 BMA: FAIL (BSS delta X/6 folds >= +0.001). Provenance: results/phase7/e1_bma_vs_beta_abm.tsv`

- [ ] **Step 10: Commit**

```
git add tests/unit/test_phase7_e1_bma.py scripts/phase7_e1_bma_calibrator.py
git add pattern_engine/matcher.py pattern_engine/config.py CLAUDE.md
git add results/phase7/e1_bma_vs_beta_abm.tsv results/phase7/enhancement_summary.tsv
git commit -m "feat(phase7/e1): BMA calibrator — PASS/FAIL [update with actual verdict]"
```

---

## Task 3: T7.2 — E2: OWA Feature Weighting

**Files:**
- Create: `pattern_engine/owa_weights.py`
- Create: `tests/unit/test_phase7_e2_owa.py`
- Create: `scripts/phase7_e2_owa_weighting.py`
- Modify: `pattern_engine/matcher.py` (`_prepare_features`)

**Context:** OWA assigns higher distance-space weights to features with higher mutual information to the target. Global weights (not per-regime). Alpha controls concentration. Use existing `feature_weights` dict infrastructure — OWA produces a weight dict that `apply_feature_weights()` already handles.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_phase7_e2_owa.py
"""TDD tests for E2: OWA Feature Weighting."""
from __future__ import annotations
import numpy as np
import pytest


class TestOWAWeights:
    def test_weights_sum_to_n_features(self):
        """Weights sum to n_features for any alpha."""
        from pattern_engine.owa_weights import owa_weights
        w = owa_weights(23, alpha=1.0)
        assert abs(w.sum() - 23) < 1e-9

    def test_alpha_zero_gives_uniform(self):
        """Alpha=0 produces all-ones weights (uniform)."""
        from pattern_engine.owa_weights import owa_weights
        w = owa_weights(23, alpha=0.0)
        np.testing.assert_allclose(w, np.ones(23), atol=1e-9)

    def test_weights_monotone_decreasing(self):
        """Weights are non-increasing (highest-ranked features get highest weight)."""
        from pattern_engine.owa_weights import owa_weights
        w = owa_weights(23, alpha=1.0)
        assert all(w[i] >= w[i+1] for i in range(len(w)-1))

    def test_no_negative_weights(self):
        """All weights >= 0 for any alpha > 0."""
        from pattern_engine.owa_weights import owa_weights
        for alpha in [0.5, 1.0, 2.0, 4.0]:
            w = owa_weights(23, alpha=alpha)
            assert (w >= 0).all()


class TestMIRanking:
    def test_mi_ranking_returns_indices(self):
        """compute_mi_ranking returns integer indices of length n_features."""
        from pattern_engine.owa_weights import compute_mi_ranking
        rng = np.random.RandomState(0)
        X = rng.randn(500, 23)
        y = rng.randint(0, 2, 500)
        ranking = compute_mi_ranking(X, y)
        assert len(ranking) == 23
        assert set(ranking) == set(range(23))

    def test_mi_ranking_uses_training_only(self):
        """Two calls on different subsets give different rankings (no leakage check)."""
        from pattern_engine.owa_weights import compute_mi_ranking
        rng = np.random.RandomState(42)
        X1 = rng.randn(500, 23)
        X2 = rng.randn(500, 23)
        y1 = rng.randint(0, 2, 500)
        y2 = rng.randint(0, 2, 500)
        r1 = compute_mi_ranking(X1, y1)
        r2 = compute_mi_ranking(X2, y2)
        assert not np.array_equal(r1, r2), "Rankings should differ on different data"


class TestOWAGate:
    def test_gate_passes_when_3_folds_improve_and_worst_ok(self):
        """Gate passes: >=3 folds delta>=+0.001 AND worst fold delta>=-0.0005."""
        from pattern_engine.owa_weights import evaluate_owa_gate
        baseline = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        enhanced = np.array([0.003, 0.003, 0.003, 0.001, 0.000, 0.002])
        result, reason = evaluate_owa_gate(baseline, enhanced)
        assert result == "PASS"

    def test_gate_fails_on_worst_fold_degradation(self):
        """Gate fails even with 4 improved folds if worst degrades >0.0005."""
        from pattern_engine.owa_weights import evaluate_owa_gate
        baseline = np.array([0.001, 0.001, 0.001, 0.001, 0.010, 0.010])
        enhanced = np.array([0.003, 0.003, 0.003, 0.003, 0.009, 0.010])
        # worst fold delta = 0.009 - 0.010 = -0.001 < -0.0005
        result, reason = evaluate_owa_gate(baseline, enhanced)
        assert result == "FAIL"
        assert "worst" in reason.lower()
```

- [ ] **Step 2: Run failing tests**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_phase7_e2_owa.py -v
```

Expected: ImportError on `pattern_engine.owa_weights` — module doesn't exist yet.

- [ ] **Step 3: Create `pattern_engine/owa_weights.py`**

```python
"""
pattern_engine/owa_weights.py — OWA feature weighting for E2.

Computes global OWA weights from mutual information rankings.
Used in PatternMatcher._prepare_features() when use_owa=True.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def compute_mi_ranking(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Return feature indices sorted by MI with y_train (descending).

    Args:
        X_train: (N, D) training features.
        y_train: (N,) binary labels.

    Returns:
        ranking: (D,) array of feature indices, highest MI first.
    """
    mi = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=3)
    return np.argsort(mi)[::-1]  # descending MI order


def owa_weights(n_features: int, alpha: float) -> np.ndarray:
    """Compute OWA weight vector.

    w_i = ((n_features - i) / n_features) ** alpha  for rank i in [0, n_features-1]
    Normalized so weights sum to n_features (mean weight = 1.0, no-op at alpha=0).

    Args:
        n_features: Number of features.
        alpha: Concentration exponent. 0 = uniform, 4 = aggressive concentration.

    Returns:
        weights: (n_features,) array, index 0 = weight for highest-MI feature.
    """
    ranks = np.arange(n_features)  # 0 = best, n-1 = worst
    raw = ((n_features - ranks) / n_features) ** alpha
    if raw.sum() < 1e-12:
        return np.ones(n_features, dtype=float)
    return raw * n_features / raw.sum()


def mi_to_weight_dict(
    feature_cols: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Compute OWA weight dict keyed by column name.

    Args:
        feature_cols: Ordered list of feature column names.
        X_train: (N, D) training features (already in feature_cols order).
        y_train: (N,) binary labels.
        alpha: OWA concentration exponent.

    Returns:
        dict mapping col -> weight (compatible with EngineConfig.feature_weights).
    """
    ranking = compute_mi_ranking(X_train, y_train)  # indices sorted by MI desc
    weights_ordered = owa_weights(len(feature_cols), alpha)  # weight by rank
    weight_by_col = {}
    for rank_pos, feat_idx in enumerate(ranking):
        col = feature_cols[feat_idx]
        weight_by_col[col] = float(weights_ordered[rank_pos])
    return weight_by_col


def evaluate_owa_gate(
    baseline_bss: np.ndarray,
    enhanced_bss: np.ndarray,
    gate_delta: float = 0.001,
    gate_min_folds: int = 3,
    worst_fold_max_degradation: float = 0.0005,
) -> tuple[str, str]:
    """Evaluate OWA gate. Returns ('PASS'/'FAIL', reason_str)."""
    deltas = enhanced_bss - baseline_bss
    n_improved = (deltas >= gate_delta).sum()
    worst_delta = deltas.min()

    if n_improved < gate_min_folds:
        return "FAIL", f"only {n_improved}/{len(deltas)} folds improved by >= +{gate_delta}"
    if worst_delta < -worst_fold_max_degradation:
        return "FAIL", f"worst fold degraded by {worst_delta:.5f} (limit: -{worst_fold_max_degradation})"
    return "PASS", f"{n_improved}/{len(deltas)} folds improved, worst_delta={worst_delta:.5f}"
```

- [ ] **Step 4: Wire OWA into `PatternMatcher._prepare_features()`**

In `matcher.py`, at the end of `_prepare_features()` (after `apply_feature_weights`), add:

```python
# OWA: compute MI-based weights from training data (fit_scaler=True path only)
# During query (fit_scaler=False), use weights computed during fit.
# Weights are stored in self.config.feature_weights after fit.
# No change needed here — OWA populates cfg.feature_weights during fit().
```

In `fit()`, after `X_weighted = self._prepare_features(X_raw, fit_scaler=True)` and before `_build_index()`, add:

```python
# E2: OWA feature weighting — compute MI-based global weights on training data
if getattr(cfg, 'use_owa', False):
    from pattern_engine.owa_weights import mi_to_weight_dict
    _y_train = (
        train_db[cfg.projection_horizon].values.astype(np.float64)
        if cfg.projection_horizon in train_db.columns
        else np.zeros(len(train_db), dtype=np.float64)
    )
    # IMPORTANT: compute MI on X_weighted (already scaled), NOT on X_raw.
    # The index is built on scaled features — MI ranking must match that space.
    cfg.feature_weights = mi_to_weight_dict(
        feature_cols, X_weighted, _y_train, alpha=getattr(cfg, 'owa_alpha', 1.0)
    )
    # Re-run prepare_features with OWA weights applied (scaler already fitted)
    X_weighted = self._prepare_features(X_raw, fit_scaler=False)
```

- [ ] **Step 5: Run the OWA tests**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_phase7_e2_owa.py -v
```

Expected: all pass.

- [ ] **Step 6: Run full suite**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"
```

Expected: 846+ passed.

- [ ] **Step 7: Write `scripts/phase7_e2_owa_weighting.py`**

Pattern: same as E1 script. Key additions:
- Alpha sweep: [0.5, 1.0, 2.0, 4.0] via leave-one-fold-out CV (use 5 folds as inner train, evaluate on held-out 6th fold → pick alpha with best mean held-out BSS)
- AvgK check after applying OWA weights: compute mean neighbours per query. If any fold < 20, print warning and record new max_distance if re-sweep was needed.
- Gate: `evaluate_owa_gate(baseline_bss_array, owa_bss_array)`
- Append to `enhancement_summary.tsv`

- [ ] **Step 8: Run E2 walk-forward**

```
PYTHONUTF8=1 py -3.12 scripts/phase7_e2_owa_weighting.py
```

- [ ] **Step 9: Apply gate result**

**PASS:** `use_owa: bool = True`, `owa_alpha: float = <selected_alpha>` in config.py. Update CLAUDE.md.
**FAIL:** Keep False. One-line CLAUDE.md note.

- [ ] **Step 10: Commit**

```
git add pattern_engine/owa_weights.py tests/unit/test_phase7_e2_owa.py
git add scripts/phase7_e2_owa_weighting.py pattern_engine/matcher.py
git add pattern_engine/config.py CLAUDE.md results/phase7/
git commit -m "feat(phase7/e2): OWA feature weighting — PASS/FAIL"
```

---

## Task 4: T7.3 — E3: DTW Reranker

**Files:**
- Create: `research/wfa_reranker.py`
- Create: `tests/unit/test_phase7_e3_dtw.py`
- Create: `scripts/phase7_e3_dtw_reranker.py`
- Create: `scripts/benchmarks/b5_dtw_vs_euclidean.py`
- Modify: `pattern_engine/matcher.py`

**Critical path:** Check Spearman ρ on 100 random queries BEFORE running full walk-forward. If ρ > 0.95, FAIL immediately.

- [ ] **Step 1: Install `dtw-python` if not present**

```
py -3.12 -m pip install dtw-python
```

Verify: `py -3.12 -c "from dtw import dtw; print('dtw ok')"`. If import fails, try `pip install dtaidistance` as fallback (uses `dtaidistance.dtw.distance` instead).

- [ ] **Step 2: Write failing tests**

```python
# tests/unit/test_phase7_e3_dtw.py
"""TDD tests for E3: DTW Reranker."""
from __future__ import annotations
import numpy as np
import pytest
import time


class TestDTWReranker:
    def test_reranker_output_has_k_elements(self):
        """dtw_rerank returns exactly k neighbour indices."""
        from research.wfa_reranker import dtw_rerank
        rng = np.random.RandomState(0)
        query_23d = rng.randn(23)
        neighbours_23d = rng.randn(50, 23)
        indices = np.arange(50)
        reranked_idx, reranked_dists = dtw_rerank(query_23d, neighbours_23d, indices, k=20)
        assert len(reranked_idx) == 20
        assert len(reranked_dists) == 20

    def test_dtw_uses_return_columns_only(self):
        """DTW distance ignores columns 8:23 (candlestick proportions)."""
        from research.wfa_reranker import dtw_rerank
        rng = np.random.RandomState(1)
        query_23d = rng.randn(23)
        neighbours_23d = rng.randn(50, 23)
        indices = np.arange(50)
        # Perturb only candlestick columns — ranking should not change
        neighbours_perturbed = neighbours_23d.copy()
        neighbours_perturbed[:, 8:] *= 1000
        idx1, _ = dtw_rerank(query_23d, neighbours_23d, indices, k=20)
        idx2, _ = dtw_rerank(query_23d, neighbours_perturbed, indices, k=20)
        np.testing.assert_array_equal(idx1, idx2)

    def test_dtw_latency_under_500ms_per_50_computations(self):
        """50 DTW computations on 8-point series completes in < 500ms."""
        from research.wfa_reranker import dtw_rerank
        rng = np.random.RandomState(2)
        query_23d = rng.randn(23)
        neighbours_23d = rng.randn(50, 23)
        indices = np.arange(50)
        t0 = time.time()
        dtw_rerank(query_23d, neighbours_23d, indices, k=20)
        elapsed = time.time() - t0
        assert elapsed < 0.5, f"DTW too slow: {elapsed:.3f}s for 50 computations"

    def test_spearman_fast_fail_threshold(self):
        """Gate logic: if Spearman rho > 0.95, fail without full walk-forward."""
        from scipy.stats import spearmanr
        # Identical rankings → rho = 1.0
        euclidean_ranks = np.arange(50)
        dtw_ranks = np.arange(50)
        rho, _ = spearmanr(euclidean_ranks, dtw_ranks)
        assert rho > 0.95  # should trigger fast-fail


class TestDTWGate:
    def test_gate_passes_with_bss_improvement(self):
        """Gate passes when >=3 folds improve by >=+0.001."""
        baseline = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        enhanced = np.array([0.003, 0.003, 0.003, 0.001, 0.001, 0.001])
        deltas = enhanced - baseline
        assert (deltas >= 0.001).sum() >= 3
```

- [ ] **Step 3: Run failing tests**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_phase7_e3_dtw.py -v
```

Expected: ImportError on `research.wfa_reranker`.

- [ ] **Step 4: Create `research/wfa_reranker.py`**

```python
"""
research/wfa_reranker.py — DTW post-retrieval reranker for E3.

Reranks top-50 Euclidean neighbours by DTW distance on the 8 return
columns only (indices 0:8). Candlestick columns (8:23) are excluded —
bounded ratios have no meaningful temporal warping.

Usage:
    reranked_idx, reranked_dists = dtw_rerank(query_23d, nbrs_23d, indices, k=20)
"""
from __future__ import annotations
import numpy as np

RETURN_COLS_SLICE = slice(0, 8)  # Only the 8 VOL_NORM return columns


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic DTW distance between two equal-length 1D sequences."""
    try:
        from dtw import dtw as dtw_fn
        result = dtw_fn(a, b, keep_internals=False)
        return float(result.distance)
    except ImportError:
        pass
    # Fallback: dtaidistance
    try:
        from dtaidistance import dtw as dtai_dtw
        return float(dtai_dtw.distance(a, b))
    except ImportError:
        pass
    # Last resort: Euclidean distance (no warping — will likely trigger Spearman fast-fail)
    return float(np.linalg.norm(a - b))


def dtw_rerank(
    query_23d: np.ndarray,
    neighbours_23d: np.ndarray,
    neighbour_indices: np.ndarray,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Rerank top-50 neighbours by DTW on return columns.

    Args:
        query_23d:          (23,) full feature vector.
        neighbours_23d:     (50, 23) top-50 neighbour features.
        neighbour_indices:  (50,) original training indices.
        k:                  Number of neighbours to return.

    Returns:
        (reranked_indices, reranked_dtw_distances): both shape (k,).
    """
    query_ret = query_23d[RETURN_COLS_SLICE]            # (8,)
    nbr_ret   = neighbours_23d[:, RETURN_COLS_SLICE]    # (N, 8)
    n = len(neighbour_indices)

    dtw_dists = np.array([
        _dtw_distance(query_ret, nbr_ret[i]) for i in range(n)
    ])

    order = np.argsort(dtw_dists)[:k]
    return neighbour_indices[order], dtw_dists[order]


def check_spearman_fast_fail(
    matcher,
    val_db,
    n_queries: int = 100,
    rho_threshold: float = 0.95,
) -> tuple[bool, float]:
    """Check if DTW ordering is near-identical to Euclidean (fast-fail gate).

    Returns:
        (should_fail, mean_rho): True if mean Spearman rho > rho_threshold.
    """
    from scipy.stats import spearmanr
    import numpy as np

    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(val_db), size=min(n_queries, len(val_db)), replace=False)
    sample_db = val_db.iloc[sample_idx]

    rhos = []
    feature_cols = matcher._feature_cols
    X_raw = sample_db[feature_cols].values
    X_scaled = matcher._prepare_features(X_raw, fit_scaler=False)

    for i in range(len(X_scaled)):
        distances_b, indices_b = matcher._query_batch(X_scaled[i:i+1])
        euclidean_order = np.argsort(distances_b[0])

        nbrs_scaled = matcher._X_train_weighted[indices_b[0]]
        _, dtw_dists = dtw_rerank(X_scaled[i], nbrs_scaled, indices_b[0], k=len(indices_b[0]))
        dtw_order = np.argsort(dtw_dists)

        rho, _ = spearmanr(euclidean_order[:50], dtw_order[:50])
        rhos.append(rho)

    mean_rho = float(np.mean(rhos))
    return mean_rho > rho_threshold, mean_rho
```

- [ ] **Step 5: Wire DTW into `matcher.py`**

In `_package_results()`, add DTW reranking BEFORE probability computation:

```python
# E3: DTW reranker — rerank accepted neighbours before prob computation
if getattr(cfg, 'use_dtw_reranker', False) and self._X_train_weighted is not None:
    from research.wfa_reranker import dtw_rerank
    _k = getattr(cfg, 'dtw_rerank_k', 20)
    # Rebuild top_mask for DTW-reranked neighbours
    _new_mask = np.zeros_like(top_mask)
    for i in range(B):
        _accepted_idx = indices_b[i][top_mask[i]]
        if len(_accepted_idx) == 0:
            continue
        _nbrs_feat = self._X_train_weighted[_accepted_idx]  # (n_acc, D)
        _reranked, _ = dtw_rerank(X_batch[i], _nbrs_feat, _accepted_idx, k=_k)
        # Mark reranked indices as True in new mask
        for orig_pos in range(indices_b.shape[1]):
            if indices_b[i, orig_pos] in _reranked:
                _new_mask[i, orig_pos] = True
    top_mask = _new_mask
```

Note: `_package_results` doesn't currently receive `X_batch`. Thread it from `query()`. **Batch loop awareness:** `query()` iterates over `val_db` in slices of `batch_size`. `X_batch` and `indices_b` are already per-batch variables inside that loop — thread them directly into `_package_results` in each loop iteration. The DTW reranking `for i in range(B)` loop inside `_package_results` already operates on the per-batch `indices_b[i]` slice correctly. Results accumulate in the existing `all_*` lists that `query()` concatenates after the batch loop — no additional concatenation logic is needed.

- [ ] **Step 6: Run DTW tests and full suite**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_phase7_e3_dtw.py -v
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"
```

- [ ] **Step 7: Write E3 walk-forward script**

In `scripts/phase7_e3_dtw_reranker.py`:
1. Load fold 1 only
2. Run `check_spearman_fast_fail()` — if mean ρ > 0.95, print "FAST-FAIL: DTW ≈ Euclidean" and write FAIL row to summary TSV. Exit.
3. If ρ ≤ 0.95, run full 6-fold walk-forward with `use_dtw_reranker=True`
4. Evaluate gate: BSS delta ≥ +0.001 on ≥ 3/6 folds AND runtime < 60s additional
5. Write `results/phase7/e3_dtw_vs_baseline.tsv`
6. Append to `enhancement_summary.tsv`

Also write `scripts/benchmarks/b5_dtw_vs_euclidean.py` — three configs on 2024 fold only:
- Config 1: KNN, no calibration (`calibration_method="none"`)
- Config 2: KNN + beta_abm, no DTW
- Config 3: KNN + beta_abm + DTW

- [ ] **Step 8: Run E3**

```
PYTHONUTF8=1 py -3.12 scripts/phase7_e3_dtw_reranker.py
PYTHONUTF8=1 py -3.12 scripts/benchmarks/b5_dtw_vs_euclidean.py
```

- [ ] **Step 9: Apply gate result and commit**

```
git add research/wfa_reranker.py tests/unit/test_phase7_e3_dtw.py
git add scripts/phase7_e3_dtw_reranker.py scripts/benchmarks/b5_dtw_vs_euclidean.py
git add pattern_engine/matcher.py pattern_engine/config.py CLAUDE.md results/
git commit -m "feat(phase7/e3): DTW reranker — PASS/FAIL"
```

---

## Task 5: T7.4 — E4: Conformal Prediction Intervals

**Files:**
- Modify: `pattern_engine/conformal_hooks.py`
- Create: `tests/unit/test_phase7_e4_conformal.py`
- Create: `scripts/phase7_e4_conformal.py`

**Context:** The existing `NaiveConformalCalibrator` is explicitly marked as a stub without coverage guarantees. Promote `conformal_hooks.py` to contain a real `AdaptiveConformalPredictor` (ACI — Gibbs & Candès, 2021). Keep the existing stub classes for backward compatibility.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_phase7_e4_conformal.py
"""TDD tests for E4: Adaptive Conformal Prediction."""
from __future__ import annotations
import numpy as np
import pytest


class TestAdaptiveConformalPredictor:
    def _make_predictor(self):
        from pattern_engine.conformal_hooks import AdaptiveConformalPredictor
        return AdaptiveConformalPredictor(nominal_alpha=0.10, gamma=0.05)

    def test_calibrate_and_predict_interval(self):
        """predict_interval returns (lower, upper) with lower < upper."""
        pred = self._make_predictor()
        rng = np.random.RandomState(0)
        cal_probs = rng.uniform(0.4, 0.7, 200)
        cal_labels = rng.randint(0, 2, 200).astype(float)
        pred.calibrate(cal_probs, cal_labels)
        lo, hi = pred.predict_interval(0.6)
        assert lo < hi
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_interval_width_positive(self):
        """Interval width > 0 for all test probs."""
        pred = self._make_predictor()
        rng = np.random.RandomState(1)
        pred.calibrate(rng.uniform(0, 1, 200), rng.randint(0, 2, 200).astype(float))
        for p in [0.4, 0.5, 0.6, 0.7]:
            lo, hi = pred.predict_interval(p)
            assert hi - lo > 0

    def test_aci_alpha_adjusts_after_update(self):
        """alpha_t changes after calling update()."""
        pred = self._make_predictor()
        rng = np.random.RandomState(2)
        pred.calibrate(rng.uniform(0, 1, 200), rng.randint(0, 2, 200).astype(float))
        alpha_before = pred.alpha_t
        pred.update(0.7, 1)
        assert pred.alpha_t != alpha_before

    def test_coverage_on_well_calibrated_data(self):
        """Coverage >= 88% on synthetic well-calibrated data at nominal 90%."""
        pred = self._make_predictor()
        rng = np.random.RandomState(3)
        # Well-calibrated: prob near true frequency
        probs = rng.uniform(0.3, 0.8, 500)
        labels = (rng.uniform(size=500) < probs).astype(float)
        # Use first 250 for calibration, last 250 for test
        pred.calibrate(probs[:250], labels[:250])
        covered = 0
        for p, y in zip(probs[250:], labels[250:]):
            lo, hi = pred.predict_interval(p)
            if lo <= y <= hi:
                covered += 1
        coverage = covered / 250
        assert coverage >= 0.88, f"Coverage {coverage:.3f} < 0.88"

    def test_mean_interval_width_computable(self):
        """mean_interval_width() returns a float."""
        pred = self._make_predictor()
        rng = np.random.RandomState(4)
        pred.calibrate(rng.uniform(0, 1, 200), rng.randint(0, 2, 200).astype(float))
        test_probs = rng.uniform(0.4, 0.7, 50)
        width = pred.mean_interval_width(test_probs)
        assert isinstance(width, float)
        assert width > 0


class TestConformalGate:
    def test_gate_requires_all_folds_above_88(self):
        """Gate fails if any single fold has coverage < 88%."""
        coverages = np.array([0.91, 0.89, 0.92, 0.87, 0.90, 0.93])
        gate = (coverages >= 0.88).all()
        assert not gate  # fold 4 has 0.87 < 0.88

    def test_gate_passes_when_all_folds_above_88(self):
        coverages = np.array([0.91, 0.89, 0.92, 0.90, 0.90, 0.93])
        widths = np.array([0.20, 0.22, 0.18, 0.25, 0.21, 0.19])
        gate = (coverages >= 0.88).all() and (widths < 0.30).all()
        assert gate
```

- [ ] **Step 2: Run failing tests**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_phase7_e4_conformal.py -v
```

Expected: ImportError on `AdaptiveConformalPredictor`.

- [ ] **Step 3: Add `AdaptiveConformalPredictor` to `conformal_hooks.py`**

Append to the existing file (do NOT remove existing classes):

```python
class AdaptiveConformalPredictor:
    """Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021).

    Provides distribution-free prediction intervals for time-series data
    by dynamically adjusting the quantile level based on recent coverage errors.

    Args:
        nominal_alpha: Target miscoverage rate (default 0.10 → 90% coverage).
        gamma: Learning rate for alpha_t adjustment (default 0.05).
    """

    def __init__(self, nominal_alpha: float = 0.10, gamma: float = 0.05) -> None:
        self.nominal_alpha = nominal_alpha
        self.gamma = gamma
        self.alpha_t = nominal_alpha
        self._scores: Optional[np.ndarray] = None

    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> None:
        """Compute non-conformity scores on calibration set.

        Score = |predicted_prob - actual_outcome|
        """
        self._scores = np.sort(np.abs(cal_probs - cal_labels))

    def predict_interval(self, prob: float) -> Tuple[float, float]:
        """Return (lower, upper) prediction interval for a single probability."""
        if self._scores is None:
            raise RuntimeError("calibrate() must be called before predict_interval().")
        n = len(self._scores)
        idx = int(np.ceil((1 - self.alpha_t) * (n + 1))) - 1
        idx = int(np.clip(idx, 0, n - 1))
        threshold = float(self._scores[idx])
        lower = max(0.0, prob - threshold)
        upper = min(1.0, prob + threshold)
        return lower, upper

    def update(self, prob: float, actual: int) -> None:
        """ACI online update: adjust alpha_t based on observed coverage error."""
        lower, upper = self.predict_interval(prob)
        err = 1 if not (lower <= float(actual) <= upper) else 0
        self.alpha_t = self.alpha_t + self.gamma * (err - self.nominal_alpha)
        self.alpha_t = float(np.clip(self.alpha_t, 0.01, 0.50))

    def mean_interval_width(self, test_probs: np.ndarray) -> float:
        """Compute mean interval width over a set of test probabilities."""
        widths = [
            self.predict_interval(float(p))[1] - self.predict_interval(float(p))[0]
            for p in test_probs
        ]
        return float(np.mean(widths))
```

- [ ] **Step 4: Run conformal tests and full suite**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_phase7_e4_conformal.py -v
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"
```

- [ ] **Step 5: Write `scripts/phase7_e4_conformal.py`**

For each fold:
1. Fit matcher on training data (beta_abm, locked config)
2. Split training data into fit portion and calibration portion (use cal_frac=0.76)
3. Get calibrated probs for calibration portion
4. Sweep gamma = [0.01, 0.05, 0.10] — for each gamma:
   - `pred.calibrate(cal_probs, cal_labels)`
   - For each validation row (in temporal order):
     - `lo, hi = pred.predict_interval(prob)`
     - Check coverage (does true outcome fall in [lo, hi]?)
     - `pred.update(prob, actual)` (ACI step)
   - Record: empirical_coverage, mean_interval_width
5. Select gamma with coverage closest to 0.90
6. Gate: coverage >= 0.88 on ALL 6 folds AND mean_width < 0.30
7. Write `results/phase7/e4_conformal_coverage.tsv`
8. Append to `enhancement_summary.tsv`
9. If PASS: document Phase 8 integration point in a comment (do NOT modify `position_sizer.py`)

- [ ] **Step 6: Run E4**

```
PYTHONUTF8=1 py -3.12 scripts/phase7_e4_conformal.py
```

- [ ] **Step 7: Apply gate and commit**

```
git add pattern_engine/conformal_hooks.py tests/unit/test_phase7_e4_conformal.py
git add scripts/phase7_e4_conformal.py pattern_engine/config.py CLAUDE.md results/
git commit -m "feat(phase7/e4): adaptive conformal prediction — PASS/FAIL"
```

---

## Task 6: T7.5 — E5: LOF Anomaly Filter

**Files:**
- Create: `pattern_engine/anomaly_filter.py`
- Create: `tests/unit/test_phase7_e5_cpod.py`
- Create: `scripts/phase7_e5_cpod_filter.py`
- Modify: `pattern_engine/signal_pipeline.py`

**Critical:** This is NOT a 6-fold BSS walk-forward. Evaluated on 585T Platt signals from `results/cached_signals_2024.csv` (159 BUY signals). Read that CSV and understand its columns first.

- [ ] **Step 1: Inspect the signals CSV and locate training data**

```
PYTHONUTF8=1 py -3.12 -c "import pandas as pd; df = pd.read_csv('results/cached_signals_2024.csv'); print(df.columns.tolist()); print(df.head(3)); print(df.shape)"
```

Note: identify columns for signal direction, confidence, and query features (the 23D feature vector columns needed to fit LOF).

Then locate available training data:
```
PYTHONUTF8=1 py -3.12 -c "from pathlib import Path; print([str(p) for p in Path('data').iterdir() if p.is_dir()])"
```

If `data/585t_features/` exists → use it for LOF training (full universe, 2024 fold).
If only `data/52t_features/` exists → use the 52T training data for LOF fit. LOF learns the training distribution; the 52T features still characterize what "normal" patterns look like, even if the signals themselves come from 585T. Document which data source was used in the TSV.

- [ ] **Step 2: Write failing tests**

```python
# tests/unit/test_phase7_e5_cpod.py
"""TDD tests for E5: LOF Anomaly Filter."""
from __future__ import annotations
import numpy as np
import pytest


class TestAnomalyFilter:
    def _make_filter(self, contamination=0.05, penalty=0.50):
        from pattern_engine.anomaly_filter import AnomalyFilter
        return AnomalyFilter(contamination=contamination, penalty=penalty)

    def test_filter_preserves_normal_confidence(self):
        """Inlier query keeps original confidence."""
        filt = self._make_filter()
        rng = np.random.RandomState(0)
        X_train = rng.randn(500, 23)
        filt.fit(X_train)
        # Query identical to a training point (should be inlier)
        normal_query = X_train[0].copy()
        conf = filt.apply_to_query(normal_query, confidence=0.70)
        assert conf == 0.70  # unchanged

    def test_filter_reduces_outlier_confidence(self):
        """Outlier query gets penalized confidence."""
        filt = self._make_filter(contamination=0.10, penalty=0.50)
        rng = np.random.RandomState(1)
        X_train = rng.randn(500, 23)
        filt.fit(X_train)
        # Far-off-distribution query
        outlier_query = np.full(23, 100.0)
        conf_original = 0.70
        conf_filtered = filt.apply_to_query(outlier_query, confidence=conf_original)
        assert conf_filtered < conf_original

    def test_anomaly_rate_below_30pct(self):
        """< 30% of normal distribution queries flagged as anomalies."""
        filt = self._make_filter(contamination=0.05)
        rng = np.random.RandomState(2)
        X_train = rng.randn(500, 23)
        filt.fit(X_train)
        X_test = rng.randn(200, 23)
        n_anomalies = sum(
            1 for x in X_test if filt.apply_to_query(x, 0.70) < 0.70
        )
        assert n_anomalies / 200 < 0.30

    def test_lof_requires_fit_before_apply(self):
        """apply_to_query before fit raises RuntimeError."""
        filt = self._make_filter()
        with pytest.raises(RuntimeError):
            filt.apply_to_query(np.zeros(23), 0.70)


class TestCPODGate:
    def test_gate_requires_fpr_reduction(self):
        """Gate fails if FPR reduction < 5%."""
        original_fpr = 0.20
        filtered_fpr = 0.18  # only 10% reduction — gate requires 25% reduction (5pp)
        reduction_pct_points = original_fpr - filtered_fpr
        assert not (reduction_pct_points >= 0.05)

    def test_gate_rejects_high_tpr_loss(self):
        """Gate fails if TPR loss > 2 percentage points."""
        original_tpr = 0.80
        filtered_tpr = 0.77  # 3pp loss
        loss_pct_points = original_tpr - filtered_tpr
        assert loss_pct_points > 0.02

    def test_gate_passes_on_good_results(self):
        """Gate passes with >=5pp FPR reduction and <=2pp TPR loss."""
        original_fpr, filtered_fpr = 0.20, 0.14   # 6pp reduction ✓
        original_tpr, filtered_tpr = 0.80, 0.79   # 1pp loss ✓
        fpr_reduction = original_fpr - filtered_fpr
        tpr_loss = original_tpr - filtered_tpr
        assert fpr_reduction >= 0.05 and tpr_loss <= 0.02
```

- [ ] **Step 3: Create `pattern_engine/anomaly_filter.py`**

```python
"""
pattern_engine/anomaly_filter.py — LOF-based anomaly pre-filter for E5.

Detects query patterns far from the training distribution and applies a
confidence penalty. Reduces false positive rate on production signals.
"""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from pattern_engine.signal_filter_base import SignalFilterBase
import pandas as pd


class AnomalyFilter(SignalFilterBase):
    """LOF-based anomaly detection pre-filter.

    Args:
        contamination: Expected fraction of outliers (default 0.05).
        penalty: Confidence multiplier for flagged outliers (default 0.50).
                 Set to 0.0 to suppress outliers entirely.
    """

    def __init__(self, contamination: float = 0.05, penalty: float = 0.50) -> None:
        self.contamination = contamination
        self.penalty = penalty
        self._lof: LocalOutlierFactor | None = None
        self._fitted = False

    def fit(self, X_train: np.ndarray) -> "AnomalyFilter":
        """Fit LOF on training feature matrix.

        Args:
            X_train: (N, D) training features.
        """
        # novelty=True: enables predict() on new (unseen) points
        self._lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True,
        )
        # LOF is O(N^2) — subsample if N > 10_000 to stay tractable
        if len(X_train) > 10_000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), 10_000, replace=False)
            self._lof.fit(X_train[idx])
        else:
            self._lof.fit(X_train)
        self._fitted = True
        return self

    def apply_to_query(self, query_features: np.ndarray, confidence: float) -> float:
        """Return adjusted confidence. Outliers get multiplied by penalty.

        Args:
            query_features: (D,) feature vector for a single query.
            confidence: Original confidence score.

        Returns:
            Adjusted confidence (unchanged for inliers, penalized for outliers).
        """
        if not self._fitted:
            raise RuntimeError("AnomalyFilter.fit() must be called first.")
        pred = self._lof.predict(query_features.reshape(1, -1))
        if pred[0] == -1:  # outlier
            return confidence * self.penalty
        return confidence

    # SignalFilterBase interface
    def apply(
        self,
        probs: np.ndarray,
        signals: list[str],
        val_db: pd.DataFrame,
        feature_cols: list[str] | None = None,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """Apply anomaly filter to a batch of signals.

        Signals where the query is an outlier AND confidence < threshold
        are reverted to HOLD.
        """
        if not self._fitted or feature_cols is None:
            return signals, np.zeros(len(signals), dtype=bool)

        changed = np.zeros(len(signals), dtype=bool)
        new_signals = list(signals)
        threshold = kwargs.get("confidence_threshold", 0.65)

        for i, (prob, sig) in enumerate(zip(probs, signals)):
            if sig != "BUY":
                continue
            features = val_db.iloc[i][feature_cols].values.astype(float)
            adj_prob = self.apply_to_query(features, float(prob))
            if adj_prob < threshold:
                new_signals[i] = "HOLD"
                changed[i] = True

        return new_signals, changed
```

- [ ] **Step 4: Run failing tests**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_phase7_e5_cpod.py -v
```

Expected: mostly pass after module creation. Fix any interface issues.

- [ ] **Step 5: Write `scripts/phase7_e5_cpod_filter.py`**

```python
"""
scripts/phase7_e5_cpod_filter.py — E5: LOF anomaly filter on 585T signals.

Evaluates FPR/TPR impact of LOF pre-filtering on 159 BUY signals from
results/cached_signals_2024.csv.

Gate: FPR reduction >= 5pp, TPR loss <= 2pp, anomaly rate < 30%.
"""
# 1. Load cached_signals_2024.csv
# 2. Identify feature columns (the 23D returns_candle columns)
# 3. Load training data (585T, 2024 fold: train_end=2023-12-31)
# 4. Fit LOF on training features
# 5. For each signal: compute LOF prediction, apply penalty
# 6. Determine which signals still fire after penalty (conf >= 0.65)
# 7. Compute FPR/TPR before and after:
#    - TP = BUY signal that correctly predicted up-move
#    - FP = BUY signal that incorrectly predicted up-move
#    FPR = FP / (FP + TN);  TPR = TP / (TP + FN)
# 8. Sweep: contamination in [0.01, 0.05, 0.10, 0.20]
#           penalty in [0.00, 0.25, 0.50, 0.75]
#    16 combinations — pick best (max FPR reduction s.t. TPR loss <= 2pp)
# 9. Write results/phase7/e5_cpod_fpr_tpr.tsv
# 10. Append to enhancement_summary.tsv
```

Implement this fully. The 585T training data location may differ from 52T — check `data/` for available parquet directories.

- [ ] **Step 6: Run E5**

```
PYTHONUTF8=1 py -3.12 scripts/phase7_e5_cpod_filter.py
```

- [ ] **Step 7: Apply gate and commit**

```
git add pattern_engine/anomaly_filter.py tests/unit/test_phase7_e5_cpod.py
git add scripts/phase7_e5_cpod_filter.py pattern_engine/config.py CLAUDE.md results/
git commit -m "feat(phase7/e5): LOF anomaly filter — PASS/FAIL"
```

---

## Task 7: T7.6 — E6: STUMPY Matrix Profile

**Files:**
- Create: `research/stumpy_matcher.py`
- Create: `tests/unit/test_phase7_e6_stumpy.py`
- Create: `scripts/phase7_e6_stumpy.py`
- Create: `scripts/benchmarks/b6_stumpy_vs_knn.py`

**Primary risk:** Compute budget. Check fold 1 timing before running all 6. If fold 1 exceeds 60s after pre-computation optimization → FAIL immediately without spending more time.

- [ ] **Step 1: Install STUMPY**

```
py -3.12 -m pip install stumpy
py -3.12 -c "import stumpy; print('stumpy ok')"
```

- [ ] **Step 2: Write failing tests**

```python
# tests/unit/test_phase7_e6_stumpy.py
"""TDD tests for E6: STUMPY Matrix Profile."""
from __future__ import annotations
import numpy as np
import pytest
import time


class TestSTUMPYSignalGenerator:
    def test_returns_valid_probability(self):
        """Output in [0, 1]."""
        from research.stumpy_matcher import STUMPYSignalGenerator
        rng = np.random.RandomState(0)
        gen = STUMPYSignalGenerator(subsequence_length=10)
        query = rng.randn(100)
        candidates = [rng.randn(100) for _ in range(3)]
        outcomes = [rng.randint(0, 2, 100).astype(float) for _ in range(3)]
        p = gen.generate_signal(query, candidates, outcomes)
        assert 0.0 <= p <= 1.0

    def test_returns_base_rate_on_empty_candidates(self):
        """If no candidates, return 0.5."""
        from research.stumpy_matcher import STUMPYSignalGenerator
        gen = STUMPYSignalGenerator(subsequence_length=10)
        p = gen.generate_signal(np.zeros(100), [], [])
        assert p == 0.5

    def test_returns_base_rate_on_too_short_series(self):
        """If candidate shorter than subsequence_length, skip it → 0.5."""
        from research.stumpy_matcher import STUMPYSignalGenerator
        gen = STUMPYSignalGenerator(subsequence_length=50)
        query = np.zeros(100)
        candidates = [np.zeros(30)]  # too short
        outcomes = [np.zeros(30)]
        p = gen.generate_signal(query, candidates, outcomes)
        assert p == 0.5

    def test_compute_time_reasonable_on_small_data(self):
        """Single generate_signal call < 2s on short series."""
        from research.stumpy_matcher import STUMPYSignalGenerator
        rng = np.random.RandomState(1)
        gen = STUMPYSignalGenerator(subsequence_length=10)
        t0 = time.time()
        gen.generate_signal(
            rng.randn(200),
            [rng.randn(200) for _ in range(3)],
            [rng.randint(0, 2, 200).astype(float) for _ in range(3)],
        )
        assert time.time() - t0 < 2.0


class TestSTUMPYGate:
    def test_gate_requires_low_correlation(self):
        """Gate fails if KNN-STUMPY Pearson correlation >= 0.50."""
        knn_preds = np.array([0.6, 0.7, 0.5, 0.6, 0.7])
        stumpy_preds = np.array([0.6, 0.7, 0.5, 0.6, 0.7])  # identical → corr=1.0
        corr = np.corrcoef(knn_preds, stumpy_preds)[0, 1]
        assert corr >= 0.50  # should fail gate

    def test_gate_passes_on_complementary_signals(self):
        """Gate passes when correlation < 0.50."""
        rng = np.random.RandomState(0)
        knn = rng.randn(100)
        stumpy = rng.randn(100)  # independent → corr ≈ 0
        corr = abs(np.corrcoef(knn, stumpy)[0, 1])
        assert corr < 0.50
```

- [ ] **Step 3: Create `research/stumpy_matcher.py`**

```python
"""
research/stumpy_matcher.py — STUMPY AB-join cross-ticker pattern matching.

For a query ticker+date, finds conserved subsequence patterns in same-sector
tickers and derives a directional prediction from post-match outcomes.
"""
from __future__ import annotations
import numpy as np


class STUMPYSignalGenerator:
    """Cross-ticker pattern matching via STUMPY AB-join."""

    def __init__(self, subsequence_length: int = 50) -> None:
        self.m = subsequence_length

    def generate_signal(
        self,
        query_series: np.ndarray,
        candidate_series: list[np.ndarray],
        candidate_outcomes: list[np.ndarray],
    ) -> float:
        """Return probability estimate from matrix profile matches.

        Args:
            query_series:       (T,) return series for the query.
            candidate_series:   list of (T_i,) return series.
            candidate_outcomes: list of (T_i,) binary outcome arrays.

        Returns:
            p in [0.5, 0.5] if no matches, otherwise fraction of UP outcomes.
        """
        import stumpy

        all_outcomes = []
        for cand, fwd in zip(candidate_series, candidate_outcomes):
            if len(cand) < self.m + 1 or len(query_series) < self.m + 1:
                continue
            try:
                mp = stumpy.stump(query_series, cand, self.m)
                last_idx = len(mp) - 1
                match_idx = int(mp[last_idx, 1])
                match_dist = mp[last_idx, 0]
                if np.isfinite(match_dist) and match_idx + self.m < len(fwd):
                    all_outcomes.append(float(fwd[match_idx + self.m]))
            except Exception:
                continue

        if not all_outcomes:
            return 0.5
        return float(np.mean(all_outcomes))
```

- [ ] **Step 4: Run STUMPY tests**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_phase7_e6_stumpy.py -v
```

- [ ] **Step 5: Write E6 walk-forward script with compute budget check**

In `scripts/phase7_e6_stumpy.py`:
1. **Mini-benchmark first (before fold 1):** Run STUMPY on 5 random queries × 3 candidate tickers on a 200-bar price slice. Time it. Extrapolate to full fold 1 (≈ n_val_queries × n_candidates / 5 × t_per_5). If the extrapolation exceeds 120s, write FAIL to summary and exit immediately — do not run fold 1.
2. **Fold 1 compute check:** If mini-benchmark looks viable, run STUMPY on fold 1 only, time it. If actual fold 1 > 60s after any pre-computation optimization → write FAIL to summary, exit.
2. If compute OK, run all 6 folds.
3. Signal blend: `p_combined = (1 - stumpy_weight) * p_knn + stumpy_weight * p_stumpy`
4. Compute Pearson correlation between p_knn and p_stumpy per fold.
5. Gate: BSS ≥ +0.001 on ≥ 3/6 folds, mean correlation < 0.50, fold 1 < 30s.
6. Write `results/phase7/e6_stumpy_vs_baseline.tsv`
7. Append to `enhancement_summary.tsv`

Also write `scripts/benchmarks/b6_stumpy_vs_knn.py` — 3 configs on 2024 fold:
1. KNN only, 2. STUMPY only, 3. Blended

- [ ] **Step 6: Run E6**

```
PYTHONUTF8=1 py -3.12 scripts/phase7_e6_stumpy.py
PYTHONUTF8=1 py -3.12 scripts/benchmarks/b6_stumpy_vs_knn.py
```

- [ ] **Step 7: Apply gate and commit**

```
git add research/stumpy_matcher.py tests/unit/test_phase7_e6_stumpy.py
git add scripts/phase7_e6_stumpy.py scripts/benchmarks/b6_stumpy_vs_knn.py
git add pattern_engine/config.py CLAUDE.md results/
git commit -m "feat(phase7/e6): STUMPY matrix profile — PASS/FAIL"
```

---

## Task 8: Phase 7 Completion Gate

**Files:**
- Verify: `results/phase7/enhancement_summary.tsv` has 6 rows
- Verify: `CLAUDE.md` Current Phase updated
- Verify: test suite passes

- [ ] **Step 1: Verify `enhancement_summary.tsv`**

```
PYTHONUTF8=1 py -3.12 -c "
import pandas as pd
df = pd.read_csv('results/phase7/enhancement_summary.tsv', sep='\t')
print(df[['enhancement', 'gate_result']].to_string())
assert len(df) == 6, f'Expected 6 rows, got {len(df)}'
assert df['gate_result'].isin(['PASS', 'FAIL']).all()
print('CHECK 1+2: OK')
"
```

- [ ] **Step 2: Verify provenance files exist**

```
PYTHONUTF8=1 py -3.12 -c "
from pathlib import Path
import pandas as pd
df = pd.read_csv('results/phase7/enhancement_summary.tsv', sep='\t')
for _, row in df.iterrows():
    p = Path(row['provenance_file'])
    assert p.exists(), f'Missing: {p}'
print('CHECK 3: all provenance files present')
"
```

- [ ] **Step 3: Run final cumulative walk-forward**

Re-run `scripts/phase7_baseline.py` with all PASS flags enabled in config (set them temporarily or use a one-off config object). Record the final cumulative BSS. This is the number to report in CLAUDE.md.

- [ ] **Step 4: Update `CLAUDE.md` Current Phase section**

Replace the Phase 6 "Current Phase" block with:

```
**Phase 7 Model Enhancements — COMPLETE (2026-04-09)**
- E1 BMA:      PASS/FAIL (BSS delta X/6 folds, provenance: results/phase7/e1_bma_vs_beta_abm.tsv)
- E2 OWA:      PASS/FAIL (BSS delta X/6 folds, alpha=X, provenance: results/phase7/e2_owa_vs_baseline.tsv)
- E3 DTW:      PASS/FAIL (BSS delta X/6 folds, Spearman ρ=X, provenance: results/phase7/e3_dtw_vs_baseline.tsv)
- E4 Conformal: PASS/FAIL (coverage X%, provenance: results/phase7/e4_conformal_coverage.tsv)
- E5 CPOD:     PASS/FAIL (FPR reduction X%, provenance: results/phase7/e5_cpod_fpr_tpr.tsv)
- E6 STUMPY:   PASS/FAIL (BSS delta X/6 folds, corr=X, provenance: results/phase7/e6_stumpy_vs_baseline.tsv)
- Cumulative BSS: X.XXXXX (Y/6 positive folds)
- Tests passing: NNN
```

- [ ] **Step 5: Run final test suite**

```
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"
```

Expected: 900+ passed (846 baseline + ~6 new test classes × ~5 tests each).

- [ ] **Step 6: Final commit**

```
git add CLAUDE.md results/phase7/ results/benchmarks/
git commit -m "feat(phase7): COMPLETE — enhancement_summary and CLAUDE.md final update"
```

---

## Diagnostics Quick Reference

| Symptom | Action |
|---------|--------|
| BMA: NaN in transform output | Weights or variances collapsed. Check `_MIN_VAR` floor. |
| BMA: EM slow (>5min per fold) | Increase `_MIN_VAR`. Reduce `n_iter` to 15. |
| OWA: AvgK drops below 20 | Mini-sweep max_distance [2.5, 3.0, 3.5, 4.0]. |
| DTW: Spearman ρ > 0.95 | FAIL immediately. Expected outcome. |
| DTW: import error | Try `pip install dtaidistance` as fallback. |
| Conformal: coverage = 100% | Intervals trivially wide. Reduce `alpha_t` floor from 0.01 to 0.001. |
| LOF: anomaly rate > 30% | Reduce contamination to 0.01. |
| STUMPY: fold 1 > 60s | FAIL immediately. Document runtime in TSV. |
| Any: NaN/Inf in probs | Fix first, counts as one strike toward 3-strike rule. |
| Three strikes on one enhancement | STOP. flag=False. Log in session log. Move on. |
