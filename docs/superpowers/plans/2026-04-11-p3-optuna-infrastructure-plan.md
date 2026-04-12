# R3 Optuna Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract walk-forward logic from `scripts/phase7_baseline.py` into production modules and build a general-purpose Optuna/Grid sweep framework for all future hyperparameter experiments.

**Architecture:** Three new modules in `pattern_engine/`: `walkforward.py` (fold runner + data loading), `sweep.py` (OptunaSweep + GridSweep), `experiment_log.py` (TSV provenance logger). `phase7_baseline.py` becomes a thin re-export wrapper so E1–E4 scripts continue working unchanged.

**Tech Stack:** Python 3.12, Optuna ≥3.4.0, betacal, scipy.stats.wilcoxon, pandas, numpy. All already installed in venv.

**Spec:** `docs/superpowers/specs/2026-04-10-optuna-infrastructure-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `pattern_engine/walkforward.py` | Data loading, fold splitting, BSS/Murphy computation, beta_abm calibrator, H7 HOLD regime, candlestick augmentation. All extracted from `scripts/phase7_baseline.py` |
| `pattern_engine/sweep.py` | `OptunaSweep` (Bayesian via TPE), `GridSweep` (exhaustive), `SweepResult` dataclass, pre-built search spaces |
| `pattern_engine/experiment_log.py` | `ExperimentLogger` — incremental TSV writer with metadata headers and trial rows |
| `tests/unit/conftest.py` | Shared synthetic database fixtures for walkforward/sweep tests |
| `tests/unit/test_walkforward.py` | Unit tests for walkforward module (~12 tests) |
| `tests/unit/test_sweep.py` | Unit tests for sweep module (~10 tests) |
| `tests/unit/test_experiment_log.py` | Unit tests for experiment_log module (~5 tests) |

### Modified Files

| File | Change |
|------|--------|
| `scripts/phase7_baseline.py` | Replace inline logic with imports from `pattern_engine.walkforward`; keep `run_fold_with_config()` as thin wrapper + re-exports |

---

## Task 0: Test Fixtures

**Files:**
- Create: `tests/unit/conftest.py`

This task creates shared pytest fixtures that Tasks 1–5 depend on. The synthetic DB must include all columns walkforward needs: `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `ret_Xd` (8 return features), `ret_90d`, `fwd_7d_up`, plus enough rows/tickers to exercise fold splitting and regime logic.

- [ ] **Step 0.1: Create conftest.py with synthetic_full_db fixture**

```python
# tests/unit/conftest.py
"""Shared fixtures for walkforward / sweep / experiment_log tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


_RET_WINDOWS = [1, 3, 7, 14, 30, 45, 60, 90]


def _make_synthetic_db(
    n_tickers: int = 3,
    start: str = "2017-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal synthetic database with all columns walkforward needs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    tickers = [f"T{i}" for i in range(1, n_tickers + 1)]
    # Add SPY — required for H7 HOLD regime
    tickers = ["SPY"] + tickers

    rows = []
    for ticker in tickers:
        for dt in dates:
            row = {"Date": dt, "Ticker": ticker}
            row["Open"] = 100.0 + rng.normal(0, 5)
            row["High"] = row["Open"] + abs(rng.normal(1, 0.5))
            row["Low"] = row["Open"] - abs(rng.normal(1, 0.5))
            row["Close"] = row["Open"] + rng.normal(0, 2)
            for w in _RET_WINDOWS:
                row[f"ret_{w}d"] = rng.normal(0, 0.02 * np.sqrt(w))
            # ret_90d used by H7 HOLD regime — make SPY oscillate around threshold
            if ticker == "SPY":
                row["ret_90d"] = rng.choice([0.02, 0.08], p=[0.3, 0.7])
            row["fwd_7d_up"] = float(rng.random() > 0.5)
            rows.append(row)

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@pytest.fixture
def synthetic_full_db() -> pd.DataFrame:
    """Full synthetic DB spanning 2017-2024 with SPY + 3 tickers."""
    return _make_synthetic_db()


@pytest.fixture
def small_synthetic_db() -> pd.DataFrame:
    """Smaller DB (2 tickers, 2018-2020) for fast sweep tests."""
    return _make_synthetic_db(n_tickers=2, start="2018-01-01", end="2020-12-31", seed=99)
```

- [ ] **Step 0.2: Verify fixture loads without error**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/conftest.py --collect-only -q`
Expected: "no tests ran" (conftest has no tests, just fixtures) — no import errors.

- [ ] **Step 0.3: Commit**

```bash
git add tests/unit/conftest.py
git commit -m "test: add synthetic DB fixtures for P3 walkforward/sweep tests"
```

---

## Task 1: walkforward.py — BSS, Murphy, BetaCalibrator

**Files:**
- Create: `pattern_engine/walkforward.py`
- Create: `tests/unit/test_walkforward.py`

Extract the pure-computation helpers first: `_bss()`, `_murphy_decomposition()`, `_BetaCalibrator`. These have no dependencies on fold splitting or data loading, making them testable in isolation with synthetic data.

- [ ] **Step 1.1: Write failing tests for _bss()**

```python
# tests/unit/test_walkforward.py
"""Tests for pattern_engine.walkforward module."""
from __future__ import annotations

import numpy as np
import pytest


class TestBSS:
    """Tests for _bss() Brier Skill Score computation."""

    def test_perfect_predictions(self):
        """Perfect predictions → BSS = 1.0."""
        from pattern_engine.walkforward import _bss
        y = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        probs = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        assert _bss(probs, y) == pytest.approx(1.0)

    def test_climatological_predictions(self):
        """Base-rate predictions → BSS = 0.0."""
        from pattern_engine.walkforward import _bss
        y = np.array([1.0, 0.0, 1.0, 0.0])
        base = y.mean()  # 0.5
        probs = np.full(len(y), base)
        assert _bss(probs, y) == pytest.approx(0.0, abs=1e-10)

    def test_worse_than_climatology(self):
        """Inverted predictions → BSS < 0."""
        from pattern_engine.walkforward import _bss
        y = np.array([1.0, 0.0, 1.0, 0.0])
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        assert _bss(probs, y) < 0.0

    def test_constant_labels_returns_zero(self):
        """All-same labels → base_rate variance ~0 → BSS = 0.0."""
        from pattern_engine.walkforward import _bss
        y = np.array([1.0, 1.0, 1.0, 1.0])
        probs = np.array([0.8, 0.9, 0.7, 0.85])
        assert _bss(probs, y) == pytest.approx(0.0)
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_walkforward.py -v -x`
Expected: FAIL with `ModuleNotFoundError: No module named 'pattern_engine.walkforward'`

- [ ] **Step 1.3: Create walkforward.py with _bss, _murphy_decomposition, _BetaCalibrator**

```python
# pattern_engine/walkforward.py
"""
pattern_engine/walkforward.py — Walk-forward fold runner + data loading.

Extracted from scripts/phase7_baseline.py for R3 Optuna Infrastructure.
Provides reusable fold execution, BSS computation, and data loading for
any sweep framework (OptunaSweep, GridSweep, or manual scripts).

Usage:
    from pattern_engine.walkforward import load_and_augment_db, run_fold, run_walkforward
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import pattern_engine.matcher as _matcher_module
from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS
from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import FeatureRegistry
from pattern_engine.candlestick import compute_candlestick_features, CANDLE_COLS

try:
    from betacal import BetaCalibration
except ImportError:
    raise RuntimeError("betacal is not installed. Run: py -3.12 -m pip install betacal")

# ── Constants ────────────────────────────────────────────────────────────────
HORIZON = "fwd_7d_up"
SPY_THRESHOLD = 0.05
DATA_DIR = Path("data/52t_features")
FEATURE_COLS = list(FeatureRegistry.get("returns_candle").columns)  # 23 columns
CANDLE_FEATURE_COLS = list(CANDLE_COLS)
MURPHY_BINS = 10

# Keys consumed by matcher via getattr(), NOT EngineConfig fields
_NON_CONFIG_KEYS = {"cal_frac"}


# ── Beta calibrator (beta_abm — locked since H5) ────────────────────────────

class _BetaCalibrator:
    """Drop-in replacement for _PlattCalibrator using BetaCalibration(parameters='abm')."""

    def __init__(self) -> None:
        self._cal = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_BetaCalibrator":
        self._cal = BetaCalibration(parameters="abm")
        self._cal.fit(raw.reshape(-1, 1), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self._cal.predict(raw.reshape(-1, 1))


# ── BSS and Murphy decomposition ────────────────────────────────────────────

def _bss(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Brier Skill Score vs climatological baseline."""
    base_rate = float(y_true.mean())
    bs_ref = base_rate * (1.0 - base_rate)
    if bs_ref < 1e-10:
        return 0.0
    brier = float(np.mean((probs - y_true) ** 2))
    return 1.0 - brier / bs_ref


def _murphy_decomposition(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = MURPHY_BINS,
) -> tuple[float, float, float]:
    """Murphy decomposition of Brier Score into reliability + resolution + uncertainty.

    Returns:
        reliability:  calibration term (lower = better)
        resolution:   discrimination term (higher = better)
        uncertainty:  climatological variance = base_rate*(1-base_rate)
    """
    base_rate = float(y_true.mean())
    uncertainty = base_rate * (1.0 - base_rate)

    if len(probs) == 0 or uncertainty < 1e-10:
        return float("nan"), float("nan"), uncertainty

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(probs, bins, right=True)
    bin_idx = np.clip(bin_idx, 1, n_bins)

    reliability = 0.0
    resolution = 0.0

    for b in range(1, n_bins + 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        n_b = mask.sum()
        mean_p = float(probs[mask].mean())
        mean_y = float(y_true[mask].mean())
        w = n_b / len(probs)
        reliability += w * (mean_p - mean_y) ** 2
        resolution += w * (mean_y - base_rate) ** 2

    return reliability, resolution, uncertainty
```

**Note:** This is the initial skeleton. Steps 1.5 and Task 2 add the remaining functions. We create the file now so the BSS/Murphy tests can import.

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_walkforward.py::TestBSS -v`
Expected: 4 PASSED

- [ ] **Step 1.5: Add Murphy decomposition tests**

Append to `tests/unit/test_walkforward.py`:

```python
class TestMurphyDecomposition:
    """Tests for _murphy_decomposition()."""

    def test_uncertainty_matches_base_rate(self):
        """Uncertainty = base_rate * (1 - base_rate)."""
        from pattern_engine.walkforward import _murphy_decomposition
        y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        probs = np.array([0.6, 0.4, 0.7, 0.3, 0.55, 0.45])
        rel, res, unc = _murphy_decomposition(probs, y)
        assert unc == pytest.approx(0.25)  # 0.5 * 0.5

    def test_perfect_predictions_have_zero_reliability(self):
        """Perfect calibration → reliability ≈ 0."""
        from pattern_engine.walkforward import _murphy_decomposition
        # 10 bins, all predictions exactly match outcomes
        y = np.array([1.0, 0.0, 1.0, 0.0])
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        rel, res, unc = _murphy_decomposition(probs, y)
        assert rel == pytest.approx(0.0, abs=1e-6)

    def test_constant_labels_return_nan(self):
        """All-same labels → uncertainty ≈ 0 → returns NaN for rel/res."""
        from pattern_engine.walkforward import _murphy_decomposition
        y = np.ones(10)
        probs = np.full(10, 0.9)
        rel, res, unc = _murphy_decomposition(probs, y)
        assert np.isnan(rel)
        assert np.isnan(res)
```

- [ ] **Step 1.6: Run Murphy tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_walkforward.py::TestMurphyDecomposition -v`
Expected: 3 PASSED

- [ ] **Step 1.7: Add BetaCalibrator test**

Append to `tests/unit/test_walkforward.py`:

```python
class TestBetaCalibrator:
    """Tests for _BetaCalibrator drop-in replacement."""

    def test_fit_transform_returns_probabilities(self):
        """BetaCalibrator.transform() outputs values in [0, 1]."""
        from pattern_engine.walkforward import _BetaCalibrator
        rng = np.random.default_rng(42)
        raw = rng.uniform(0.3, 0.7, size=200)
        y = (rng.random(200) > 0.5).astype(float)
        cal = _BetaCalibrator()
        cal.fit(raw, y)
        out = cal.transform(raw)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
        assert len(out) == 200
```

- [ ] **Step 1.8: Run BetaCalibrator test**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_walkforward.py::TestBetaCalibrator -v`
Expected: 1 PASSED

- [ ] **Step 1.9: Commit**

```bash
git add pattern_engine/walkforward.py tests/unit/test_walkforward.py
git commit -m "feat(p3): add walkforward.py with _bss, _murphy_decomposition, _BetaCalibrator"
```

---

## Task 2: walkforward.py — H7 HOLD, Candlestick, Config Builder

**Files:**
- Modify: `pattern_engine/walkforward.py`
- Modify: `tests/unit/test_walkforward.py`

Add the remaining private helpers: `_apply_h7_hold_regime()`, `_impute_candle_nans()`, `_augment_with_candlestick()`, `_build_cfg()`.

- [ ] **Step 2.1: Write failing test for _apply_h7_hold_regime**

Append to `tests/unit/test_walkforward.py`:

```python
class TestH7HoldRegime:
    """Tests for _apply_h7_hold_regime()."""

    def test_bear_rows_get_base_rate(self):
        """Rows where SPY ret_90d < 0.05 should have probs replaced with base_rate."""
        from pattern_engine.walkforward import _apply_h7_hold_regime

        dates = pd.to_datetime(["2020-03-01", "2020-03-02", "2020-06-01", "2020-06-02"])
        # Val DB: 2 tickers (SPY + T1), 2 dates each
        val_db = pd.DataFrame({
            "Date": dates.repeat(2),
            "Ticker": ["SPY", "T1"] * 2 + ["SPY", "T1"] * 2,
        })
        # Flatten: need SPY rows with ret_90d
        val_db = pd.DataFrame({
            "Date": ["2020-03-01", "2020-03-01", "2020-06-01", "2020-06-01"],
            "Ticker": ["SPY", "T1", "SPY", "T1"],
            "ret_90d": [-0.10, 0.05, 0.10, 0.05],  # SPY: bear then bull
        })
        val_db["Date"] = pd.to_datetime(val_db["Date"])

        train_db = val_db.copy()  # Not used when SPY present in val
        probs = np.array([0.7, 0.8, 0.6, 0.9])
        base_rate = 0.5

        probs_hold, bear_mask = _apply_h7_hold_regime(val_db, train_db, base_rate, probs)

        # Bear dates (2020-03-01): SPY ret_90d = -0.10 < 0.05 → bear
        assert bear_mask[0] == True   # SPY on bear date
        assert bear_mask[1] == True   # T1 on bear date
        # Bull dates (2020-06-01): SPY ret_90d = 0.10 >= 0.05 → not bear
        assert bear_mask[2] == False
        assert bear_mask[3] == False
        # Bear rows should have base_rate
        assert probs_hold[0] == pytest.approx(base_rate)
        assert probs_hold[1] == pytest.approx(base_rate)
        # Bull rows should keep original probs
        assert probs_hold[2] == pytest.approx(0.6)
        assert probs_hold[3] == pytest.approx(0.9)
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_walkforward.py::TestH7HoldRegime -v -x`
Expected: FAIL with `ImportError: cannot import name '_apply_h7_hold_regime'`

- [ ] **Step 2.3: Add _apply_h7_hold_regime, _impute_candle_nans, _augment_with_candlestick, _build_cfg to walkforward.py**

Append to `pattern_engine/walkforward.py`:

```python
# ── Regime labeler — H7 HOLD mode ────────────────────────────────────────────

def _apply_h7_hold_regime(
    val_db: pd.DataFrame,
    train_db: pd.DataFrame,
    base_rate: float,
    probs: np.ndarray,
    threshold: float = SPY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply H7 HOLD regime: bear rows get base_rate instead of model probability.

    Bear row: SPY ret_90d on that date < threshold (+0.05).
    """
    spy_val = val_db[val_db["Ticker"] == "SPY"].copy()
    if spy_val.empty:
        spy_val = train_db[train_db["Ticker"] == "SPY"].copy()

    spy_val["Date"] = pd.to_datetime(spy_val["Date"])
    spy_val = spy_val.set_index("Date").sort_index()
    spy_ret90 = spy_val["ret_90d"]

    row_dates = pd.to_datetime(val_db["Date"])
    mapped = spy_ret90.reindex(row_dates.values, method="nearest")
    bear_mask = mapped.values < threshold

    probs_hold = probs.copy()
    probs_hold[bear_mask] = base_rate

    return probs_hold, bear_mask


# ── Candlestick augmentation ────────────────────────────────────────────────

def _impute_candle_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs in candlestick columns with neutral values."""
    df = df.copy()
    prop_fill = {c: 0.0 for c in CANDLE_COLS if "direction" not in c}
    dir_fill = {c: 1.0 for c in CANDLE_COLS if "direction" in c}
    df.fillna({**prop_fill, **dir_fill}, inplace=True)
    return df


def _augment_with_candlestick(full_db: pd.DataFrame) -> pd.DataFrame:
    """Compute and append candlestick features to full_db."""
    print("Computing candlestick features...", flush=True)
    t0 = time.time()
    candle_df = compute_candlestick_features(full_db)
    print(f"  {len(CANDLE_COLS)} candle columns computed in {time.time() - t0:.1f}s")
    augmented = pd.concat(
        [full_db.reset_index(drop=True), candle_df.reset_index(drop=True)], axis=1
    )
    augmented = _impute_candle_nans(augmented)
    remaining = int(augmented[CANDLE_COLS].isna().values.sum())
    if remaining > 0:
        raise RuntimeError(
            f"Candlestick imputation failed: {remaining} NaNs remain after fillna."
        )
    return augmented


# ── Config builder ────────────────────────────────────────────────────────────

def _build_cfg(cfg_overrides: dict | None = None) -> EngineConfig:
    """Build locked EngineConfig, then apply any overrides dict.

    Non-EngineConfig keys (e.g. cal_frac) are stripped from overrides,
    applied via setattr after config construction, so matcher.py can
    access them via getattr(cfg, 'cal_frac', 0.76).
    """
    overrides = dict(cfg_overrides) if cfg_overrides else {}

    # Strip non-config keys before building EngineConfig
    extra = {k: overrides.pop(k) for k in list(_NON_CONFIG_KEYS) if k in overrides}

    cfg = EngineConfig()
    cfg.max_distance = 2.5
    cfg.top_k = 50
    cfg.distance_weighting = "uniform"
    cfg.distance_metric = "euclidean"
    cfg.confidence_threshold = 0.65
    cfg.agreement_spread = 0.05
    cfg.min_matches = 5
    cfg.exclude_same_ticker = True
    cfg.same_sector_only = False
    cfg.regime_filter = False
    cfg.regime_fallback = False
    cfg.projection_horizon = HORIZON
    cfg.cal_max_samples = 100_000
    cfg.use_hnsw = True
    cfg.nn_jobs = 1
    cfg.feature_set = "returns_candle"
    cfg.use_sax_filter = False
    cfg.use_wfa_rerank = False
    cfg.use_ib_compression = False
    cfg.use_sector_conviction = False
    cfg.use_momentum_filter = False
    cfg.use_sentiment_veto = False
    cfg.use_bma = False
    cfg.use_owa = False
    cfg.use_dtw_reranker = False
    cfg.use_conformal = False
    cfg.use_anomaly_filter = False
    cfg.use_stumpy = False

    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise RuntimeError(
                f"_build_cfg: unknown EngineConfig field {k!r}. "
                "Check cfg_overrides keys."
            )
        setattr(cfg, k, v)

    # Inject non-config keys after config construction
    for k, v in extra.items():
        setattr(cfg, k, v)

    return cfg
```

- [ ] **Step 2.4: Run H7 regime test**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_walkforward.py::TestH7HoldRegime -v`
Expected: 1 PASSED

- [ ] **Step 2.5: Add _build_cfg test**

Append to `tests/unit/test_walkforward.py`:

```python
class TestBuildCfg:
    """Tests for _build_cfg()."""

    def test_locked_defaults(self):
        """Default config matches locked settings."""
        from pattern_engine.walkforward import _build_cfg
        cfg = _build_cfg()
        assert cfg.max_distance == 2.5
        assert cfg.top_k == 50
        assert cfg.nn_jobs == 1
        assert cfg.use_hnsw is True

    def test_overrides_applied(self):
        """cfg_overrides dict sets specified fields."""
        from pattern_engine.walkforward import _build_cfg
        cfg = _build_cfg({"max_distance": 1.5, "top_k": 30})
        assert cfg.max_distance == 1.5
        assert cfg.top_k == 30

    def test_unknown_field_raises(self):
        """Unknown field in overrides raises RuntimeError."""
        from pattern_engine.walkforward import _build_cfg
        with pytest.raises(RuntimeError, match="unknown EngineConfig field"):
            _build_cfg({"nonexistent_field": 42})

    def test_cal_frac_stripped_and_injected(self):
        """cal_frac is not an EngineConfig field but gets injected via setattr."""
        from pattern_engine.walkforward import _build_cfg
        cfg = _build_cfg({"cal_frac": 0.80})
        assert getattr(cfg, "cal_frac", None) == 0.80
```

- [ ] **Step 2.6: Run _build_cfg tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_walkforward.py::TestBuildCfg -v`
Expected: 4 PASSED

- [ ] **Step 2.7: Commit**

```bash
git add pattern_engine/walkforward.py tests/unit/test_walkforward.py
git commit -m "feat(p3): add H7 HOLD regime, candlestick augmentation, config builder to walkforward"
```

---

## Task 3: walkforward.py — Public API (run_fold, load_and_augment_db, run_walkforward)

**Files:**
- Modify: `pattern_engine/walkforward.py`
- Modify: `tests/unit/test_walkforward.py`

Add the three public functions. `run_fold()` is the core — it must produce identical output to `scripts/phase7_baseline.py:run_fold_with_config()`. `load_and_augment_db()` handles data loading. `run_walkforward()` is the 6-fold convenience wrapper with Wilcoxon p-value.

- [ ] **Step 3.1: Write failing test for run_fold return shape**

Append to `tests/unit/test_walkforward.py`:

```python
class TestRunFold:
    """Tests for run_fold() — requires mocking PatternMatcher."""

    def test_return_dict_keys(self, synthetic_full_db, monkeypatch):
        """run_fold returns dict with all required keys."""
        from pattern_engine.walkforward import run_fold, FEATURE_COLS

        # Mock PatternMatcher to avoid real KNN computation
        class _MockMatcher:
            def __init__(self, cfg):
                pass
            def fit(self, train_db, feature_cols):
                pass
            def query(self, val_db, verbose=0):
                n = len(val_db)
                probs = np.full(n, 0.55)
                signals = ["HOLD"] * n
                return probs, signals, [None]*n, [50]*n, [None]*n, [None]*n

        import pattern_engine.walkforward as wf_mod
        monkeypatch.setattr(wf_mod, "PatternMatcher", _MockMatcher)
        # Also mock _BetaCalibrator monkey-patch target
        monkeypatch.setattr(wf_mod._matcher_module, "_PlattCalibrator", wf_mod._BetaCalibrator)

        fold = {"label": "2019", "train_end": "2018-12-31",
                "val_start": "2019-01-01", "val_end": "2019-12-31"}
        result = run_fold(fold, synthetic_full_db)

        expected_keys = {"fold", "bss", "n_scored", "n_total", "base_rate",
                         "mean_prob", "reliability", "resolution", "uncertainty"}
        assert set(result.keys()) == expected_keys
        assert result["fold"] == "2019"
        assert isinstance(result["bss"], float)
        assert result["n_total"] > 0

    def test_empty_val_returns_nan(self, synthetic_full_db, monkeypatch):
        """Fold with no validation rows returns NaN BSS."""
        from pattern_engine.walkforward import run_fold

        # Use a date range with no data
        fold = {"label": "1990", "train_end": "1989-12-31",
                "val_start": "1990-01-01", "val_end": "1990-12-31"}
        result = run_fold(fold, synthetic_full_db)

        assert result["fold"] == "1990"
        assert np.isnan(result["bss"])
        assert result["n_total"] == 0
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_walkforward.py::TestRunFold -v -x`
Expected: FAIL with `ImportError: cannot import name 'run_fold'`

- [ ] **Step 3.3: Add run_fold() to walkforward.py**

Append to `pattern_engine/walkforward.py`:

```python
# ── Core fold runner ─────────────────────────────────────────────────────────

def run_fold(
    fold: dict,
    full_db: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cfg_overrides: dict | None = None,
) -> dict:
    """Run one walk-forward fold and return a result dict.

    Args:
        fold:          One entry from WALKFORWARD_FOLDS.
        full_db:       Augmented full database (with candlestick features).
        feature_cols:  Feature column list. Defaults to locked 23-col set.
        cfg_overrides: Dict of EngineConfig field overrides.

    Returns:
        Dict with keys: fold, bss, n_scored, n_total, base_rate,
                        mean_prob, reliability, resolution, uncertainty.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_db = (
        full_db[full_db["Date"] <= train_end]
        .dropna(subset=[HORIZON])
        .copy()
    )
    val_db = (
        full_db[
            (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
        ]
        .dropna(subset=[HORIZON])
        .copy()
    )

    n_total = len(val_db)
    y_true = val_db[HORIZON].values.astype(float)
    base_rate = float(y_true.mean()) if n_total > 0 else 0.0

    nan_row = {
        "fold": fold["label"],
        "bss": float("nan"),
        "n_scored": 0,
        "n_total": n_total,
        "base_rate": round(base_rate, 6),
        "mean_prob": float("nan"),
        "reliability": float("nan"),
        "resolution": float("nan"),
        "uncertainty": round(base_rate * (1.0 - base_rate), 6),
    }

    if n_total == 0:
        return nan_row

    cfg = _build_cfg(cfg_overrides)

    # Monkey-patch _PlattCalibrator → _BetaCalibrator for this fold
    original_calibrator = _matcher_module._PlattCalibrator
    try:
        _matcher_module._PlattCalibrator = _BetaCalibrator

        matcher = PatternMatcher(cfg)
        matcher.fit(train_db, list(feature_cols))
        probs_raw, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)

    finally:
        _matcher_module._PlattCalibrator = original_calibrator

    probs = np.asarray(probs_raw)

    # H7 HOLD regime: bear rows → base_rate probability
    probs_hold, bear_mask = _apply_h7_hold_regime(
        val_db=val_db,
        train_db=train_db,
        base_rate=base_rate,
        probs=probs,
    )

    scored_mask = ~bear_mask
    n_scored = int(scored_mask.sum())
    mean_prob = float(probs_hold[scored_mask].mean()) if scored_mask.any() else float("nan")

    bss_val = _bss(probs_hold[scored_mask], y_true[scored_mask])

    if scored_mask.any():
        rel, res, unc = _murphy_decomposition(probs_hold[scored_mask], y_true[scored_mask])
    else:
        rel, res, unc = float("nan"), float("nan"), base_rate * (1.0 - base_rate)

    return {
        "fold": fold["label"],
        "bss": round(bss_val, 6),
        "n_scored": n_scored,
        "n_total": n_total,
        "base_rate": round(base_rate, 6),
        "mean_prob": round(mean_prob, 6) if not np.isnan(mean_prob) else float("nan"),
        "reliability": round(rel, 8) if not np.isnan(rel) else float("nan"),
        "resolution": round(res, 8) if not np.isnan(res) else float("nan"),
        "uncertainty": round(unc, 6),
    }
```

- [ ] **Step 3.4: Run run_fold tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_walkforward.py::TestRunFold -v`
Expected: 2 PASSED

- [ ] **Step 3.5: Write failing test for run_walkforward**

Append to `tests/unit/test_walkforward.py`:

```python
class TestRunWalkforward:
    """Tests for run_walkforward() convenience wrapper."""

    def test_return_dict_shape(self, synthetic_full_db, monkeypatch):
        """run_walkforward returns dict with aggregates + fold_results."""
        from pattern_engine.walkforward import run_walkforward

        # Mock run_fold to return synthetic results
        call_count = [0]
        def _mock_run_fold(fold, full_db, feature_cols=None, cfg_overrides=None):
            call_count[0] += 1
            bss = 0.001 * call_count[0]  # All positive
            return {
                "fold": fold["label"], "bss": bss, "n_scored": 100,
                "n_total": 200, "base_rate": 0.5, "mean_prob": 0.55,
                "reliability": 0.01, "resolution": 0.02, "uncertainty": 0.25,
            }

        import pattern_engine.walkforward as wf_mod
        monkeypatch.setattr(wf_mod, "run_fold", _mock_run_fold)

        result = run_walkforward(synthetic_full_db)

        assert "mean_bss" in result
        assert "trimmed_mean_bss" in result
        assert "positive_folds" in result
        assert "fold_results" in result
        assert "wilcoxon_p" in result
        assert len(result["fold_results"]) == 6
        assert result["positive_folds"] == 6

    def test_trimmed_mean_drops_worst(self, synthetic_full_db, monkeypatch):
        """trimmed_mean_bss drops the worst fold."""
        from pattern_engine.walkforward import run_walkforward

        bss_values = [0.01, 0.02, 0.03, -0.05, 0.015, 0.025]
        idx = [0]
        def _mock_run_fold(fold, full_db, feature_cols=None, cfg_overrides=None):
            i = idx[0]; idx[0] += 1
            return {
                "fold": fold["label"], "bss": bss_values[i], "n_scored": 100,
                "n_total": 200, "base_rate": 0.5, "mean_prob": 0.55,
                "reliability": 0.01, "resolution": 0.02, "uncertainty": 0.25,
            }

        import pattern_engine.walkforward as wf_mod
        monkeypatch.setattr(wf_mod, "run_fold", _mock_run_fold)

        result = run_walkforward(synthetic_full_db)

        # Worst fold is -0.05, trimmed mean drops it
        trimmed = sorted(bss_values)[1:]  # drop -0.05
        expected_trimmed = np.mean(trimmed)
        assert result["trimmed_mean_bss"] == pytest.approx(expected_trimmed)
        assert result["mean_bss"] == pytest.approx(np.mean(bss_values))
```

- [ ] **Step 3.6: Add run_walkforward and load_and_augment_db to walkforward.py**

Append to `pattern_engine/walkforward.py`:

```python
# ── Data loading ─────────────────────────────────────────────────────────────

def load_and_augment_db(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load train + val parquets, concatenate, augment with candlestick features.

    Called ONCE per sweep, not per trial.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    t_path = data_dir / "train_db.parquet"
    v_path = data_dir / "val_db.parquet"

    if not t_path.exists() or not v_path.exists():
        raise RuntimeError(
            f"52T features data not found in {data_dir}.\n"
            "Run scripts/build_52t_features.py first."
        )

    full_db = pd.concat(
        [pd.read_parquet(t_path), pd.read_parquet(v_path)], ignore_index=True
    )
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    full_db = _augment_with_candlestick(full_db)
    return full_db


# ── Walk-forward convenience wrapper ─────────────────────────────────────────

def run_walkforward(
    full_db: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cfg_overrides: dict | None = None,
    folds: list[dict] | None = None,
) -> dict:
    """Run all 6 folds, compute aggregates including Wilcoxon p-value.

    Returns:
        Dict with mean_bss, trimmed_mean_bss, positive_folds, fold_results,
        wilcoxon_p (None if < 6 non-zero BSS values).
    """
    if folds is None:
        folds = WALKFORWARD_FOLDS

    fold_results = []
    for fold in folds:
        result = run_fold(fold, full_db, feature_cols=feature_cols,
                          cfg_overrides=cfg_overrides)
        fold_results.append(result)

    bss_values = [r["bss"] for r in fold_results if not np.isnan(r["bss"])]

    if not bss_values:
        return {
            "mean_bss": float("nan"),
            "trimmed_mean_bss": float("nan"),
            "positive_folds": 0,
            "fold_results": fold_results,
            "wilcoxon_p": None,
        }

    mean_bss = float(np.mean(bss_values))
    positive_folds = sum(1 for b in bss_values if b > 0)

    # Trimmed mean: drop worst fold
    sorted_bss = sorted(bss_values)
    trimmed_mean_bss = float(np.mean(sorted_bss[1:])) if len(sorted_bss) > 1 else mean_bss

    # Wilcoxon signed-rank p-value (one-sided, H₀: median BSS ≤ 0)
    wilcoxon_p = None
    non_zero = [b for b in bss_values if b != 0.0]
    if len(non_zero) >= 6:
        from scipy.stats import wilcoxon as _wilcoxon
        _, wilcoxon_p = _wilcoxon(non_zero, alternative="greater")
        wilcoxon_p = float(wilcoxon_p)

    return {
        "mean_bss": mean_bss,
        "trimmed_mean_bss": trimmed_mean_bss,
        "positive_folds": positive_folds,
        "fold_results": fold_results,
        "wilcoxon_p": wilcoxon_p,
    }
```

- [ ] **Step 3.7: Run run_walkforward tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_walkforward.py::TestRunWalkforward -v`
Expected: 2 PASSED

- [ ] **Step 3.8: Run full walkforward test suite**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_walkforward.py -v`
Expected: 12+ PASSED

- [ ] **Step 3.9: Commit**

```bash
git add pattern_engine/walkforward.py tests/unit/test_walkforward.py
git commit -m "feat(p3): add run_fold, run_walkforward, load_and_augment_db to walkforward"
```

---

## Task 4: experiment_log.py — ExperimentLogger

**Files:**
- Create: `pattern_engine/experiment_log.py`
- Create: `tests/unit/test_experiment_log.py`

TSV provenance logger. Writes incrementally (mode="a") so partial sweep results are preserved if killed.

- [ ] **Step 4.1: Write failing tests for ExperimentLogger**

```python
# tests/unit/test_experiment_log.py
"""Tests for pattern_engine.experiment_log module."""
from __future__ import annotations

import pytest


class TestExperimentLogger:
    """Tests for ExperimentLogger TSV writer."""

    def test_header_writes_metadata(self, tmp_path):
        """log_header writes comment block + column header."""
        from pattern_engine.experiment_log import ExperimentLogger

        logger = ExperimentLogger(output_dir=str(tmp_path), experiment_name="test_sweep")
        logger.log_header(["trial", "max_distance", "mean_bss", "gate_pass"])

        tsv = (tmp_path / "test_sweep.tsv").read_text()
        assert "# experiment: test_sweep" in tsv
        assert "# started:" in tsv
        assert "trial\tmax_distance\tmean_bss\tgate_pass" in tsv

    def test_log_trial_appends_row(self, tmp_path):
        """log_trial appends a TSV row with correct values."""
        from pattern_engine.experiment_log import ExperimentLogger

        logger = ExperimentLogger(output_dir=str(tmp_path), experiment_name="test_sweep")
        cols = ["trial", "max_distance", "mean_bss"]
        logger.log_header(cols)
        logger.log_trial(trial_id=0, config={"max_distance": 2.5}, result={"mean_bss": 0.001})

        lines = (tmp_path / "test_sweep.tsv").read_text().strip().split("\n")
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == 2  # header + 1 data row
        assert "0\t" in data_lines[1]

    def test_incremental_append_survives_reopen(self, tmp_path):
        """Multiple log_trial calls produce valid TSV."""
        from pattern_engine.experiment_log import ExperimentLogger

        logger = ExperimentLogger(output_dir=str(tmp_path), experiment_name="append_test")
        cols = ["trial", "param_a", "score"]
        logger.log_header(cols)
        logger.log_trial(0, {"param_a": 1.0}, {"score": 0.5})
        logger.log_trial(1, {"param_a": 2.0}, {"score": 0.6})
        logger.log_trial(2, {"param_a": 3.0}, {"score": 0.7})

        lines = (tmp_path / "append_test.tsv").read_text().strip().split("\n")
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == 4  # header + 3 rows

    def test_finalize_writes_summary(self, tmp_path):
        """finalize writes a summary comment block."""
        from pattern_engine.experiment_log import ExperimentLogger

        logger = ExperimentLogger(output_dir=str(tmp_path), experiment_name="final_test")
        logger.log_header(["trial", "score"])
        logger.log_trial(0, {}, {"score": 0.5})

        # Create a mock SweepResult-like object
        class _MockResult:
            best_config = {"max_distance": 2.5}
            best_bss = 0.001
            elapsed_minutes = 5.0

        logger.finalize(best_config={"max_distance": 2.5}, sweep_result=_MockResult())

        tsv = (tmp_path / "final_test.tsv").read_text()
        assert "# best_config:" in tsv
        assert "# best_bss:" in tsv

    def test_missing_dir_created(self, tmp_path):
        """Logger creates output_dir if it doesn't exist."""
        from pattern_engine.experiment_log import ExperimentLogger

        nested = tmp_path / "sub" / "dir"
        logger = ExperimentLogger(output_dir=str(nested), experiment_name="nested")
        logger.log_header(["trial"])
        assert (nested / "nested.tsv").exists()
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_experiment_log.py -v -x`
Expected: FAIL with `ModuleNotFoundError: No module named 'pattern_engine.experiment_log'`

- [ ] **Step 4.3: Implement experiment_log.py**

```python
# pattern_engine/experiment_log.py
"""
pattern_engine/experiment_log.py — TSV provenance logger for sweep experiments.

Writes trial results incrementally to a TSV file with metadata comment headers.
Designed to survive interruption — each row is appended individually.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ExperimentLogger:
    """Incremental TSV logger for hyperparameter sweep experiments.

    Args:
        output_dir:      Directory for TSV output (created if missing).
        experiment_name: Experiment identifier; used as filename stem.
    """

    def __init__(self, output_dir: str = "results", experiment_name: str = "") -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._name = experiment_name or "experiment"
        self._path = self._dir / f"{self._name}.tsv"
        self._columns: list[str] = []

    @property
    def path(self) -> Path:
        return self._path

    def log_header(
        self,
        columns: list[str],
        search_space: dict | None = None,
        locked_settings: str | None = None,
    ) -> None:
        """Write metadata comment block + column header."""
        self._columns = list(columns)
        with open(self._path, "w", encoding="utf-8") as f:
            f.write(f"# experiment: {self._name}\n")
            f.write(f"# started: {datetime.now(timezone.utc).isoformat()}\n")
            if search_space:
                f.write(f"# search_space: {json.dumps(search_space)}\n")
            if locked_settings:
                f.write(f"# locked: {locked_settings}\n")
            f.write("\t".join(columns) + "\n")

    def log_trial(self, trial_id: int, config: dict, result: dict) -> None:
        """Append one TSV row for a completed trial."""
        row_data = {"trial": trial_id}
        row_data.update(config)
        row_data.update(result)

        values = []
        for col in self._columns:
            val = row_data.get(col, "")
            if isinstance(val, float):
                values.append(f"{val:+.6f}" if val != 0.0 else "+0.000000")
            elif isinstance(val, bool):
                values.append(str(val))
            else:
                values.append(str(val))

        with open(self._path, "a", encoding="utf-8") as f:
            f.write("\t".join(values) + "\n")

    def finalize(self, best_config: dict, sweep_result: object) -> None:
        """Append summary comment block after sweep completion."""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(f"# best_config: {json.dumps(best_config)}\n")
            bss = getattr(sweep_result, "best_bss", None)
            if bss is not None:
                f.write(f"# best_bss: {bss:+.6f}\n")
            elapsed = getattr(sweep_result, "elapsed_minutes", None)
            if elapsed is not None:
                f.write(f"# elapsed_minutes: {elapsed:.1f}\n")
```

- [ ] **Step 4.4: Run experiment_log tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_experiment_log.py -v`
Expected: 5 PASSED

- [ ] **Step 4.5: Commit**

```bash
git add pattern_engine/experiment_log.py tests/unit/test_experiment_log.py
git commit -m "feat(p3): add ExperimentLogger TSV provenance writer"
```

---

## Task 5: sweep.py — SweepResult, OptunaSweep, GridSweep

**Files:**
- Create: `pattern_engine/sweep.py`
- Create: `tests/unit/test_sweep.py`

This is the largest task. `OptunaSweep` wraps Optuna's TPE sampler; `GridSweep` does exhaustive enumeration. Both share the same objective function contract. Tests use mock objective functions — no real PatternMatcher.

- [ ] **Step 5.1: Write failing tests for SweepResult and GridSweep**

```python
# tests/unit/test_sweep.py
"""Tests for pattern_engine.sweep module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _mock_objective(config: dict, full_db: pd.DataFrame) -> dict:
    """Mock objective: BSS = -|max_distance - 2.5| (peaks at 2.5)."""
    md = config.get("max_distance", 2.5)
    bss = -abs(md - 2.5) + 0.001
    return {
        "mean_bss": bss,
        "trimmed_mean_bss": bss + 0.0001,
        "positive_folds": 4 if bss > 0 else 1,
        "fold_results": [
            {"fold": f"fold_{i}", "bss": bss + 0.0001 * i}
            for i in range(6)
        ],
        "wilcoxon_p": 0.03 if bss > 0 else 0.5,
    }


class TestSweepResult:
    """Tests for SweepResult dataclass."""

    def test_fields_exist(self):
        from pattern_engine.sweep import SweepResult
        sr = SweepResult(
            best_config={"a": 1},
            best_bss=0.001,
            best_positive_folds=4,
            best_wilcoxon_p=0.03,
            results_df=pd.DataFrame(),
            elapsed_minutes=1.0,
            study=None,
        )
        assert sr.best_config == {"a": 1}
        assert sr.best_bss == 0.001


class TestGridSweep:
    """Tests for GridSweep exhaustive search."""

    def test_exhaustive_enumeration(self):
        """GridSweep runs all combinations."""
        from pattern_engine.sweep import GridSweep

        gs = GridSweep(
            objective_fn=_mock_objective,
            param_grid={"max_distance": [1.0, 2.0, 2.5, 3.0]},
        )
        fake_db = pd.DataFrame({"x": [1]})
        result = gs.run(fake_db, verbose=0)

        assert len(result.results_df) == 4
        assert result.best_config["max_distance"] == pytest.approx(2.5)
        assert result.best_bss > 0

    def test_gate_fn_filters(self):
        """Custom gate_fn penalizes failing trials."""
        from pattern_engine.sweep import GridSweep

        def strict_gate(r: dict) -> bool:
            return r["positive_folds"] >= 5

        gs = GridSweep(
            objective_fn=_mock_objective,
            param_grid={"max_distance": [1.0, 2.5]},
            gate_fn=strict_gate,
        )
        fake_db = pd.DataFrame({"x": [1]})
        result = gs.run(fake_db, verbose=0)

        assert len(result.results_df) == 2
        # max_distance=2.5 has positive_folds=4, still fails strict gate
        assert "gate_pass" in result.results_df.columns


class TestOptunaSweep:
    """Tests for OptunaSweep Bayesian optimization."""

    def test_basic_run(self):
        """OptunaSweep completes n_trials and returns SweepResult."""
        from pattern_engine.sweep import OptunaSweep

        sweep = OptunaSweep(
            study_name="test_basic",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=5,
            max_hours=1.0,
        )
        fake_db = pd.DataFrame({"x": [1]})
        result = sweep.run(fake_db, verbose=0)

        assert len(result.results_df) == 5
        assert result.best_bss > -1.0
        assert result.study is not None

    def test_sqlite_persistence(self, tmp_path):
        """OptunaSweep with storage_path creates SQLite file."""
        from pattern_engine.sweep import OptunaSweep

        db_path = str(tmp_path / "test.db")
        sweep = OptunaSweep(
            study_name="test_persist",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=3,
            storage_path=db_path,
        )
        fake_db = pd.DataFrame({"x": [1]})
        result = sweep.run(fake_db, verbose=0)

        assert (tmp_path / "test.db").exists()
        assert len(result.results_df) == 3

    def test_resume(self, tmp_path):
        """resume() continues from existing study."""
        from pattern_engine.sweep import OptunaSweep

        db_path = str(tmp_path / "resume.db")
        sweep = OptunaSweep(
            study_name="test_resume",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=5,
            storage_path=db_path,
        )
        fake_db = pd.DataFrame({"x": [1]})

        # Run 3 initially
        sweep._n_trials = 3
        sweep.run(fake_db, verbose=0)

        # Resume to 5 total
        sweep2 = OptunaSweep(
            study_name="test_resume",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=5,
            storage_path=db_path,
        )
        result = sweep2.resume(fake_db, verbose=0)

        assert len(result.results_df) >= 5

    def test_resume_requires_storage(self):
        """resume() without storage_path raises RuntimeError."""
        from pattern_engine.sweep import OptunaSweep

        sweep = OptunaSweep(
            study_name="test_no_storage",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0)},
        )
        with pytest.raises(RuntimeError, match="storage_path"):
            sweep.resume(pd.DataFrame(), verbose=0)

    def test_exception_in_objective_handled(self):
        """Objective that raises → trial recorded with penalty BSS."""
        from pattern_engine.sweep import OptunaSweep

        call_count = [0]
        def _bad_objective(config, full_db):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("simulated crash")
            return _mock_objective(config, full_db)

        sweep = OptunaSweep(
            study_name="test_exception",
            objective_fn=_bad_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=3,
        )
        fake_db = pd.DataFrame({"x": [1]})
        result = sweep.run(fake_db, verbose=0)

        # All 3 trials should complete (exception handled gracefully)
        assert len(result.results_df) == 3

    def test_to_tsv(self, tmp_path):
        """to_tsv() writes valid TSV output."""
        from pattern_engine.sweep import OptunaSweep

        sweep = OptunaSweep(
            study_name="test_tsv",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=3,
        )
        fake_db = pd.DataFrame({"x": [1]})
        sweep.run(fake_db, verbose=0)

        out = tmp_path / "output.tsv"
        sweep.to_tsv(str(out))

        assert out.exists()
        df = pd.read_csv(out, sep="\t")
        assert len(df) == 3

    def test_int_param_uses_suggest_int(self):
        """Integer search space bounds use suggest_int."""
        from pattern_engine.sweep import OptunaSweep

        sweep = OptunaSweep(
            study_name="test_int",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0), "top_k": (20, 100)},
            n_trials=3,
        )
        fake_db = pd.DataFrame({"x": [1]})
        result = sweep.run(fake_db, verbose=0)

        # top_k should be integer in all results
        for _, row in result.results_df.iterrows():
            if "top_k" in row:
                assert float(row["top_k"]).is_integer()
```

- [ ] **Step 5.2: Run tests to verify they fail**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_sweep.py -v -x`
Expected: FAIL with `ModuleNotFoundError: No module named 'pattern_engine.sweep'`

- [ ] **Step 5.3: Implement sweep.py**

```python
# pattern_engine/sweep.py
"""
pattern_engine/sweep.py — Hyperparameter sweep framework (Optuna + Grid).

Provides OptunaSweep (Bayesian TPE) and GridSweep (exhaustive) with a
unified objective function contract. Algorithm-agnostic — KNN, LightGBM,
HMM, or any future model provides its own objective function.
"""
from __future__ import annotations

import itertools
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


# ── SweepResult ──────────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    """Results from a completed sweep."""
    best_config: dict
    best_bss: float
    best_positive_folds: int
    best_wilcoxon_p: float | None
    results_df: pd.DataFrame
    elapsed_minutes: float
    study: "optuna.Study | None" = field(default=None, repr=False)


# ── Pre-built search spaces ─────────────────────────────────────────────────

KNN_SEARCH_SPACE = {
    "max_distance": (1.0, 4.0),
    "top_k": (20, 100),
    "cal_frac": (0.5, 0.95),
    "confidence_threshold": (0.55, 0.80),
}


# ── Default gate ─────────────────────────────────────────────────────────────

def _default_gate(result: dict) -> bool:
    """Default gate: positive_folds >= 3 (TPE steering only, NOT significance)."""
    return result.get("positive_folds", 0) >= 3


# ── OptunaSweep ──────────────────────────────────────────────────────────────

class OptunaSweep:
    """Bayesian hyperparameter sweep using Optuna TPE sampler.

    Args:
        study_name:   Identifies study for persistence.
        objective_fn: (config_dict, full_db) -> dict with mean_bss, trimmed_mean_bss,
                      positive_folds, fold_results, wilcoxon_p.
        search_space: {"param": (lo, hi) | ["cat1", "cat2"]}
        n_trials:     Trial budget (default 80).
        max_hours:    Wall-clock budget (default 16.0).
        storage_path: SQLite path; None = in-memory.
        gate_fn:      (result_dict) -> bool; default: positive_folds >= 3.
        seed:         TPE sampler seed.
    """

    def __init__(
        self,
        study_name: str,
        objective_fn: Callable,
        search_space: dict,
        n_trials: int = 80,
        max_hours: float = 16.0,
        storage_path: str | None = None,
        gate_fn: Callable | None = None,
        seed: int = 42,
    ) -> None:
        self._study_name = study_name
        self._objective_fn = objective_fn
        self._search_space = search_space
        self._n_trials = n_trials
        self._max_hours = max_hours
        self._storage_path = storage_path
        self._gate_fn = gate_fn or _default_gate
        self._seed = seed
        self._study: optuna.Study | None = None
        self._trial_records: list[dict] = []

    def _sample_config(self, trial) -> dict:
        """Sample config from search space using Optuna trial."""
        config = {}
        for name, spec in self._search_space.items():
            if isinstance(spec, list):
                config[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                lo, hi = spec
                if isinstance(lo, int) and isinstance(hi, int):
                    config[name] = trial.suggest_int(name, lo, hi)
                else:
                    config[name] = trial.suggest_float(name, float(lo), float(hi))
            else:
                raise RuntimeError(f"Invalid search space spec for {name!r}: {spec}")
        return config

    def _wrapped_objective(self, trial, full_db: pd.DataFrame) -> float:
        """Optuna objective wrapper: sample, evaluate, gate, record."""
        config = self._sample_config(trial)

        try:
            result = self._objective_fn(config, full_db)
            tmb = result.get("trimmed_mean_bss", float("nan"))
            if np.isnan(tmb):
                raise ValueError("trimmed_mean_bss is NaN")
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            result = {
                "mean_bss": float("nan"),
                "trimmed_mean_bss": -0.10,
                "positive_folds": 0,
                "fold_results": [],
                "wilcoxon_p": None,
            }
            tmb = -0.10

        # Gate penalization
        gate_pass = self._gate_fn(result)
        if not gate_pass:
            tmb = max(tmb - 0.05, -0.10)

        # Store fold-level results as trial user attributes
        trial.set_user_attr("config", config)
        trial.set_user_attr("mean_bss", result.get("mean_bss", float("nan")))
        trial.set_user_attr("positive_folds", result.get("positive_folds", 0))
        trial.set_user_attr("wilcoxon_p", result.get("wilcoxon_p"))
        trial.set_user_attr("gate_pass", gate_pass)
        trial.set_user_attr("fold_results", result.get("fold_results", []))

        # Record for results_df
        record = {"trial": trial.number, **config}
        record["mean_bss"] = result.get("mean_bss", float("nan"))
        record["trimmed_mean_bss"] = result.get("trimmed_mean_bss", float("nan"))
        record["positive_folds"] = result.get("positive_folds", 0)
        record["wilcoxon_p"] = result.get("wilcoxon_p")
        record["gate_pass"] = gate_pass
        record["objective_value"] = tmb

        # Flatten fold BSS values
        for fr in result.get("fold_results", []):
            col = f"bss_{fr['fold']}"
            record[col] = fr.get("bss", float("nan"))

        self._trial_records.append(record)

        return tmb

    def run(self, full_db: pd.DataFrame, verbose: int = 1) -> SweepResult:
        """Execute the full study."""
        import optuna

        if verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        storage = f"sqlite:///{self._storage_path}" if self._storage_path else None
        sampler = optuna.samplers.TPESampler(seed=self._seed)

        self._study = optuna.create_study(
            study_name=self._study_name,
            sampler=sampler,
            direction="maximize",
            storage=storage,
            load_if_exists=True,
        )

        self._trial_records = []
        t0 = time.time()

        self._study.optimize(
            lambda trial: self._wrapped_objective(trial, full_db),
            n_trials=self._n_trials,
            timeout=self._max_hours * 3600,
        )

        elapsed = (time.time() - t0) / 60.0
        return self._build_result(elapsed)

    def resume(self, full_db: pd.DataFrame, verbose: int = 1) -> SweepResult:
        """Resume an existing SQLite-persisted study."""
        import optuna

        if not self._storage_path:
            raise RuntimeError(
                "resume() requires storage_path to be set (cannot resume in-memory study)."
            )

        if verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        storage = f"sqlite:///{self._storage_path}"
        self._study = optuna.load_study(
            study_name=self._study_name,
            storage=storage,
        )

        existing = len(self._study.trials)
        remaining = max(0, self._n_trials - existing)

        self._trial_records = []
        t0 = time.time()

        if remaining > 0:
            self._study.optimize(
                lambda trial: self._wrapped_objective(trial, full_db),
                n_trials=remaining,
                timeout=self._max_hours * 3600,
            )

        # Rebuild records from all trials (including pre-existing)
        self._trial_records = []
        for trial in self._study.trials:
            record = {"trial": trial.number}
            config = trial.user_attrs.get("config", {})
            record.update(config)
            record["mean_bss"] = trial.user_attrs.get("mean_bss", float("nan"))
            record["trimmed_mean_bss"] = trial.value if trial.value is not None else float("nan")
            record["positive_folds"] = trial.user_attrs.get("positive_folds", 0)
            record["wilcoxon_p"] = trial.user_attrs.get("wilcoxon_p")
            record["gate_pass"] = trial.user_attrs.get("gate_pass", False)
            record["objective_value"] = trial.value if trial.value is not None else float("nan")
            for fr in trial.user_attrs.get("fold_results", []):
                record[f"bss_{fr['fold']}"] = fr.get("bss", float("nan"))
            self._trial_records.append(record)

        elapsed = (time.time() - t0) / 60.0
        return self._build_result(elapsed)

    def _build_result(self, elapsed_minutes: float) -> SweepResult:
        """Construct SweepResult from trial records."""
        results_df = pd.DataFrame(self._trial_records)
        best_trial = self._study.best_trial

        best_config = best_trial.user_attrs.get("config", {})
        best_bss = best_trial.user_attrs.get("mean_bss", float("nan"))
        best_pf = best_trial.user_attrs.get("positive_folds", 0)
        best_wp = best_trial.user_attrs.get("wilcoxon_p")

        return SweepResult(
            best_config=best_config,
            best_bss=best_bss,
            best_positive_folds=best_pf,
            best_wilcoxon_p=best_wp,
            results_df=results_df,
            elapsed_minutes=elapsed_minutes,
            study=self._study,
        )

    def best(self) -> dict:
        """Return best trial's config + metrics."""
        if self._study is None:
            raise RuntimeError("No study available. Call run() first.")
        bt = self._study.best_trial
        return {
            "config": bt.user_attrs.get("config", {}),
            "mean_bss": bt.user_attrs.get("mean_bss"),
            "trimmed_mean_bss": bt.value,
            "positive_folds": bt.user_attrs.get("positive_folds"),
            "wilcoxon_p": bt.user_attrs.get("wilcoxon_p"),
        }

    def to_tsv(self, path: str) -> None:
        """Export all completed trials to provenance TSV."""
        results_df = pd.DataFrame(self._trial_records)
        results_df.to_csv(path, sep="\t", index=False, float_format="%.6f")


# ── GridSweep ────────────────────────────────────────────────────────────────

class GridSweep:
    """Exhaustive grid search over parameter combinations.

    Same objective function contract as OptunaSweep.
    """

    def __init__(
        self,
        objective_fn: Callable,
        param_grid: dict,
        gate_fn: Callable | None = None,
    ) -> None:
        self._objective_fn = objective_fn
        self._param_grid = param_grid
        self._gate_fn = gate_fn or _default_gate
        self._trial_records: list[dict] = []

    def run(self, full_db: pd.DataFrame, verbose: int = 1) -> SweepResult:
        """Run all parameter combinations."""
        param_names = list(self._param_grid.keys())
        param_values = list(self._param_grid.values())
        combos = list(itertools.product(*param_values))

        self._trial_records = []
        t0 = time.time()

        best_tmb = float("-inf")
        best_idx = 0

        for i, combo in enumerate(combos):
            config = dict(zip(param_names, combo))

            try:
                result = self._objective_fn(config, full_db)
                tmb = result.get("trimmed_mean_bss", float("nan"))
                if np.isnan(tmb):
                    raise ValueError("trimmed_mean_bss is NaN")
            except Exception as e:
                logger.warning(f"Grid trial {i} failed: {e}")
                result = {
                    "mean_bss": float("nan"),
                    "trimmed_mean_bss": -0.10,
                    "positive_folds": 0,
                    "fold_results": [],
                    "wilcoxon_p": None,
                }
                tmb = -0.10

            gate_pass = self._gate_fn(result)
            if not gate_pass:
                tmb = max(tmb - 0.05, -0.10)

            record = {"trial": i, **config}
            record["mean_bss"] = result.get("mean_bss", float("nan"))
            record["trimmed_mean_bss"] = result.get("trimmed_mean_bss", float("nan"))
            record["positive_folds"] = result.get("positive_folds", 0)
            record["wilcoxon_p"] = result.get("wilcoxon_p")
            record["gate_pass"] = gate_pass
            record["objective_value"] = tmb

            for fr in result.get("fold_results", []):
                record[f"bss_{fr['fold']}"] = fr.get("bss", float("nan"))

            self._trial_records.append(record)

            if tmb > best_tmb:
                best_tmb = tmb
                best_idx = i

            if verbose >= 1:
                print(f"  [{i+1}/{len(combos)}] {config} → tmb={tmb:+.6f} gate={gate_pass}")

        elapsed = (time.time() - t0) / 60.0

        best_record = self._trial_records[best_idx]
        best_config = {k: best_record[k] for k in param_names}

        return SweepResult(
            best_config=best_config,
            best_bss=best_record.get("mean_bss", float("nan")),
            best_positive_folds=best_record.get("positive_folds", 0),
            best_wilcoxon_p=best_record.get("wilcoxon_p"),
            results_df=pd.DataFrame(self._trial_records),
            elapsed_minutes=elapsed,
            study=None,
        )
```

- [ ] **Step 5.4: Run sweep tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_sweep.py -v`
Expected: 10 PASSED

- [ ] **Step 5.5: Commit**

```bash
git add pattern_engine/sweep.py tests/unit/test_sweep.py
git commit -m "feat(p3): add OptunaSweep, GridSweep, SweepResult, KNN_SEARCH_SPACE"
```

---

## Task 6: Re-export Wrapper (phase7_baseline.py)

**Files:**
- Modify: `scripts/phase7_baseline.py`

Replace inline logic with imports from `pattern_engine.walkforward`. Keep `run_fold_with_config()` as a thin wrapper. Re-export all symbols E1–E4 scripts depend on.

**Critical:** E1–E4 scripts must continue to work unchanged. The re-export surface is:
- `run_fold_with_config` → thin wrapper calling `walkforward.run_fold()`
- `_BetaCalibrator` → re-exported from `walkforward`
- `_augment_with_candlestick` → re-exported from `walkforward`
- `_apply_h7_hold_regime` → re-exported from `walkforward`
- `_build_cfg` → re-exported from `walkforward`
- `DATA_DIR` → re-exported from `walkforward` (but computed via `_find_data_dir()`)
- `FEATURE_COLS` → re-exported from `walkforward`
- `HORIZON` → re-exported from `walkforward`
- `SPY_THRESHOLD` → re-exported from `walkforward`

- [ ] **Step 6.1: Rewrite phase7_baseline.py as thin wrapper**

Replace the body of `scripts/phase7_baseline.py` with:

```python
"""
scripts/phase7_baseline.py — Task T7.0: Phase 7 baseline re-establishment.

Runs 6-fold walk-forward with locked Phase 6 settings.
Core logic now lives in pattern_engine.walkforward (R3 extraction).

This file re-exports symbols for backwards compatibility with E1–E4 scripts.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/phase7_baseline.py

Provenance: Phase 7 implementation plan, Task T7.0 (2026-04-09)
             R3 Optuna Infrastructure extraction (2026-04-11)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Repo root resolution ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Re-exports from pattern_engine.walkforward ──────────────────────────────
# E1–E4 scripts import these symbols from phase7_baseline.
# Keep this list in sync with the spec's "Full re-export list".
from pattern_engine.walkforward import (      # noqa: F401 — re-exports
    _BetaCalibrator,
    _augment_with_candlestick,
    _apply_h7_hold_regime,
    _build_cfg,
    _bss,
    _murphy_decomposition,
    _impute_candle_nans,
    FEATURE_COLS,
    HORIZON,
    SPY_THRESHOLD,
    run_fold as _run_fold,
)
from pattern_engine.config import WALKFORWARD_FOLDS

# ── DATA_DIR with worktree fallback ─────────────────────────────────────────
def _find_data_dir() -> Path:
    candidate = REPO_ROOT / "data" / "52t_features"
    if candidate.exists():
        return candidate
    main_repo = REPO_ROOT.parent.parent
    alt = main_repo / "data" / "52t_features"
    if alt.exists():
        return alt
    raise RuntimeError(
        f"52T features data not found.\n"
        f"  Tried: {candidate}\n"
        f"  Tried: {alt}\n"
        f"  Run scripts/build_52t_features.py first."
    )

DATA_DIR    = _find_data_dir()
RESULTS_DIR = REPO_ROOT / "results" / "phase7"
OUTPUT_TSV  = RESULTS_DIR / "baseline_23d.tsv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Thin wrapper for backwards compatibility ─────────────────────────────────

def run_fold_with_config(
    fold: dict,
    full_db: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    cfg_overrides: dict | None = None,
) -> dict:
    """Thin wrapper — delegates to pattern_engine.walkforward.run_fold().

    Kept for backwards compatibility with E1–E4 enhancement scripts.
    """
    return _run_fold(fold, full_db, feature_cols=feature_cols,
                     cfg_overrides=cfg_overrides)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()
    from pattern_engine.features import FeatureRegistry

    FEATURE_SET = "returns_candle"
    feature_cols = list(FeatureRegistry.get(FEATURE_SET).columns)

    print("=" * 72)
    print("  Task T7.0 — Phase 7 Baseline: returns_candle (23D)")
    print("=" * 72)
    print(f"  feature_set   : {FEATURE_SET} ({len(feature_cols)} columns)")
    print(f"  max_distance  : 2.5")
    print(f"  calibration   : beta_abm (H5 locked)")
    print(f"  regime        : H7 HOLD (SPY ret_90d < +{SPY_THRESHOLD:.2f} → base_rate)")
    print(f"  horizon       : {HORIZON}")
    print(f"  data          : {DATA_DIR}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    t_path = DATA_DIR / "train_db.parquet"
    v_path = DATA_DIR / "val_db.parquet"

    if not t_path.exists() or not v_path.exists():
        print(f"ERROR: 52T features data not found in {DATA_DIR}")
        sys.exit(1)

    print(f"Loading 52T features from {DATA_DIR} ...", flush=True)
    full_db = pd.concat(
        [pd.read_parquet(t_path), pd.read_parquet(v_path)], ignore_index=True
    )
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    print(f"  {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers, "
          f"date range {full_db['Date'].min().date()} – {full_db['Date'].max().date()}")

    full_db = _augment_with_candlestick(full_db)

    # ── Run 6-fold walk-forward ───────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  Running 6-fold walk-forward (locked baseline config)")
    print(f"{'─'*72}")

    rows: list[dict] = []
    for fi, fold in enumerate(WALKFORWARD_FOLDS):
        print(f"\n  [{fi+1}/6] {fold['label']}", flush=True)
        t0 = time.time()
        result = run_fold_with_config(fold=fold, full_db=full_db)
        elapsed = time.time() - t0
        bss_s = f"{result['bss']:+.5f}" if not np.isnan(result["bss"]) else "  N/A"
        print(f"         BSS={bss_s}  n_scored={result['n_scored']}  "
              f"n_total={result['n_total']}  base_rate={result['base_rate']:.4f}  "
              f"t={elapsed:.0f}s")
        rows.append(result)

    # ── Write TSV ─────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_TSV, sep="\t", index=False, float_format="%.8f")
    print(f"\nSaved: {OUTPUT_TSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    bss_vals = [r["bss"] for r in rows if not np.isnan(r["bss"])]
    mean_bss = float(np.mean(bss_vals)) if bss_vals else float("nan")
    pos_folds = sum(1 for b in bss_vals if b > 0)

    total = time.time() - t_total
    print()
    print("=" * 72)
    print("  BASELINE SUMMARY")
    print("=" * 72)
    print(f"  {'Fold':<14}  {'BSS':>9}  {'n_scored':>8}  {'n_total':>7}  {'base_rate':>9}")
    print(f"  {'─'*14}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*9}")
    for r in rows:
        bss_s = f"{r['bss']:+.5f}" if not np.isnan(r["bss"]) else "     N/A"
        print(f"  {r['fold']:<14}  {bss_s:>9}  {r['n_scored']:>8}  "
              f"{r['n_total']:>7}  {r['base_rate']:>9.4f}")
    print(f"  {'─'*14}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*9}")
    mean_s = f"{mean_bss:+.5f}" if not np.isnan(mean_bss) else "     N/A"
    print(f"  {'MEAN':<14}  {mean_s:>9}  {pos_folds}/6 folds positive")
    print()
    print(f"  Total runtime: {total:.0f}s")
    print(f"  Output: {OUTPUT_TSV}")
    print()

    if np.isnan(mean_bss):
        print("[WARN] Mean BSS is NaN — check fold results above.")
    elif pos_folds >= 3:
        print(f"[PASS] {pos_folds}/6 positive folds (expected ≥ 3 from H7 baseline).")
    else:
        print(f"[WARN] Only {pos_folds}/6 positive folds — baseline may differ.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Run existing test suite to verify no regressions**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: 870+ tests pass (858 existing + ~27 new), 0 failures.

- [ ] **Step 6.3: Commit**

```bash
git add scripts/phase7_baseline.py
git commit -m "refactor(p3): phase7_baseline → thin wrapper over pattern_engine.walkforward"
```

---

## Task 7: Parity Test

**Files:**
- Modify: `tests/unit/test_walkforward.py`

This is the **regression gate**: verify `walkforward.run_fold()` produces identical BSS to the old `phase7_baseline.run_fold_with_config()` on real data. Marked `@pytest.mark.slow` since it requires 52T data and runs a real PatternMatcher fold.

- [ ] **Step 7.1: Add parity test**

Append to `tests/unit/test_walkforward.py`:

```python
@pytest.mark.slow
class TestParity:
    """Parity: walkforward.run_fold() == old phase7_baseline behavior on real data."""

    def test_fold_2019_parity(self):
        """BSS from walkforward.run_fold on fold 2019 matches baseline TSV."""
        from pattern_engine.walkforward import load_and_augment_db, run_fold
        from pattern_engine.config import WALKFORWARD_FOLDS

        full_db = load_and_augment_db()
        fold = WALKFORWARD_FOLDS[0]  # 2019
        result = run_fold(fold, full_db)

        # Compare against known baseline from results/phase7/baseline_23d.tsv
        import csv
        baseline_path = Path("results/phase7/baseline_23d.tsv")
        if not baseline_path.exists():
            pytest.skip("Baseline TSV not found — run phase7_baseline.py first")

        with open(baseline_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["fold"] == "2019":
                    expected_bss = float(row["bss"])
                    break
            else:
                pytest.fail("2019 fold not found in baseline TSV")

        assert result["bss"] == pytest.approx(expected_bss, abs=1e-6), \
            f"Parity failure: walkforward BSS={result['bss']}, baseline BSS={expected_bss}"
```

- [ ] **Step 7.2: Run parity test (slow)**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_walkforward.py::TestParity -v -m slow`
Expected: 1 PASSED (requires 52T data + baseline TSV)

- [ ] **Step 7.3: Commit**

```bash
git add tests/unit/test_walkforward.py
git commit -m "test(p3): add parity test — walkforward.run_fold vs baseline TSV"
```

---

## Task 8: Integration Test — End-to-End OptunaSweep

**Files:**
- Modify: `tests/unit/test_sweep.py`

One `@pytest.mark.slow` test that runs a 3-trial OptunaSweep with `run_fold` on real 52T data. Validates the full pipeline: data loading → Optuna TPE → fold execution → results.

- [ ] **Step 8.1: Add slow integration test**

Append to `tests/unit/test_sweep.py`:

```python
@pytest.mark.slow
class TestIntegration:
    """End-to-end integration test on real 52T data."""

    def test_optuna_3_trials_on_real_data(self):
        """3-trial OptunaSweep with real data produces valid SweepResult."""
        from pattern_engine.walkforward import load_and_augment_db, run_walkforward
        from pattern_engine.sweep import OptunaSweep

        full_db = load_and_augment_db()

        def knn_objective(config, db):
            return run_walkforward(db, cfg_overrides=config)

        sweep = OptunaSweep(
            study_name="integration_test",
            objective_fn=knn_objective,
            search_space={"max_distance": (1.5, 3.5)},
            n_trials=3,
            max_hours=2.0,
        )
        result = sweep.run(full_db, verbose=1)

        assert len(result.results_df) == 3
        assert result.best_config is not None
        assert isinstance(result.best_bss, float)
        assert result.elapsed_minutes > 0
```

- [ ] **Step 8.2: Run integration test**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_sweep.py::TestIntegration -v -m slow`
Expected: 1 PASSED (takes ~6 min for 3 trials × 6 folds × ~20s/fold)

- [ ] **Step 8.3: Commit**

```bash
git add tests/unit/test_sweep.py
git commit -m "test(p3): add slow integration test — 3-trial OptunaSweep on real data"
```

---

## Task 9: Final Validation + Full Suite

**Files:** None new — validation only.

- [ ] **Step 9.1: Run full fast test suite**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: 885+ tests pass (858 existing + ~27 new), 0 failures.

- [ ] **Step 9.2: Verify module imports work cleanly**

Run:
```bash
PYTHONUTF8=1 py -3.12 -c "from pattern_engine.walkforward import run_fold, run_walkforward, load_and_augment_db; print('walkforward OK')"
PYTHONUTF8=1 py -3.12 -c "from pattern_engine.sweep import OptunaSweep, GridSweep, SweepResult, KNN_SEARCH_SPACE; print('sweep OK')"
PYTHONUTF8=1 py -3.12 -c "from pattern_engine.experiment_log import ExperimentLogger; print('experiment_log OK')"
```
Expected: All three print "OK".

- [ ] **Step 9.3: Verify E1–E4 re-exports still work**

Run:
```bash
PYTHONUTF8=1 py -3.12 -c "import sys; sys.path.insert(0, 'scripts'); from phase7_baseline import run_fold_with_config, _BetaCalibrator, _augment_with_candlestick, _apply_h7_hold_regime, _build_cfg, DATA_DIR, FEATURE_COLS, HORIZON, SPY_THRESHOLD; print('re-exports OK')"
```
Expected: Prints "re-exports OK".

- [ ] **Step 9.4: Final commit with any fixes**

If any validation step found issues, fix and commit. Otherwise, no commit needed.

---

## Success Criteria Checklist

| # | Criterion | Verified By |
|---|-----------|-------------|
| 1 | `walkforward.run_fold()` passes parity test vs baseline TSV | Task 7 |
| 2 | `OptunaSweep` completes trials with SQLite persistence | Task 5 (test_sqlite_persistence) |
| 3 | `GridSweep` reproduces exhaustive enumeration | Task 5 (test_exhaustive_enumeration) |
| 4 | TSV output follows provenance format with metadata headers | Task 4 (test_header_writes_metadata) |
| 5 | E1–E4 scripts work via re-export wrapper | Task 6 + Step 9.3 |
| 6 | 885+ tests pass (`pytest tests/ -q -m "not slow"`) | Task 9 |
| 7 | `@pytest.mark.slow` integration test validates end-to-end | Task 8 |
