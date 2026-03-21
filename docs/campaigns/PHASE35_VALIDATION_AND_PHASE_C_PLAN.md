# Phase 3.5 Validation, Risk Tuning & Phase C Kickoff — Execution Plan

**Status:** Active campaign
**Branch convention:** `validate/phase35-*` per experiment
**Provenance rule:** Every BSS/Sharpe number must cite terminal output or a TSV row. Fabricated metrics are forbidden.

---

## Promotion Gate (blocks all Phase C work)

> Phase C CANNOT begin until **both** of the following are true:
> 1. At least one research module clears **BSS ≥ 0.02 above baseline (+0.00103)** on the 2024 fold
> 2. All 574 tests still pass with the wired module

---

## Phase 1: Validation Sprint — Research vs. Locked Settings

### 1A: EMDDistance + BMACalibrator vs. Euclidean + Platt

**Why a standalone script (not WalkForwardRunner):**
`EngineConfig.distance_metric` passes a string directly to `sklearn.NearestNeighbors(metric=...)`. It cannot accept an `EMDDistance` object without modifying `matching.py` — which is read-only until promotion. The validation script runs EMD outside the normal pipeline.

**Create:** `scripts/validate_emd_bma.py`

```python
"""
validate_emd_bma.py — Standalone fold comparison: EMDDistance + BMACalibrator vs baseline.

Logs results to ExperimentLogger (experiments.tsv) for provenance.
Usage: python scripts/validate_emd_bma.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from pattern_engine.experiment_logging import ExperimentLogger
from pattern_engine.data import load_db           # adjust to actual import
from research.emd_distance import EMDDistance
from research.bma_calibrator import BMACalibrator

# --- Config (mirrors locked settings) ---
FOLD_TRAIN_END   = "2023-12-31"
FOLD_VAL_START   = "2024-01-01"
FOLD_VAL_END     = "2024-12-31"
TOP_K            = 50
MAX_DISTANCE     = 1.1019
FEATURE_COLS     = [f"fwd_{w}d_ret" for w in [1,3,7,14,30,45,60,90]]  # returns_only(8)
TARGET_COL       = "fwd_7d_up"

logger = ExperimentLogger()
db = load_db()  # full historical DB

train = db[db.index <= FOLD_TRAIN_END]
val   = db[(db.index >= FOLD_VAL_START) & (db.index <= FOLD_VAL_END)]

X_train = train[FEATURE_COLS].values   # (N_train, 8)
X_val   = val[FEATURE_COLS].values     # (N_val, 8)
y_train = train[TARGET_COL].values
y_val   = val[TARGET_COL].values

# --- EMDDistance: fit and compute pairwise distances ---
metric = EMDDistance(time_penalty=0.5, price_penalty=1.0)
metric.fit(X_train)

raw_probs = np.zeros((len(X_val), TOP_K))
for i, query in enumerate(X_val):
    distances = metric.compute(query, X_train)         # (N_train,)
    within_max = distances <= MAX_DISTANCE
    top_idx = np.argsort(distances[within_max])[:TOP_K]
    analogues = np.where(within_max)[0][top_idx]
    # Raw prob = fraction of analogues with positive outcome
    raw_probs[i] = y_train[analogues[:TOP_K]] if len(analogues) >= TOP_K else np.nan

# Drop rows with insufficient analogues
valid = ~np.isnan(raw_probs).any(axis=1)
raw_probs_valid = raw_probs[valid]
y_val_valid     = y_val[valid]

# --- BMACalibrator ---
# CRITICAL: Mirror the production cal_frac=0.76 chronological split to prevent
# temporal leakage. The Platt baseline respects this split; BMA must too or
# any BSS improvement is a false positive.
CAL_FRAC = 0.76
cal_split = int(len(X_train) * CAL_FRAC)
X_fit, X_cal = X_train[:cal_split], X_train[cal_split:]
y_fit, y_cal  = y_train[:cal_split], y_train[cal_split:]

cal = BMACalibrator(n_iter=30)
# Build cal raw_probs: for each cal sample, top-K neighbours from fit portion only
cal_raw = np.zeros((len(X_cal), TOP_K))
for i, query in enumerate(X_cal):
    distances = metric.compute(query, X_fit)            # query against fit only
    top_idx = np.argsort(distances)[:TOP_K]
    cal_raw[i] = y_fit[top_idx]

cal.fit(cal_raw, y_cal)

# Calibrated probs: one scalar per val row
cal_probs = np.array([float(cal.transform(row)) for row in raw_probs_valid])

# --- Metrics ---
baseline_bss = 0.00103   # fold 6, walkforward terminal output — do NOT update without new run
brier_emd_bma = brier_score_loss(y_val_valid, cal_probs)
brier_ref     = brier_score_loss(y_val_valid, np.full(len(y_val_valid), y_val_valid.mean()))
bss_emd_bma   = 1 - (brier_emd_bma / brier_ref)

print(f"BSS (EMD+BMA):  {bss_emd_bma:+.5f}")
print(f"BSS (baseline): {baseline_bss:+.5f}")
print(f"Delta:          {bss_emd_bma - baseline_bss:+.5f}")
print(f"Gate cleared:   {bss_emd_bma - baseline_bss >= 0.02}")

# Log to TSV — provenance
logger.log(
    experiment_name="validate_emd_bma_2024fold",
    fold="2024",
    brier_skill_score=bss_emd_bma,
    notes=f"EMDDistance(tp=0.5,pp=1.0)+BMA(df=3,n_iter=30) vs baseline {baseline_bss}",
)
```

**Run:**
```bash
python scripts/validate_emd_bma.py
```

**Success gate:** `BSS (EMD+BMA) ≥ +0.02103`
**Failure path:** If BSS < gate, try tuning `time_penalty` (0.0–2.0) and `price_penalty` before concluding EMD adds no edge. Log each run.

---

### 1B: Feature Set Comparison — `returns_overnight` and `returns_session`

`EngineConfig.feature_set` is already pluggable with no code changes. Use `WalkForwardRunner` directly.

**Run baseline (locked):**
```bash
python -c "
from pattern_engine.config import EngineConfig
from pattern_engine.walkforward import WalkForwardRunner
from pattern_engine.data import load_db
from pattern_engine.experiment_logging import ExperimentLogger

logger = ExperimentLogger()
db = load_db()

for fs in ['returns_only', 'returns_overnight', 'returns_session']:
    cfg = EngineConfig(
        feature_set=fs,
        nn_jobs=1,                  # LOCKED — Windows/Py3.12 deadlock prevention
        top_k=50,
        max_distance=1.1019,
        confidence_threshold=0.65,
    )
    runner = WalkForwardRunner(config=cfg, logger=logger)
    runner.run(db, experiment_name=f'feature_set_{fs}_2024')
"
```

**Expected output per run:** Fold-by-fold BSS printed to terminal + logged to `results/experiments.tsv`

**Success gate:** Any feature set showing mean BSS improvement ≥ 0.02 over `returns_only` baseline
**Constraint:** `nn_jobs=1` in every config — never change this

---

## Phase 2: Risk Engine Tuning & Slip-Deficit Integration

### 2A: ATR Stop Sweep — 3.0× to 4.0×

**Context:** Phase 2 results showed `stop_loss_atr_multiple=2.0` caused 38% premature exit rate and Sharpe regression (1.82 → 1.16). Target range: 3.0–4.0.

**Run using BayesianSweepRunner or a manual grid:**
```bash
python -c "
from pattern_engine.sweep import BayesianSweepRunner
from pattern_engine.config import EngineConfig
from trading_system.config import RiskConfig
from pattern_engine.experiment_logging import ExperimentLogger
from pattern_engine.data import load_db

logger = ExperimentLogger()
db = load_db()

for atr_mult in [3.0, 3.25, 3.5, 3.75, 4.0]:
    risk_cfg = RiskConfig(stop_loss_atr_multiple=atr_mult)
    eng_cfg  = EngineConfig(nn_jobs=1)
    # Wire risk_cfg into BacktestEngine via trading system config
    # (exact param name: see BacktestConfig or trading_system/__init__.py)
    logger.log(
        experiment_name=f'atr_sweep_{atr_mult}x',
        fold='2024',
        notes=f'stop_loss_atr_multiple={atr_mult}',
    )
    print(f'Running ATR={atr_mult}x ...')
    # [run backtest here — see BacktestEngine usage in test_backtest_engine.py]
"
```

**Log all runs to TSV.** Do not declare a winner without provenance.
**Decision criteria:** Choose `atr_mult` that maximises Sharpe without MaxDD > 8%.

---

### 2B: SlipDeficit Wiring into `backtest_engine.py`

**Integration point:** `backtest_engine.py` line ~536, immediately before `decision = size_position(...)`.

**Exact change:**
```python
# --- SlipDeficit TTF gate (Phase 3.5 research integration) ---
# Tighten stop if short-term vol Z-score signals elevated risk.
# SlipDeficit requires max(sma_window=200, vol_lookback+10=70) rows = 200 rows.
from research.slip_deficit import SlipDeficit as _SlipDeficit  # import at top of file

_sd = _SlipDeficit()                         # stateless; instantiate once per backtest run (move to __init__)
_ticker_close = price_history[["close"]].copy()   # price_history is already a DataFrame — no rebuild
try:
    _overlay = _sd.compute(_ticker_close)
    _effective_atr_mult = (
        1.5 if _overlay.ttf_probability > 0.8
        else cfg_risk.stop_loss_atr_multiple
    )
except ValueError:
    # Insufficient history — use configured multiple unchanged
    _effective_atr_mult = cfg_risk.stop_loss_atr_multiple

# Pass tightened multiple to risk engine via a modified config
import dataclasses as _dc
_cfg_risk_effective = _dc.replace(cfg_risk, stop_loss_atr_multiple=_effective_atr_mult)

decision = size_position(
    ticker=ticker,
    entry_price=entry_price,
    current_equity=equity,
    price_history=price_history,
    risk_state=risk_state if risk_state is not None else RiskState.initial(equity),
    config=_cfg_risk_effective,          # ← was cfg_risk
    position_limits=cfg_pos,
    sector_map=self.config.sector_map,
    open_positions=open_positions,
    fractional_shares=self.config.capital.fractional_shares,
)
```

**Notes:**
- Move `_SlipDeficit()` instantiation to `BacktestEngine.__init__` (not inside the loop)
- `dataclasses.replace()` creates a new `RiskConfig` without mutating the shared instance
- `price_history` already pulled above (line ~519); reuse it — no extra data fetch
- Add a test to `tests/test_backtest_engine.py` verifying that high-vol ticker gets `stop_loss_price` closer to entry than low-vol ticker at same ATR multiple

**Run after wiring:**
```bash
python -m pytest tests/test_backtest_engine.py -v
python -m pytest tests/ -q --tb=no   # must still show 574+ passed
```

---

## Phase 3: Phase C Kickoff (Post-Promotion-Gate Only)

> **Do not begin this phase until Phase 1 or Phase 2A has a confirmed TSV-logged result that clears the promotion gate.**

### Domain 1 (Priority): FAISS / HNSW Approximate NN

**Trigger:** EMDDistance proves predictive edge (Phase 1A clears gate) but O(N) scan is too slow for overnight runner (target < 10 ms/query on 50 k fingerprints).

**Integration path:**

1. **Install (prefer `hnswlib` over `faiss-cpu`):**
   ```bash
   python -m pip install hnswlib
   ```
   `faiss-cpu` can fail to compile or lack AVX2 support in mixed Windows/WSL environments.
   `hnswlib` is pip-friendly on Windows and does not require a C++ build chain.
   Fall back to `faiss-cpu` only if `hnswlib` recall benchmarks are insufficient.

2. Implement `research/hnsw_distance.py` subclassing `BaseDistanceMetric`:
   - `fit()`: build `hnswlib.Index` (space=`l2`, dim=8) on `X_train`
   - `compute()`: query index, return distances shape `(N,)`

3. Validate recall@50 ≥ 0.95 vs exact ball_tree on held-out set

4. Only then: modify `matching.py` to accept `BaseDistanceMetric` objects (Phase C promotion unlocks this file)

5. Swap `EngineConfig.distance_metric` from string → `BaseDistanceMetric` instance

**Keep `nn_jobs=1`** — HNSW handles its own threading; joblib parallelism is still forbidden.

---

### Domain 2: Hawkes Process + Multiplex Contagion

**Trigger:** Signal clustering observed in live runs (≥ 3 BUY signals within 2 days on correlated tickers).

**Sequence:**
1. Implement `research/hawkes_overlay.py` subclassing `BaseRiskOverlay`
2. Fit Hawkes process on training signal timestamps (`mu`, `alpha`, `beta` via `scipy.optimize.minimize`)
3. Wire as a second overlay in `backtest_engine.py` alongside SlipDeficit
4. Promotion gate: reduces max-drawdown in 2024 backtest vs baseline (logged to TSV)

---

### Domain 3: OODA / CPOD Circuit Breakers

**Trigger:** Regime-break events (2008/2020-style) detected in live monitoring.

**Sequence:**
1. Implement `research/cpod_detector.py` — CUSUM-based change-point on rolling BSS
2. Integrate at `matching.py` query stage as an outlier pre-filter (flagged fingerprints skipped)
3. Promotion gate: CPOD fires during known stress periods (2020 COVID month), < 5% false-positive rate on normal data

---

### Domain 4: Case-Based Reasoning + OWA Feature Weighting

**Trigger:** BSS plateau — locked `returns_only(8)` stops improving after ATR tuning.

**Sequence:**
1. Implement `research/owa_weighter.py` using `pattern_engine/regime.py` (existing) for regime state
2. Apply as pre-transform on fingerprints before distance computation
3. **Locked setting change required:** `Features=returns_only(8)` → `Features=owa_weighted(8)` — must cite walk-forward experiment log in `CLAUDE.md` update
4. Promotion gate: BSS ≥ 0.02 improvement on held-out fold

---

## Constraints Checklist (enforce at every step)

| Constraint | Rule |
|-----------|------|
| `nn_jobs=1` | Never change. Windows/Py3.12 joblib deadlock. |
| `matching.py` read-only | Stays unchanged until a module clears the BSS gate |
| `calibration.py` read-only | Same |
| Numbers require provenance | Every result cites `experiments.tsv` row or terminal output |
| Test suite | Must show ≥ 574 passed after each code change before commit |
| Locked settings | `CLAUDE.md` entries unchanged until experiment evidence cited |
