# PROJECT_GUIDE.md — Multi-AI Collaboration Reference
# Last Updated: 2026-03-20
# Owner: Sleep (Isaia)
# Primary AI: Claude (Anthropic) | Supporting: Gemini, ChatGPT

---

## QUICK CONTEXT FOR ANY AI READING THIS

You are helping build **FPPE (Financial Pattern Prediction Engine)** — an autonomous
financial prediction system. The core engine uses K-nearest-neighbor historical
analogue matching on return fingerprints to generate probabilistic BUY/SELL/HOLD
signals across a 52-ticker universe.

The project has three codebases:
1. **`pattern_engine/`** — Python package (21 modules; bulk of the automated test suite)
2. **`trading_system/`** — Four-layer trading system built on FPPE signals (**Phase 1 complete; Phase 2 risk engine integrated** — see `docs/PHASE2_RESULTS.md`)
3. **`pattern-engine-v2.1.jsx`** — Standalone React demo (claude.ai artifact, no HTTP calls)

**Key design documents (always reference before modifying related code):**
- `docs/FPPE_TRADING_SYSTEM_DESIGN.md` v0.3 — Architecture spec for all four trading layers
- `docs/PHASE1_FILE_REVIEW.md` — Structural stability review; all Phase 1 bugs documented and fixed
- `docs/CANDLESTICK_CATEGORIZATION_DESIGN.md` v0.2 — Future K-NN pre-filtering module (Phase 6)
- `docs/PHASE2_SYSTEM_DESIGN.md` — Phase 2 system design

**Do NOT modify `prepare.py` or this file unless explicitly asked.**

**When starting any AI session, provide:**
1. This file (PROJECT_GUIDE.md)
2. The relevant source files for the task at hand
3. Terminal output or screenshots from the most recent run

**NUMBERS REQUIRE PROVENANCE.** Any claimed metric must trace to walk-forward
results or experiment logs. If it cannot be traced, it is fabricated.

---

## 1. PRODUCTION VISION

The final system operates as a fully automated overnight pipeline:

1. **4:00 PM ET** — US markets close
2. **4:05 PM ET** — Pipeline fetches end-of-day OHLCV data for all tickers via yfinance
3. **4:10 PM ET** — Compute 8-feature return fingerprint for each ticker
4. **4:15 PM ET** — Run analogue search: find K nearest historical twins across 25-year database, filtered to same macro regime
5. **4:30 PM ET** — Apply calibrated probabilities, generate BUY/SELL/HOLD signals via three-filter gate
6. **4:40 PM ET** — Save overnight report (JSON + text) and deliver alerts
7. **9:15 AM ET** — Predictions delivered before market open

**Ticker universe:** Starting at 52, expanding to 500+ (S&P 500).
The pipeline is designed for arbitrary scale from the start.
The three-filter signal gate keeps trade volume manageable at any universe size.

---

## 2. SYSTEM ARCHITECTURE

### Package Structure (v2.1)

```
pattern_engine/                    # 21 modules, version 2.1.0
├── __init__.py                    # Public API: PatternEngine, EngineConfig, CrossValidator
├── config.py                      # EngineConfig frozen dataclass (all proven defaults)
├── features.py                    # FeatureSet/FeatureRegistry (7 pluggable sets + hybrid)
├── matching.py                    # Core KNN matcher (ball_tree, batched, 52-ticker SECTOR_MAP)
├── projection.py                  # Forward projection + three-filter signal gate
├── calibration.py                 # Platt/isotonic/none probability calibrators
├── regime.py                      # Binary/multi/octet market regime detection
├── scoring.py                     # Brier Score, BSS, CRPS, calibration buckets
├── evaluation.py                  # Signal-aligned + probabilistic evaluation
├── candlestick.py                 # Continuous multi-timeframe encoding (1d/3d/5d)
├── sector.py                      # Cross-asset sector features (vectorized)
├── engine.py                      # PatternEngine class (fit/predict/evaluate/save/load)
├── schema.py                      # Native DataFrame validation at engine boundaries
├── cross_validation.py            # Cross-model validation & integrity checking
├── data.py                        # DataLoader pipeline (yfinance → features → parquet)
├── experiment_logging.py          # TSV experiment logger with full config hash (all ~20 fields)
├── walkforward.py                 # 6-fold walk-forward with optional cross-validation
├── sweep.py                       # Grid sweep + Bayesian (Optuna TPE) sweep
├── live.py                        # Production EOD signal generator with consensus mode
├── overnight.py                   # Multi-phase runner (static + Bayesian modes)
└── reliability.py                 # Atomic writes, lock files, progress logging
```

### Trading System Package (Phase 1 complete)

```
trading_system/                    # 4 modules, 88 tests, Phase 1 backtest complete
├── __init__.py                    # Full package exports (all primary symbols)
├── config.py                      # TradingConfig: 7 frozen sub-configs, from_profile()
├── signal_adapter.py              # Normalizes FPPE K-NN / DL outputs → UnifiedSignal
├── backtest_engine.py             # Layer 1: Trade simulation, friction model, P&L
└── run_phase1.py                  # Phase 1 entry point (equal-weight, cached signals)
```

**Trading system design reference:** `docs/FPPE_TRADING_SYSTEM_DESIGN.md` v0.3
**Phase 1 bug audit:** `docs/PHASE1_FILE_REVIEW.md` — 9 findings, all fixed

### Legacy Files (reference only — do not modify)

| File | Purpose |
|------|---------|
| `strategy.py` | Original monolithic engine (2,498 lines) — superseded by pattern_engine/ |
| `strategy_overnight.py` | Original overnight runner — superseded by overnight.py |
| `strategyv1.py` through `strategyv4.py` | Historical iterations |
| `prepare.py` | Data pipeline (downloads OHLCV, builds analogue DB) — **human-only** |

### Hardware
- CPU: AMD Ryzen 9 5900X (12-core, 24 threads, 3.70 GHz)
- RAM: 32GB DDR4, OS: Windows 10, Python 3.12
- venv: `C:\Users\Isaia\.claude\financial-research\venv`

---

## 3. CORE API

```python
from pattern_engine import PatternEngine, EngineConfig

# Default usage (all proven research defaults)
engine = PatternEngine()
engine.fit(train_db)                    # Fit scaler, NN index, calibrator
result = engine.predict(query_db)       # Generate predictions
metrics = engine.evaluate(val_db)       # Predict + score

# Method chaining
metrics = PatternEngine().fit(train_db).evaluate(val_db)

# Persistence
engine.save("engine_state.pkl")
engine = PatternEngine.load("engine_state.pkl")

# Feature swapping (config change, not code change)
config = EngineConfig(feature_set="returns_candle")
engine = PatternEngine(config)

# Walk-forward validation
from pattern_engine.walkforward import WalkForwardRunner
from pattern_engine.config import WALKFORWARD_FOLDS
runner = WalkForwardRunner(config, folds=WALKFORWARD_FOLDS)
fold_metrics = runner.run(full_db)

# Grid sweep
from pattern_engine.sweep import SweepRunner
configs = SweepRunner.grid(max_distance=[0.8, 1.0, 1.1019], regime_mode=["binary", "multi"])
results = SweepRunner(configs).run(full_db)

# Bayesian optimization (NEW v2.1)
from pattern_engine.sweep import BayesianSweepRunner
runner = BayesianSweepRunner(n_trials=50, storage="sqlite:///study.db")
results = runner.run(full_db)

# Overnight runner (Bayesian mode)
from pattern_engine.overnight import OvernightRunner
runner = OvernightRunner(bayesian_mode=True, n_trials=50)
results = runner.run(full_db)
```

---

## 4. DATASET

### Current Tickers (52 across 7 sectors)
**Index (2):** SPY, QQQ
**Tech (19):** AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, ORCL, ADBE,
CRM, AMD, NFLX, INTC, CSCO, QCOM, TXN, MU, PYPL
**Finance (9):** JPM, BAC, WFC, GS, MS, V, MA, AXP, BRK-B
**Health (10):** LLY, UNH, JNJ, ABBV, MRK, PFE, TMO, ISRG, AMGN, GILD
**Consumer (7):** WMT, COST, PG, KO, PEP, HD, DIS *(DIS moved from Industrial — C1 fix)*
**Industrial (3):** CAT, BA, GE
**Energy (2):** XOM, CVX

### Data Split (Temporal — NO leakage)
- **Training:** 2010-01-01 → 2023-12-31 (~175,605 rows)
- **Validation:** 2024-01-01 → 2024-12-31 (~13,104 rows)
- **Test:** 2025-01-01 → 2026-01-28 (13,936 rows — held out, do not touch)

### Active Feature Set: Return-Only (8 features) — LOCKED
Full 16-feature configs produced BSS ≈ −0.135, 3.5× worse. Only these 8 used:
```
ret_1d, ret_3d, ret_7d, ret_14d, ret_30d, ret_45d, ret_60d, ret_90d
```

### Feature Weights (v2.1)
```python
feature_weights = {
    "ret_7d": 1.5,           # Medium-term trend (proven signal) — weighted UP
    "ret_14d": 1.5,          # Medium-term trend (proven signal) — weighted UP
    "ret_90d": 0.5,          # Long-term secular (too noisy) — weighted DOWN
    "vol_10d": 1.2,          # Volatility context
    "price_vs_sma20": 1.2,   # Mean-reversion signal
    # All others default to 1.0
}
```

### Additional Feature Sets (available, pluggable via config)
| Set | Features | Status |
|-----|----------|--------|
| `returns_only` | 8 trailing returns | **Default — proven** |
| `returns_candle` | 8 returns + 15 candlestick | Available |
| `returns_vol` | 8 returns + 4 volatility | Available |
| `returns_sector` | 8 returns + 3 sector-relative | Available |
| `returns_overnight` | 8 returns + 4 overnight/session | New — research-driven |
| `returns_session` | 8 returns + 4 overnight + 2 weekend | New — research-driven |
| `full` | 34 features combined | Experimental |
| `returns_hybrid` | 8 returns + 16 LSTM latent dims | Placeholder (needs trained model) |

---

## 5. THE ANALOGUE ENGINE PIPELINE

### 7-Step Process
1. **Data Ingestion** (`data.py`) — Load/normalize analogue DB from CSV/parquet
2. **Feature Engineering** (`features.py`) — Compute 8 return windows + optional features
3. **K-NN Matching** (`matching.py`) — Ball_tree spatial index, weighted Euclidean distance, top-K within max_distance
4. **Regime Filtering** (`regime.py`) — Classify market into 8 states, filter matches to same regime
5. **Probability Calibration** (`calibration.py`) — Platt scaling double-pass (train-as-query)
6. **Signal Generation** (`projection.py`) — Three-filter gate → BUY/SELL/HOLD
7. **Evaluation** (`scoring.py`) — Brier Score, BSS, CRPS

### Signal Generation — Three Required Filters (all must pass)
```python
if n_matches < MIN_MATCHES (10):       → HOLD
if agreement < AGREEMENT_SPREAD (0.10): → HOLD
if prob >= CONFIDENCE_THRESHOLD (0.65): → BUY
if prob <= 1 - CONFIDENCE_THRESHOLD:    → SELL
else:                                   → HOLD
```

### 8-State Regime Detection
Three orthogonal dimensions → 2×2×2 = 8 states:

| Dimension | Metric | Threshold | States |
|-----------|--------|-----------|--------|
| Direction | SPY 90-day return | > 0 | Bull / Bear |
| Volatility | SPY 30-day stdev | < historical median | Low Vol / High Vol |
| Trend | ADX(14) | > 25.0 | Trending / Ranging |

Fallback chain when insufficient matches: octet (8) → multi (4) → binary (2) → unfiltered

### Calibration Double-Pass
The calibrator trains on training data querying itself (not val data):
1. Build K-NN index from training data
2. Query training data against itself (cal_frac=0.76 holdout)
3. Fit Platt sigmoid on self-query frequencies + known outcomes
4. No look-ahead bias — calibrator sees same distribution as inference

### Locked Hyperparameters
```python
distance_metric = "euclidean"        # Cosine collapsed at 93.3% saturation
max_distance = 1.1019                # Quantile-calibrated (AvgK ~42)
top_k = 50                           # Neighbourhood ceiling
distance_weighting = "uniform"       # Beats inverse (sweep 1)
projection_horizon = "fwd_7d_up"     # Best BSS across horizons
confidence_threshold = 0.65          # Best accuracy trade-off (sweep 1)
agreement_spread = 0.10              # Minimum directional agreement
min_matches = 10                     # Minimum analogues required
calibration_method = "platt"         # Generalises best across folds
cal_frac = 0.76                      # Best calibration holdout
regime_filter = True                 # Binary mode (Bull/Bear)
regime_mode = "binary"               # Proven default
nn_jobs = 1                          # Prevents Win/Py3.12 joblib deadlock
batch_size = 256                     # Memory-efficient batched queries
```

---

## 6. EVALUATION METRICS

### Primary: Brier Skill Score (BSS)
```
BSS = 1 - (Brier_model / Brier_climatology)
BSS > 0    = beats base rate  ← ACHIEVED on 2024 fold (+0.00103)
BSS > 0.05 = materially useful ← next target
BSS < 0    = worse than base rate
```

### Secondary
- **CRPS**: Continuous Ranked Probability Score (lower = better)
- **Accuracy**: % correct on confident BUY/SELL trades only
- **AvgK**: Average analogues after all filters (target ~42)

### Walk-Forward Results (6 Expanding Folds)

| Fold | Period | BSS | Accuracy | Avg Matches |
|------|--------|-----|----------|-------------|
| 1 | 2019 | −0.01211 | 61.2% | 39.3 |
| 2 | 2020 (COVID) | −0.00142 | 57.4% | 25.1 |
| 3 | 2021 | −0.00131 | 57.7% | 42.6 |
| 4 | 2022 (Bear) | −0.03208 | 48.7% | 21.2 |
| 5 | 2023 | −0.00029 | 56.2% | 40.1 |
| **6 (Standard)** | **2024** | **+0.00103** | **56.6%** | **41.9** |

**Key observations:**
- 2024 fold = positive BSS — genuine predictive skill on unseen data
- 2022 bear market = worst fold (regime shift, fewer bear training analogues)
- Expanding window means later folds have more training data → better performance
- 458 automated tests all passing (pattern_engine + expanded trading_system including Phase 2 risk tests)

---

## 7. PHASE 1 TRADING SYSTEM RESULTS

**Status:** Phase 1 complete (all bugs fixed). **Phase 2 risk layer** (`risk_engine.py`, `risk_state.py`, backtest integration) is **shipped** — see §7.6 and `docs/PHASE2_RESULTS.md`.
**Full specification:** `docs/FPPE_TRADING_SYSTEM_DESIGN.md` v0.3
**Structural audit:** `docs/PHASE1_FILE_REVIEW.md` — 9 findings (3 critical, 5 significant, 1 deferred)

### 7.1 Phase 1 Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Capital | $10,000 | Paper trading baseline |
| Position sizing | Equal-weight 5% | Phase 1 only; Phase 2 adds volatility-based sizing |
| Confidence threshold | 0.60 | Empirically optimal (threshold_sweep.py, 5 values) |
| Max holding days | 14 | Empirically optimal (holding_period_sweep.py, 7 windows) |
| Transaction friction | 26 bps round-trip | 10 slippage + 3 spread, both sides |
| Signal source | Cached FPPE K-NN | 2024 validation signals |

### 7.2 Phase 1 Results (2024 validation year)

| Metric | FPPE System | SPY Buy & Hold |
|--------|-------------|----------------|
| Annual return | 22.3% | 25.4% |
| Sharpe ratio | **1.82** | 1.52 |
| Max drawdown | **6.9%** | 8.4% |
| Win rate | 60.8% | — |
| Net expectancy | $7.39/trade | — |
| Profit factor | 1.83 | — |
| Total trades | 278 | 1 |
| Avg idle cash | ~25% | 0% |

**v1 success criteria: ALL PASSED.** Net expectancy > $0 after friction ✓, confidence calibration within expected range ✓, drawdown ≤ 25% ✓, beats SPY on risk-adjusted basis (Sharpe 1.82 > 1.52) ✓.

**CAUTION:** 2024 was a strong bull year. Longer hold (14d) captures market beta. Re-validate max_holding_days in bear-market conditions before treating as permanent.

### 7.3 Risk Profiles (empirically validated)

| Profile | Threshold | Trades | Annual | Sharpe | Max DD | NE/Trade |
|---------|-----------|--------|--------|--------|--------|----------|
| Conservative | 0.68 | ~90 est. | ~10-12% est. | ~1.1 est. | ~3% est. | ~$5+ est. |
| Moderate | 0.63 | 255 | 14.4% | 1.39 | 4.5% | $3.90 |
| **Aggressive** | **0.60** | **278** | **22.3%** | **1.82** | **6.9%** | **$7.39** |

*Conservative estimates are interpolated — run threshold_sweep.py at 0.68 to confirm empirically.*

### 7.4 Phase 1 Bugs Fixed

All findings from `docs/PHASE1_FILE_REVIEW.md`:

| ID | File | Finding | Status |
|----|------|---------|--------|
| C1 | config.py | DIS misclassified as Industrial (should be Consumer) | **Fixed** |
| C2 | config.py | validate() missing SignalConfig checks | **Fixed** |
| C3 | backtest_engine.py | "Unknown" sector bypassed concentration limits | **Fixed** |
| S1 | backtest_engine.py | iterrows() → to_dict('records') (~5× faster lookup construction) | **Fixed** |
| S2 | config.py | from_profile() described in design doc but not implemented | **Fixed** |
| S3 | All modules | Stale design doc version references (v0.2 → v0.3) | **Fixed** |
| S4 | __init__.py | Only 2 symbols exported; full API needed manual submodule imports | **Fixed** |
| S5 | backtest_engine.py | P&L double-counting of entry friction (understated $6.65→$7.39/trade) | **Fixed** |
| D1 | backtest_engine.py | Force-close exit friction not subtracted from final_equity() | **Fixed** |
| D2 | backtest_engine.py | _advance_trading_days() silently truncated year-boundary cooldowns | **Fixed** |
| D3 | backtest_engine.py | strategy_return_excl_cash is approximate (error < 1%) | Deferred to Phase 4 |

### 7.5 Open Items After Phase 2

| Item | Priority |
|------|----------|
| Conservative profile empirical validation (run threshold_sweep.py at 0.68) | Medium |
| D3: TWRR decomposition in strategy_evaluator.py | Low — Phase 4 |

### 7.6 Phase 2 — Risk engine (complete)

**Delivered:** `trading_system/risk_engine.py`, `trading_system/risk_state.py`, `use_risk_engine` integration in `backtest_engine.py`, `run_phase2.py`, unit + integration tests (`tests/test_risk_*.py`, `tests/test_phase2_integration.py`). **Spec / approval:** `docs/SLE-9_APPROVED_IMPLEMENTATION_PLAN.md`, `docs/PHASE2_SYSTEM_DESIGN.md`, cross-review `docs/SLE-28_PHASE_2_CROSS_AGENT_REVIEW.md`. **Validation notes:** `docs/PHASE2_RESULTS.md`.

---

## 8. v2.1 NEW FEATURES

### 7.1 Bayesian Optimization (Optuna TPE)

Replaces exhaustive grid search with intelligent exploration. Optuna's Tree-structured
Parzen Estimator (TPE) models good vs. bad parameter regions and focuses trials on
promising areas.

**Search space:**
| Parameter | Type | Range |
|-----------|------|-------|
| max_distance | Float | [0.8, 2.0] |
| top_k | Integer | [20, 100] |
| calibration_method | Categorical | {platt, isotonic} |
| cal_frac | Float | [0.5, 0.95] |
| confidence_threshold | Float | [0.55, 0.80] |
| regime_mode | Categorical | {binary, multi, octet, off} |

**Key features:**
- **MedianPruner**: Kills unpromising trials after 2 folds if BSS < median (40-60% compute savings)
- **SQLite persistence**: `storage="sqlite:///study.db"` enables cross-session resume
- **Time-budget callbacks**: Configurable wall-clock limits
- **Per-trial isolation**: `catch=(Exception,)` prevents one bad trial from killing the study

**Implementation:** `BayesianSweepRunner` class in `sweep.py` (~220 lines)

### 7.2 Schema Validation

Native DataFrame validation at `engine.fit()` and `engine.predict()` boundaries.
Implemented in `schema.py` without external dependencies (no pandera).

**Checks:**
- Required columns present (Date, Ticker)
- Feature columns exist and are numeric
- Projection horizon column exists with binary values (fit only)
- No NaN in features or target
- Minimum 50 rows for meaningful matching
- All violations aggregated into single SchemaError (not fail-fast)

**Technical detail:** Uses `pd.api.types.is_numeric_dtype()` instead of
`np.issubdtype()` to handle pandas Arrow-backed extension types (StringDtype).

### 7.3 Overnight Runner — Bayesian Mode

`OvernightRunner` gained `bayesian_mode=True` parameter:
- Static mode: cycles through predefined phase configs
- Bayesian mode: creates `BayesianSweepRunner` with SQLite in results dir
- Both modes support checkpoint resume and per-phase isolation

### 7.4 Cross-Model Validation Framework

`CrossValidator` enables models to check each other's outputs:
- Agreement matrix across config variants
- Disagreement flagging (min_spread threshold)
- Consensus signals (majority-vote filtering)
- Pipeline integrity checks (determinism, persistence, calibration sanity)

### 7.5 Reliability Infrastructure

For unattended 6+ hour overnight runs:
- **Atomic writes**: temp + fsync + rename (crash-safe)
- **Lock files**: PID-based, stale lock detection
- **Checkpoint resume**: JSON-serialized phase IDs (static) or SQLite (Bayesian)
- **Per-phase isolation**: try/except per phase, errors logged, next phase continues
- **Progress logging**: Structured append-only log with timestamps

---

## 9. BUGS FIXED (history)

### pattern_engine/ bugs

1. Evaluator/signal mismatch — confident trades counted by threshold only
2. Python default-arg binding — generate_signal called bare
3. Global mutation in sweep — swept wrong module object
4. BSS falsiness — 0.0 or -99 = -99
5. Calibration edge case — probability=1.0 outside all buckets
6. Euclidean threshold arithmetic — wrong values in early sweeps
7. Scaler mismatch (FATAL) — 16-feature saved scaler on 8-column matrix
8. exclude_same_ticker global — now explicit kwarg
9. TSV column misalignment — new fields inserted mid-dict
10. Walk-forward continue trap — BSS/CRPS skipped for zero-trade folds
11. joblib deadlock — n_jobs=-1 + brute + Euclidean on Windows (fix: nn_jobs=1)
12. NaN poisoning — np.nan passes `is not None`
13. Winner printout incomplete — missing distance_metric and feature_set_name
14. Warning watchdog — sklearn flood caused overnight hang
15. Overnight identical cycles — fixed with schedule rotation
16. Regime labeling bug v3→v4 — val queries labeled with train_db SPY rows
17. **hash(config) TypeError** — EngineConfig contains dict (feature_weights), `hash()` fails. Fix: `repr(config)` for Optuna trial user attributes
18. **SQLite WAL PermissionError** — Windows holds SQLite file handles after close. Fix: `TemporaryDirectory(ignore_cleanup_errors=True)` in tests
19. **np.issubdtype() TypeError** — pandas StringDtype breaks numpy dtype check. Fix: `pd.api.types.is_numeric_dtype()` in schema.py
20. **overnight.py checkpoint-on-failure** — `completed.add(phase_id)` executed unconditionally even after exception. Failed phases were permanently marked done and silently skipped on resume. Fix: moved checkpoint write inside `try` (success only); `except` block saves checkpoint WITHOUT the failed phase.
21. **experiment_logging.py hash collision** — `_config_hash()` only hashed 8 of ~20 EngineConfig fields. Two configs differing in `confidence_threshold`, `feature_weights`, etc. shared the same hash. Fix: `dataclasses.asdict()` + `json.dumps(sort_keys=True)` covers all fields.
22. **engine.py cal_frac no-op** — `cal_frac` declared in EngineConfig with comment "Platt cal_frac=0.76 best" but never used in any code path. Bayesian sweep wasted trials optimizing a ghost parameter. Fix: implemented chronological calibration split (earlier `1-cal_frac` → NN index, later `cal_frac` → calibration query) eliminating temporal adjacency leakage.
23. **assert → RuntimeError** (`engine.py`, `matching.py`) — `assert self._fitted` is stripped under `-O` (optimized execution). Public API guards must raise `RuntimeError`. Fix applied to both modules; tests updated accordingly.

### trading_system/ bugs (documented in docs/PHASE1_FILE_REVIEW.md)

24. **C1: DIS sector misclassification** — Disney classified as Industrial; corrected to Consumer. Sector concentration enforcement was silently incorrect.
25. **C2: validate() missing SignalConfig checks** — confidence_threshold=0.30 passed validation. Added bounds checks for confidence_threshold, min_matches, max_holding_days.
26. **C3: Unknown sector bypass** — "Unknown" sector had no concentration tracking; unlimited positions could accumulate silently. Added `warnings.warn()` on unknown sector fallback.
27. **S1: iterrows() in price lookup** — Replaced with `to_dict('records')` (~5× faster for large DataFrames).
28. **S2: from_profile() not implemented** — Design doc described method; code didn't have it. Implemented all three profiles (aggressive/moderate/conservative).
29. **S5: P&L double-counting** — `gross_pnl` used friction-inclusive `entry_price`; `total_costs` re-added entry friction. Net expectancy understated by ~$0.74/trade ($203.71 total across 277 trades). Fix: stored `raw_entry_price` separately, gross_pnl uses raw prices only.
30. **D1: Force-close exit friction in final_equity()** — Last daily MTM record reflected close price without exit friction on force-closed positions. final_equity() overstated by ~$10.50. Fix: accumulate and subtract `force_close_exit_friction`.
31. **D2: Year-end cooldown truncation** — `_advance_trading_days()` silently returned last available date when cooldown exceeded data range. Fix: calendar estimate (`remaining × 1.4 + 1` days) when insufficient trading dates remain.

---

## 9. RUNNING THE SYSTEM

### Environment Setup
```cmd
cd C:\Users\Isaia\.claude\financial-research
venv\Scripts\activate
pip install -r requirements.txt  # includes optuna
```

### Run Tests (always first)
```cmd
python -m pytest tests/ -v
```
All **458 tests** must pass (27 test files under `tests/`, including Phase 2 risk and integration coverage).

### Key Entry Points

| Command | Purpose |
|---------|---------|
| `python -m pattern_engine.live` | Production EOD signals (after market close) |
| `python -m pattern_engine.overnight` | 6-hour overnight runner |
| `python -m pytest tests/ -v` | Full test suite |
| `python -m pattern_engine.walkforward` | Walk-forward validation |
| `python quick_sweep.py` | Quick parameter sweep |

### Bayesian Optimization
```python
from pattern_engine.sweep import BayesianSweepRunner
from pattern_engine.config import WALKFORWARD_FOLDS

runner = BayesianSweepRunner(
    n_trials=50,
    storage="sqlite:///results/bayesian_study.db",  # persist across sessions
    study_name="fppe_v2.1_optimization",
)
results = runner.run(full_db, folds=WALKFORWARD_FOLDS)
print(results.sort_values("mean_bss", ascending=False).head(10))
```

---

## 10. LOCKED SETTINGS (do not re-test without strong evidence)

| Setting | Value | Evidence |
|---------|-------|----------|
| Distance metric | Euclidean | Cosine saturated at 93.3% |
| Distance weighting | "uniform" | Beats inverse (sweep 1) |
| Feature set | returns_only (8) | Full 16-feature 3.5× worse |
| Calibration method | Platt | Generalises best across folds |
| cal_frac | 0.76 | Zero-crossing point for positive BSS |
| max_distance | 1.1019 | Quantile-calibrated, AvgK ~42 |
| top_k | 50 | Neighbourhood ceiling |
| confidence_threshold | 0.65 | Best accuracy trade-off |
| regime_filter | True (binary) | Prevents cross-regime contamination |
| nn_jobs | 1 | Prevents Win/Py3.12 joblib deadlock |
| Horizon | fwd_7d_up | Best BSS across horizons |

---

## 11. HALLUCINATION PREVENTION

### Documented Incidents
- **ChatGPT ×3:** BSS=+0.0483 (fabricated); walk-forward positive BSS (no such run); cross-sectional encoding 16.8% (actual: 92.4%)
- **Gemini ×1:** Walk-forward Mean BSS=+0.0303 — session primed with description of intended steps rather than raw data

### Rules
1. Numbers require provenance — no experiment log row = not real
2. Anchor sessions with raw results, not prose summaries
3. "Positive BSS on walk-forward" requires ≥3/6 folds positive
4. "System complete" requires stable walk-forward
5. "Production-ready" requires BSS > 0.05 and validated pipeline
6. AvgK=50.0 in any fold = regime filter not working — investigate

---

## 12. DELIVERABLES PRODUCED

### 12.1 Python Package (`pattern_engine/`)
21 modules, 300 tests, all passing. Fully modular replacement of monolithic strategy.py.
Merged to main branch 2026-03-18. Legacy files (strategy.py, strategyv1-v4.py) preserved
but marked superseded.

### 12.2 Trading System (`trading_system/`)
4 modules, 88 tests, all passing. Phase 1 (equal-weight backtest) complete.
- Annualized return 22.3%, Sharpe 1.82, Max DD 6.9% on 2024 validation year
- All 9 Phase 1 bugs fixed and documented in `docs/PHASE1_FILE_REVIEW.md`
- `config.py`: 7 frozen sub-configs, 3 risk profiles, full validate() coverage
- `backtest_engine.py`: Correct P&L (no double-counting), corrected final_equity(), calendar-aware cooldowns

### 12.3 Design Documents

| Document | Version | Status | Purpose |
|----------|---------|--------|---------|
| `docs/FPPE_TRADING_SYSTEM_DESIGN.md` | v0.3 | Active | Architecture spec for all 4 trading layers |
| `docs/PHASE1_FILE_REVIEW.md` | — | Complete | Structural audit; all Phase 1 findings and fixes |
| `docs/CANDLESTICK_CATEGORIZATION_DESIGN.md` | v0.2 | Design phase | Future K-NN pre-filtering module (Phase 6) |
| `docs/PHASE2_SYSTEM_DESIGN.md` | — | Active | Phase 2 system design |
| `docs/TECH_DEBT_AUDIT.md` | — | Complete | Tech debt audit |

### 12.4 React Frontend Demo (`pattern-engine-v2.1.jsx`)
Standalone React artifact for claude.ai (~1500 lines). No HTTP calls — all backend
concepts embedded in JavaScript:
- 52 tickers matching SECTOR_MAP exactly
- Feature-vector matching with weighted Euclidean distance
- 8-state regime detection (direction × vol × trend)
- BSS computation and display per horizon tab
- Three-filter signal gate with SignalBadge components
- Walk-forward panel showing 6 folds with BSS per fold
- Feature weight visualization chart
- Methodology section (4 columns: K-NN, Calibration, Regime, Validation)

### 12.5 Project Report (`FPPE_v2.1_Project_Report.docx`)
Professional DOCX report (~20 pages, 4 sections):
1. **System Overview** — pipeline architecture, module map, headline results
2. **Technical Deep Dive** — K-NN matching, calibration double-pass, regime detection, Bayesian optimization, walk-forward, schema validation, reliability
3. **Comparison to Established Firms** — Berkshire, Renaissance, Two Sigma, AQR, D.E. Shaw, Citadel, BlackRock with comparative analysis
4. **Roadmap & Evolution** — v3.0 (neural hybrid) → v4.0 (real-time) → v5.0 (multi-asset)

---

## 13. ROADMAP

### v2.1 — pattern_engine Complete
- [x] Modular pattern_engine package (21 modules)
- [x] K-NN analogue matching with weighted Euclidean distance
- [x] 8-state regime detection with fallback chain
- [x] Platt calibration double-pass (chronological split, cal_frac active)
- [x] Three-filter signal gate (min_matches, agreement, confidence)
- [x] 6-fold walk-forward validation (BSS +0.00103 on 2024)
- [x] Bayesian optimization via Optuna TPE
- [x] Native schema validation at fit/predict boundaries
- [x] SQLite-persistent study state for cross-session resume
- [x] Reliability infrastructure (atomic writes, lock files, checkpoints)
- [x] overnight.py: failed phases correctly retried on resume (not silently skipped)
- [x] experiment_logging.py: full config hash (all ~20 fields, no dedup collisions)
- [x] React frontend demo (JSX)
- [x] Professional project report (DOCX)
- [x] Merged modular package to main (superseding strategy.py)
- [x] Overnight/session feature sets (research-driven, pluggable)
- [x] Research integration: 24/7 equities overnight drift analysis

### Phase 1 Trading System — Complete
- [x] trading_system/config.py — all 7 sub-configs frozen, 3 risk profiles, full validate()
- [x] trading_system/signal_adapter.py — UnifiedSignal, adapt_knn/dl_signals, import guard
- [x] trading_system/backtest_engine.py — P&L (no double-count), D1/D2 fixes, all bugs patched
- [x] 88 trading_system tests (test_trading_config, test_signal_adapter, test_backtest_engine)
- [x] Phase 1 results: 22.3% annual, Sharpe 1.82, Max DD 6.9% (beats SPY risk-adjusted)
- [x] docs/FPPE_TRADING_SYSTEM_DESIGN.md v0.3
- [x] docs/PHASE1_FILE_REVIEW.md — structural audit complete
- [x] docs/CANDLESTICK_CATEGORIZATION_DESIGN.md v0.2 — future scaling module designed

### Phase 2 — Risk Engine — **Complete**
- [x] `trading_system/risk_engine.py` — ATR stop-losses, volatility-based position sizing
- [x] Drawdown brake (linear scalar at 15%) and halt (full stop at 20%)
- [x] Integration: backtest consumes risk sizing when `use_risk_engine=True`
- [x] Validation suite + results documented (`docs/PHASE2_RESULTS.md`)
- [ ] Re-validate max_holding_days in bear-market or multi-year data (ongoing research)
- [ ] Run threshold_sweep.py at 0.68 (conservative profile empirical validation)

### Phase 3 — Portfolio Manager
- [ ] `trading_system/portfolio_manager.py` — signal ranking, sector allocation, capital queue
- [ ] Idle cash target ≤ 20% (from current ~25%)

### Phase 4 — Strategy Evaluator
- [ ] `trading_system/strategy_evaluator.py` — rolling metrics, RED/YELLOW/GREEN status
- [ ] D3: TWRR decomposition separating trading alpha from cash yield

### v3.0 — Neural Hybrid (Target: Q3-Q4 2026)
- [ ] CONV_LSTM trained on price windows → 16-dim latent embeddings
- [ ] `returns_hybrid` feature set activated (8 returns + 16 LSTM dims)
- [ ] Faiss approximate NN (IndexIVFFlat) for 500+ tickers
- [ ] Multi-horizon consensus signals (agree on 7d + 14d + 30d → higher conviction)
- [ ] ≥3/6 walk-forward folds with positive BSS
- [ ] BSS > 0.05 target (materially useful)

### v4.0 — Real-Time Layer (Target: 2027)
- [ ] FastAPI service for on-demand predictions
- [ ] Live trading integration (Interactive Brokers)
- [ ] Position sizing (Kelly criterion)
- [ ] Transaction cost model (spread + commission + impact)
- [ ] Risk management (position limits, sector caps, drawdown circuit-breakers)
- [ ] Paper trading validation on 2025-2026 test set

### v5.0 — Multi-Asset Intelligence (Target: 2027-2028)
- [ ] Multi-asset expansion (bonds, commodities, FX)
- [ ] Alternative data integration (NLP sentiment, satellite, options IV)
- [ ] Reinforcement learning adaptation (dynamic feature weights)
- [ ] Ensemble meta-learner across multiple strategies
- [ ] $2K starting capital → live deployment with kill switches

---

## 14. DEPENDENCIES

### Core (requirements.txt)
```
pandas, numpy, scikit-learn, yfinance, ta, pyarrow, optuna
```

### Testing
```
pytest
```

### Optional
```
scoringrules    # CRPS computation
python-docx     # Report generation
```

### Future
```
torch           # CONV_LSTM feature extraction
faiss-cpu       # Approximate nearest neighbors at scale
fastapi         # Real-time prediction API
```

---

## 15. GLOSSARY

| Term | Definition |
|------|-----------|
| AvgK | Average analogues per query after all filters. Target: ~42. |
| BSS | Brier Skill Score. Positive = beats base rate. Best: +0.00103 (2024 fold). |
| Calibration double-pass | Train data queries itself during fit() to build unbiased calibrator. |
| EngineConfig | Frozen dataclass holding all hyperparameters. Immutable after creation. |
| Feature weights | Multiplicative scaling applied before NN index build. ret_7d/ret_14d at 1.5×. |
| FPPE | Financial Pattern Prediction Engine. The project name. |
| K-NN | K-nearest neighbors. Non-parametric — finds similar historical episodes. |
| Optuna TPE | Tree-structured Parzen Estimator. Bayesian hyperparameter optimization. |
| Platt scaling | Logistic sigmoid mapping raw frequencies → calibrated probabilities. |
| Regime | Market state classification. Binary (Bull/Bear), Multi (4), Octet (8). |
| Schema validation | DataFrame checks at engine boundaries. Native Python, no pandera. |
| Signal gate | Three-filter rule: min_matches ≥ 10, agreement ≥ 0.10, confidence ≥ 0.65. |
| Walk-forward | 6-fold expanding validation 2019-2024. Standard fold = 2024. |

---

## 16. RESEARCH CONTEXT: 24/7 EQUITIES & OVERNIGHT DRIFT

### Source
"The Financial Microstructure of 24/7 Equities: An Exhaustive Analysis of the S&P 500
and Trade Hyperliquid Partnership" (March 18, 2026). S&P 500 licensed to Trade/Hyperliquid
for 24/7 perpetual contracts — first time in 69-year history the benchmark trades
continuously on decentralized infrastructure.

### Key Findings Relevant to FPPE
1. **Overnight drift** (Fed NY Staff Report 917): US equities show ~3.7% annualized
   positive return during off-hours, primarily at 02:00-03:00 ET when European liquidity
   arrives. Driven by dealer inventory risk — market makers absorb selling pressure at
   close and extract compensation overnight.
2. **Gap risk as signal**: Weekend/overnight gaps represent forced price discovery
   compression. Magnitude and direction contain predictive information.
3. **Session decomposition**: Total daily return = overnight (gap) + intraday (session).
   These components carry orthogonal information about different market mechanisms.
4. **Cross-timezone correlation gaps**: Asian markets systematically overreact to US
   movements during 21:00-02:00 HKT, creating predictable lag patterns.

### Implementation in FPPE
Two new pluggable feature sets added to `features.py`:

| Feature Set | Columns | Description |
|-------------|---------|-------------|
| `returns_overnight` | 12 (8 returns + 4 overnight) | Adds ret_overnight, ret_intraday, gap_magnitude, gap_direction_streak |
| `returns_session` | 14 (8 returns + 6 session) | Adds above + weekend_gap, weekend_gap_magnitude |

Usage: `EngineConfig(feature_set="returns_overnight")` — default `returns_only` unchanged.

### Feature Definitions
- `ret_overnight`: Close[t-1] -> Open[t] (the gap component)
- `ret_intraday`: Open[t] -> Close[t] (the session component)
- `gap_magnitude`: |ret_overnight| (absolute gap size)
- `gap_direction_streak`: Consecutive same-sign gaps (momentum signal)
- `weekend_gap`: Friday close -> Monday open (only non-zero on Mondays)
- `weekend_gap_magnitude`: |weekend_gap|

### Status
Features implemented and tested (24 new tests). Walk-forward validation pending.
These are opt-in experimental features — default `returns_only` set is LOCKED.
