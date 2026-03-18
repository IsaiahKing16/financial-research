# PROJECT_GUIDE.md — Multi-AI Collaboration Reference
# Last Updated: 2026-03-18
# Owner: Sleep (Isaia)
# Primary AI: Claude (Anthropic) | Supporting: Gemini, ChatGPT

---

## QUICK CONTEXT FOR ANY AI READING THIS

You are helping build **FPPE (Financial Pattern Prediction Engine)** — an autonomous
financial prediction system. The core engine uses K-nearest-neighbor historical
analogue matching on return fingerprints to generate probabilistic BUY/SELL/HOLD
signals across a 52-ticker universe.

The project has two codebases:
1. **`pattern_engine/`** — Python package (21 modules, 218 tests, all passing)
2. **`pattern-engine-v2.1.jsx`** — Standalone React demo (claude.ai artifact, no HTTP calls)

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
├── features.py                    # FeatureSet/FeatureRegistry (5 pluggable sets + hybrid)
├── matching.py                    # Core KNN matcher (ball_tree, batched, 52-ticker SECTOR_MAP)
├── projection.py                  # Forward projection + three-filter signal gate
├── calibration.py                 # Platt/isotonic/none probability calibrators
├── regime.py                      # Binary/multi/octet market regime detection
├── scoring.py                     # Brier Score, BSS, CRPS, calibration buckets
├── evaluation.py                  # Signal-aligned + probabilistic evaluation
├── candlestick.py                 # Continuous multi-timeframe encoding (1d/3d/5d)
├── sector.py                      # Cross-asset sector features (vectorized)
├── engine.py                      # PatternEngine class (fit/predict/evaluate/save/load)
├── schema.py                      # NEW v2.1: Native DataFrame validation at boundaries
├── cross_validation.py            # Cross-model validation & integrity checking
├── data.py                        # DataLoader pipeline (yfinance → features → parquet)
├── experiment_logging.py          # TSV experiment logger with config provenance
├── walkforward.py                 # 6-fold walk-forward with optional cross-validation
├── sweep.py                       # Grid sweep + NEW Bayesian (Optuna TPE) sweep
├── live.py                        # Production EOD signal generator with consensus mode
├── overnight.py                   # Multi-phase runner (static + Bayesian modes)
└── reliability.py                 # Atomic writes, lock files, progress logging
```

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
**Consumer (6):** WMT, COST, PG, KO, PEP, HD
**Industrial (4):** DIS, CAT, BA, GE
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
- 218 automated tests all passing

---

## 7. v2.1 NEW FEATURES

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

## 8. BUGS FIXED (history)

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
All **218 tests** must pass (14 test files).

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
21 modules, 218 tests, all passing. Fully modular replacement of monolithic strategy.py.

### 12.2 React Frontend Demo (`pattern-engine-v2.1.jsx`)
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

### 12.3 Project Report (`FPPE_v2.1_Project_Report.docx`)
Professional DOCX report (~20 pages, 4 sections):
1. **System Overview** — pipeline architecture, module map, headline results
2. **Technical Deep Dive** — K-NN matching, calibration double-pass, regime detection, Bayesian optimization, walk-forward, schema validation, reliability
3. **Comparison to Established Firms** — Berkshire, Renaissance, Two Sigma, AQR, D.E. Shaw, Citadel, BlackRock with comparative analysis
4. **Roadmap & Evolution** — v3.0 (neural hybrid) → v4.0 (real-time) → v5.0 (multi-asset)

---

## 13. ROADMAP

### v2.1 — Current (Complete)
- [x] Modular pattern_engine package (21 modules)
- [x] K-NN analogue matching with weighted Euclidean distance
- [x] 8-state regime detection with fallback chain
- [x] Platt calibration double-pass
- [x] Three-filter signal gate (min_matches, agreement, confidence)
- [x] 6-fold walk-forward validation (BSS +0.00103 on 2024)
- [x] Bayesian optimization via Optuna TPE
- [x] Native schema validation at fit/predict boundaries
- [x] SQLite-persistent study state for cross-session resume
- [x] Reliability infrastructure (atomic writes, lock files, checkpoints)
- [x] 218 tests all passing
- [x] React frontend demo (JSX)
- [x] Professional project report (DOCX)

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
