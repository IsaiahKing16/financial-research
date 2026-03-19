# PROJECT_GUIDE.md — Multi-AI Collaboration Reference
# Last Updated: 2026-03-18 (v2.2 + manifest system)
# Owner: Sleep (Isaia)
# Primary AI: Claude (Anthropic) | Supporting: Gemini, ChatGPT

---

## QUICK CONTEXT FOR ANY AI READING THIS

You are helping build **FPPE (Financial Pattern Prediction Engine)** — an autonomous
financial prediction system. The core engine uses K-nearest-neighbor historical
analogue matching on return fingerprints to generate probabilistic BUY/SELL/HOLD
signals across a 52-ticker universe.

The project has two codebases:
1. **`pattern_engine/`** — Python package (22 modules, 294 tests, all passing) — **v2.2**
2. **`pattern-engine-v2.1.jsx`** — Standalone React demo (claude.ai artifact, no HTTP calls)
3. **`archive/`** — Legacy Phase 1 files (strategy*.py, prepare.py, etc.)

**Do NOT modify this file unless explicitly asked.**

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

### Package Structure (v2.2)

```
pattern_engine/                    # 22 modules, version 2.2.0
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
├── schema.py                      # Native DataFrame validation at boundaries
├── cross_validation.py            # Cross-model validation & integrity checking
├── data.py                        # DataLoader pipeline (yfinance → features → parquet)
├── experiment_logging.py          # TSV experiment logger with full-config SHA-256 hashing
├── walkforward.py                 # 6-fold walk-forward with deadline support
├── sweep.py                       # Grid sweep + Bayesian (Optuna TPE) sweep
├── live.py                        # Production EOD signal generator with consensus mode
├── overnight.py                   # Multi-phase runner with checkpoint state machine
├── reliability.py                 # Atomic writes, lock files, fsync'd progress logging
└── manifest.py                    # NEW v2.2: Run manifests, data versioning, prior-run memory
```

### Legacy Files (moved to archive/ — do not modify)

All Phase 1 files are now in `archive/`:
- `strategy.py`, `strategyv1-v4.py`, `strategy_overnight.py`
- `prepare.py`, `oldstrategy.py`, `oldstrategy1.py`
- `dedup.py`, `diagnose_distances.py`, `quick_sweep.py`, `test_strategy.py`

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
2. Query training data against itself
3. Fit Platt sigmoid on self-query frequencies + known outcomes
4. No look-ahead bias — calibrator sees same distribution as inference

**Note:** `cal_frac` was removed in v2.2 (was a no-op — never used by engine.fit()).
If calibration holdout is needed in future, implement proper train/cal split in engine.py first.

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
- 294 automated tests all passing

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
| confidence_threshold | Float | [0.55, 0.80] |
| regime_mode | Categorical | {binary, multi} |

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
All **294 tests** must pass (17 test files).

### Key Entry Points

| Command | Purpose |
|---------|---------|
| `python -m pattern_engine.live` | Production EOD signals (after market close) |
| `python -m pattern_engine.overnight` | 6-hour overnight runner |
| `python -m pytest tests/ -v` | Full test suite |
| `python -m pattern_engine.walkforward` | Walk-forward validation |
| (see sweep.py API below) | Quick parameter sweep |

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
22 modules, 294 tests, all passing. Fully modular replacement of monolithic strategy.py.
Merged to main branch 2026-03-18. Legacy files moved to `archive/` directory.
v2.2 code review fixes applied 2026-03-18. Manifest system added 2026-03-18.

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

### v2.1 — Complete
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
- [x] React frontend demo (JSX)
- [x] Professional project report (DOCX)
- [x] Merged modular package to main (superseding strategy.py)
- [x] Overnight/session feature sets (research-driven, pluggable)
- [x] Research integration: 24/7 equities overnight drift analysis

### v2.2 — Current (Complete)
Cross-AI code review fixes (Gemini + ChatGPT reviewed, Claude implemented):
- [x] P0-1: Overnight checkpoint state machine (failed phases retryable, not permanently skipped)
- [x] P0-2: Full-config SHA-256 hashing for experiment deduplication (was only 8/20+ fields)
- [x] P0-3: Removed cal_frac no-op parameter (was optimized by Optuna but never used)
- [x] P0-4: Deadline propagation from overnight runner into walkforward folds
- [x] P0-5: Phase health evaluation (completed/partial/failed based on fold outcomes)
- [x] P1-6: ProgressLog._write() now calls os.fsync() for crash durability
- [x] P1-8: DataLoader guards against empty universe with RuntimeError
- [x] P1-9: All public assert statements replaced with RuntimeError/ValueError
- [x] README rewritten for pattern_engine architecture (was describing Phase 1 Conv1D/LSTM)
- [x] 13 legacy files moved to archive/
- [x] Run manifest system (manifest.py): immutable provenance per run
- [x] Data version fingerprinting for cache staleness detection
- [x] Prior-run context loading for proactive memory injection
- [x] 294 tests all passing (17 test files)

### v2.3 — Agentic Village Phase 1 (Next — see Section 17)
Based on "Agentic Village" research. Smallest useful implementation:
- [ ] Integrate data version fingerprinting into DataLoader.build_database()
- [ ] Create risk_gate.py: pre-signal governance checks before live emission
- [ ] Add SignalEvidence structured output to live.py
- [ ] Extend cross_validation.py into synthesis layer (ensemble weighting)
- [ ] Auto-warm Bayesian sweeps from prior best configs via manifests
- [ ] Add run_id and data_version columns to experiments.tsv

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

---

## 17. AGENTIC VILLAGE ARCHITECTURE REFERENCE

### Source
"Collaborative AI Architecture Research Report" (March 2026) — enterprise multi-agent
orchestration patterns distilled for single-developer quant research context.

### Core Principle: Structured State Over Free-Text Handoffs

The key insight: when multiple AI agents (or overnight automation sessions) collaborate on
a codebase, **structured artifacts** (JSON manifests, typed configs, schema-validated DataFrames)
dramatically outperform free-text summaries for continuity and correctness. This is why FPPE v2.2
introduced `RunManifest` as a dataclass → JSON rather than appending to a log file.

### Architecture Pattern: Blackboard

FPPE uses a simplified **blackboard architecture** where shared state lives in well-defined
files rather than a message bus:

| Blackboard Artifact | File Location | Schema |
|---------------------|---------------|--------|
| Run manifests | `data/runs/<run_id>/manifest.json` | `RunManifest` dataclass |
| Data version | `data/processed/data_version.json` | `{version, tickers, n_features, created_at}` |
| Experiment log | `data/results/experiments.tsv` | 20+ columns, TSV with header |
| Checkpoints | `data/runs/<run_id>/checkpoint.json` | `{phase_statuses: {name: {status, attempts, ...}}}` |
| Config | `EngineConfig` frozen dataclass | 20+ typed fields, hashable |

### File-Level Mapping (What Research Concepts → FPPE Modules)

| Research Concept | FPPE Implementation | Module |
|-----------------|---------------------|--------|
| Shared memory / blackboard | Run manifests + experiment TSV | `manifest.py`, `experiment_logging.py` |
| Private agent memory | Per-run checkpoint state | `overnight.py` |
| Proactive context injection | `load_prior_context()` at run start | `manifest.py` |
| Dependency-aware cache invalidation | `compute_data_version()` + `check_data_staleness()` | `manifest.py` |
| Structured evidence | `SignalEvidence` (planned) | `live.py` (v2.3) |
| Pre-action governance | `risk_gate.py` (planned) | New module (v2.3) |
| Synthesis / ensemble layer | Weighted fold combination (planned) | `cross_validation.py` (v2.3) |
| Config warm-start | Auto-seed Bayesian sweeps from prior best (planned) | `sweep.py` (v2.3) |

### Shared Memory Schema

Every run writes a `RunManifest` with these fields:

```
run_id          : str   — YYYYMMDD_HHMMSS_<6hex>
mode            : str   — "static" | "bayesian" | "walkforward" | "live"
started_at      : str   — ISO 8601 timestamp
ended_at        : str   — ISO 8601 timestamp
git_sha         : str   — Short commit hash for provenance
data_version    : str   — 12-char SHA-256 of ticker universe + features
config_hash     : str   — 16-char SHA-256 of full EngineConfig
phases_completed: int   — Count of phases with all folds succeeding
phases_failed   : int   — Count of phases with all folds failing
phases_partial  : int   — Count of phases with mixed fold outcomes
best_bss        : float — Best Brier Skill Score achieved
total_folds     : int   — Total walk-forward folds executed
elapsed_minutes : float — Wall-clock runtime
artifact_paths  : dict  — Paths to saved models, plots, reports
notes           : str   — Free-text annotation
```

### Proactive Context Flow

```
overnight_runner.run()
  ├── load_prior_context(runs_dir)          # Read shared memory
  │     ├── Find N most recent manifests
  │     ├── Extract best_bss, best_config_hash
  │     ├── Identify failed_runs for quarantine
  │     └── Return structured context dict
  ├── check_data_staleness(current_version) # Cache validation
  │     ├── Compare vs data/processed/data_version.json
  │     └── Return {stale: bool, reason: str}
  ├── [Execute phases with checkpoint state machine]
  └── manifest.save(runs_dir)               # Write to shared memory
```

### Deferred Items (Research → Not Yet Needed)

These research concepts were evaluated and intentionally deferred:

| Concept | Reason for Deferral |
|---------|---------------------|
| Service mesh / MCP protocol | Single-process; no inter-service communication needed |
| A2A (Agent-to-Agent) protocol | Single developer; Claude Code sessions are sequential |
| GPU orchestration / CUDA scheduling | K-NN is CPU-bound; no neural training yet |
| LLM debate / multi-model consensus | One model (Claude) is sufficient for code generation |
| Vector DB for agent memory | File-based manifests scale to thousands of runs |
| Real-time event bus | Overnight batch processing; no streaming requirements |
| Kubernetes / container orchestration | Runs on single machine, <10min per overnight cycle |

### Implementation Phases

**Phase 1 (v2.2 — COMPLETE):**
- `manifest.py`: RunManifest, generate_run_id, compute_data_version, load_prior_context, check_data_staleness, save_data_version
- `overnight.py`: Checkpoint state machine, manifest integration, prior context loading
- `test_manifest.py`: 15 tests for manifest system
- `test_review_fixes.py`: 37 tests for all v2.2 fixes

**Phase 2 (v2.3 — PLANNED):**
- Integrate `compute_data_version()` into `DataLoader.build_database()` flow
- Create `risk_gate.py` for pre-signal governance checks
- Add `SignalEvidence` structured output replacing raw dicts in `live.py`
- Extend `cross_validation.py` synthesis layer for ensemble weighting
- Auto-warm Bayesian sweeps from prior best configs
- Add `run_id` and `data_version` columns to experiments.tsv

**Phase 3 (v3.0+ — FUTURE):**
- Neural hybrid features would benefit from structured caching
- Multi-model consensus if ensemble strategy proves viable
- Vector DB migration if manifest count exceeds ~10K runs
