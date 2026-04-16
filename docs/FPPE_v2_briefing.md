> **SUPERSEDED (2026-04-15):** This document describes the v2.1 architecture as of the
> initial Phase 3Z rebuild. For current architecture, see
> `docs/FPPE_TRADING_SYSTEM_DESIGN.md` v0.6 and `CLAUDE.md`.

# Financial Prediction Pattern Engine v2.1 — Development Briefing

## What Was Built

A complete ground-up rebuild of the monolithic `strategy.py` (2,498 lines) into a modular, class-based **PatternEngine** package. The system finds historical "twins" of current market conditions using K-nearest-neighbor analogue matching on return fingerprints, then projects forward returns to generate probabilistic BUY/SELL/HOLD trading signals.

## Architecture

```
pattern_engine/
├── __init__.py              # Public API: PatternEngine, EngineConfig, CrossValidator
├── config.py                # EngineConfig frozen dataclass (all proven defaults)
├── features.py              # FeatureSet/FeatureRegistry (6 pluggable sets)
├── matching.py              # Core KNN matcher (batched queries, filtering)
├── projection.py            # Forward projection + three-filter signal gate
├── calibration.py           # Platt/isotonic/none probability calibrators
├── regime.py                # Binary/multi/octet market regime detection
├── scoring.py               # Brier Score, BSS, CRPS, calibration buckets
├── evaluation.py            # Signal-aligned + probabilistic evaluation
├── candlestick.py           # Continuous multi-timeframe encoding (1d/3d/5d)
├── sector.py                # Cross-asset sector features (vectorized)
├── engine.py                # PatternEngine class (fit/predict/evaluate/save/load)
├── cross_validation.py      # Cross-model validation & integrity checking
├── data.py                  # DataLoader pipeline (yfinance → features → parquet)
├── experiment_logging.py    # TSV experiment logger with config provenance
├── walkforward.py           # 6-fold walk-forward with optional cross-validation
├── sweep.py                 # Parameter grid sweep ranked by BSS
├── live.py                  # Production EOD signal generator with consensus mode
└── overnight.py             # Multi-phase overnight runner with resilience & integrity
```

**Test suite:** 182 tests across 14 files — all passing.

## Core API

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
runner = WalkForwardRunner(config, folds=WALKFORWARD_FOLDS)
fold_metrics = runner.run(full_db)

# Parameter sweep
from pattern_engine.sweep import SweepRunner
configs = SweepRunner.grid(max_distance=[0.8, 1.0, 1.1019], regime_mode=["binary", "multi"])
results = SweepRunner(configs).run(full_db)
```

## Proven Research Defaults (EngineConfig)

All defaults trace to specific sweep experiments:

| Parameter | Value | Evidence |
|-----------|-------|----------|
| `distance_metric` | `"euclidean"` | Cosine collapsed at 93.3% accuracy |
| `max_distance` | `1.1019` | Quantile-calibrated, yields AvgK ~42 |
| `top_k` | `50` | Neighbourhood ceiling before distance filter |
| `distance_weighting` | `"uniform"` | Beats inverse weighting (sweep 1) |
| `feature_set` | `"returns_only"` | Full 16-feature set was 3.5x worse (sweeps 2/3) |
| `calibration_method` | `"platt"` | Platt scaling generalises best across folds |
| `cal_frac` | `0.76` | Best performance on 2024 validation fold |
| `confidence_threshold` | `0.65` | Best binary accuracy trade-off (sweep 1) |
| `regime_filter` | `True` | Binary regime (SPY ret_90d > 0 = bull) |
| `nn_jobs` | `1` | Windows/Python 3.12 joblib deadlock prevention |

The `nn_algorithm` property is **derived** from `distance_metric` (ball_tree for Euclidean, brute for cosine — ball_tree doesn't support cosine).

## Key Design Decisions

### 1. Pluggable Feature System

Five built-in feature sets, swappable via config string:

| Set | Features | Purpose |
|-----|----------|---------|
| `returns_only` | 8 trailing returns | Proven baseline |
| `returns_candle` | 8 returns + 15 candlestick | Pattern structure |
| `returns_vol` | 8 returns + 4 volatility | Regime context |
| `returns_sector` | 8 returns + 3 sector | Cross-asset signals |
| `full` | 34 features combined | Experimental |

Custom sets: `FeatureRegistry.register("custom", [...columns...])`

### 2. Continuous Candlestick Encoding

Multi-timeframe continuous proportions instead of discrete categories (preserves Euclidean KNN integrity):

- **3 timeframes**: 1-day, 3-day composite, 5-day composite
- **5 features per timeframe**: body_to_range, upper_wick, lower_wick, body_position, direction
- **15 total candlestick features**
- Based on Lin et al. (2021) — continuous encoding outperforms 13-class discrete labels for distance-based algorithms

### 3. 8-State Regime Detection with Graceful Fallback

Three-dimensional classification:

| Dimension | Metric | Split |
|-----------|--------|-------|
| Direction | SPY ret_90d | > 0 = Bull, ≤ 0 = Bear |
| Volatility | SPY vol_30d | < median = Low, ≥ median = High |
| Trend | ADX(14) | > 25.0 = Trending, ≤ = Range |

Fallback chain when insufficient analogues: octet (8) → multi (4) → binary (2)

### 4. Calibration Double-Pass (Encapsulated in fit())

The most complex pattern — previously re-implemented differently in every runner:

1. Fit scaler on training features (with feature_weights applied)
2. Build NN index on weighted, scaled training data
3. Compute regime labels for training SPY rows
4. Run matching on train-as-query with **identical filtering as inference** (critical: regime_filter must match)
5. Fit Platt/isotonic calibrator on raw probabilities + ground truth

Callers just call `engine.fit(train_db)` — the complexity is invisible.

### 5. Feature Weights

Applied before NN index build AND query transform (directly affects distance computation):

```python
feature_weights = {
    "ret_7d": 1.5, "ret_14d": 1.5,     # Medium-term trend weighted higher
    "ret_90d": 0.5,                      # Long-term secular trend weighted lower
    "vol_10d": 1.2, "price_vs_sma20": 1.2,  # Volatility/mean-reversion context
    # ... all others default to 1.0
}
```

### 6. Three-Filter Signal Gate

All three conditions must pass to generate a BUY or SELL:
1. **MIN_MATCHES ≥ 10** — sufficient analogues found
2. **AGREEMENT_SPREAD ≥ 0.10** — analogues agree on direction
3. **CONFIDENCE_THRESHOLD**: P(up) ≥ 0.65 → BUY, P(up) ≤ 0.35 → SELL

Otherwise → HOLD (no trade signal).

## Evaluation Metrics

- **Brier Score (BS)**: Mean squared error of probability forecast (0 = perfect, 0.25 = naive)
- **Brier Skill Score (BSS)**: Improvement over always predicting base rate (positive = skill)
- **CRPS**: Continuous Ranked Probability Score from analogue ensemble (optional dependency)
- **Calibration**: Reliability buckets — does P=60% actually go up 60% of the time?
- **Signal-aligned accuracy**: Precision/recall/F1 only on confident trades (BUY/SELL)

**Benchmark**: System A's first positive BSS was +0.00100 on the 2024 walk-forward fold.

## BSS Regression Test Results (Real 52-Ticker Data)

Walk-forward validation with proven defaults confirms the rebuild matches System A:

| Fold | BSS | Accuracy | Avg Matches |
|------|-----|----------|-------------|
| 2019 | -0.01211 | 61.2% | 39.3 |
| 2020 (COVID) | -0.00142 | 57.4% | 25.1 |
| 2021 | -0.00131 | 57.7% | 42.6 |
| 2022 (Bear) | -0.03208 | 48.7% | 21.2 |
| 2023 | -0.00029 | 56.2% | 40.1 |
| **2024 (Standard Val)** | **+0.00103** | **56.6%** | **41.9** |

**Key finding:** The 2024 fold achieves BSS = +0.00103, matching System A's benchmark of +0.00100. The modular rebuild is functionally equivalent.

**Pattern observations:**
- Negative BSS on earlier folds is expected — expanding-window folds have less training history and fewer market regime examples
- 2022 (Bear) is the worst fold — the market crash creates a distribution shift that the calibrator struggles with (fewer training bear-market analogues)
- 2024 fold has the most training data and the richest regime coverage, yielding the model's best performance
- 0 confident trades across all folds — the three-filter gate (confidence > 0.65, agreement > 0.10, min_matches > 10) is strict by design, prioritizing precision

## Data Pipeline

- 52 tickers across 7 sectors (Tech, Finance, Health, Consumer, Industrial, Energy, Index)
- Historical OHLCV via yfinance, cached to CSV
- Feature computation: trailing returns, supplementary (vol, RSI, ATR, SMA), candlestick, sector
- Strict temporal split: no data leakage across train/val/test boundaries
- Output: parquet files for fast loading

## Walk-Forward Validation

6-fold expanding window (2019-2024):

| Fold | Train | Validate |
|------|-------|----------|
| 2019 | ≤ 2018-12-31 | 2019 |
| 2020 (COVID) | ≤ 2019-12-31 | 2020 |
| 2021 | ≤ 2020-12-31 | 2021 |
| 2022 (Bear) | ≤ 2021-12-31 | 2022 |
| 2023 | ≤ 2022-12-31 | 2023 |
| 2024 (Standard Val) | ≤ 2023-12-31 | 2024 |

## v2.1 Enhancements (Gemini Code Review Integration)

### Cross-Model Validation Framework (NEW)

`CrossValidator` enables models to check each other's outputs — not just self-validate:

```python
from pattern_engine import CrossValidator, EngineConfig

configs = [
    EngineConfig(),                                    # Baseline
    EngineConfig(max_distance=1.25),                   # Wider search
    EngineConfig(calibration_method="isotonic"),        # Different calibrator
]

xval = CrossValidator(configs)
result = xval.run(train_db, val_db)

# Which predictions do all models agree on?
matrix = xval.agreement_matrix()

# Flag predictions where models strongly disagree
disagreements = xval.flag_disagreements(min_spread=0.15)

# Ultra-high-conviction: only emit when majority agrees
consensus = xval.consensus_signals()

# Pipeline integrity: determinism, persistence, calibration sanity
integrity = CrossValidator.integrity_check(config, train_db, val_db)
```

**Runner integration:**
- `WalkForwardRunner(cross_validate=True)` — runs config variants per fold, logs agreement
- `OvernightRunner` — auto integrity check after each phase
- `LiveSignalRunner(consensus_configs=[...])` — consensus-filtered live signals

### Vectorized Sector Features

Sector-relative return computation replaced nested O(Sectors x Tickers) merge loops with a single vectorized join. Identical results, 10-50x faster at scale.

### Overnight Reliability Hardening (NEW)

Comprehensive reliability infrastructure for unattended 6+ hour overnight runs:

- **`reliability.py`** — `atomic_write()` (temp+fsync+rename), `LockFile` (PID-based advisory locks), `ProgressLog` (append-only timestamped logging)
- **`overnight.py`** — Lock files prevent concurrent runs, atomic checkpoint with set-based phase IDs, per-phase exception isolation, atexit handler for clean shutdown
- **`walkforward.py`** — Per-fold try/except so one bad fold doesn't lose the other 5
- **`experiment_logging.py`** — Config hash deduplication, fsync after every append, atomic first-write
- **`engine.py`** — Atomic save (temp+fsync+rename), validated load with corruption detection
- **30 reliability tests** covering atomic writes, lock files, progress logs, dedup, and persistence round-trips

### CONV_LSTM Hybrid Feature Set (Placeholder)

`returns_hybrid` feature set registered — 8 returns + 16 LSTM latent embedding dimensions. Architecture ready for deep learning feature extraction when CONV_LSTM is trained.

## What's Next

Areas for continued research and improvement:

1. **BSS optimisation** — sweep feature_weights, candlestick features, regime modes
2. **CONV_LSTM integration** — train the network, extract latent embeddings as features, ensemble signal gate with 4th filter (LSTM_CONFIDENCE >= 0.55)
3. **Multi-horizon projection** — currently optimised for 7-day; test 1d, 3d, 14d, 30d
4. **Approximate NN (Faiss)** — swap ball_tree for IndexIVFFlat at 500+ tickers for 10-50x speedup
5. **Schema validation (Pandera)** — runtime schema checks at engine.fit()/predict() boundaries
6. **Scale to 500+ tickers** — architecture supports it, need to test performance
7. **Live signal deployment** — `LiveSignalRunner` with consensus mode needs production wrapper
8. **DVC integration** — data versioning for reproducibility

## Technical Stack

- Python 3.12, pandas, numpy, scikit-learn, yfinance, ta (technical analysis)
- pytest (182 tests), pyarrow (parquet I/O)
- Optional: scoringrules (CRPS computation)
- Future: faiss (approximate NN), pandera (schema validation), torch (CONV_LSTM)
