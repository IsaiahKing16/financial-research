# Phase 3Z – FPPE Full Rebuild Plan

**Date:** March 21, 2026
**Author:** Claude Opus 4.6 (Lead Architect)
**Owner:** Sleep (Isaia)
**Status:** DRAFT — requires owner review before execution begins
**Applies to:** `pattern_engine/`, `trading_system/`, `research/`, root-level files, `tests/`

---

## 1. Executive Summary

### Purpose

Phase 3Z is a controlled architectural reset of the Financial Pattern Prediction Engine (FPPE). It merges the strongest elements of the current production build (Phases 1–3), Phase 3.5 research integrations (EMD, BMA, SlipDeficit), and Phase C research roadmap into a single coherent framework that can serve as the permanent architectural foundation for all future development.

### Why a full rebuild is justified

FPPE has grown through five sequential phases (1, 2, 3, 3.5, C-roadmap), each adding capability but also accumulating structural debt:

1. **Boundary ambiguity.** No formal contracts exist between pattern_engine and trading_system. Data flows through implicit DataFrame conventions with no schema enforcement at package boundaries.
2. **Legacy contamination.** `strategy.py` (2,499 lines, superseded) is still imported by `signal_adapter.py`. Seven additional legacy scripts (11,379 total lines) remain in the root directory.
3. **Missing infrastructure.** No CI pipeline, no frozen dependency lock, no dataset manifests, no parity harnesses, no SharedState implementation (referenced in design docs but never built).
4. **Schema validation gap.** `schema.py` uses hand-rolled validation. No Pydantic models, no Pandera schemas. Silent coercion and missing-column fallback are possible in multiple paths.
5. **Test coverage holes.** `data.py` (entire feature pipeline) and `live.py` (production signal runner) have zero test coverage. SECTOR_MAP has diverged across three files.
6. **Phase 4 does not exist.** `strategy_evaluator.py` (rolling metrics, RED/YELLOW/GREEN status, TWRR) is designed but unbuilt — it is the next critical layer.
7. **Research promotion path is ad-hoc.** Phase 3.5 modules (EMD, BMA, SlipDeficit) have ABCs but no feature flags, no parity harnesses, and no rollback mechanism.

Without Phase 3Z, each future phase will compound this debt. A controlled rebuild now — while the system is small enough to reconstruct in days rather than weeks — prevents the codebase from reaching a state where rebuilds become infeasible.

### Intended end-state architecture

After Phase 3Z, FPPE will be:

- **Contract-driven**: Every cross-module boundary enforced by Pydantic (config/state objects) or Pandera (DataFrames)
- **Parity-tested**: Frozen baseline data + documented numerical tolerances ensure the rebuild matches prior validated results
- **Feature-flagged**: All research modules (EMD, BMA, SlipDeficit, future HNSW) live behind explicit flags with rollback paths
- **Four-layer complete**: Layer 4 (strategy_evaluator) exists and emits operational status
- **Reproducible**: Dataset manifests, config hashes, and run provenance make any result auditable from raw data to final report
- **Simpler**: Legacy scripts retired, SECTOR_MAP centralized, SharedState implemented, dead code removed

---

## 2. Current-State Assessment

### 2.1 Codebase inventory

| Package | Modules | LOC | Tests | Status |
|---------|---------|-----|-------|--------|
| `pattern_engine/` | 22 | ~4,300 | 300 | Production — clean, no prototype contamination |
| `trading_system/` | 9 | ~3,600 | 256 | Production — Phases 1–3 complete |
| `research/` | 4 | ~480 | 21 | Phase 3.5 complete; Phase C roadmap only |
| `scripts/` | 4 | ~650 | 0 | Validation scripts (SLE-43–46), not yet executed |
| Root legacy | 12 | ~11,400 | 0 | Superseded — `strategy.py` still imported |
| **Total** | **51** | **~20,430** | **577** | |

### 2.2 Strengths worth preserving

1. **Frozen EngineConfig dataclass** — deterministic hashing, full provenance, no mutation. This is the gold standard for the project.
2. **Calibration double-pass** — train-as-query with regime filtering to match inference distribution. Prevents calibration leakage.
3. **Three-filter signal gate** — min_matches + min_agreement + confidence_threshold. Simple, interpretable, validated.
4. **Four-layer trading architecture** — clean separation of simulation (L1), risk (L2), portfolio (L3), evaluation (L4). Design doc v0.4 is excellent.
5. **Crash-safe infrastructure** — `reliability.py` provides atomic writes, lock files, progress logging. Production-grade.
6. **Walk-forward validation framework** — expanding-window temporal splits with deadline support and error isolation per fold.
7. **Experiment logging with deduplication** — TSV-based, fsync-on-write, config-hash-keyed. Simple and reliable.
8. **Research ABCs** — `BaseDistanceMetric`, `BaseCalibrator`, `BaseRiskOverlay` define clean promotion contracts.
9. **Defense-in-depth validation** — PM checks count constraints, risk engine re-checks independently. Intentional, documented.
10. **Frozen dataclasses for state** — `PortfolioSnapshot`, `RankedSignal`, `AllocationDecision`, `PositionDecision` are all immutable.

### 2.3 Architectural weaknesses

#### Critical (must fix in rebuild)

| ID | Issue | Impact | Location |
|----|-------|--------|----------|
| W1 | **No cross-package contracts** | Silent schema drift between pattern_engine and trading_system | Package boundary |
| W2 | **SECTOR_MAP defined in 3 files, already diverged** | DIS misclassified in matching.py (silent wrong matches) | matching.py:26, sector.py:13, trading_system/config.py:21 |
| W3 | **SharedState never implemented** | Design doc v0.4 §2.1 references it; code uses ad-hoc mutable dicts | backtest_engine.py |
| W4 | **strategy.py still imported** | signal_adapter.simulate_signals_from_val_db() depends on legacy code | signal_adapter.py:175 |
| W5 | **data.py has zero test coverage** | Entire feature pipeline (269 lines) untested; silent corruption possible | pattern_engine/data.py |
| W6 | **live.py has zero test coverage** | Production signal runner untested; still uses assert guards | pattern_engine/live.py |
| W7 | **No dataset manifests or frozen data** | Parity cannot be verified against a known-good baseline | Project-wide |
| W8 | **No CI pipeline** | 577 tests run manually or not at all | Infrastructure |
| W9 | **Layer 4 (strategy_evaluator) doesn't exist** | No rolling metrics, no operational status, no calibration drift tracking | trading_system/ |
| W10 | **No feature flags for research modules** | SlipDeficit is hard-wired into backtest_engine with no disable switch | backtest_engine.py:538 |

#### Moderate (address during rebuild)

| ID | Issue | Impact |
|----|-------|--------|
| W11 | `returns_hybrid` feature set references nonexistent LSTM columns | Confusing error if accidentally selected |
| W12 | No engine state migration path (v2.1 → v2.2 will break all saved states) | Re-fit required on every version bump |
| W13 | Parquet files not in .gitignore | Potential repo bloat |
| W14 | No lock file for dependencies | `pip install` on new machine may get breaking versions |
| W15 | `cal_frac` ambiguity | Docs say 0.76; production Platt uses train-as-query on all X_train |

### 2.4 Where Phase 3.5 improves upon the current build

Phase 3.5 introduced three research modules with formal ABCs:

1. **EMDDistance** — Optimal transport distance as an alternative to Euclidean. Well-structured, but O(N×M) runtime makes it impractical as a primary retriever. Best used as a post-retrieval reranker.
2. **BMACalibrator** — Bayesian Model Averaging via Student's t mixture. Shape asymmetry with PlattCalibrator (expects (N,K) vs (N,)) requires migration work. Validation pending.
3. **SlipDeficit** — Seismic-inspired TTF volatility overlay. Already wired into backtest_engine. Proven concept with graceful degradation. Most production-ready of the three.

**Key Phase 3.5 contribution:** The ABC pattern (`BaseDistanceMetric`, `BaseCalibrator`, `BaseRiskOverlay`) is the right abstraction for research module promotion. The rebuild should formalize this into a proper plugin system with feature flags.

### 2.5 Phase C assessment

Phase C exists only as a roadmap document (`research/phase_c_roadmap.md`). No code has been written. Four domains are planned:

1. **HNSW approximate NN** — High priority. Enables scaling beyond 10k fingerprints. Should be implemented as a `Matcher` backend option, not a `BaseDistanceMetric` subclass (the roadmap's integration suggestion is architecturally wrong — HNSW is an index strategy, not a distance metric).
2. **Hawkes Process** — Medium priority. Risk overlay for cross-asset contagion. Fits the `BaseRiskOverlay` ABC.
3. **CPOD/EILOF anomaly detection** — Medium priority. Pre-filter for regime outliers. Fits as a matching.py pre-filter.
4. **OWA dynamic feature weighting** — Medium priority. Touches locked settings. Requires strong experiment evidence.

**Key Phase C contribution to the rebuild:** Domain 1 (HNSW) should be prioritized and built during the rebuild as a guarded pilot. Domains 2–4 should remain roadmap items.

---

## 3. What to Preserve from Each Source

### 3.1 Current FPPE build

#### Preserve as-is
- `EngineConfig` frozen dataclass with all locked defaults (config.py)
- `FeatureRegistry` pluggable feature set system (features.py)
- `RegimeLabeler` with binary/multi/octet modes (regime.py)
- `PlattCalibrator` / `IsotonicCalibrator` / `NoCalibrator` factory (calibration.py)
- Three-filter signal gate in `generate_signal()` (projection.py)
- `Matcher.query()` batched KNN with regime filtering (matching.py — logic, not current interface)
- All scoring functions: `brier_score`, `brier_skill_score`, `compute_crps` (scoring.py)
- `evaluate_probabilistic()` combined evaluation suite (evaluation.py)
- `WalkForwardRunner` with deadline support (walkforward.py)
- `SweepRunner` + `BayesianSweepRunner` (sweep.py)
- Crash-safe I/O: `atomic_write`, `LockFile`, `ProgressLog` (reliability.py)
- `ExperimentLogger` with deduplication (experiment_logging.py)
- `RunManifest` provenance tracking (manifest.py)
- Candlestick continuous encoding (candlestick.py)
- Sector mapping and cross-asset features (sector.py — canonical source)
- `TradingConfig` hierarchy with frozen sub-configs (trading_system/config.py)
- `RiskEngine.size_position()` with ATR-based sizing (risk_engine.py)
- `PortfolioManager` ranking + allocation (portfolio_manager.py)
- All frozen state dataclasses (portfolio_state.py, risk_state.py)
- `BacktestEngine` simulation loop with friction model (backtest_engine.py)

#### Rebuild (preserve logic, improve interface)
- `PatternEngine.fit()` / `predict()` — add contract validation at boundaries
- `DataLoader` — add Pandera validation, test coverage, lineage metadata
- `schema.py` — replace hand-rolled validation with Pandera
- `Matcher` — split into retrieval stages (prep → index → query → filter → package)
- `signal_adapter.py` — remove strategy.py dependency, add UnifiedSignal Pydantic model
- `backtest_engine.py` — extract SharedState, add trade-event schema, feature-flag SlipDeficit
- Inter-layer communication — implement actual SharedState object

#### Discard
- All root-level legacy scripts (`strategy.py`, `strategyv1-v4.py`, `oldstrategy*.py`, etc.)
- `returns_hybrid` feature set placeholder (replace with proper `NotImplementedError` guard)
- `simulate_signals_from_val_db()` in signal_adapter.py (replace with PatternEngine-native path)
- `dedup.py`, `diagnose_distances.py`, `quick_sweep.py` (root-level ad-hoc scripts)
- All scratch files (`freshcmdwindow.txt`, `experiment_log.md.txt`, `newprogram.md`, `program - Copy.md`)

### 3.2 Phase 3.5

#### Preserve
- ABC system (`BaseDistanceMetric`, `BaseCalibrator`, `BaseRiskOverlay`) — formalize as plugin contracts
- `RiskOverlayResult` dataclass — standardized overlay output
- `SlipDeficit` module — most production-ready research module
- All 21 Phase 3.5 tests

#### Keep as guarded pilot (feature-flagged)
- `EMDDistance` — useful as reranker, not as primary retriever
- `BMACalibrator` — pending validation results; shape asymmetry needs resolution

#### Risks
- BMA shape asymmetry ((N,K) vs (N,)) may contaminate the calibration contract if not handled carefully
- SlipDeficit is currently hard-wired with no disable switch
- EMD's O(N×M) runtime prevents use at scale without HNSW pre-filtering

### 3.3 Phase C

#### Preserve (implement during rebuild)
- HNSW approximate NN concept — but as a `Matcher` backend option, not a `BaseDistanceMetric` subclass
- Promotion gate criteria (BSS ≥ 0.02, all tests pass, locked settings updated with evidence)

#### Keep as roadmap (do not implement during rebuild)
- Hawkes Process (Domain 2) — research spike only
- CPOD/EILOF (Domain 3) — research spike only
- OWA weighting (Domain 4) — touches locked settings, requires strong evidence

#### Discard from architecture
- Suggestion to subclass `BaseDistanceMetric` for HNSW — wrong abstraction level. HNSW is an index strategy. The distance metric is still Euclidean (or cosine). HNSW should be a `Matcher` backend that implements `fit()` / `kneighbors()` with an sklearn-compatible interface.

---

## 4. Proposed Target Architecture

### 4.1 Top-level system layout

```
FPPE/
├── pattern_engine/           # Core analogue engine (Lane A — stable)
│   ├── contracts/            # Pydantic + Pandera boundary schemas
│   ├── config.py             # EngineConfig (frozen, validated)
│   ├── schema.py             # Pandera DataFrame schemas
│   ├── data.py               # DataLoader (ingestion + features)
│   ├── features.py           # Feature registry + column definitions
│   ├── matching.py           # Retriever (exact + ANN backends)
│   ├── calibration.py        # Platt/Isotonic/None calibrators
│   ├── regime.py             # Market regime labeling
│   ├── projection.py         # Forward projection + signal gate
│   ├── scoring.py            # Proper scoring rules
│   ├── evaluation.py         # Combined eval suite
│   ├── cross_validation.py   # Multi-config validation
│   ├── sweep.py              # Grid + Bayesian sweeps
│   ├── walkforward.py        # Walk-forward validator
│   ├── engine.py             # PatternEngine (main entry)
│   ├── live.py               # Production EOD runner
│   ├── overnight.py          # Multi-phase research runner
│   ├── reliability.py        # Atomic I/O + crash recovery
│   ├── experiment_logging.py # Provenance logging
│   ├── manifest.py           # Run manifests
│   ├── candlestick.py        # Candlestick features
│   └── sector.py             # Sector mapping (SINGLE SOURCE)
│
├── trading_system/           # Four-layer trading system (Lane A — stable)
│   ├── contracts/            # Pydantic trade/risk/allocation schemas
│   ├── config.py             # TradingConfig hierarchy
│   ├── shared_state.py       # NEW: SharedState (inter-layer bus)
│   ├── signal_adapter.py     # FPPE → UnifiedSignal bridge
│   ├── backtest_engine.py    # Layer 1: trade simulation
│   ├── risk_state.py         # Layer 2 state tracking
│   ├── risk_engine.py        # Layer 2: volatility sizing + stops
│   ├── portfolio_state.py    # Layer 3 frozen state objects
│   ├── portfolio_manager.py  # Layer 3: ranking + allocation
│   ├── strategy_evaluator.py # NEW: Layer 4: rolling metrics + status
│   ├── run_phase1.py         # Phase 1 runner
│   └── run_phase2.py         # Phase 2 runner
│
├── research/                 # Guarded pilots + research spikes (Lane B/C)
│   ├── __init__.py           # ABCs: BaseDistanceMetric, BaseCalibrator, BaseRiskOverlay
│   ├── hnsw_distance.py      # NEW: HNSW ANN backend (Lane B — guarded pilot)
│   ├── emd_distance.py       # EMD reranker (Lane B — guarded pilot)
│   ├── bma_calibrator.py     # BMA calibrator (Lane C — research spike)
│   ├── slip_deficit.py       # SlipDeficit TTF overlay (Lane B — guarded pilot)
│   └── phase_c_roadmap.md    # Deferred domains (Hawkes, CPOD, OWA)
│
├── tests/
│   ├── unit/                 # Per-module unit tests
│   ├── integration/          # Cross-module integration tests
│   ├── parity/               # NEW: Baseline parity harnesses
│   ├── performance/          # NEW: Throughput + latency benchmarks
│   └── regression/           # Bug-fix regression tests
│
├── scripts/                  # Validation + utility scripts
│   ├── validate_emd_bma.py
│   ├── feature_set_comparison.py
│   ├── atr_sweep.py
│   └── hash_baseline_inputs.py  # NEW: dataset fingerprinting
│
├── artifacts/                # NEW: Versioned outputs
│   ├── manifests/            # Baseline manifests, dataset hashes
│   ├── baselines/            # Frozen baseline reports
│   └── benchmarks/           # Performance benchmarks
│
├── archive/                  # Retired code (git history preserved)
│   ├── strategy.py
│   ├── strategyv1-v4.py
│   ├── oldstrategy*.py
│   └── ...
│
├── docs/
│   ├── design/               # Active design docs
│   ├── campaigns/            # Campaign execution plans
│   ├── session-logs/         # Session handoff logs
│   ├── research/             # Research specs
│   └── rebuild/              # NEW: Phase 3Z rebuild artifacts
│
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Dev/test dependencies
├── requirements.lock.txt     # NEW: Pinned versions
├── prepare.py                # LOCKED — do not modify
├── CLAUDE.md                 # Session protocol
└── AGENTS.md                 # Multi-agent routing
```

### 4.2 Major modules and responsibilities

#### Pattern Engine layer

| Module | Responsibility | Inputs | Outputs |
|--------|---------------|--------|---------|
| `config.py` | Frozen hyperparameters with validated defaults | Constructor args | `EngineConfig` (frozen) |
| `contracts/` | Pydantic models for cross-boundary data | N/A | Validation + serialization |
| `schema.py` | Pandera schemas for all DataFrame boundaries | DataFrames | Validated DataFrames or SchemaError |
| `data.py` | Ingestion, feature computation, temporal splitting | Ticker list, date range | Validated DataFrames with lineage metadata |
| `features.py` | Feature set registry, column definitions | Feature set name | Ordered column list + metadata |
| `matching.py` | KNN retrieval with pluggable backends (exact, HNSW) | Scaled feature matrix | Neighbor indices, distances, metadata |
| `calibration.py` | Probability calibration (Platt, Isotonic, None) | Raw probabilities | Calibrated probabilities [0,1] |
| `regime.py` | Market regime classification (binary/multi/octet) | SPY price data | Regime labels per row |
| `projection.py` | Forward projection + three-filter signal gate | Match set | Signal (BUY/SELL/HOLD) + metadata |
| `engine.py` | Orchestrates fit → predict → evaluate cycle | Config + DataFrames | PredictionResult |

#### Trading System layer

| Module | Responsibility | Inputs | Outputs |
|--------|---------------|--------|---------|
| `shared_state.py` | Central state bus for inter-layer communication | Layer updates | Typed state snapshots |
| `signal_adapter.py` | Normalize model outputs to UnifiedSignal | FPPE/DL predictions | `UnifiedSignal` (Pydantic) |
| `backtest_engine.py` | Trade simulation with realistic friction | Signals + prices | Trade log, equity curve, P&L |
| `risk_engine.py` | Volatility sizing, stops, drawdown brake | Signal + ATR + equity | `PositionDecision` (frozen) |
| `portfolio_manager.py` | Ranking, sector limits, allocation checks | Signals + portfolio state | `AllocationDecision` (frozen) |
| `strategy_evaluator.py` | Rolling metrics, status flags, baseline comparison | Equity curve + trade log | RED/YELLOW/GREEN + commands |

### 4.3 How components communicate

```
                    ┌──────────────────────────┐
                    │      SharedState         │
                    │  (Pydantic, versioned)    │
                    │                          │
                    │  equity_state             │
                    │  position_state           │
                    │  risk_state               │
                    │  portfolio_state          │
                    │  evaluator_state          │
                    │  commands[]               │
                    └──────┬───────────────────┘
                           │
          ┌────────────────┼────────────────────┐
          │                │                    │
    reads/writes     reads/writes         reads/writes
          │                │                    │
   ┌──────▼──────┐  ┌─────▼──────┐  ┌──────────▼──────────┐
   │ L1 Backtest │  │ L2 Risk    │  │ L3 Portfolio         │
   │ Engine      │──│ Engine     │──│ Manager              │
   └─────────────┘  └────────────┘  └──────────────────────┘
          │                                     │
          └─────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ L4 Strategy │
                    │ Evaluator   │
                    └─────────────┘
```

**Rules:**
- Layers never import each other directly. All inter-layer data flows through `SharedState`.
- `SharedState` is a Pydantic model with typed sub-states. Each layer reads the full state but writes only to its own section.
- Commands flow downward: L4 can emit `HALT` or `REDUCE_EXPOSURE` commands that L1 reads on the next iteration.

### 4.4 Where configuration belongs

- **Engine hyperparameters:** `pattern_engine/config.py` → `EngineConfig` (frozen dataclass, existing)
- **Trading parameters:** `trading_system/config.py` → `TradingConfig` (frozen dataclass, existing)
- **Feature flags for research modules:** `pattern_engine/config.py` → new fields on `EngineConfig`:
  ```python
  # Research feature flags (Lane B pilots)
  use_hnsw: bool = False
  use_slip_deficit: bool = False
  use_emd_rerank: bool = False
  ```
- **Runtime overrides:** Environment variables or CLI args → never in code

### 4.5 Where experimental logic belongs

Research modules live in `research/` and are accessed only through:

1. **Feature flags** on `EngineConfig` or `TradingConfig`
2. **ABC contracts** defined in `research/__init__.py`
3. **Conditional imports** guarded by flag checks

```python
# Example: matching.py
if config.use_hnsw:
    from research.hnsw_distance import HNSWIndex
    index = HNSWIndex(ef_construction=200, M=16)
else:
    index = NearestNeighbors(algorithm=config.nn_algorithm, n_jobs=1)
```

Research code must never be imported at module level in production modules.

### 4.6 Contract enforcement

**Pydantic models** (new `contracts/` directories):

```python
# pattern_engine/contracts/signals.py
class NeighborResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    query_index: int
    neighbor_indices: list[int]
    distances: list[float]
    distance_metric: str  # "euclidean" | "cosine"
    regime_filtered: bool

# trading_system/contracts/trades.py
class TradeEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    ticker: str
    signal_date: date
    signal_direction: SignalDirection
    ordered_quantity: float
    target_price: float
    fill_quantity: float
    fill_price: float
    fill_ratio: float  # fill_quantity / ordered_quantity
    order_timestamp: datetime
    execution_timestamp: datetime
    execution_latency_seconds: float
    entry_friction_bps: float
    exit_friction_bps: float
```

**Pandera schemas** (in `schema.py`):

```python
import pandera as pa

class TrainDBSchema(pa.DataFrameModel):
    Date: pa.Index[pa.DateTime]
    Ticker: pa.Column[str] = pa.Field(nullable=False)
    ret_1d: pa.Column[float] = pa.Field(nullable=False)
    ret_3d: pa.Column[float] = pa.Field(nullable=False)
    # ... all 8 return columns
    fwd_7d_up: pa.Column[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True  # No extra columns allowed
        coerce = False  # No silent type coercion
```

### 4.7 Dataset freezing and artifact manifests

```python
# artifacts/manifests/baseline_manifest_20260321.json
{
    "manifest_version": "1.0",
    "created_at": "2026-03-21T00:00:00Z",
    "git_sha": "3aec621...",
    "python_version": "3.12.x",
    "datasets": {
        "train_db": {
            "path": "data/train_db.parquet",
            "sha256": "abc123...",
            "row_count": 45000,
            "column_count": 35,
            "date_range": ["2015-01-02", "2023-12-29"],
            "ticker_count": 52,
            "schema_version": "1.0"
        },
        "val_db": {
            "path": "data/val_db.parquet",
            "sha256": "def456...",
            "row_count": 12000,
            "column_count": 35,
            "date_range": ["2024-01-02", "2024-12-31"],
            "ticker_count": 52,
            "schema_version": "1.0"
        }
    },
    "test_count": 577,
    "baseline_metrics": {
        "bss_2024_fold": 0.00103,
        "sharpe_phase2": 1.93,
        "annual_return_phase2": 0.151,
        "max_drawdown_phase2": 0.039
    },
    "locked_config": {
        "distance_metric": "euclidean",
        "feature_set": "returns_only",
        "top_k": 50,
        "max_distance": 1.1019,
        "confidence_threshold": 0.65,
        "calibration_method": "platt",
        "regime_mode": "binary",
        "stop_loss_atr_multiple": 3.0
    }
}
```

### 4.8 Parity harness location

```
tests/parity/
├── test_matcher_parity.py       # Vectorized vs trusted exact output
├── test_hnsw_parity.py          # HNSW recall vs exact (when implemented)
├── test_baseline_metrics.py     # Rebuild metrics vs frozen manifest values
├── test_walkforward_parity.py   # Phase 1/2/3 outcomes vs preserved manifests
└── conftest.py                  # Loads frozen data slices + tolerance config
```

**Tolerance policy** (defined in `tests/parity/conftest.py`):

```python
PARITY_TOLERANCES = {
    "integer_outputs": {"method": "exact"},
    "categorical_outputs": {"method": "exact"},
    "ranking_identity": {"method": "exact", "note": "unless ties documented"},
    "float_outputs": {"method": "np.allclose", "rtol": 1e-7, "atol": 1e-8},
    "bss_scores": {"method": "np.isclose", "rtol": 1e-5, "atol": 1e-6},
}
```

Any relaxation of these tolerances requires:
1. Justification documented in the test file
2. Entry in the test manifest
3. Code review approval

### 4.9 Rollback boundaries

Rollback checkpoints are placed after each rebuild phase:

| Phase | Checkpoint | Rollback action |
|-------|-----------|-----------------|
| Phase 0 | Baseline captured | N/A (starting point) |
| Phase 1 | Contracts defined | Revert to Phase 0 commit |
| Phase 2 | Data layer rebuilt | Revert to Phase 1 commit |
| Phase 3 | Feature layer rebuilt | Revert to Phase 2 commit |
| Phase 4 | Matcher rebuilt | Revert to Phase 3 commit |
| Phase 5 | Calibration + signals rebuilt | Revert to Phase 4 commit |
| Phase 6 | Trading layers rebuilt | Revert to Phase 5 commit |
| Phase 7 | SharedState + orchestration | Revert to Phase 6 commit |
| Phase 8 | Research pilots staged | Revert to Phase 7 commit |

Each checkpoint requires: all tests pass + parity harnesses pass + no metric degradation beyond documented tolerances.

---

## 5. Architectural Principles for Phase 3Z

### P1. Validated baseline first, experimental research second
No research feature enters the default code path until it clears a promotion gate (BSS ≥ 0.02 above baseline, all tests pass, locked settings updated with evidence). The rebuild sequence prioritizes restoring baseline parity before staging any pilot.

### P2. Contracts at every boundary
Every function that crosses a module or package boundary must validate its inputs and outputs against a typed schema (Pydantic for objects, Pandera for DataFrames). No silent coercion, no implicit fallback, no missing-column tolerance.

### P3. Single source of truth for every constant
SECTOR_MAP, feature column lists, locked hyperparameters, and config defaults each have exactly one canonical definition. All other references are imports from the canonical source.

### P4. Feature-flag isolation for all research code
Research modules are never imported at module level in production code. They are loaded conditionally behind explicit boolean flags on the config object. Disabling a flag must restore exact baseline behavior.

### P5. Frozen data for frozen metrics
Every performance claim is tied to a specific (git_sha, dataset_hash) pair. Parity tests run against frozen data slices stored in manifests. No live data is used for parity verification.

### P6. Deterministic pipelines
Given the same config + input data, the system must produce bit-identical outputs for integer/categorical values and `np.allclose`-identical outputs for floating-point values. Any source of non-determinism (random seeds, parallel execution, hash ordering) must be controlled or documented.

### P7. Fail fast, fail loud
No silent fallback. If a required column is missing, raise immediately. If a config field is stale, raise immediately. If a shape mismatch occurs, raise immediately. The cost of a clear error is always lower than the cost of silent wrong results.

### P8. Configuration-driven execution
Behavior is controlled by config objects, not by code branches. Switching from exact to HNSW retrieval, from Platt to Isotonic calibration, or from Phase 1 to Phase 3 mode should be a config change, not a code change.

### P9. Parity-first rebuild sequencing
Each rebuild phase must re-establish parity with the frozen baseline before proceeding to the next phase. If parity cannot be restored within 5 working days, the phase enters review mode. A smaller mergeable rebuild is always preferred over a larger unmergeable one.

### P10. Test-first for core flows
Every core flow (fit → predict → evaluate, signal → backtest → P&L) has an integration test that runs on frozen data before any refactoring begins. The test is written first; the refactoring is proven correct by the test continuing to pass.

---

## 6. Rebuild Workstreams

### WS1: Repository Restructuring

**Purpose:** Remove legacy code, organize test directories, create artifact directories, centralize constants.

**Major tasks:**
1. Move all root-level legacy scripts to `archive/` (strategy.py, strategyv1-v4.py, oldstrategy*.py, dedup.py, diagnose_distances.py, quick_sweep.py, test_strategy.py)
2. Delete scratch files (freshcmdwindow.txt, experiment_log.md.txt, newprogram.md, program - Copy.md)
3. Create `tests/unit/`, `tests/integration/`, `tests/parity/`, `tests/performance/`, `tests/regression/` directories
4. Reorganize existing tests into subdirectories by category
5. Create `artifacts/manifests/`, `artifacts/baselines/`, `artifacts/benchmarks/`
6. Create `pattern_engine/contracts/`, `trading_system/contracts/`
7. Centralize SECTOR_MAP: delete from matching.py and trading_system/config.py, import from sector.py
8. Update .gitignore for parquet files, results, and scratch files
9. Create `requirements.lock.txt` from current working environment

**Dependencies:** None (can start immediately)

**Risks:**
- Moving `strategy.py` to archive breaks `signal_adapter.simulate_signals_from_val_db()`. This function must be rewritten to use PatternEngine before the move, or wrapped with a deprecation shim.

### WS2: Baseline Capture and Frozen Data Manifests

**Purpose:** Lock down the exact state (code + data + results) before any reconstruction begins.

**Major tasks:**
1. Run full test suite and record exact pass count (currently 577)
2. Generate SHA-256 hashes for all baseline datasets (train_db.parquet, val_db.parquet, etc.)
3. Export current config defaults (EngineConfig, TradingConfig) to JSON
4. Run walk-forward validation on frozen data and record per-fold BSS
5. Record all locked settings and experiment-proven thresholds
6. Write `scripts/hash_baseline_inputs.py` to automate dataset fingerprinting
7. Generate `artifacts/manifests/baseline_manifest_20260321.json`
8. Generate `artifacts/baselines/rebuild_start_report.md`

**Dependencies:** None (can run in parallel with WS1)

**Risks:**
- Dataset files may not be in the repo (gitignored). Must ensure the exact dataset is available and pinned before proceeding. If datasets must be regenerated, hash the regenerated output and document the process.

### WS3: Contract and Schema Layer

**Purpose:** Define typed interfaces for all cross-module boundaries.

**Major tasks:**
1. Add `pydantic` and `pandera` to requirements.txt
2. Create `pattern_engine/contracts/`:
   - `datasets.py` — TrainDBContract, QueryDBContract, FeatureMatrixContract
   - `signals.py` — NeighborResult, CalibrationResult, UnifiedSignal
   - `state.py` — EngineState (serializable engine checkpoint)
3. Create `trading_system/contracts/`:
   - `trades.py` — TradeEvent (with future-ready fields: ordered_quantity, fill_quantity, fill_price, fill_ratio, order_timestamp, execution_timestamp, execution_latency_seconds)
   - `decisions.py` — PositionDecision, AllocationDecision, EvaluatorStatus
   - `state.py` — SharedState (inter-layer bus)
4. Rebuild `schema.py` with Pandera:
   - TrainDBSchema, QueryDBSchema, FeatureMatrixSchema
   - Strict mode (no extra columns), no coercion
5. Add `@validate_call` decorators to key public functions where practical
6. Define `CalibrationResult` with explicit `is_calibrated: bool` field

**Dependencies:** WS1 (directory structure must exist)

**Risks:**
- Pydantic/Pandera may introduce import-time overhead. Benchmark before committing to `@validate_call` on hot paths.
- Trade-event schema must include future-ready fields even if v1 defaults them to simple values.

### WS4: Core Engine Redesign

**Purpose:** Rebuild PatternEngine internals around contracts and pluggable backends.

**Major tasks:**
1. Split `Matcher` into staged pipeline:
   - `_prepare_features()` — scaling, weighting
   - `_build_index()` — NearestNeighbors or HNSW (config-driven)
   - `_query_batch()` — batched KNN query
   - `_post_filter()` — distance, regime, ticker exclusion
   - `_package_results()` — structured NeighborResult output
2. Add feature-flag support for HNSW backend in config.py
3. Ensure `PatternEngine.fit()` validates inputs via Pandera schemas
4. Ensure `PatternEngine.predict()` validates outputs via Pydantic contracts
5. Fix `live.py` assert guards → RuntimeError (tech debt D2)
6. Add engine state migration path for version bumps (tech debt A2)
7. Guard `returns_hybrid` with `NotImplementedError` (tech debt C2)

**Dependencies:** WS3 (contracts must be defined)

**Risks:**
- Matcher split may introduce floating-point drift in vectorized paths. Parity tests (WS2 baseline) must pass after split.

### WS5: Data Pipeline Redesign

**Purpose:** Make ingestion, feature computation, and persistence deterministic, validated, and auditable.

**Major tasks:**
1. Add Pandera validation at DataLoader boundaries:
   - After raw data download (validate OHLCV columns, dtypes, no NaNs in price fields)
   - After feature computation (validate return columns, forward targets, supplementary features)
   - Before persistence (validate schema before writing parquet)
2. Add lineage metadata to DataFrames (`DataFrame.attrs` with source, timestamp, hash)
3. Separate raw ingestion from engineered dataset generation
4. Make every write atomic (use reliability.py's `atomic_write`)
5. Add deterministic temporal split with leakage guard assertions
6. Write `tests/unit/test_data.py` — the most critical missing test coverage
7. Remove any silent datetime coercion paths

**Dependencies:** WS3 (Pandera schemas must exist)

**Risks:**
- DataLoader.download() depends on yfinance, which is network-dependent and version-sensitive. Tests must mock the download path.

### WS6: Trading System Rebuild

**Purpose:** Implement SharedState, build strategy_evaluator, feature-flag research overlays.

**Major tasks:**
1. **Implement SharedState** (`trading_system/shared_state.py`):
   - Pydantic model with typed sub-states (equity, positions, risk, portfolio, evaluator)
   - Each layer reads full state, writes only to its own section
   - Command queue for downward control flow (HALT, REDUCE_EXPOSURE)
   - Serializable for checkpointing
2. **Rewrite signal_adapter.py**:
   - Remove strategy.py dependency
   - Use PatternEngine API directly for signal generation
   - Convert UnifiedSignal to Pydantic model with validation
3. **Feature-flag SlipDeficit in backtest_engine.py**:
   - Move from hard-wired to config-driven (`TradingConfig.use_slip_deficit: bool = False`)
   - Conditional import from research/ only when flag is True
4. **Build strategy_evaluator.py** (Layer 4):
   - Rolling metrics: 30/90/252/all-time windows
   - Status logic: GREEN/YELLOW/RED with explicit threshold definitions
   - Baseline comparison table (vs SPY buy-and-hold)
   - Rolling BSS tracking
   - Calibration drift detection
   - Per-sector hit-rate view
   - Insufficient-data handling (< 30 trades → YELLOW)
   - TWRR decomposition
   - Command outputs back into SharedState
5. **Upgrade TradeEvent schema** to support future-ready fields:
   - `ordered_quantity`, `target_price`, `fill_quantity`, `fill_price`
   - `fill_ratio`, `order_timestamp`, `execution_timestamp`
   - v1 defaults: fill_ratio=1.0, execution_latency=next-open

**Dependencies:** WS3 (contracts), WS4 (engine redesign for signal generation path)

**Risks:**
- SharedState introduces a new coupling point. Must be carefully typed to prevent accidental cross-layer mutation.
- strategy_evaluator is the largest new module. Break into sub-tasks with independent testability.

### WS7: Testing Strategy

**Purpose:** Reorganize tests, fill coverage gaps, add parity and performance harnesses.

**Major tasks:**
1. Reorganize existing 577 tests into `tests/unit/`, `tests/integration/`, `tests/regression/`
2. Write `tests/unit/test_data.py` (zero coverage today — tech debt T1)
3. Write `tests/unit/test_live.py` (zero coverage today — tech debt T2)
4. Write `tests/parity/test_matcher_parity.py` — vectorized vs trusted exact
5. Write `tests/parity/test_baseline_metrics.py` — rebuild vs frozen manifest
6. Write `tests/parity/conftest.py` — tolerance policy definitions
7. Write `tests/performance/test_matcher_throughput.py` — regression guard
8. Write `tests/regression/test_sector_map.py` — SECTOR_MAP divergence prevention
9. Write `tests/regression/test_year_boundary.py` — cooldown across year boundaries
10. Write `tests/regression/test_no_ghost_params.py` — verify every config field affects behavior

**Dependencies:** WS2 (frozen baselines must exist for parity tests)

**Risks:**
- Test reorganization may break import paths. Must update all test discovery configuration.

### WS8: Configuration Management

**Purpose:** Centralize, validate, and lock all configuration.

**Major tasks:**
1. Generate `requirements.lock.txt` from current verified environment
2. Split requirements: `requirements.txt` (runtime), `requirements-dev.txt` (testing)
3. Audit all config fields against actual usage:
   - Verify every EngineConfig field affects behavior (no ghost params)
   - Verify every TradingConfig field affects behavior
   - Remove or document any no-op fields
4. Add `cal_frac` resolution: document whether it is 0.76 or "all X_train" and update locked settings accordingly

**Dependencies:** None (can run in parallel)

**Risks:**
- Discovering ghost parameters may require config schema changes that affect test fixtures.

### WS9: Documentation and Developer Workflow

**Purpose:** Update all design docs to reflect rebuilt architecture, add CI pipeline.

**Major tasks:**
1. Create `.github/workflows/test.yml` (Windows runner, Python 3.12, full test suite)
2. Update `FPPE_TRADING_SYSTEM_DESIGN.md` to v0.5 reflecting SharedState and Layer 4
3. Update `CLAUDE.md` with post-rebuild module counts and test counts
4. Write `docs/rebuild/PHASE_3Z_COMPLETION_REPORT.md` after rebuild
5. Update `TECH_DEBT_AUDIT.md` to close resolved items
6. Add pre-commit hook for `pytest tests/ -v` (optional, user preference)

**Dependencies:** All other workstreams complete

**Risks:**
- CI on Windows runners may have availability/cost constraints.

---

## 7. Step-by-Step Implementation Roadmap

### Phase 0 — Freeze, Snapshot, and Baseline Capture (Day 1)

**Objective:** Lock down the current state before reconstructing anything.

**Tasks:**
1. Create a dedicated rebuild branch: `git checkout -b phase3z-full-rebuild`
2. Run `python -m pytest tests/ -v` and record exact pass count
3. Export `pip freeze > artifacts/manifests/requirements_lock.txt`
4. Write `scripts/hash_baseline_inputs.py`:
   ```python
   import hashlib, pathlib, json
   datasets = ["data/train_db.parquet", "data/val_db.parquet", ...]
   hashes = {}
   for path in datasets:
       h = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
       hashes[path] = {"sha256": h, "size_bytes": pathlib.Path(path).stat().st_size}
   ```
5. Run `python scripts/hash_baseline_inputs.py > artifacts/manifests/baseline_dataset_hashes_20260321.json`
6. Export config defaults: `python -c "from pattern_engine.config import EngineConfig; print(EngineConfig().__dict__)"` → save to manifest
7. Identify and document all feature flags and dead config fields
8. Write `artifacts/baselines/rebuild_start_report.md` containing:
   - Test count, benchmark count
   - Walk-forward summary, Phase 1/2/3 summary
   - Known defects (from TECH_DEBT_AUDIT.md)
   - Known doc inconsistencies

**Gate:** Nothing is rewritten until the baseline can be reproduced from frozen code + frozen input data.

**Parallel work:** WS8 (requirements lock, config audit) can proceed in parallel.

---

### Phase 1 — Contracts First, Logic Second (Day 2)

**Objective:** Define immutable interfaces before rewriting implementation.

**Tasks:**
1. Add `pydantic>=2.0` and `pandera>=0.18` to requirements.txt
2. Create `pattern_engine/contracts/__init__.py`
3. Define dataset contracts (Pandera):
   - `TrainDBSchema` — all 8 return columns, forward targets, Date index, Ticker column
   - `QueryDBSchema` — same minus forward targets
   - `FeatureMatrixSchema` — numeric columns only, no NaN
4. Define signal contracts (Pydantic):
   - `NeighborResult` — query_index, neighbor_indices, distances, metric, regime_filtered
   - `CalibratedProbability` — value: float, is_calibrated: bool, method: str
   - `UnifiedSignal` — date, ticker, direction, confidence, source, sector, raw_metadata
5. Define trade contracts (Pydantic):
   - `TradeEvent` — with future-ready fields (ordered_quantity, fill_quantity, fill_price, fill_ratio, order_timestamp, execution_timestamp, execution_latency_seconds)
   - `RiskDecision` — approved, ticker, position_pct, shares, stop_price, rejection_reason
   - `AllocationDecision` — approved, ticker, rank, confidence, sector, rejection_reason
   - `EvaluatorStatus` — status (GREEN/YELLOW/RED), metrics dict, commands list
6. Define state contracts (Pydantic):
   - `SharedState` — equity_state, position_state, risk_state, portfolio_state, evaluator_state, commands
7. Write unit tests for all contracts (validation passes for valid data, raises for invalid)

**Gate:** Every core module can be described in terms of contracts without reading its body.

**Abort criterion:** If contract definitions reveal fundamental incompatibilities between pattern_engine and trading_system schemas, pause and resolve before proceeding.

---

### Phase 2 — Data Layer Rebuild (Day 3)

**Objective:** Make ingestion, dataset preparation, and persistence deterministic and auditable.

**Tasks:**
1. Rebuild `schema.py` using Pandera schemas from Phase 1
2. Add validation calls at DataLoader boundaries:
   - After `download()`: validate OHLCV schema
   - After `compute_features()`: validate feature matrix schema
   - Before parquet write: validate full schema
3. Add lineage metadata (`DataFrame.attrs`) with source info, timestamp, row count hash
4. Ensure temporal split has explicit leakage guards (no future data in training set)
5. Make all writes atomic via `reliability.py`
6. Write `tests/unit/test_data.py` with:
   - Synthetic OHLCV data (30 rows per ticker)
   - Return column value verification at spot-check rows
   - Overnight feature sign verification
   - Temporal split leakage guard
   - Mock yfinance download

**Gate:** A dataset can be loaded, validated, fingerprinted, and persisted with deterministic outputs and explicit manifests.

**Parity check:** Run walk-forward on frozen data; BSS must match baseline manifest within tolerance.

---

### Phase 3 — Feature Layer Rebuild (Day 4)

**Objective:** Normalize feature registration and prevent silent misapplication.

**Tasks:**
1. Add promotion states to FeatureSet: `stable`, `available`, `experimental`, `placeholder`, `retired`
2. Tag current sets: `returns_only` → stable, others → available or experimental
3. Guard `returns_hybrid` with `NotImplementedError` + clear error message
4. Ensure each feature set returns deterministic ordered columns (sorted, frozen)
5. Add feature-family manifest generation
6. Add scaler/feature-set compatibility check (prevent mismatched scaling)
7. Centralize SECTOR_MAP: single import from `sector.py` everywhere

**Gate:** Feature generation is reproducible, order-stable, and impossible to misapply silently.

**Parity check:** Feature matrix on frozen data must be bit-identical to baseline.

---

### Phase 4 — Retrieval Layer Rebuild (Day 5)

**Objective:** Split Matcher into staged pipeline, prepare for HNSW backend.

**Tasks:**
1. Split `Matcher` into 5 stages (prep → build → query → filter → package)
2. Each stage has typed inputs/outputs matching Phase 1 contracts
3. Add `use_hnsw: bool` flag to EngineConfig
4. When `use_hnsw=False` (default): exact same ball_tree behavior as baseline
5. Prepare HNSW interface (implement in Phase 8, not here)
6. Add parity test: rebuilt Matcher vs baseline on frozen data
7. Add performance regression test: throughput must not degrade > 5%

**Gate:** Exact parity harness passes. Vectorized throughput benchmark passes.

**Parity check:** Neighbor indices and distances on frozen data must match baseline within tolerance (rtol=1e-7, atol=1e-8 for distances; exact match for indices unless ties documented).

---

### Phase 5 — Calibration and Signal-Generation Rebuild (Day 6)

**Objective:** Clean calibration contracts, verify no ghost params, preserve three-filter gate.

**Tasks:**
1. Ensure every calibrator exposes: fit source, inference source, assumptions, output type
2. Resolve `cal_frac` ambiguity: document actual behavior, update locked settings if needed
3. Remove any ghost parameters that don't affect behavior
4. Isolate regime classification from retrieval logic (already mostly done)
5. Add end-to-end audit log: one signal from raw data → calibrated probability → BUY/SELL/HOLD
6. Add explicit distinction in evaluation between:
   - Probabilistic quality (BSS, CRPS, calibration)
   - Directional signal accuracy (precision, recall)
   - Trade-layer outcomes (Sharpe, P&L)

**Gate:** A full training/evaluation cycle can be audited from raw dataset to calibrated BUY/SELL/HOLD output without hidden state.

**Parity check:** BSS on frozen 2024 validation fold must match baseline manifest.

**ABORT CRITERION:** If Phase 5 cannot re-establish baseline parity within 5 working days after implementation start:
1. Freeze further feature work
2. Generate divergence report identifying root cause (data drift, contract mismatch, numerical tolerance, logic regression)
3. Choose: revert offending changes, split into smaller sub-phases, or archive branch
4. If still broken after 2 additional remediation days: branch is declared non-mergeable

---

### Phase 6 — Trading System Rebuild (Days 7–8)

**Objective:** Implement SharedState, feature-flag research overlays, build Layer 4.

**Sub-phases:**

#### 6A — Signal Adapter (Day 7, morning)
- Remove strategy.py dependency
- Convert UnifiedSignal to Pydantic model
- Rewrite `simulate_signals_from_val_db()` to use PatternEngine API
- Move strategy.py to archive/

#### 6B — Backtest Engine (Day 7, afternoon)
- Feature-flag SlipDeficit (`TradingConfig.use_slip_deficit: bool = False`)
- Upgrade trade records to use TradeEvent contract (with future-ready fields)
- Wire SharedState for daily state publishing

#### 6C — Risk Engine (Day 7–8)
- No major logic changes (already clean)
- Add RiskDecision Pydantic output
- Wire to SharedState

#### 6D — Portfolio Manager (Day 8)
- No major logic changes (already clean)
- Add AllocationDecision Pydantic output
- Wire to SharedState

#### 6E — Strategy Evaluator (Days 8–9)
- Implement rolling metrics (30/90/252/all-time)
- Implement status logic (GREEN/YELLOW/RED) with explicit thresholds:
  - RED: 90-day Sharpe < 0 OR max_drawdown > 15% OR halt triggered
  - YELLOW: 30-day Sharpe < 0.5 OR insufficient data (< 30 trades)
  - GREEN: all metrics within acceptable envelopes
- Baseline comparison table (vs SPY)
- Rolling BSS tracking
- Calibration drift detection (bucket shift > 10% → warning)
- Per-sector hit-rate view
- TWRR decomposition
- Command outputs into SharedState

**Gate:** Trade log, equity curve, and P&L attribution all reconcile exactly. Evaluator correctly determines operational status.

**Parity check:** Phase 2/3 backtest results on frozen data must match baseline.

---

### Phase 7 — Orchestration and Integration (Day 9)

**Objective:** Reconnect all layers using SharedState and validate end-to-end.

**Tasks:**
1. Centralize orchestration sequence in engine runners
2. Make each layer's read/write dependencies on SharedState explicit
3. Prohibit direct cross-layer imports
4. Require SharedState update ordering (L1 → L2 → L3 → L4)
5. Make overnight runner use SharedState for multi-phase orchestration
6. Keep static and Bayesian sweep modes separate but contract-compatible
7. Run full end-to-end integration test: FPPE signal → adapter → backtest → risk → PM → evaluator

**Gate:** A full overnight simulation can run with traceable state transitions from close to report output.

---

### Phase 8 — Research Pilot Staging (Days 10–11)

**Objective:** Stage guarded pilots without contaminating stable path.

**Tasks:**
1. Implement `research/hnsw_distance.py`:
   - sklearn-compatible `fit()` / `kneighbors()` interface
   - `hnswlib` backend (pip install guard with clear error message)
   - L2-distance parity behavior
   - Query-time `ef` parameter support
   - Dimensional mismatch guards
2. Wire HNSW into Matcher behind `use_hnsw` flag
3. Add parity test: HNSW recall@50 vs exact (target > 0.95)
4. Add performance test: HNSW ms/query on 50k fingerprints (target < 10ms)
5. Verify SlipDeficit feature flag works correctly (enable/disable produces expected behavior)
6. Run EMD/BMA validation scripts (SLE-43, SLE-44) if not already run
7. Document promotion gate status for each pilot

**Gate:** All pilots work behind flags, stable path is unaffected when flags are off, parity harnesses pass.

---

### Phase 9 — Parity Validation and Cleanup (Days 11–12)

**Objective:** Final validation that the rebuild matches the frozen baseline.

**Tasks:**
1. Run full test suite — all must pass
2. Run all parity tests against frozen baseline data
3. Run walk-forward validation — BSS per fold must match baseline manifest within tolerance
4. Run Phase 1/2/3 backtests — P&L must match baseline within tolerance
5. Run performance benchmarks — no regression > 5% vs baseline
6. Update CLAUDE.md with new module counts and test counts
7. Update TECH_DEBT_AUDIT.md to close resolved items
8. Write `docs/rebuild/PHASE_3Z_COMPLETION_REPORT.md`
9. Create CI pipeline (`.github/workflows/test.yml`)

**Gate:** The rebuild is complete only when all parity tests pass, all documentation is updated, and the rebuild can be reproduced from scratch on a clean clone.

---

### Where parallel work is safe vs unsafe

| Safe to parallelize | Unsafe to parallelize |
|--------------------|-----------------------|
| WS1 (repo restructure) + WS2 (baseline capture) | WS4 (matcher) depends on WS3 (contracts) |
| WS8 (config audit) + WS2 (baseline capture) | WS5 (calibration) depends on WS4 (matcher) |
| Phase 0 tasks (all independent) | Phase 6 sub-phases have dependencies (6A before 6B) |
| HNSW implementation + strategy_evaluator | Parity tests depend on all rebuild phases |

---

## 8. Stability and Reliability Plan

### 8.1 Runtime stability

- **Atomic writes everywhere:** All disk I/O uses `reliability.py`'s `atomic_write()`. No partial writes, no corrupt checkpoints.
- **Lock files for exclusive operations:** Overnight runner, sweep runner use advisory locks. Stale-PID detection prevents deadlock.
- **Exception isolation per fold/phase:** One fold's failure doesn't kill a walk-forward run. One phase's failure doesn't kill an overnight session.

### 8.2 Error handling

- **assert → RuntimeError** for all public API guards (already in progress, some gaps remain in live.py)
- **Pydantic validation** at contract boundaries: invalid data raises `ValidationError` with clear field-level messages
- **Pandera validation** at DataFrame boundaries: schema violations raise `SchemaError` with column/row details
- **No silent fallback:** Missing columns, wrong dtypes, NaN in required fields all raise immediately

### 8.3 Checkpointing and recovery

- **Walk-forward:** Per-fold results saved atomically. Resume skips completed folds.
- **Overnight runner:** Phase checkpoints saved after each phase. Resume loads last checkpoint.
- **Backtest:** Daily equity snapshots enable mid-run inspection. Final reconciliation validates consistency.
- **SharedState:** Serializable to JSON for daily checkpointing.

### 8.4 Reproducibility

- **Frozen data manifests:** Every run references a specific dataset hash
- **Config hashing:** `EngineConfig` and `TradingConfig` produce deterministic SHA-256 hashes
- **Git SHA in manifests:** Every result links to the exact code version
- **Seed control:** Any randomized operation (Bayesian sweep) uses explicit seeds
- **nn_jobs=1 always:** Prevents non-deterministic parallel execution on Windows

### 8.5 Debugging

- **End-to-end audit trail:** For any signal, trace from raw data → features → neighbors → projection → calibration → gate → trade → P&L
- **Structured rejection reasons:** Every rejected signal/trade/allocation carries a string reason
- **Experiment logging:** All results logged to TSV with config hash, fold label, deduplication

### 8.6 Branch safety during rebuild

- **Commit after each phase gate:** Every successful parity check gets a commit on the rebuild branch
- **No force-push:** All commits are additive
- **Tag parity-safe commits:** `git tag phase3z-parity-phaseN` after each gate
- **Rollback = revert to tagged commit:** Simple, safe, auditable

---

## 9. Testing and Validation Strategy

### 9.1 Unit tests (per module, isolated)

| Module | Key tests | Priority |
|--------|-----------|----------|
| `config.py` (PE) | Frozen enforcement, default values, nn_algorithm derivation | Existing |
| `schema.py` | Pandera schema pass/fail for valid/invalid DataFrames | NEW |
| `data.py` | Feature computation on synthetic data, temporal split leakage guard | NEW (critical gap) |
| `features.py` | Column order determinism, feature set registration, promotion states | Existing + NEW |
| `matching.py` | Staged pipeline, distance filtering, regime filtering, batch behavior | Existing + NEW |
| `calibration.py` | Platt/Isotonic/None factory, fit/transform shapes | Existing |
| `regime.py` | Binary/multi/octet modes, SPY fallback, v4 leakage fix | Existing |
| `projection.py` | Three-filter gate, distance weighting | Existing |
| `live.py` | RuntimeError guards, output schema, signal sorting | NEW (critical gap) |
| `config.py` (TS) | TradingConfig validation, frozen enforcement | Existing |
| `shared_state.py` | State isolation, command queue, serialization | NEW |
| `signal_adapter.py` | UnifiedSignal validation, PatternEngine integration | Existing + NEW |
| `backtest_engine.py` | Trade lifecycle, friction, cooldown, force-close, feature-flag toggle | Existing |
| `risk_engine.py` | ATR sizing, drawdown scalar, stop computation | Existing |
| `portfolio_manager.py` | Ranking, sector limits, allocation checks | Existing |
| `strategy_evaluator.py` | Rolling metrics, status logic, baseline comparison | NEW |

### 9.2 Integration tests (cross-module)

- Train → predict → evaluate cycle on synthetic data
- FPPE signal → signal adapter → backtest → P&L reconciliation
- Backtest → risk → portfolio → evaluator → SharedState round trip
- Overnight run artifact generation and checkpoint recovery
- Feature-flag toggle: SlipDeficit on/off produces expected behavior difference

### 9.3 Parity tests (rebuild correctness)

- **Matcher parity:** Rebuilt vectorized matcher vs prior trusted exact output on frozen data
- **HNSW parity:** HNSW recall@50 vs exact neighbors (when implemented)
- **Baseline metrics parity:** BSS, Sharpe, annual return, max drawdown vs frozen manifest
- **Walk-forward parity:** Per-fold BSS vs preserved manifest values
- **Tolerance enforcement test:** Verify that tolerance widening is detected and fails

### 9.4 Performance tests (regression guards)

- Matcher throughput: queries/second on standard workload
- HNSW ms/query on 50k fingerprints (when implemented)
- Full daily 52-ticker inference time
- Memory footprint under batch_size=256
- Evaluator computation time on rolling windows

### 9.5 Regression tests (bug-fix guards)

- SECTOR_MAP single-source enforcement (no divergence possible)
- `cal_frac` / ghost parameter prevention (every config field must alter output)
- Windows/joblib deadlock prevention (nn_jobs must equal 1)
- Year-boundary cooldown behavior
- Final-equity friction subtraction
- Same-day churn prohibition

### 9.6 Numerical tolerance rules

Defined in `tests/parity/conftest.py` and enforced across all parity tests:

| Output type | Comparison method | Tolerance | Override procedure |
|------------|-------------------|-----------|-------------------|
| Integer outputs | Exact equality | None | Not allowed |
| Categorical outputs | Exact equality | None | Not allowed |
| Ranking identity | Exact match | None (unless ties mathematically valid and documented) | Document ties |
| Float outputs (vectorized math) | `np.allclose` | rtol=1e-7, atol=1e-8 | Justify per metric, record in test manifest |
| BSS scores | `np.isclose` | rtol=1e-5, atol=1e-6 | Justify, record |
| P&L values | `np.isclose` | rtol=1e-6, atol=0.01 (penny-level) | Justify, record |

**No silent widening.** Any relaxed tolerance requires a code comment justifying it and an entry in the test manifest.

### 9.7 Acceptance criteria for replacing legacy systems

A legacy system (e.g., `strategy.py`) can only be retired when:
1. All functionality is replicated in the rebuilt system
2. All tests pass with the rebuilt system
3. Parity tests confirm equivalent outputs on frozen data
4. The legacy system is moved to `archive/` (not deleted — git history preserved)
5. No active code imports from it

---

## 10. Migration Plan

### 10.1 Migrate directly (minimal changes)

These modules are clean enough to keep with only contract additions:

| Module | Migration action |
|--------|-----------------|
| `config.py` (PE) | Add feature-flag fields for research modules |
| `features.py` | Add promotion states, guard `returns_hybrid` |
| `calibration.py` | No changes needed |
| `regime.py` | No changes needed |
| `projection.py` | No changes needed |
| `scoring.py` | No changes needed |
| `evaluation.py` | No changes needed |
| `cross_validation.py` | No changes needed |
| `sweep.py` | No changes needed |
| `walkforward.py` | No changes needed |
| `reliability.py` | No changes needed |
| `experiment_logging.py` | No changes needed |
| `manifest.py` | No changes needed |
| `candlestick.py` | No changes needed |
| `sector.py` | No changes needed (already canonical) |
| `overnight.py` | Wire SharedState (minor) |
| `risk_engine.py` | Add Pydantic output type |
| `portfolio_manager.py` | Add Pydantic output type |
| `portfolio_state.py` | No changes needed |
| `risk_state.py` | No changes needed |

### 10.2 Rewrite from scratch

| Module | Reason |
|--------|--------|
| `schema.py` | Replace hand-rolled validation with Pandera |
| `shared_state.py` | Does not exist — must be built |
| `strategy_evaluator.py` | Does not exist — must be built |
| `pattern_engine/contracts/` | Does not exist — must be built |
| `trading_system/contracts/` | Does not exist — must be built |
| `tests/parity/` | Does not exist — must be built |
| `scripts/hash_baseline_inputs.py` | Does not exist — must be built |

### 10.3 Wrap temporarily for compatibility

| Module | Wrapper strategy |
|--------|-----------------|
| `signal_adapter.py` | Keep `simulate_signals_from_val_db()` as deprecated shim until PatternEngine path is verified, then remove |
| `matching.py` | Keep existing Matcher interface while adding staged pipeline internally. External callers don't change. |

### 10.4 Retire immediately

| Item | Action |
|------|--------|
| `strategy.py` → `archive/strategy.py` | After signal_adapter.py rewrite |
| `strategyv1-v4.py`, `oldstrategy*.py` | → `archive/` |
| `test_strategy.py` | → `archive/` |
| `dedup.py`, `diagnose_distances.py`, `quick_sweep.py` | → `archive/` |
| `freshcmdwindow.txt`, `experiment_log.md.txt`, `newprogram.md`, `program - Copy.md` | Delete |
| SECTOR_MAP copies in matching.py, trading_system/config.py | Delete (import from sector.py) |

### 10.5 Avoiding breakage during transition

1. **Signal adapter first:** Rewrite signal_adapter.py to use PatternEngine before moving strategy.py. Run all tests between these two changes.
2. **SECTOR_MAP centralization:** Change imports, run tests, then delete copies. Never delete first.
3. **Schema migration:** Add Pandera schemas alongside existing validation first. Run both. Remove old validation after Pandera is verified.
4. **Test reorganization:** Move tests to subdirectories and update test discovery. Verify `pytest tests/ -v` still finds all tests before and after.
5. **SharedState:** Build as a new module that wraps existing mutable dicts in backtest_engine. Verify behavior equivalence before removing old patterns.

---

## 11. Risks and Failure Modes

### R1. Parity drift from changing data

**Risk:** Datasets are regenerated from yfinance during the rebuild, producing different values than the frozen baseline (stock splits, data corrections, API changes).

**Mitigation:** Hash all baseline datasets before the rebuild begins (Phase 0). Parity tests run against frozen data only. Never use live data for parity verification. If frozen data is unavailable, regenerate once, hash, and treat as the new baseline.

### R2. Silent schema/coercion failures

**Risk:** Pandera migration introduces coercion where none existed, or fails to catch edge cases the hand-rolled validation handled.

**Mitigation:** Run both old and new validation in parallel during migration. Diff the results. Remove old validation only after equivalence is proven on the full test suite plus production data.

### R3. Numerical drift from vectorization

**Risk:** Matcher split or HNSW integration introduces micro-variations in floating-point results that compound into different neighbor rankings or BSS scores.

**Mitigation:** Strict tolerance policy (rtol=1e-7, atol=1e-8) enforced in parity tests. Any drift beyond tolerance triggers investigation before proceeding. Ranking identity must be exact unless ties are mathematically valid and explicitly documented.

### R4. Rebuild branch sprawl

**Risk:** The rebuild takes longer than expected, diverges from main, and becomes unmergeable.

**Mitigation:**
- **Timebox:** If Phase 5 (calibration) cannot restore parity within 5 working days, enter review mode.
- **Branch health check:** If parity is broken for > 7 calendar days + > 3 critical modules changed + no merge path visible → stop and review immediately.
- **Incremental commits:** Tag parity-safe commits after each phase. Rollback = revert to tag.
- **Smaller is better:** A merged Phase 0–5 rebuild is better than an unmergeable Phase 0–9 rebuild.

### R5. Incomplete execution-layer abstractions

**Risk:** TradeEvent schema lacks fields needed for future live execution, requiring another schema rebuild.

**Mitigation:** TradeEvent Pydantic model includes all future-ready fields from day one:
- `ordered_quantity`, `target_price`, `fill_quantity`, `fill_price`, `fill_ratio`
- `order_timestamp`, `execution_timestamp`, `execution_latency_seconds`
- v1 defaults: `fill_ratio=1.0`, latency=fixed next-open. No schema changes needed for v2 live execution.

### R6. Contamination from prototype code

**Risk:** Research modules (EMD, BMA, SlipDeficit) leak into the default code path without clearing promotion gates.

**Mitigation:** All research modules behind explicit boolean flags on config objects. Flags default to False. Conditional imports only — no module-level imports from `research/` in production code. Parity tests verify that all flags off = exact baseline behavior.

### R7. SharedState over-coupling

**Risk:** SharedState becomes a god object that every module depends on, creating implicit coupling worse than the current ad-hoc approach.

**Mitigation:** SharedState is typed with sub-states. Each layer writes only to its own section. Cross-layer reads are explicit. The Pydantic model enforces field-level typing. If any layer needs to read another layer's internal state, that's a design smell — flag and resolve.

---

## 12. Final Recommendation

### Best architectural direction

**Incremental contract-driven rebuild** rather than a big-bang rewrite. The current codebase is cleaner than many projects at this stage — the core logic (matching, calibration, signal generation, risk sizing, portfolio allocation) is well-structured and validated. The weakness is at boundaries and infrastructure, not in algorithms.

The rebuild should focus on:
1. **Wrapping existing logic in contracts** (Pydantic + Pandera) rather than rewriting it
2. **Filling the two critical gaps** (SharedState + strategy_evaluator) rather than restructuring what works
3. **Isolating research from production** via feature flags rather than separate branches
4. **Proving correctness at every step** via parity tests against frozen baselines

### Biggest mistakes to avoid

1. **Do not rewrite the Matcher from scratch.** Split it into stages, add contracts at boundaries, but preserve the core KNN logic. The vectorized batch query path is the hottest path in the system — introducing bugs here is catastrophic and hard to detect.

2. **Do not let the rebuild branch live longer than 2 weeks.** If Phase 5 breaks parity and can't recover within 7 days, archive the branch and try a smaller scope. An unmergeable rebuild teaches nothing; a merged partial rebuild teaches a lot.

3. **Do not introduce Pydantic validation on hot paths.** Validate at ingress (when data enters a module) and egress (when data leaves a package). Do not validate inside loops or per-row operations. Benchmark the overhead before committing.

4. **Do not build HNSW before the stable rebuild passes parity.** HNSW is a performance optimization, not a correctness requirement. It belongs in Phase 8, not Phase 4.

5. **Do not promote EMD or BMA to mainline during the rebuild.** Both have unresolved issues (EMD runtime, BMA shape asymmetry) and unvalidated promotion gates. They belong behind flags as guarded pilots.

### Highest-value early wins

1. **SECTOR_MAP centralization (30 minutes).** Fixes an active silent bug (DIS misclassification in matching.py). Immediate correctness improvement.

2. **Baseline data manifests (2 hours).** Enables all future parity testing. Without this, no rebuild phase can prove it hasn't regressed.

3. **`tests/unit/test_data.py` (4 hours).** Closes the single most dangerous coverage gap. The entire pipeline runs on features computed by untested code.

4. **SharedState implementation (1 day).** Unblocks Layer 4 (strategy_evaluator) which is the most important missing capability.

5. **Signal adapter rewrite (half day).** Removes the last dependency on legacy code, enabling archive of 11,379 lines of dead weight.

---

## Phase 3Z Core Build Priorities

1. **Freeze baseline data and code** — hash all datasets, record all metrics, tag the starting commit. Nothing moves until parity is reproducible.

2. **Define contracts at every boundary** — Pydantic for objects, Pandera for DataFrames. No cross-module call without typed validation. Silent coercion is the enemy.

3. **Centralize SECTOR_MAP immediately** — fix the active divergence bug. Single source in `sector.py`, imports everywhere else.

4. **Fill the two critical test gaps** — `data.py` and `live.py` have zero coverage. These are the most dangerous untested paths in the system.

5. **Build SharedState** — replace ad-hoc mutable dicts with a typed Pydantic state bus. This is the architectural prerequisite for Layer 4.

6. **Build strategy_evaluator.py (Layer 4)** — rolling metrics, GREEN/YELLOW/RED status, TWRR, baseline comparison. The system cannot self-monitor without this.

7. **Feature-flag all research modules** — SlipDeficit, EMD, BMA, future HNSW must all be behind explicit boolean flags defaulting to off. No research code in the default path.

8. **Rewrite signal_adapter.py and retire legacy code** — remove strategy.py dependency, move 11,379 lines of dead code to archive.

9. **Add parity harnesses with strict numerical tolerances** — every rebuild phase must prove equivalence to the frozen baseline. No "close enough" — document rtol/atol for every comparison.

10. **Define and enforce rebuild abort criteria** — 5-day timebox on Phase 5 parity, 7-day branch health check, tagged rollback commits after every phase gate. A smaller merged rebuild beats a larger zombie branch.

---

## Proposed Target Module Structure (Post-Rebuild)

```
FPPE/
├── pattern_engine/              # 22→24 modules
│   ├── contracts/
│   │   ├── __init__.py
│   │   ├── datasets.py          # Pandera schemas for DataFrames
│   │   └── signals.py           # Pydantic models for predictions
│   ├── __init__.py              # Public API: PatternEngine, EngineConfig, CrossValidator
│   ├── config.py                # EngineConfig + feature flags + WALKFORWARD_FOLDS
│   ├── schema.py                # Pandera-based validation (rebuilt)
│   ├── data.py                  # DataLoader + lineage metadata
│   ├── features.py              # FeatureRegistry + promotion states
│   ├── matching.py              # Staged Matcher (prep→build→query→filter→package)
│   ├── calibration.py           # Platt/Isotonic/None factory
│   ├── regime.py                # RegimeLabeler (binary/multi/octet)
│   ├��─ projection.py            # Forward projection + three-filter gate
│   ├── scoring.py               # BSS, Brier, CRPS
│   ├── evaluation.py            # Combined eval suite
│   ├── cross_validation.py      # Multi-config validation
│   ├── sweep.py                 # Grid + Bayesian sweeps
│   ├── walkforward.py           # Walk-forward validator
│   ├── engine.py                # PatternEngine (main entry)
│   ├── live.py                  # Production EOD runner (RuntimeError guards)
│   ├── overnight.py             # Multi-phase research runner
│   ├── reliability.py           # Atomic I/O + crash recovery
│   ├── experiment_logging.py    # Provenance logging
│   ├── manifest.py              # Run manifests
│   ├── candlestick.py           # Candlestick features
│   └── sector.py                # SECTOR_MAP (SINGLE canonical source)
│
├── trading_system/              # 9→12 modules
│   ├── contracts/
│   │   ├── __init__.py
│   │   ├── trades.py            # TradeEvent (future-ready)
│   │   ├── decisions.py         # PositionDecision, AllocationDecision
│   │   └── state.py             # SharedState Pydantic model
│   ├── __init__.py
│   ├── config.py                # TradingConfig + feature flags
│   ├── shared_state.py          # NEW: Inter-layer state bus
│   ├── signal_adapter.py        # FPPE→UnifiedSignal (no legacy deps)
│   ├── backtest_engine.py       # Layer 1 (feature-flagged overlays)
│   ├── risk_state.py            # Layer 2 state
│   ├── risk_engine.py           # Layer 2 logic
│   ├── portfolio_state.py       # Layer 3 state
│   ├── portfolio_manager.py     # Layer 3 logic
│   ├── strategy_evaluator.py    # NEW: Layer 4
│   ├── run_phase1.py
│   └── run_phase2.py
│
├── research/                    # 4→5 modules (guarded pilots)
│   ├── __init__.py              # ABCs
│   ├── hnsw_distance.py         # NEW: HNSW ANN backend
│   ├── emd_distance.py          # EMD reranker
│   ├── bma_calibrator.py        # BMA calibrator
│   ├── slip_deficit.py          # SlipDeficit TTF overlay
│   └── phase_c_roadmap.md       # Deferred domains
│
├── tests/                       # 577→700+ tests
│   ├── unit/                    # Per-module isolation tests
│   ├── integration/             # Cross-module flow tests
│   ├── parity/                  # Baseline comparison harnesses
│   ├── performance/             # Throughput + latency benchmarks
│   └── regression/              # Bug-fix guards
│
├── scripts/                     # Validation + utilities
├── artifacts/                   # Versioned outputs + manifests
├── archive/                     # Retired legacy code
├── docs/                        # Design docs + session logs
├── prepare.py                   # LOCKED
├── CLAUDE.md                    # Session protocol
├── AGENTS.md                    # Multi-agent routing
├── requirements.txt             # Runtime deps
├── requirements-dev.txt         # Dev deps
└── requirements.lock.txt        # Pinned versions
```

**Estimated post-rebuild metrics:**
- Production modules: 36 (up from 31)
- Tests: 700+ (up from 577)
- Legacy lines removed: ~11,400
- New contracts: ~15 Pydantic models, ~5 Pandera schemas
- Research modules behind flags: 4 (SlipDeficit, EMD, BMA, HNSW)
- Ghost parameters: 0
- Diverged constants: 0
- Untested production modules: 0

---

*This plan is a living document. Update it as phases complete and new findings emerge during the rebuild.*
