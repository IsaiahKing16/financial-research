# Phase 3Z Execution Plan

**Date:** 2026-03-21
**Linear Project:** FPPE Phase 3Z Rebuild (SLE-51 → SLE-86)
**Baseline Manifest:** `rebuild_phase_3z/artifacts/manifests/baseline_manifest_20260321.json`
**Status:** ACTIVE — M1, M2, M3 complete (794 tests). M4 next (SLE-65–67).

---

## 1. Current-State Findings

### Codebase Snapshot (2026-03-21)

| Metric | Value |
|--------|-------|
| Total modules | 51 |
| Total LOC | ~20,430 |
| Tests passing | 596 / 596 |
| Test framework | pytest, Python 3.12 |
| Data files | 52 ticker CSVs + 1 cached signals CSV |
| Backtest annual return | 18.5% (2024 fold, aggressive profile) |
| Backtest Sharpe | 1.16 |
| Backtest max drawdown | 6.7% |
| Walk-forward BSS | +0.00103 (2024 fold) |

### Architectural Deltas (Current → Target)

| Area | Current State | Target State | Gap |
|------|---------------|--------------|-----|
| Cross-package contracts | Implicit DataFrame conventions | Pydantic models + Pandera schemas | **No contracts exist** |
| HNSW integration | `BaseDistanceMetric` subclass (wrong abstraction) | Matcher backend with `fit()`/`kneighbors()` | **Repositioning required** |
| Feature flags | Research modules hard-wired | Explicit `bool` flags in config, conditional imports | **No flags exist** |
| SharedState | Referenced in design docs, never built | Pydantic model with typed sub-states | **Missing entirely** |
| Layer 4 (strategy_evaluator) | Designed in v0.4, unbuilt | Rolling metrics, RED/YELLOW/GREEN, TWRR | **Missing entirely** |
| Dataset manifests | None | SHA-256 hash manifest per freeze | **Baseline manifest created (this session)** |
| CI pipeline | Manual `pytest` runs | GitHub Actions with parity gate | **No CI** |
| SECTOR_MAP | Diverged across 3 files | Single source in `pattern_engine/sector.py` | **Partially fixed** (trading_system imports from sector.py) |
| Legacy code | `strategy.py` still imported by `signal_adapter.py` | All legacy retired | **Active dependency** |
| Schema validation | Hand-rolled `schema.py` | Pandera `DataFrameSchema(strict=True, coerce=False)` | **No Pandera** |

---

## 2. Linear Issue Breakdown

### Milestone M1: Baseline Freeze (SLE-52 → SLE-56) — COMPLETE
**Gate:** All 596 legacy tests pass, manifest verified, git tag created. ✓

| Issue | Title | Status |
|-------|-------|--------|
| SLE-52 | Freeze baseline dataset hashes (SHA-256 manifest) | ✓ Done |
| SLE-53 | Snapshot EngineConfig + TradingConfig defaults | ✓ Done |
| SLE-54 | Record walk-forward BSS + backtest metrics | ✓ Done |
| SLE-55 | Create `scripts/hash_baseline_inputs.py` | ✓ Done |
| SLE-56 | Tag git baseline: `phase3z-baseline-v1` | ✓ Done |

### Milestone M2: Contracts & Schemas (SLE-57 → SLE-59) — COMPLETE
**Gate:** All boundaries have Pydantic/Pandera contracts. 771 tests passing. ✓

| Issue | Title | Status |
|-------|-------|--------|
| SLE-57 | pattern_engine contracts: NeighborResult, CalibratedProbability, EngineState + Pandera schemas | ✓ Done |
| SLE-58 | trading_system contracts: EvaluatorStatus, PositionDecision, SharedState, TradeEvent | ✓ Done |
| SLE-59 | Replace hand-rolled schema.py with Pandera (SchemaError backward compat) | ✓ Done |

### Milestone M3: Matcher / HNSW Foundation (SLE-60 → SLE-64) — COMPLETE
**Gate:** PatternMatcher 5-stage parity verified. Promotion gate defined. 794 tests passing. ✓

| Issue | Title | Status |
|-------|-------|--------|
| SLE-60 | 5-stage PatternMatcher + features.py (RETURNS_ONLY_COLS, apply_feature_weights) | ✓ Done |
| SLE-61 | HNSW backend wired in Stage 2 via cfg.use_hnsw; lazy import; num_threads=1 | ✓ Done |
| SLE-62 | Parity tests: Level 1 (exact BallTree) + Level 2 (HNSW approx recall@50 ≥ 0.995) | ✓ Done |
| SLE-63 | Performance benchmarks: p95 < 0.1ms gate, 20× speedup gate, artifact JSON | ✓ Done |
| SLE-64 | HNSW promotion gate: 5-gate doc + machine-checkable script | ✓ Done |

### Milestone M4: Data Pipeline Hardening (SLE-65 → SLE-67)
**Gate:** data.py has ≥90% coverage, all paths schema-validated.

| Issue | Title | Priority | Depends On |
|-------|-------|----------|------------|
| SLE-65 | Add comprehensive tests for `data.py` (feature pipeline) | High | M2 |
| SLE-66 | Add tests for `live.py` (production signal runner) | High | M2 |
| SLE-67 | Replace assert guards with RuntimeError in public APIs | Medium | — |

### Milestone M5: Trading System Rebuild (SLE-68 → SLE-71)
**Gate:** Backtest parity within tolerance (Sharpe ±0.01, return ±0.1%).

| Issue | Title | Priority | Depends On |
|-------|-------|----------|------------|
| SLE-68 | Implement SharedState (Pydantic model with typed sub-states) | High | SLE-57 |
| SLE-69 | Build Layer 4: strategy_evaluator (rolling metrics, status) | High | SLE-68 |
| SLE-70 | Add feature flags for all research modules (EMD, BMA, SlipDeficit) | High | SLE-57 |
| SLE-71 | Break `signal_adapter.py` dependency on legacy `strategy.py` | Medium | SLE-70 |

### Milestone M6: Research Pilot Integrations (SLE-72 → SLE-79)
**Gate:** Each pilot passes A/B parity test (flag on vs off).

| Issue | Title | Priority | Tier | Depends On |
|-------|-------|----------|------|------------|
| SLE-72 | SAX discretization for pattern compression | Medium | T0 | M3 |
| SLE-73 | Walk-forward alignment for strategy_evaluator | Medium | T0 | SLE-69 |
| SLE-74 | Liquidity gate (volume/spread filter) | Medium | T1 | M4 |
| SLE-75 | Signal fatigue overlay (decay after N consecutive signals) | Low | T1 | SLE-70 |
| SLE-76 | Interactive Brokers paper trading bridge | Low | T2 | SLE-69 |
| SLE-77 | Calibration drift monitoring | Low | T2 | SLE-69 |
| SLE-78 | Conformal prediction UQ bands | Low | T3 | M3 |
| SLE-79 | Streaming / real-time scaffolding | Low | T3 | M5 |

### Milestone M7: Parity & CI (SLE-80 → SLE-83)
**Gate:** CI green, parity tests pass, no manual steps required.

| Issue | Title | Priority | Depends On |
|-------|-------|----------|------------|
| SLE-80 | Build parity test harness (frozen baseline ↔ rebuild comparison) | High | M5 |
| SLE-81 | Create GitHub Actions CI pipeline | Medium | SLE-80 |
| SLE-82 | Add regression test suite (numeric drift detection) | Medium | SLE-80 |
| SLE-83 | Generate `requirements.lock` for reproducible installs | Medium | — |

### Milestone M8: Migration & Documentation (SLE-84 → SLE-86)
**Gate:** All legacy code retired, CLAUDE.md updated, clean import graph.

| Issue | Title | Priority | Depends On |
|-------|-------|----------|------------|
| SLE-84 | Retire legacy scripts (strategy.py, 11 root files) | Medium | M5 |
| SLE-85 | Update CLAUDE.md, PROJECT_GUIDE.md, design docs | Medium | M7 |
| SLE-86 | Final migration: move rebuild_phase_3z/ → production paths | Medium | M7, M8 |

---

## 3. Dependency Order

```
M1 (Baseline Freeze)
 ├─→ M2 (Contracts/Schemas)
 │    ├─→ M3 (Matcher/HNSW) ─→ M6.T0 (SAX) ─→ M6.T3 (Conformal)
 │    ├─→ M4 (Data Pipeline) ─→ M6.T1 (Liquidity, Fatigue)
 │    └─→ M5 (Trading System) ─→ M6.T2 (IB, Drift) ─→ M6.T3 (Streaming)
 │         └─→ M7 (Parity/CI) ─→ M8 (Migration)
 └─→ SLE-67 (assert→RuntimeError, independent)
```

**Critical path:** M1 → M2 → M3 → M5 → M7 → M8
**HNSW urgency:** M3 is on the critical path. SLE-60 (Matcher ABC) unblocks all retrieval work.

---

## 4. HNSW Integration Plan

### Current Architecture (Wrong)
```
BaseDistanceMetric (ABC)
 ├── EuclideanDistance     ← production default
 └── HNSWIndex            ← WRONG: HNSW is not a distance metric, it's a retrieval backend
```

`HNSWIndex` inherits `BaseDistanceMetric` but overrides `fit()` and `kneighbors()` — methods that don't belong on a distance metric class. The sqrt() correction for hnswlib's L2 convention is correct but architecturally misplaced.

### Target Architecture (Correct)
```
BaseMatcher (ABC)                     BaseDistanceMetric (ABC)
 ├── BallTreeMatcher                   └── EuclideanDistance
 │    uses → EuclideanDistance
 └── HNSWMatcher
      uses → hnswlib (internal L2)
      applies → sqrt() correction
      guarantees → recall@50 ≥ 0.9996
```

### Implementation Steps

1. **SLE-60:** Define `BaseMatcher` ABC in `rebuild_phase_3z/fppe/pattern_engine/contracts/matcher.py`:
   ```python
   class BaseMatcher(ABC):
       @abstractmethod
       def fit(self, X: np.ndarray) -> None: ...

       @abstractmethod
       def kneighbors(self, X: np.ndarray, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]: ...

       @abstractmethod
       def get_params(self) -> dict: ...
   ```

2. **SLE-61:** `BallTreeMatcher` wraps `sklearn.neighbors.BallTree` with `EuclideanDistance`. This is the current production path, just properly encapsulated.

3. **SLE-62:** `HNSWMatcher` wraps `hnswlib.Index`. Moves the existing `HNSWIndex` code from `research/hnsw_distance.py` into the new abstraction. Preserves the sqrt() correction. Adds `ef_construction`, `M`, `ef_search` as config params.

4. **SLE-63:** Parity test: for 1000 random queries, verify BallTree and HNSW return the same top-50 neighbors with recall ≥ 0.9996 and distance RMSE < 1e-6.

5. **SLE-64:** Add `matcher_backend: str = "balltree"` to `EngineConfig`. Wire into `matching.py` as a factory dispatch.

---

## 5. Research Additions Placement

### Tier 0 — Immediate (during M3/M5)

| Module | Where It Goes | Why Now |
|--------|---------------|---------|
| **SAX discretization** | `fppe/pattern_engine/sax.py` | Pattern compression for HNSW pre-filtering; directly enables faster retrieval |
| **WFA for evaluator** | `fppe/trading_system/strategy_evaluator.py` | Layer 4 needs walk-forward alignment to compute meaningful rolling metrics |

### Tier 1 — After M4

| Module | Where It Goes | Why Then |
|--------|---------------|----------|
| **Liquidity gate** | `fppe/pattern_engine/filters/liquidity.py` | Requires validated data pipeline (M4) to compute volume/spread metrics |
| **Signal fatigue overlay** | `fppe/research/fatigue_overlay.py` | Requires feature flag infrastructure (SLE-70) |

### Tier 2 — After M5

| Module | Where It Goes | Why Then |
|--------|---------------|----------|
| **IB paper trading** | `fppe/trading_system/bridges/ib_bridge.py` | Requires complete trading system with SharedState |
| **Calibration drift** | `fppe/trading_system/drift_monitor.py` | Requires strategy_evaluator (SLE-69) for metric comparison |

### Tier 3 — Deferred (After M7)

| Module | Where It Goes | Why Deferred |
|--------|---------------|--------------|
| **Conformal UQ** | `fppe/research/conformal_uq.py` | Speculative; needs HNSW + calibration stability first |
| **Streaming scaffolding** | `fppe/trading_system/streaming/` | Architectural investment with no immediate payoff; wait for IB bridge learnings |

---

## 6. Explicit Deferrals

These items are **out of scope** for Phase 3Z:

| Item | Reason |
|------|--------|
| Multi-horizon prediction (14d, 30d) | Requires new walk-forward validation folds; Phase 4+ |
| Short selling / leverage | v1 is long-only by design; max_gross_exposure=1.0 is a constraint |
| Candlestick categorization (Phase 6) | Design doc exists but depends on Phase 5 data infrastructure |
| LSTM feature set (`returns_hybrid`) | References nonexistent columns; shelved until data pipeline can generate them |
| Ticker universe expansion (52 → 500+) | Depends on HNSW performance at scale; Tier 2 at earliest |
| Web dashboard / React integration | `pattern-engine-v2.1.jsx` is a standalone demo; not part of rebuild scope |
| Multi-asset (crypto, forex) | Entirely different data pipeline; not in scope |

---

## 7. Immediate Blockers

| Blocker | Impact | Resolution |
|---------|--------|------------|
| No Pydantic/Pandera in `requirements.txt` | Cannot define contracts | `pip install pydantic pandera` + add to requirements |
| `hnswlib` may not be installed | HNSW matcher implementation blocked | `pip install hnswlib` (already in research venv) |
| `strategy.py` import in `signal_adapter.py` | Cannot retire legacy code until replaced | SLE-71 provides replacement path |
| No git tag for baseline | Cannot reference frozen state | SLE-56: `git tag phase3z-baseline-v1` |

---

## 8. Current Status & Next Slice

**M1, M2, M3 are complete (794 tests passing). M4 is next.**

### M4: Data Pipeline Hardening (SLE-65 → SLE-67) — NEXT
- [ ] SLE-65: Add comprehensive tests for `data.py` (feature pipeline, ≥80% coverage)
- [ ] SLE-66: Add tests for `live.py` (production signal runner)
- [ ] SLE-67: Replace `assert` guards with `RuntimeError` in all public APIs

### Completed Slices
- [x] M1: Baseline freeze, 596 legacy tests, git tag `phase3z-baseline-v1`
- [x] M2: All cross-package contracts (Pydantic + Pandera), 771 tests
- [x] M3: 5-stage PatternMatcher, HNSW backend, parity + benchmark gates, 794 tests

---

## 9. Parity Testing Strategy

Every milestone gate requires a parity check:

```
BASELINE (frozen manifest)
    ↓ compare
REBUILD (new code, same data)
    ↓ verify
TOLERANCES:
  - Float values: rtol=1e-7
  - Integer counts: exact
  - Categorical labels: exact
  - Signal counts: exact
  - Sharpe ratio: rtol=1e-4
  - Annual return: rtol=1e-4
```

**Red line:** If any parity test fails beyond tolerance, the rebuild step is REJECTED. Debug before proceeding. No exceptions.

---

## 10. Non-Negotiable Rules

1. **Parity-first.** Every change must prove it doesn't break validated results.
2. **Contracts at boundaries.** No implicit DataFrame passing between packages.
3. **Single source of truth.** SECTOR_MAP in `sector.py`. Configs in frozen dataclasses.
4. **Fail-fast.** RuntimeError, not assert. Pandera strict=True, coerce=False.
5. **Deterministic execution.** nn_jobs=1. No randomness without seeded RNG.
6. **Isolated workspace.** All rebuild work in `rebuild_phase_3z/` until M8 migration.
7. **Feature flags for research.** Every research module behind an explicit bool flag.
8. **3-strike rule.** Three failed attempts → stop, log, escalate.

---

*Generated: 2026-03-21 | Linear: SLE-51 → SLE-86 | Baseline: 596 tests, Sharpe 1.16, BSS +0.00103*
