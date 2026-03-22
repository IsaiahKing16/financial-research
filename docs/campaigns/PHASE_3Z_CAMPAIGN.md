# Phase 3Z Full Rebuild Campaign

**Status:** COMPLETE — M1–M8 all done (2026-03-22)
**Started:** 2026-03-21
**Git tag:** `phase3z-baseline-v1`
**Linear Project:** FPPE Phase 3Z Rebuild (SLE-51 → SLE-89)
**Final tests:** 596 legacy + 543 new = 1139 total

---

## Milestone Status

| Milestone | Issues | Status | Gate |
|-----------|--------|--------|------|
| M1: Baseline Freeze | SLE-52–56 | **COMPLETE** | 644 tests, manifest verified, git tag |
| M2: Contracts/Schemas | SLE-57–59 | **COMPLETE** | Contracts at all boundaries, 771 tests |
| M3: Matcher/HNSW | SLE-60–64 | **COMPLETE** | 794 tests, parity + benchmark gates defined |
| M4: Data Pipeline | SLE-65–67 | **COMPLETE** | 841 tests, hardened DataLoader, leakage guard, atomic writes |
| M5: Trading System | SLE-68–71 | **COMPLETE** | 972 tests, SharedState isolation, StrategyEvaluator, signal_adapter rebuilt, SlipDeficit flag |
| M6: Research Pilots | SLE-72–79 | **COMPLETE** | 1137 tests, all pilots behind flags, A/B parity fixtures |
| M7: Parity/CI | SLE-80–83 | **COMPLETE** | 999 tests, snapshot regression, CI pipeline, HNSW gate |
| M8: Migration/Docs | SLE-84–86, SLE-89 | **COMPLETE** | Legacy archived, docs updated, Platt calibration wired (SLE-89) |

---

## M1 Deliverables (Complete)

- `rebuild_phase_3z/artifacts/manifests/baseline_manifest_20260321.json` — 53-file SHA-256 manifest
- `rebuild_phase_3z/artifacts/baselines/rebuild_start_report.md` — Phase 1/2/3 baselines
- `requirements.lock.txt` — 89 packages pinned
- `requirements.txt` — updated with pydantic, pandera, hnswlib
- `scripts/hash_baseline_inputs.py` — automated parity verification
- `rebuild_phase_3z/` workspace — full directory structure + __init__.py files
- git tag `phase3z-baseline-v1` at HEAD

## M2 Deliverables (Complete)

**SLE-57 — pattern_engine contracts:**
- `fppe/pattern_engine/contracts/datasets.py` — Pandera: TrainDBSchema, QueryDBSchema, FeatureMatrixSchema, OHLCVSchema, SignalsOutputSchema
- `fppe/pattern_engine/contracts/signals.py` — SignalRecord (existing) + NeighborResult + CalibratedProbability
- `fppe/pattern_engine/contracts/state.py` — EngineState (serializable checkpoint: scaler params, config hash, fit timestamp, JSON round-trip)
- `fppe/pattern_engine/contracts/matcher.py` — BaseMatcher ABC (from M1 head start)
- `fppe/pattern_engine/contracts/matchers/balltree_matcher.py` — BallTreeMatcher
- `fppe/pattern_engine/contracts/matchers/hnsw_matcher.py` — HNSWMatcher

**SLE-58 — trading_system contracts:**
- `fppe/trading_system/contracts/decisions.py` — EvaluatorStatus (GREEN/YELLOW/RED) + EvaluatorSnapshot + PositionDecision + AllocationDecision + RejectionReason
- `fppe/trading_system/contracts/state.py` — SharedState inter-layer bus with typed sub-states (EquityState, PositionsState, RiskState, PortfolioState, EvaluatorState) + SystemCommand enum
- `fppe/trading_system/contracts/trades.py` — TradeEvent with future-ready fields (fill_quantity, fill_price, fill_ratio, execution_timestamp, execution_latency_seconds) + backtest_fill factory

**SLE-59 — Pandera schema.py:**
- `fppe/pattern_engine/schema.py` — Pandera-backed validate_train_db / validate_query_db preserving SchemaError(ValueError) backward compatibility

**Tests (all passing):**
- `tests/unit/test_contracts.py` — 27 tests (SignalRecord, TradeRecord, PositionRecord, DailySnapshot)
- `tests/unit/test_matchers.py` — 17 tests (BallTree + HNSW)
- `tests/parity/test_matcher_parity.py` — 4 parity tests (recall@50 ≥ 0.999 confirmed)
- `tests/unit/test_datasets.py` — 27 tests (Pandera schema validation)
- `tests/unit/test_signals_extended.py` — 15 tests (NeighborResult, CalibratedProbability)
- `tests/unit/test_engine_state.py` — 14 tests (EngineState: construction, serialization, factory)
- `tests/unit/test_trading_contracts.py` — 33 tests (decisions, state, TradeEvent)
- `tests/unit/test_pandera_schema.py` — 21 tests (SLE-59 semantic parity with legacy schema.py)
- **Total: 771 tests (596 legacy + 175 new) — all passing**

## M3 Deliverables (Complete)

**SLE-60/61 — PatternMatcher 5-stage + HNSW backend wiring:**
- `fppe/pattern_engine/features.py` — `RETURNS_ONLY_COLS`, `get_feature_cols()`, `apply_feature_weights()`
- `fppe/pattern_engine/matcher.py` — `PatternMatcher` with 5 explicit stages (fit/query), config-driven BallTree/HNSW selection
- `conftest.py` — pytest config, registers `slow` mark

**SLE-62 — Parity tests (Level 1 + Level 2):**
- `tests/parity/test_matcher_parity_staged.py` — 17 tests: Level 1 (exact BallTree parity), Level 2 (HNSW approx parity: recall@50 ≥ 0.995, signal agreement ≥ 99%, BSS delta ≤ 0.001)

**SLE-63 — Performance benchmarks:**
- `tests/performance/test_hnsw_benchmark.py` — build time, per-row latency (p95 gate), batch speedup (20× gate), artifact JSON

**SLE-64 — HNSW promotion gate:**
- `docs/rebuild/HNSW_PROMOTION_GATE.md` — 5-gate promotion criteria
- `scripts/check_hnsw_promotion_gate.py` — machine-checkable gate script (exit 0=approved, exit 1=blocked)

**Tests (all passing):**
- `tests/parity/test_matcher_parity_staged.py` — 17 tests (Level 1 + Level 2 parity)
- `tests/performance/test_hnsw_benchmark.py` — 6 tests (build, latency, speedup)
- **Total: 794 tests (596 legacy + 198 new M2/M3) — all passing**

## M4 Deliverables (Complete)

**SLE-65 — Hardened DataLoader:**
- `fppe/pattern_engine/data.py` — `DataLoaderHardened` with 3 Pandera validation checkpoints, lineage metadata via `DataFrame.attrs`, atomic parquet writes, temporal leakage guard
- `DataPipelineError(ValueError)` — schema failures wrapped with checkpoint context; backward-compatible with `except ValueError`

**SLE-66 — test_data.py (47 tests):**
- `tests/unit/test_data.py` — `TestAtomicWriteParquet`, `TestLineage`, `TestOHLCVValidation`, `TestFeatureDBValidation`, `TestDataLoaderHardenedInit`, `TestDownload`, `TestComputeFeatures`, `TestTemporalSplit`, `TestBuildDatabase`, `TestDataPipelineError`
- All acceptance criteria met: mocked yfinance, 2 leakage guard cases, feature spot-checks, ≥80% coverage

**SLE-67 — Assert scan:**
- Zero `assert` statements found in `pattern_engine/`, `trading_system/`, `research/`, `rebuild_phase_3z/fppe/`
- No code changes required — already clean from Phase 1–3 enforcement

**Tests (all passing):**
- `tests/unit/test_data.py` — 47 tests
- **Total: 841 tests (596 legacy + 245 new M2/M3/M4) — all passing**

## M5 Deliverables (Complete)

**SLE-68 — SharedStateManager:**
- `rebuild_phase_3z/fppe/trading_system/shared_state.py` — Layer isolation enforcement, JSON checkpoint serialization, command queue helpers
- `rebuild_phase_3z/fppe/trading_system/config.py` — Rebuild TradingConfig (isolated from production `pattern_engine.sector` import)
- `LayerTag` enum (PATTERN_ENGINE / RISK_ENGINE / PORTFOLIO_MANAGER / STRATEGY_EVALUATOR)
- Write-permit table: `_LAYER_WRITE_PERMITS[LayerTag] → frozenset[str]`
- `manager.update()` — validates permits before applying model_copy(update={...})
- `manager.enqueue_command()` / `manager.drain_commands()` — evaluator→PM command bus
- `manager.to_json()` / `manager.from_json()` — Pydantic v2 JSON with schema version guard

**SLE-69 — signal_adapter.py rebuild:**
- `rebuild_phase_3z/fppe/trading_system/signal_adapter.py` — Strategy.py dependency eliminated
- `UnifiedSignal` upgraded from `@dataclass` to Pydantic `BaseModel` (frozen, validated, JSON-serializable)
- `simulate_signals_from_val_db()` — now uses `PatternMatcher` directly; no legacy `strategy.py` import
- `adapt_knn_signals()` / `adapt_dl_signals()` — retained, updated for Pydantic UnifiedSignal
- `load_cached_signals()` / `save_signals()` — CSV caching helpers retained

**SLE-70 — StrategyEvaluator (Layer 4):**
- `rebuild_phase_3z/fppe/trading_system/strategy_evaluator.py` — Rolling metrics + RED/YELLOW/GREEN
- `ClosedTrade` / `EvaluatorConfig` dataclasses
- Pure math helpers: `_annualized_sharpe()` (Bessel-corrected, √252 annualized), `_drawdown_from_peak()`, `_linear_slope()`
- Rolling windows: 30d / 90d / 252d / all-time Sharpe
- Status priority: RED (drawdown>15% OR 90d Sharpe<0) → YELLOW (trades<30 OR 30d Sharpe<0.5 OR BSS drift) → GREEN
- Command emission on status transitions only (not repeated on stable status)
- `evaluate_and_update_state()` — writes EvaluatorState + enqueues commands to SharedState
- BSS calibration drift detection: linear slope of BSS series, threshold=-0.001/period

**SLE-71 — ResearchFlagsConfig (SlipDeficit flag):**
- `rebuild_phase_3z/fppe/trading_system/config.py` — `ResearchFlagsConfig(use_slip_deficit=False)`
- Flag defaults False (safe baseline identical to pre-SLE-71 behavior)
- Enable via `dataclasses.replace(cfg.research_flags, use_slip_deficit=True)` — immutable pattern

**Tests (all passing):**
- `tests/unit/test_shared_state.py` — 33 tests (layer permits, command queue, JSON round-trip)
- `tests/unit/test_trading_config.py` — 24 tests (ResearchFlags, defaults parity, validation, immutability)
- `tests/unit/test_strategy_evaluator.py` — 43 tests (math helpers, status logic, command emission, BSS drift)
- `tests/unit/test_signal_adapter.py` — 31 tests (UnifiedSignal, KNN/DL adapters, CSV helpers)
- **Total: 972 tests (596 legacy + 376 new M2–M5) — all passing**

## M7 Deliverables (Complete)

**SLE-80 — End-to-end parity test harness (3 levels):**
- `scripts/generate_parity_snapshot.py` — runs PatternMatcher on seeded synthetic data, writes frozen JSON artifact
- `rebuild_phase_3z/artifacts/baselines/parity_snapshot.json` — committed snapshot (schema SLE-80-v1); BSS=-0.003815, 93 BUY / 104 SELL / 203 HOLD on N_QUERY=400
- `rebuild_phase_3z/tests/parity/test_end_to_end_parity.py` — 3-level parity: determinism (bit-identical), snapshot regression (vs artifact), production oracle (@slow, real data)
- `SyntheticConfig` with relaxed parameters: `max_distance=4.5`, `confidence_threshold=0.55`, `min_matches=5` (production max_distance=1.1019 filters all analogues on i.i.d. normal synthetic data)

**SLE-81 — CI pipeline:**
- `.github/workflows/ci.yml` — 5-job pipeline: lint-requirements → legacy-tests + rebuild-tests (parallel) → parity-gate → hnsw-gate (advisory, `continue-on-error: true`)
- Windows runner mandatory (nn_jobs=1, joblib deadlock prevention)
- Job dependency: parity-gate runs only after both test suites pass; hnsw-gate advisory (unblocking)
- All slow tests excluded from CI via `-m "not slow"`

**SLE-82 — Regression/numeric-drift test suite:**
- `rebuild_phase_3z/tests/regression/test_numeric_drift.py` — 4 classes:
  - `TestSharpeFormulaCorrectness` — 8 tests: known values, annualization factor, drawdown, linear slope, `±inf` edge cases
  - `TestMatcherOutputInvariants` — 6 tests: probs in [0,1], valid labels, n_matches ≥ 0 and ≤ top_k, shape, actionable count
  - `TestSnapshotDrift` — 5 tests: exact signal counts + float metrics vs committed artifact
  - `TestProductionMetricsDrift` — 2 tests: `@slow`, BSS=+0.00103 and trade count=191 on real data

**SLE-83 — requirements.lock.txt:**
- Verified: 89 packages (above CI gate of ≥80), committed at M1

**Tests (all passing):**
- `tests/parity/test_end_to_end_parity.py` — 11 tests (5 determinism + 6 snapshot regression; 4 slow excluded)
- `tests/regression/test_numeric_drift.py` — 19 tests (10 Sharpe/invariants + 5 snapshot drift + 4 slow excluded)
- **Total: 999 tests (596 legacy + 403 new M2–M7) — all passing**

## M6 Deliverables (Complete)

**SLE-72 — SAX symbolic filter:**
- `rebuild_phase_3z/fppe/pattern_engine/sax_filter.py` — PAA segmentation, symbol digitisation, MINDIST lookup table, `SAXFilter` class, `apply_sax_filter()` helper
- Flag: `getattr(config, 'use_sax_filter', False)` — applied as second-stage filter on HNSW top-K candidates after Stage 3
- `tests/unit/test_sax_filter.py` — 22 tests (PAA, digitise, dist table, SAXFilter class, batch vs per-row equivalence)

**SLE-73 — WFA DTW reranker:**
- `rebuild_phase_3z/fppe/pattern_engine/wfa_reranker.py` — Sakoe-Chiba constrained DTW, `WFAReranker` class, `rerank_mask()` for batch reordering
- Flag: `getattr(config, 'use_wfa_rerank', False)` — reranks post-filter survivors by DTW distance before Stage 5
- `tests/unit/test_wfa_reranker.py` — 18 tests (DTW known values, symmetry, window=0 is Euclidean, mask shape/count preservation)

**SLE-74 — Liquidity congestion gate:**
- `rebuild_phase_3z/fppe/trading_system/risk_overlays/base.py` — `BaseRiskOverlay` ABC (update/get_signal_multiplier/reset)
- `rebuild_phase_3z/fppe/trading_system/risk_overlays/liquidity_congestion.py` — `LiquidityCongestionGate` (ATR/price ratio, cooldown, linear throttle)
- Flag: `ResearchFlagsConfig.use_liquidity_congestion_gate = False`

**SLE-75 — Fatigue accumulation overlay:**
- `rebuild_phase_3z/fppe/trading_system/risk_overlays/fatigue_accumulation.py` — `FatigueAccumulationOverlay` (exponential-approach accumulation, regime-transition reset)
- Flag: `ResearchFlagsConfig.use_fatigue_accumulation = False`
- `tests/unit/test_risk_overlays.py` — 33 tests (ABC enforcement, gate + overlay correctness, M6 flags in ResearchFlagsConfig)

**SLE-76 — SPC/CUSUM drift monitor:**
- `rebuild_phase_3z/fppe/trading_system/drift_monitor.py` — `CUSUMState`, `EWMAState`, `CalibrationBucket`, `DriftMonitor`
- Feature mean/variance CUSUM + BSS EWMA + calibration bucket empirical hit rates
- Flag: `ResearchFlagsConfig.use_drift_monitor = False`
- `tests/unit/test_drift_monitor.py` — 28 tests (CUSUM mechanics, EWMA formula, calibration drift, DriftMonitor integration)

**SLE-77 — Conformal uncertainty stubs:**
- `rebuild_phase_3z/fppe/pattern_engine/conformal_hooks.py` — `BaseConformalCalibrator` ABC, `NaiveConformalCalibrator`, `augment_signals_with_conformal()`
- `docs/rebuild/CONFORMAL_UQ_PLAN.md` — integration plan, feasibility assessment (exchangeability concern, weighted conformal recommendation)
- `tests/unit/test_conformal_hooks.py` — 11 tests

**SLE-78 — Information Bottleneck pilot:**
- `rebuild_phase_3z/fppe/pattern_engine/ib_compression.py` — `IBCompressor` (supervised PCA / Gaussian IB lower bound), `compare_bss_with_ib()`
- Flag: `ResearchFlagsConfig.use_ib_compression = False`
- `tests/unit/test_ib_compression.py` — 14 tests (fit/transform shape, top_features ranking, BSS comparison dict)

**SLE-79 — Streaming ingestion design:**
- `docs/rebuild/STREAMING_INGESTION_DESIGN.md` — trigger types, `IngestionEvent`/`IngestionHandler`/`FeatureStore` interface definitions, incremental ATR computation, compatibility assessment

**Tests (all passing):**
- `tests/unit/test_sax_filter.py` — 22 tests
- `tests/unit/test_wfa_reranker.py` — 18 tests
- `tests/unit/test_risk_overlays.py` — 35 tests (includes M6 flags + new C3 + I5 tests)
- `tests/unit/test_drift_monitor.py` — 29 tests (includes new I3 single-row test)
- `tests/unit/test_conformal_hooks.py` — 11 tests
- `tests/unit/test_ib_compression.py` — 14 tests
- **Total: 1139 tests (596 legacy + 543 new M2–M7) — all passing**

**Post-review fixes applied (2026-03-22 continuation):**
- C1: All M6 modules wired into PatternMatcher (SAX→_post_filter, WFA→query batch, IB→fit/query, overlays→_package_results) and DriftMonitor→StrategyEvaluator via `set_drift_monitor()`
- C2: IBCompressor double-weighting fixed — `Vt[:d_out].T` only (weights already applied via `X_w = X * weights`)
- C3: `regime_duration` now counts first period as 1 (not 0); NEUTRAL→NEUTRAL with `reset_on_neutral=False` accumulates correctly
- C4: CI test-count gate now enforces minimum 556 with `exit 1` on PowerShell
- I1: `use_sax_filter` and `use_wfa_rerank` added to `ResearchFlagsConfig`
- I2: Cooldown partial-throttle proportional to remaining cooldown (was hardcoded 0.5)
- I3: DriftMonitor single-row batch clarified (ddof=0 explicitly); `numpy.bool_` → `bool()` cast in `update_features()`
- I4: NaiveConformalCalibrator.predict_set() stub disclaimer added
- I5: `reset_on_neutral=False` neutral-accumulation bug fixed; test added

## Key Decisions Made

1. **HNSW as Matcher backend** (not BaseDistanceMetric subclass) — DONE
2. **Parity tolerance policy** — float rtol=1e-7, integers exact, BSS rtol=1e-5
3. **Isolated workspace** — all Phase 3Z work in `rebuild_phase_3z/` until M8
4. **SECTOR_MAP** — W2 was already resolved in Phase C. Not a blocker.
5. **Pandera dtype=None for Date/string columns** — pandas 3 changed str dtype to string[pyarrow], which breaks Column(object, ...) and Column(str, ...) checks. Use None to skip type enforcement; content checks (NaN, bounds) remain.
6. **strict=False for train/query DBs** — these DataFrames are wide (OHLCV + all forward-return variants). strict=True on outputs (FeatureMatrix, SignalsOutput) only.
7. **SharedState as functional bus** — frozen Pydantic model; "updates" via model_copy(update={...}) create new instances. No in-place mutation.
8. **TradeEvent.backtest_fill factory** — convenience constructor for backtest mode (fill_ratio=1.0, latency=0). Live trading uses full constructor with broker fill details.
9. **np.datetime64 → datetime.date coercion in Stage 5** — `val_db["Date"].values` returns `np.datetime64`, which Pydantic v2 rejects for `datetime.date` fields. Fixed by normalising via `pd.Timestamp(raw_date).date()` before constructing `NeighborResult`. Always extract `.date()` when passing DataFrame date columns to Pydantic models.
10. **Small-N speedup threshold = 0.25×** — At N=2k, D=8, BallTree and HNSW are within noise of each other. The `test_small_scale_speedup` smoke test uses 0.25× (catastrophic failure check: HNSW must not be >4× slower). The 20× production gate only applies at N=50k where HNSW's graph-traversal amortises.
11. **@pytest.mark.skipif on fixtures removed (pytest 9)** — pytest 9 deprecated marks on fixtures. Replaced with `if not HAS_HNSWLIB: pytest.skip(...)` inside the fixture body.
12. **TA library crashes on < ~50 rows** — `ta` indicators (RSI, ATR, SMA) raise IndexError on DataFrames with fewer rows than their window. Wrap `_compute_ticker_features()` calls in `try/except` and skip the ticker. Test fixtures must use n≥300 rows (300 − 97 window = 203 surviving rows, above 50-row threshold).
13. **`DataPipelineError(ValueError)` for Pandera errors** — schema failures are data errors (bad input), not programming errors. Inheriting from ValueError (not RuntimeError) means existing `except ValueError` callers catch pipeline errors correctly without code changes.
14. **Lineage via `DataFrame.attrs`** — `DataFrame.attrs` is shallow-copied by most pandas ops (concat, copy). Always re-attach lineage at each checkpoint rather than relying on propagation. `_attach_lineage()` is called at every validation boundary.
15. **LayerTag write-permit check order** — Unknown field names must be validated BEFORE the permit check. A nonexistent field gives a clearer error ("Unknown SharedState field") rather than the confusing "does not have write permission" which implies the field exists but is forbidden.
16. **`evaluate()` does not emit commands; `evaluate_and_update_state()` does** — `evaluate()` sets `_previous_status` at the end. To detect transitions in `evaluate_and_update_state()`, the previous status must be saved BEFORE calling `evaluate()`. Otherwise `_determine_commands()` always sees no transition (old==new).
17. **UnifiedSignal upgraded to Pydantic BaseModel** — frozen, validated (confidence [0,1], ticker uppercase), JSON-serializable. The production signal_adapter uses a plain `@dataclass` — the rebuild uses Pydantic for contract enforcement at the trading layer boundary.
18. **StrategyEvaluator is stateful; SharedStateManager is stateless** — evaluator accumulates trade and return history as mutable lists; manager is a pure-function utility that accepts state and returns new state. This separation makes both independently testable.

---

## Blockers

None currently.

---

## Session Handoff Log

| Date | Session | Status | Next |
|------|---------|--------|------|
| 2026-03-21 | Phase 3Z kickoff | M1 complete, M2/M3 skeleton done | SLE-57 full Pandera schemas |
| 2026-03-21 | SLE-51 to SLE-56 | All M1 issues Done, git tag created, 644 tests | SLE-57 (Pandera) |
| 2026-03-21 | SLE-57 to SLE-59 | M2 complete, 771 tests (596+175) all passing | SLE-60 (BaseMatcher full impl) |
| 2026-03-21 | SLE-60 to SLE-64 | M3 complete, 794 tests (596+198) all passing | SLE-65 (M4 Data Pipeline) |
| 2026-03-22 | SLE-65 to SLE-67 | M4 complete, 841 tests (596+245) all passing | SLE-68 (M5 Trading System) |
| 2026-03-22 | SLE-68 to SLE-71 | M5 complete, 972 tests (596+376) all passing | SLE-72 (M6 SAX discretization) |
| 2026-03-22 | SLE-80 to SLE-83 | M7 complete (skipped M6), 999 tests all passing; 3 bugs fixed (Sharpe ±inf, snapshot path parents[3]→[2], unpack count) | SLE-72 (M6 Research Pilots) |
| 2026-03-22 | SLE-72 to SLE-79 | M6 complete, 1137 tests all passing; 8 pilots behind flags; 1 fix (pytest.approx >= TypeError) | SLE-84 (M8 Migration/Docs) |
| 2026-03-22 | M6/M7 post-review fixes | 1139 tests all passing; 12 review items resolved (C1a-f, C2, C3, C4, I1-I5); M2–M5 code review dispatched | M2–M5 review → SLE-84 (M8) |
| 2026-03-22 | M2–M5 review + M8 complete | 1139 tests all passing; all 18 review findings resolved (C1-C4, I1-I8, m2, m4); Platt calibration wired (SLE-89); CLAUDE.md + campaign updated | Phase 3Z COMPLETE |
