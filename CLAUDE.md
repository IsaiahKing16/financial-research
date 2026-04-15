# CLAUDE.md — FPPE Project Root
# This file is loaded every session. Keep it under 80 lines.
# Detailed protocols live in .claude/skills/ — Claude loads them on demand.

## Project

**FPPE (Financial Pattern Prediction Engine)** — K-nearest-neighbor historical analogue
matching on return fingerprints. Generates probabilistic BUY/SELL/HOLD signals.

## Commands
- `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"` — Run full suite (always run before committing)
- `venv\Scripts\activate` — Windows venv activation

## Codebases

- `pattern_engine/` — Phase 3Z production core: PatternMatcher, Platt calibration, contracts
  - `matcher.py`: 5-stage PatternMatcher (scale → search → filter → aggregate → calibrate)
  - `contracts/`: Pydantic schemas, BaseMatcher ABC, EngineState, SignalDirection/Source
  - `features.py`, `data.py`, `schema.py` — feature pipeline and data loading
  - `regime.py`: RegimeLabeler (SPY + VIX spike + yield curve); regime_filter=False in prod
  - `walkforward.py`: run_fold, run_walkforward, load_and_augment_db, BSS, Murphy decomposition, beta_abm calibration
  - `sweep.py`: OptunaSweep (TPE), GridSweep, SweepResult, KNN_SEARCH_SPACE
  - `experiment_log.py`: ExperimentLogger incremental TSV writer
  - `live.py`: LiveRunner (Phase 5): receives AllocationDecisions + exit_tickers, submits via OrderManager + BaseBroker DI
  - `contracts/matchers/hnsw_matcher.py`: save_index/load_index for disk persistence
  - Research pilots (behind flags): `sax_filter.py`, `wfa_reranker.py`, `ib_compression.py`, `conformal_hooks.py`
- `trading_system/` — Phase 3Z production layer: SharedState, StrategyEvaluator, risk overlays
  - `strategy_evaluator.py`: signal → position decision with risk overlays
  - `signal_adapter.py`: UnifiedSignal (Pydantic), KNN/DL adapters
  - `risk_overlays/`: fatigue accumulation, liquidity congestion
  - `position_sizer.py`: Half-Kelly with SizingConfig (Phase 2)
  - `risk_engine.py`: stateless orchestrator — compute_atr_pct, drawdown_brake_scalar, apply_risk_adjustments (Phase 3)
  - `portfolio_manager.py`: rank_signals, allocate_day — stateless PM (Phase 4, when created)
  - `portfolio_state.py`: PortfolioSnapshot, RankedSignal, PMRejection (Phase 4, when created)
  - `broker/`: BaseBroker ABC, Order/OrderResult schemas, MockBroker (Phase 5)
  - `order_manager.py`: OrderManager — AllocationDecision→Order lifecycle (Phase 5)
  - `reconciliation.py`: Position reconciliation vs broker (Phase 5)
  - `drift_monitor.py`: feature drift detection
- `research/` — pluggable ABCs + Phase C modules
  - `hnsw_distance.py`: HNSWIndex, 54.5× speedup, recall@50=0.9996 (SLE-47 ✓)
  - Enable: `EngineConfig(use_hnsw=True)` — default False (ball_tree unchanged)
- `archive/legacy_v1/` — Pre-Phase-3Z legacy code (archived, do not modify)
- `rebuild_phase_3z/` — Phase 3Z rebuild workspace (preserved for reference + parallel tests)
  - `artifacts/baselines/parity_snapshot.json` — frozen SLE-80-v1 snapshot

## Critical Rules

1. **Run tests first.** `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"` — 945 tests, all must pass.
2. **Numbers require provenance.** Any claimed metric must trace to walk-forward results
   or experiment logs. If it cannot be traced, it is fabricated. No exceptions.
3. **Do NOT modify `prepare.py` or this file** unless explicitly asked.
4. **assert → RuntimeError** for all public API guards. `assert` is stripped under `-O`.
5. **nn_jobs=1** always. Prevents Windows/Py3.12 joblib deadlock.
6. **3-strike rule:** If three consecutive attempts at the same fix fail, STOP. Log what
   was tried in the session log and escalate.

## Power of 10 Coding Standards (NASA Holzmann — P8-PRE-5/6, ADR-008–012)

All new production code in `pattern_engine/` and `trading_system/` must follow these rules.
Violations require a documented waiver in the relevant ADR.

- **R4 (function length):** New functions ≤60 lines / ≤50 statements. Splits required for
  any new function that would exceed this. Sequential-pipeline waivers must be documented.
- **R5/R10 (contracts):** Add `@icontract.require`/`@icontract.ensure` to every new public
  API function. Guard NaN/Inf inputs with `math.isfinite()`. Use `FiniteFloat` (not `float`)
  for execution-layer financial quantities in Pydantic models.
- **R7 (no silent swallows):** Never `except ...: pass`. Either re-raise, raise a typed
  exception (`from None` for translation), or log + re-raise. Bare `except Exception: pass`
  is forbidden.
- **R9 (zero new warnings):** New code must introduce 0 new ruff/mypy/bandit findings.
  Existing baseline (601 ruff) is frozen — do not add to it.
- **No recursion (R1):** Use iterative loops. No recursive calls in production paths.
- ADR index: `docs/adr/` — ADR-008 (static analysis), ADR-009 (FiniteFloat),
  ADR-011 (icontract), ADR-012 (P10 audit findings + waivers)

## Locked Settings (do not change without new experiment evidence)

Distance=Euclidean, Weighting=uniform, Features=returns_candle(23), Calibration=beta_abm,
cal_frac=0.76, max_distance=2.5, top_k=50, confidence_threshold=0.65,
regime=hold_spy_threshold+0.05, horizon=fwd_7d_up, stop_loss_atr_multiple=3.0
# Features=returns_candle(23), max_distance=2.5: Phase 6 (2026-04-09).
#   Task 6.1 sweep: winner=2.5 (smallest with AvgK≥20 all 6 folds). Provenance: results/phase6/sweep_max_distance_23d.tsv
#   Task 6.2 BSS comparison: returns_candle wins 5/6 folds vs returns_only. GATE PASS. Provenance: results/phase6/bss_comparison_candle_vs_baseline.tsv
#   Task 6.3 body_position: gate triggered DROP (3/6) but KEPT by judgment — 2023/2024-Val deltas favor 23D
#     by ×10–36 vs the gains in earlier folds. Provenance: results/phase6/redundancy_body_position.tsv
# max_distance=0.90, beta_abm (8D baseline, superseded): swept H5 (2026-04-02). Provenance: results/bss_fix_sweep_h5.tsv
# regime=hold_spy_threshold+0.05: H7 (2026-04-06). GATE MET: 3/6 positive folds, mean_BSS=+0.00033.
#   mode=hold: Bear rows (SPY ret_90d < +0.05) → base_rate prob (HOLD signal).
#   Bull mode only: KNN signal used when SPY 90d return > +5% (confirmed trend).
#   Provenance: results/bss_fix_sweep_h7.tsv. Caution: thin margin, aggressive threshold.
# stop_loss_atr_multiple: swept 2.0–4.0 on 2024 fold (2026-03-21). 3.0× won:
# Sharpe=1.53 (+32% vs 2.0×), MaxDD=5.7%, stops=28/171. Provenance: results/atr_sweep_results.tsv
# confidence_threshold=0.65 locked; use 0.55 in experiments on 585T (max Platt prob=0.6195).
# 52T beta_abm probability range: [0.50, 0.58] — below 0.65 threshold. Use production signals
#   (results/cached_signals_2024.csv, 585T Platt) for backtest work, NOT the 52T walk-forward.
# min_sector_lift=0.005 (recalibrated 2026-03-26; old 0.03 vetoed 100% on 585T).
# BSS root cause (2026-03-26): analogue pool dilution at 585T scale, NOT miscalibration.
#   Platt is helping (+0.023 BSS vs raw). Fix: tighten max_distance or same_sector_only
#   — locked settings, require new experiment evidence before changing.
# E1 BMA: FAIL (0/6 folds improved by >=+0.001). use_bma stays False.
#   BMA EM-fitted Student's t mixture degrades BSS (delta ~-0.09 to -0.14 all folds).
#   Provenance: results/phase7/e1_bma_vs_beta_abm.tsv (2026-04-09)
# E2 OWA: FAIL (0/6 folds improved by >=+0.001). use_owa stays False.
#   MI-ranked OWA weighting (best alpha=4.0) shows no BSS improvement on 23D returns_candle.
#   Deltas: [-0.00026, -0.00044, -0.00015, +0.00025, +0.00015, -0.000079] all < +0.001.
#   Provenance: results/phase7/e2_owa_vs_baseline.tsv (2026-04-09)
# E3 DTW Reranker: FAIL (Spearman fast-fail: mean rho=1.0000, rankings near-identical to Euclidean).
#   DTW on 8 return scalars is redundant with Euclidean distance (same values, no warping benefit).
#   use_dtw_reranker stays False. Provenance: results/phase7/e3_dtw_vs_baseline.tsv (2026-04-09)
# E4 Conformal: FAIL (coverage 0.814 mean, 5/6 folds >= 88%; width 1.000 mean, 0/6 < 0.30).
#   Root cause: 52T probs cluster in [0.50, 0.59]; |prob-label| scores always >=0.41;
#   threshold ~0.57 yields near-trivial [0,1] intervals (width ~1.0). 2020-COVID fold
#   coverage 0% (no gamma achieves 88% — ACI over-tightens on COVID volatility).
#   use_conformal stays False. Provenance: results/phase7/e4_conformal_coverage.tsv (2026-04-09)

## Key Design Docs (read before modifying related code)

- `docs/FPPE_TRADING_SYSTEM_DESIGN.md` v0.3 — Trading layer architecture
- `docs/campaigns/PHASE_3Z_CAMPAIGN.md` — Full Phase 3Z rebuild history (SLE-51–89)
- `docs/CANDLESTICK_CATEGORIZATION_DESIGN.md` v0.2 — Future Phase 6
- `docs/PHASE2_SYSTEM_DESIGN.md`, `docs/PHASE2_RISK_ENGINE.md` — Phase 2 spec + campaign
- Phase 4 plan: `docs/superpowers/plans/2026-04-06-phase4-portfolio-manager-plan.md`
- ADRs: `docs/adr/` — ADR-008 (Ruff/mypy/Bandit CI), ADR-009 (FiniteFloat),
  ADR-010 (structlog), ADR-011 (icontract), ADR-012 (Power of 10 audit)

## Current Phase

**NEXT: T8.1 — EOD Pipeline Automation (UNBLOCKED as of 2026-04-15)**
- Build `scripts/eod_pipeline.py` — 60-day autonomous signal-to-order pipeline
- Use structlog from day one (ADR-010). All new code must pass Ruff/mypy/Bandit baselines.
- Zero new ruff findings policy — existing baseline is 601.

**P8-PRE-6 Power of 10 Retroactive Audit — COMPLETE (2026-04-15)**
- Audited all production files in `pattern_engine/` + `trading_system/` against NASA P10 rules.
- R7 FIXED: `features.py` silent swallow. R4 improved: CC 35→18, no grade-D/E remain.
- icontract contracts added to `size_position`, `risk_engine`, `matcher.fit/query`.
- FiniteFloat added to `portfolio_state` (equity, cash, confidence, position_pct).
- ADR: `docs/adr/ADR-012-power-of-10-retroactive-audit.md`. 945 tests pass.

**P8-PRE-5 Power of 10 Hardening — COMPLETE (2026-04-15)**
- FiniteFloat type, MAX_* constants, error hierarchy, static analysis CI, icontract guards.
- ADRs: ADR-008 (Ruff/mypy/Bandit), ADR-009 (FiniteFloat), ADR-011 (icontract). 927 tests.

**R3 Optuna Infrastructure — COMPLETE (2026-04-11)**
- OptunaSweep (TPE) + GridSweep. 908 tests. Research roadmap: `docs/research/UNIFIED_RESEARCH_ROADMAP.md`

**Phase 7 Model Enhancements — COMPLETE (2026-04-10)**
- E1–E4 FAIL. Root cause: 52T probs cluster [0.50–0.59], below 0.65 threshold.
- E5/E6 deferred. All 6 flags remain False. Provenance: results/phase7/enhancement_summary.tsv

Phase 1–6 summary (for context, not active):
- Phase 2: Half-Kelly, Sharpe=2.527. Phase 3: Risk engine, Sharpe=2.659, MaxDD=4.3%.
- Phase 4: PM filter, Sharpe=2.649, MaxDD=4.4%. Phase 5: Live execution plumbing (G1–G3 ✓).
- Phase 6: returns_candle(23D), max_distance=2.5, wins 5/6 folds. 846 tests.

## Session Protocol

Every session must:
1. Read this file (automatic in Claude Code)
2. Check for active campaign: `docs/campaigns/`
3. Before ending: update session log via `/session-handoff`

## Environment

- Windows 11, Ryzen 9 5900X (12 cores), 32GB RAM, Python 3.12
- venv: `C:\Users\Isaia\.claude\financial-research\venv`
- Activate: `venv\Scripts\activate`

## Key Result Files

- `results/backtest_trades.csv` — Phase 1 flat 5% sizing, 278 trades, 2024 fold
- `results/phase2_backtest_trades.csv` — Legacy Phase 2 ATR 10% sizing, 191 trades, 2024 fold
- `results/cached_signals_2024.csv` — 585T Platt signals, 13,104 rows, 159 BUY, conf [0.65–0.75]
- `results/phase2_walkforward.tsv` — Kelly sizing, 6-fold results
- `results/phase3_walkforward.tsv` — Risk engine integration, 2024 fold
- `results/phase3_gate_check.txt` — Phase 3 gate metrics (Sharpe=2.659, MaxDD=4.3%)

## Skills Available

Claude: check `.claude/skills/` for task-specific protocols. Key skills:
run-walkforward, debug-bss, add-ticker, add-feature-set, run-backtest,
phase2-risk-engine, session-handoff
