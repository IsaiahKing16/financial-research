# CLAUDE.md — FPPE Project Root
# Loaded every session. Provenance details in docs/LOCKED_SETTINGS_PROVENANCE.md.
# Phase history in docs/PHASE_COMPLETION_LOG.md. Skills in .claude/skills/.

## Project
**FPPE (Financial Pattern Prediction Engine)** — KNN historical analogue matching on
return fingerprints. Generates probabilistic BUY/SELL/HOLD signals.

## Commands
- `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"` — 945 tests, all must pass
- `venv\Scripts\activate` — Windows venv activation
- `py -3.12 -m ruff check pattern_engine/ trading_system/` — Static analysis (baseline: 275)

## Codebases
- `pattern_engine/` — Production core: PatternMatcher, beta_abm calibration, contracts
  - `matcher.py`: 5-stage pipeline (scale > search > filter > aggregate > calibrate)
  - `contracts/`: Pydantic schemas, BaseMatcher ABC, EngineState, SignalDirection/Source
  - `contracts/finite_types.py`: FiniteFloat validated type for financial quantities
  - `features.py`, `data.py`, `schema.py` — feature pipeline and data loading
  - `regime.py`: RegimeLabeler (SPY + VIX + yield curve); regime_filter=True, mode=hold (H7)
  - `walkforward.py`: run_fold, run_walkforward, BSS, Murphy decomposition, beta_abm cal
  - `sweep.py`: OptunaSweep (TPE), GridSweep, SweepResult, KNN_SEARCH_SPACE
  - `live.py`: LiveRunner (Phase 5): AllocationDecisions + exit_tickers via OrderManager
  - Research pilots (behind flags): `sax_filter.py`, `wfa_reranker.py`, `conformal_hooks.py`
- `trading_system/` — Execution layer: SharedState, risk overlays, broker integration
  - `strategy_evaluator.py`, `signal_adapter.py`, `risk_overlays/`
  - `position_sizer.py`: Half-Kelly (Phase 2). `risk_engine.py`: stateless orchestrator (Phase 3)
  - `portfolio_manager.py`, `portfolio_state.py`: PM filter + FiniteFloat snapshots (Phase 4)
  - `broker/`: BaseBroker ABC, MockBroker. `order_manager.py`: Order lifecycle (Phase 5)
  - `reconciliation.py`, `drift_monitor.py`
  - `exceptions.py`: TradingSystemError hierarchy (P8-PRE-5)
- `research/hnsw_distance.py` — 54.5x speedup, recall@50=0.9996. Enable: `use_hnsw=True`

## Critical Rules
1. **Run tests first.** 945 tests, all must pass before committing.
2. **Numbers require provenance.** No experiment log row = not real. No exceptions.
3. **Do NOT modify `prepare.py` or this file** unless explicitly asked.
4. **assert -> RuntimeError** for all public API guards (`-O` strips assert).
5. **nn_jobs=1** always (Windows/Py3.12 joblib deadlock).
6. **3-strike rule:** 3 failed attempts -> STOP, log in session log, escalate.
7. **R4:** New functions <=60 lines / <=50 statements. Document waivers in ADR.
8. **R5/R10:** icontract @require/@ensure on new public APIs. FiniteFloat for financial Pydantic fields.
9. **R7:** No silent swallows. Never `except ...: pass`. Re-raise or log+re-raise.
10. **R9:** Zero new ruff/mypy/bandit findings. Existing baseline: 275 ruff (frozen 2026-04-15).
11. **R1:** No recursion in production paths. Use iterative loops.

## Locked Settings
Distance=Euclidean, Weighting=uniform, Features=returns_candle(23), Calibration=Platt (585T production); beta_abm (52T research only),
cal_frac=0.76, max_distance=2.5, top_k=50, confidence_threshold=0.65,
regime=hold_spy_threshold+0.05(mode=hold), horizon=fwd_7d_up, stop_loss_atr_multiple=3.0
# Full provenance: docs/LOCKED_SETTINGS_PROVENANCE.md

## Key Design Docs
- `docs/FPPE_TRADING_SYSTEM_DESIGN.md` v0.6 — Trading layer architecture
- `docs/campaigns/PHASE_3Z_CAMPAIGN.md` — Phase 3Z rebuild history (SLE-51-89)
- `docs/PHASE2_SYSTEM_DESIGN.md`, `docs/PHASE2_RISK_ENGINE.md` — Phase 2 spec
- `docs/superpowers/plans/2026-04-06-phase4-portfolio-manager-plan.md` — Phase 4 plan
- `docs/campaigns/P8_RECOVERY_CAMPAIGN.md` — Active recovery campaign (Track A/B/C)
- `docs/adr/` — ADR-007 (VOL_NORM), ADR-008 (Ruff/mypy/Bandit), ADR-009 (FiniteFloat),
  ADR-010 (structlog), ADR-011 (icontract), ADR-012 (P10 audit), ADR-013 (calibration split)
- `pyproject.toml` — Static analysis toolchain config (ruff, mypy, bandit)
- `docs/PHASE_COMPLETION_LOG.md` — Full phase history with metrics

## Current Phase
**P8-PRE-1 FAIL** (2026-04-16) — 585T full-stack walk-forward DID NOT PASS
  Failed gates: G1 (0/6 BSS positive), G2 (Sharpe=0.04), G5 (Win rate=47.5%)
  Provenance: results/phase8_pre/585t_gate_check.txt
**T8.1 BLOCKED on P8-PRE-1 failure.** Recovery campaign ACTIVE.
**NEXT: P8-RECOVERY-CAMPAIGN** — Three-track research: Track A (universe sizing),
  Track B (per-sector pools + connectors), Track C (LightGBM). See docs/campaigns/P8_RECOVERY_CAMPAIGN.md
**Phases 1-7 + P8-PRE-4/5/6 COMPLETE.** 945 tests. See docs/PHASE_COMPLETION_LOG.md.

## Session Protocol
1. Read this file (automatic). 2. Check `docs/campaigns/`. 3. End: `/session-handoff`.

## Environment
Windows 11, Ryzen 9 5900X, 32GB RAM, Python 3.12. venv: `venv\Scripts\activate`

## Key Result Files
- `results/cached_signals_2024.csv` — 585T Platt signals, 13,104 rows, 159 BUY
- `results/phase3_gate_check.txt` — Phase 3 gate metrics (Sharpe=2.659, MaxDD=4.3%)
- `results/phase6/` — Phase 6 sweep + BSS comparison results
- `results/phase7/` — Enhancement experiment results (E1-E4 all FAIL)
