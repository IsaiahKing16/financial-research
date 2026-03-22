# CLAUDE.md — FPPE Project Root
# This file is loaded every session. Keep it under 80 lines.
# Detailed protocols live in .claude/skills/ — Claude loads them on demand.

## Project

**FPPE (Financial Pattern Prediction Engine)** — K-nearest-neighbor historical analogue
matching on return fingerprints. Generates probabilistic BUY/SELL/HOLD signals.

## Commands
- `python -m pytest tests/ -q -m "not slow"` — Run full suite (always run before committing)
- `venv\Scripts\activate` — Windows venv activation

## Codebases

- `pattern_engine/` — Phase 3Z production core: PatternMatcher, Platt calibration, contracts
  - `matcher.py`: 5-stage PatternMatcher (scale → search → filter → aggregate → calibrate)
  - `contracts/`: Pydantic schemas, BaseMatcher ABC, EngineState, SignalDirection/Source
  - `features.py`, `data.py`, `schema.py` — feature pipeline and data loading
  - Research pilots (behind flags): `sax_filter.py`, `wfa_reranker.py`, `ib_compression.py`, `conformal_hooks.py`
- `trading_system/` — Phase 3Z production layer: SharedState, StrategyEvaluator, risk overlays
  - `strategy_evaluator.py`: signal → position decision with risk overlays
  - `signal_adapter.py`: UnifiedSignal (Pydantic), KNN/DL adapters
  - `risk_overlays/`: fatigue accumulation, liquidity congestion
  - `drift_monitor.py`: feature drift detection
- `research/` — pluggable ABCs + Phase C modules
  - `hnsw_distance.py`: HNSWIndex, 54.5× speedup, recall@50=0.9996 (SLE-47 ✓)
  - Enable: `EngineConfig(use_hnsw=True)` — default False (ball_tree unchanged)
- `archive/legacy_v1/` — Pre-Phase-3Z legacy code (archived, do not modify)
- `rebuild_phase_3z/` — Phase 3Z rebuild workspace (preserved for reference + parallel tests)
  - `artifacts/baselines/parity_snapshot.json` — frozen SLE-80-v1 snapshot

## Critical Rules

1. **Run tests first.** `python -m pytest tests/ -q -m "not slow"` — 543 tests, all must pass.
2. **Numbers require provenance.** Any claimed metric must trace to walk-forward results
   or experiment logs. If it cannot be traced, it is fabricated. No exceptions.
3. **Do NOT modify `prepare.py` or this file** unless explicitly asked.
4. **assert → RuntimeError** for all public API guards. `assert` is stripped under `-O`.
5. **nn_jobs=1** always. Prevents Windows/Py3.12 joblib deadlock.
6. **3-strike rule:** If three consecutive attempts at the same fix fail, STOP. Log what
   was tried in the session log and escalate.

## Locked Settings (do not change without new experiment evidence)

Distance=Euclidean, Weighting=uniform, Features=returns_only(8), Calibration=Platt,
cal_frac=0.76, max_distance=1.1019, top_k=50, confidence_threshold=0.65,
regime=binary, horizon=fwd_7d_up, stop_loss_atr_multiple=3.0
# stop_loss_atr_multiple: swept 2.0–4.0 on 2024 fold (2026-03-21). 3.0× won:
# Sharpe=1.53 (+32% vs 2.0×), MaxDD=5.7%, stops=28/171. Provenance: results/atr_sweep_results.tsv

## Key Design Docs (read before modifying related code)

- `docs/FPPE_TRADING_SYSTEM_DESIGN.md` v0.3 — Trading layer architecture
- `docs/campaigns/PHASE_3Z_CAMPAIGN.md` — Full Phase 3Z rebuild history (SLE-51–89)
- `docs/CANDLESTICK_CATEGORIZATION_DESIGN.md` v0.2 — Future Phase 6

## Current Phase

**Production Migration** — COMPLETE. Phase 3Z rebuild promoted to root.
- `pattern_engine/` and `trading_system/` are now the Phase 3Z production code.
- Legacy code archived at `archive/legacy_v1/`. Operational scripts (live.py, overnight.py)
  deferred to M9 (data ingestion scale-up).
- Test suite: `python -m pytest tests/ -q -m "not slow"` → 543 tests.
- **Next phase:** M9 (data ingestion scale-up — universe expansion to 8,000–12,000 tickers).

## Session Protocol

Every session must:
1. Read this file (automatic in Claude Code)
2. Check for active campaign: `docs/campaigns/`
3. Before ending: update session log via `/session-handoff`

## Environment

- Windows 11, Ryzen 9 5900X (12 cores), 32GB RAM, Python 3.12
- venv: `C:\Users\Isaia\.claude\financial-research\venv`
- Activate: `venv\Scripts\activate`

## Skills Available

Claude: check `.claude/skills/` for task-specific protocols. Key skills:
run-walkforward, debug-bss, add-ticker, add-feature-set, run-backtest,
phase2-risk-engine, session-handoff
