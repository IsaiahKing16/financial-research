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
  - `live.py`: LiveRunner + MockBrokerAdapter (EOD pipeline skeleton; broker DI for tests)
  - `contracts/matchers/hnsw_matcher.py`: save_index/load_index for disk persistence
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

1. **Run tests first.** `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"` — 678 tests, all must pass.
2. **Numbers require provenance.** Any claimed metric must trace to walk-forward results
   or experiment logs. If it cannot be traced, it is fabricated. No exceptions.
3. **Do NOT modify `prepare.py` or this file** unless explicitly asked.
4. **assert → RuntimeError** for all public API guards. `assert` is stripped under `-O`.
5. **nn_jobs=1** always. Prevents Windows/Py3.12 joblib deadlock.
6. **3-strike rule:** If three consecutive attempts at the same fix fail, STOP. Log what
   was tried in the session log and escalate.

## Locked Settings (do not change without new experiment evidence)

Distance=Euclidean, Weighting=uniform, Features=VOL_NORM_COLS(8), Calibration=beta_abm,
cal_frac=0.76, max_distance=0.90, top_k=50, confidence_threshold=0.65,
regime=hold_spy_threshold+0.05, horizon=fwd_7d_up, stop_loss_atr_multiple=3.0
# max_distance=0.90, beta_abm: swept H5 (2026-04-02). Best at 52T VOL_NORM. Provenance: results/bss_fix_sweep_h5.tsv
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

## Key Design Docs (read before modifying related code)

- `docs/FPPE_TRADING_SYSTEM_DESIGN.md` v0.3 — Trading layer architecture
- `docs/campaigns/PHASE_3Z_CAMPAIGN.md` — Full Phase 3Z rebuild history (SLE-51–89)
- `docs/CANDLESTICK_CATEGORIZATION_DESIGN.md` v0.2 — Future Phase 6
- `docs/PHASE2_SYSTEM_DESIGN.md`, `docs/PHASE2_RISK_ENGINE.md` — Phase 2 spec + campaign (in docs/, NOT docs/campaigns/)

## Current Phase

**Phase 3 Risk Engine Integration — COMPLETE 2026-04-06. GATE PASSED.**

Experiment history (H1–H7 complete):
- 585T (H1–H4): Resolution≈0 (pool dilution). All parametric fixes failed. 0/6 positive folds.
- 52T VOL_NORM (H5): max_d sweep → best max_d=0.90, 2/6 positive folds. Gate=3/6.
- 52T Murphy: Resolution=0.0076 (signal EXISTS), Reliability=0.0095 (dominant, esp. 2022 Bear).
- 52T H6 (feature expansion): 8D baseline still best. Cross-sectional features hurt at max_d=0.90.
- **H7 (regime HOLD): GATE MET — 3/6 positive folds, mean_BSS=+0.00033.**
  - Winner: mode=hold, spy_threshold=+0.05 (SPY ret_90d < +5% → HOLD)
  - See: `results/bss_fix_sweep_h7.tsv`, `scripts/experiments/h7_regime_filter.py`

**Phase 2 COMPLETE (2026-04-06):**
- trading_system/position_sizer.py — Half-Kelly with SizingConfig, 28 tests
- Gate: 5/6 folds positive Kelly, 2024 Sharpe=2.527, MaxDD=4.7% ✓
- Caution: 2022-Bear fold Kelly=-0.504 (b_ratio=0.661 — poor win/loss ratio in crash)
- Provenance: results/phase2_walkforward.tsv, results/phase2_gate_check.txt

**Phase 3 COMPLETE (2026-04-06):**
- `trading_system/risk_engine.py` — thin stateless orchestrator: `compute_atr_pct`,
  `drawdown_brake_scalar`, `AdjustedSizing`, `apply_risk_adjustments`
- `size_position(atr_pct=...)` — Phase 2 compat preserved (None → flat_atr_pct)
- **Phase 3 contract:** overlays multiply POSITION SIZE, not confidence (Half-Kelly
  already incorporates confidence; double-throttling would double-count).
- Gate (2024 fold): Sharpe=2.659, MaxDD=4.3%, 0 blocked, 278/278 placed ✓
- Provenance: `results/phase3_walkforward.tsv`, `results/phase3_gate_check.txt`
- **Fatigue overlay DISABLED** in walk-forward (`USE_FATIGUE_OVERLAY=False` in
  `scripts/run_phase3_walkforward.py`). SLE-75 saturates in sustained regimes:
  with H7's sticky BULL definition and `decay_rate=0.15`, multiplier collapsed to
  ~1e-13 over 181 BULL days, dropping PnL from $2053 → $408. LiquidityCongestionGate
  stays on (multiplier=1.0 on all 278 trades, zero drag). SLE-75 needs redesign
  before re-enabling. Diagnostic: `results/phase3_throttling_diagnostic.csv`.

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

## Skills Available

Claude: check `.claude/skills/` for task-specific protocols. Key skills:
run-walkforward, debug-bss, add-ticker, add-feature-set, run-backtest,
phase2-risk-engine, session-handoff
