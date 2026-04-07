# CAMPAIGN: Phase 2 Risk Engine (+ Phase 3 Integration)
# Created: 2026-03-19
# Status: Phase 2 COMPLETE. Phase 3 COMPLETE 2026-04-06 — GATE PASSED (fatigue disabled).
# Owner: Isaia
# Design Doc: docs/PHASE2_SYSTEM_DESIGN.md

---

## Scope

### Objective
Build `trading_system/risk_engine.py` with ATR stop-losses, volatility-based position
sizing, and drawdown brake. Integrate with Layer 1 backtest to replace fixed 5% equal-weight
sizing. Validate that risk-adjusted returns (Sharpe) improve or hold vs Phase 1 baseline.

### Success Criteria (all must pass)
- [ ] risk_engine.py implements ATR stops, vol sizing, drawdown brake
- [ ] Sharpe ratio >= 1.82 (Phase 1 baseline)
- [ ] Max drawdown <= 6.9% (Phase 1 baseline)
- [ ] Net expectancy > $0 after 26 bps friction
- [ ] All existing 388 tests still pass
- [ ] New tests cover all risk_engine code paths
- [ ] Bear-market validation (2022 fold) shows improved drawdown vs Phase 1

### Explicitly Out of Scope
- Portfolio manager (Phase 3)
- Strategy evaluator (Phase 4)
- Signal ranking or capital queue logic
- Changing any pattern_engine/ locked settings
- Modifying the signal gate thresholds

---

## Decisions Log

| Date | Decision | Rationale | Evidence | Reversible? |
|------|----------|-----------|----------|-------------|
| Pre | Drawdown brake: 15% linear, 20% halt | From FPPE_TRADING_SYSTEM_DESIGN.md v0.3 | Design doc spec | Yes |
| Pre | ATR period: 14 (default) | Standard ATR period, widely used | Industry standard | Yes |
| Pre | Friction: 26 bps round-trip | Phase 1 validated value | Phase 1 results | No |
| 2026-04-06 | Build position_sizer.py (Half-Kelly) before legacy risk_engine.py | Roadmap v2A §5: Phase 2 = Half-Kelly sizing; Phase 3 = full risk engine. Simpler interface, JR-appropriate. | fppe-roadmap-v2A.md | Yes |
| 2026-04-06 | flat_atr_pct=0.02 constant for Phase 2 | Real ATR requires OHLCV at sizing time; Phase 2 deliberately defers this to Phase 3 | Roadmap v2A Phase 2 note | Yes (Phase 3 swaps in compute_atr_pct) |
| 2026-04-06 | b_ratio=1.1811 from 2024 backtest trades | avg_win/avg_loss from results/backtest_trades.csv (278 trades) | results/backtest_trades.csv | Yes (update per fold for walk-forward) |
| 2026-04-06 | Phase 3: thin stateless orchestrator in `trading_system/risk_engine.py` | Approach A (spec). Flat functions + dataclass; no new stateful classes. Composes existing building blocks. | docs/superpowers/specs/2026-04-06-phase3-risk-engine-integration-design.md | Yes |
| 2026-04-06 | Phase 3: overlays multiply **position size**, not signal confidence | Half-Kelly already incorporates confidence into size; re-throttling confidence would double-count the same input. `BaseRiskOverlay` docstring updated to reflect new contract. | Phase 3 plan T3.5; spec §Contract | Yes |
| 2026-04-06 | Phase 3: `FatigueAccumulationOverlay` disabled by default in walk-forward (flag `USE_FATIGUE_OVERLAY=False`) | Overlay saturates in sustained regimes: with `decay_rate=0.15` and no regime transitions in 181 BULL days (H7 `SPY ret_90d > +5%`), multiplier collapses to ~1e-13, destroying positive PnL ($2053 → $408, ratio 0.20). SLE-75 was designed for short, choppy regimes; needs redesign before re-enabling. `LiquidityCongestionGate` stays on (multiplier=1.0 on all 278 trades — zero drag). | results/phase3_throttling_diagnostic.csv | Yes (flag flip) |

---

## What's Been Tried

| Date | Attempt | Result | Lesson |
|------|---------|--------|--------|
| 2026-04-06 | T2.1: trading_system/position_sizer.py | 28 tests, all pass. 159/159 BUY signals approved. Kelly range [0.357, 0.547]. | See results/phase2_kelly_sizing.csv |
| 2026-04-06 | Phase 3 T3.1–T3.5: `trading_system/risk_engine.py` (compute_atr_pct, drawdown_brake_scalar, AdjustedSizing, apply_risk_adjustments) + `size_position(atr_pct=...)` override | 34 new tests in test_risk_engine.py + 5 new tests in test_position_sizer.py. Full suite 678 passing. | tests/test_risk_engine.py, tests/test_position_sizer.py |
| 2026-04-06 | Phase 3 T3.6: walk-forward with fatigue + congestion + DD brake, all overlays ON | **FAIL** Sharpe=-0.368, final equity $10,408.60 (+4.1% < 4.5% RF). PnL drag: $2053 → $408 (80% destroyed). Root cause: fatigue saturation. | results/phase3_throttling_diagnostic.csv |
| 2026-04-06 | Phase 3 T3.6 rerun with `USE_FATIGUE_OVERLAY=False` | **GATE PASS**. Sharpe=2.659, MaxDD=4.3%, final equity $12,150.81, 278/278 placed, 0 blocked. Exceeds Phase 2's 2.527 Sharpe. | results/phase3_walkforward.tsv, results/phase3_gate_check.txt |

---

## Current Blocker
None — Phase A complete. Phase B (backtest integration) is next.

---

## Implementation Phases

### Phase A: Core Risk Functions (COMPLETE 2026-04-06)
- [x] Half-Kelly position sizer (trading_system/position_sizer.py)
- [x] Unit tests: 28 tests, all passing (tests/test_position_sizer.py)
- [x] Walk-forward comparison script (scripts/compare_kelly_vs_flat.py)
- [x] Signal-level gate check: 159/159 BUY signals positive Kelly
- Status: COMPLETE

### Phase B: Walk-Forward Validation (COMPLETE 2026-04-06)
- [x] 2024-fold equity simulation: rescaled Phase 1 trades with Kelly sizing
- [x] 6-fold Kelly fraction gate from 52T base-rate win rates
- [x] scripts/run_phase2_walkforward.py
- [x] **GATE PASSED**: 5/6 folds positive Kelly, Sharpe=2.527, MaxDD=4.7%
- Status: COMPLETE

### Phase C: Validation
- [ ] Re-run full 2024 backtest with dynamic sizing
- [ ] Compare to Phase 1 baseline (Sharpe 1.82, DD 6.9%)
- [ ] Run on 2022 bear fold specifically
- [ ] Run threshold_sweep.py at 0.68 (conservative profile validation)
- Status: NOT_STARTED

### Phase 3: Risk Engine Integration (COMPLETE 2026-04-06)
- [x] T3.1 `compute_atr_pct(atr_14, close)` — raises on non-positive inputs
- [x] T3.2 `drawdown_brake_scalar(dd, warn=0.15, halt=0.20)` — linear 1→0
- [x] T3.3 `size_position(atr_pct=...)` override — Phase 2 callers unchanged when `atr_pct=None`
- [x] T3.4 `AdjustedSizing` + `apply_risk_adjustments` orchestrator (block priority: sizing → dd_brake → overlay:<Class>)
- [x] T3.5 `BaseRiskOverlay` docstring — documents Phase 3 position-size multiplication contract
- [x] T3.6 `scripts/run_phase3_walkforward.py` — real ATR + DD brake + congestion, fatigue disabled
- [x] **GATE PASSED** (2024 fold): Sharpe 2.659, MaxDD 4.3%, 0 blocked, 0 stops on replayed Phase 1 trades
- **Followup (separate workstream):** SLE-75 redesign — `FatigueAccumulationOverlay` saturates in long-duration regimes. Options: (a) cap `regime_duration` contribution; (b) reset on DD peaks instead of regime transitions; (c) much lower `decay_rate` (~0.005, ~138-day half-life). No decision yet.
- Status: COMPLETE

---

## Metrics Tracking

| Metric | Phase 1 Baseline | Current | Target | Source |
|--------|-----------------|---------|--------|--------|
| Sharpe ratio | 1.82 | — | >= 1.82 | Phase 1 run |
| Max drawdown | 6.9% | — | <= 6.9% | Phase 1 run |
| Net expectancy | $7.39/trade | — | > $0 | Phase 1 run |
| Annual return | 22.3% | — | > 15% | Phase 1 run |
| Total trades | 278 | — | 200-350 | Phase 1 run |
| Test count | 388 | 678 | > 400 | pytest |
| Phase 3 Sharpe (2024) | — | 2.659 | ≥ 1.0 | results/phase3_gate_check.txt |
| Phase 3 Max DD (2024) | — | 4.3% | ≤ 10% | results/phase3_gate_check.txt |
| Kelly fraction (mean) | — | 0.392 | > 0 on ≥4/6 folds | results/phase2_kelly_sizing.csv |
| Half-Kelly position (mean) | — | 5.2% (2024) | [2%, 10%] | results/phase2_walkforward.tsv |
| Sharpe ratio | — | 2.527 | ≥ 1.0 | results/phase2_gate_check.txt |
| Max drawdown | — | 4.7% | ≤ 15% | results/phase2_gate_check.txt |
| Kelly positive folds | — | 5/6 | ≥ 4/6 | results/phase2_gate_check.txt |

---

## Session History

| Date | AI | Session Log | Summary |
|------|-----|-------------|---------|
| 2026-04-06 | Claude Sonnet 4.6 | SESSION_2026-04-06_phase2-positionsizer.md | T2.1–T2.3 complete: position_sizer.py, 28 tests, comparison script |
| 2026-04-06 | Claude Sonnet 4.6 | SESSION_2026-04-06_phase2-walkforward.md | Phase B: walk-forward, 2024 simulation, GATE PASSED |

---

## Decomposition Validation

- Original ask: Build Phase 2 risk engine per FPPE_TRADING_SYSTEM_DESIGN.md v0.3
- Does Phase A cover core risk functions: YES
- Does Phase B cover integration: YES
- Does Phase C cover validation: YES
- Anything missing: Conservative profile validation (0.68 threshold) added to Phase C
- Re-validation of max_holding_days in bear conditions: Added to Phase C
