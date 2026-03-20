# CAMPAIGN: Phase 2 Risk Engine
# Created: 2026-03-19
# Status: NOT_STARTED
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

---

## What's Been Tried

| Date | Attempt | Result | Lesson |
|------|---------|--------|--------|
| (none yet) | | | |

---

## Current Blocker
None — ready to begin implementation.

---

## Implementation Phases

### Phase A: Core Risk Functions
- [ ] ATR stop-loss calculator
- [ ] Volatility position sizer
- [ ] Drawdown brake (linear scalar)
- [ ] Unit tests for all three
- Status: NOT_STARTED

### Phase B: Integration with Backtest
- [ ] Modify backtest_engine.py to consume risk_engine sizing
- [ ] Replace fixed 5% with dynamic sizing
- [ ] ATR stops in exit logic
- [ ] Integration tests
- Status: NOT_STARTED

### Phase C: Validation
- [ ] Re-run full 2024 backtest with dynamic sizing
- [ ] Compare to Phase 1 baseline (Sharpe 1.82, DD 6.9%)
- [ ] Run on 2022 bear fold specifically
- [ ] Run threshold_sweep.py at 0.68 (conservative profile validation)
- Status: NOT_STARTED

---

## Metrics Tracking

| Metric | Phase 1 Baseline | Current | Target | Source |
|--------|-----------------|---------|--------|--------|
| Sharpe ratio | 1.82 | — | >= 1.82 | Phase 1 run |
| Max drawdown | 6.9% | — | <= 6.9% | Phase 1 run |
| Net expectancy | $7.39/trade | — | > $0 | Phase 1 run |
| Annual return | 22.3% | — | > 15% | Phase 1 run |
| Total trades | 278 | — | 200-350 | Phase 1 run |
| Test count | 388 | — | > 400 | pytest |

---

## Session History

| Date | AI | Session Log | Summary |
|------|-----|-------------|---------|
| (none yet) | | | |

---

## Decomposition Validation

- Original ask: Build Phase 2 risk engine per FPPE_TRADING_SYSTEM_DESIGN.md v0.3
- Does Phase A cover core risk functions: YES
- Does Phase B cover integration: YES
- Does Phase C cover validation: YES
- Anything missing: Conservative profile validation (0.68 threshold) added to Phase C
- Re-validation of max_holding_days in bear conditions: Added to Phase C
