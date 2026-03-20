# Phase 2 Task Breakdown — Risk Engine Implementation
## Derived from: PHASE2_SYSTEM_DESIGN.md v1.0

**Date:** March 19, 2026
**Linear Project:** FPPE Phase 2: Risk Engine
**Tracking:** All tasks tracked as Linear issues (SLE-5 through SLE-19)

---

## Overview

Phase 2 adds three capabilities to the FPPE trading system:
1. **Volatility-based position sizing** — ATR-scaled, targeting 2% max loss per trade
2. **ATR stop-losses** — Dynamic stops at entry - 2×ATR%
3. **Drawdown brake & halt** — Linear ramp from 15%→20% drawdown

### Files to Create
| File | Owner | Lines (est.) | Dependencies |
|------|-------|-------------|--------------|
| `trading_system/risk_state.py` | Codex | ~120 | stdlib only |
| `trading_system/risk_engine.py` | Opus | ~250 | risk_state, ta, pandas, numpy, config |
| `trading_system/run_phase2.py` | Sonnet | ~150 | backtest_engine, risk_engine |
| `tests/test_risk_state.py` | Codex | ~150 | risk_state |
| `tests/test_risk_engine.py` | Opus | ~400 | risk_engine, risk_state |
| `tests/test_phase2_integration.py` | Codex | ~300 | backtest_engine, risk_engine |

### Files to Modify
| File | Owner | Changes |
|------|-------|---------|
| `trading_system/backtest_engine.py` | Opus | Add use_risk_engine flag, stop-loss loop, RiskState tracking |
| `trading_system/__init__.py` | Opus | Export new symbols |
| `CLAUDE.md` | Sonnet | Test count, key files |
| `PROJECT_GUIDE.md` | Sonnet | Phase 2 results, module list |
| `docs/FPPE_TRADING_SYSTEM_DESIGN.md` | Sonnet | Bump to v0.4 |

---

## Milestone 1: Design Review & Plan Approval

**Goal:** All agents review the design, surface concerns, and produce an approved plan.
**Parallelism:** SLE-5, SLE-6, SLE-7, SLE-8 run in parallel. SLE-9 runs after all complete.

### SLE-5: Risk Model Math Review (GPT-5.4)
**Input:** PHASE2_SYSTEM_DESIGN.md Sections 4.1–4.4
**Question to answer:** Is the position sizing formula mathematically sound?
**Key checks:**
- `raw_weight = 0.02 / stop_distance_pct` — does a stop hit actually lose 2%?
- Linear interpolation at boundaries: what happens at exactly 15% and 20% DD?
- ATR% definition: is EMA the right choice vs SMA for risk management?
- Gap-through: is the max-position-pct clamp sufficient protection?

### SLE-6: Integration Feasibility (Opus)
**Input:** backtest_engine.py (42KB) + PHASE2_SYSTEM_DESIGN.md Section 2.3
**Question to answer:** Can we cleanly add risk_engine calls without breaking Phase 1?
**Key checks:**
- Map all `position_pct = 0.05` locations in backtest_engine.py
- Verify daily loop ordering: MTM → risk update → stop check → exits → entries
- Check P&L double-counting fix survives stop-loss additions
- Confirm `use_risk_engine=False` produces identical Phase 1 results

### SLE-7: Project Readiness Verification (Sonnet)
**Input:** PROJECT_GUIDE.md, CLAUDE.md
**Question to answer:** Is the documentation accurate and ready for Phase 2?
**Key checks:**
- Test count = 402, module count = 21+8 (trading_system), all paths use `docs/` prefix
- Phase 1 completion items checked off
- ISSUE 0.1 (missing files) resolved

### SLE-8: Alternative Approaches (Gemini)
**Input:** PHASE2_SYSTEM_DESIGN.md (full)
**Question to answer:** Are there better approaches the design missed?
**Key checks:**
- 2× ATR vs Kelly criterion vs risk parity
- Linear brake vs exponential/sigmoid
- Stateless functions vs class-based risk engine
- Missing rejection conditions

### SLE-9: Plan Synthesis (Composer) — BLOCKED BY SLE-5,6,7,8
**Input:** All review comments
**Output:** Approved implementation plan with any design amendments
**Gate:** Human (@Sleep) must approve before M2 begins

---

## Milestone 2: Core Implementation

**Goal:** risk_state.py and risk_engine.py created with full unit test coverage.
**Parallelism:** SLE-10 and SLE-11 can run in parallel (agreed interface). SLE-12 can start once API signatures are known.

### SLE-10: risk_state.py (Codex) — BLOCKED BY SLE-9
**Deliverables:**
```
trading_system/risk_state.py
├── PositionDecision (frozen dataclass)
│   ├── approved, ticker, position_pct, shares, dollar_amount
│   ├── stop_price, stop_distance_pct, atr_pct
│   ├── drawdown_scalar, raw_weight, rejection_reason
│   └── __post_init__ validation
├── RiskState (mutable dataclass)
│   ├── peak_equity, current_equity, current_drawdown
│   ├── drawdown_mode, sizing_scalar, active_stops, daily_atr_cache
│   ├── update(current_equity, config)
│   ├── register_stop(ticker, stop_price)
│   ├── remove_stop(ticker)
│   └── initial(starting_equity) classmethod
└── StopLossEvent (frozen dataclass)
    ├── ticker, trigger_date, stop_price, trigger_low
    ├── entry_price, exit_price, gap_through, atr_at_entry
```
**Tests:** 10 in tests/test_risk_state.py
**Acceptance:** `python -m pytest tests/test_risk_state.py -v` — 10 passed

### SLE-11: risk_engine.py (Opus) — BLOCKED BY SLE-9
**Deliverables:**
```
trading_system/risk_engine.py
├── compute_atr_pct(price_history, lookback=20) → float
│   ├── Uses ta.volatility.AverageTrueRange
│   ├── Returns ATR / current_close
│   ├── Rejects: insufficient data, zero price, zero ATR
│   └── Warns: ATR% > 20%
├── size_position(ticker, entry_price, ...) → PositionDecision
│   ├── Step 1: compute stop_distance_pct = atr_multiple × atr_pct
│   ├── Step 2: raw_weight = max_loss / stop_distance
│   ├── Step 3: apply drawdown_scalar
│   ├── Step 4: clamp to [min_pct, max_pct]
│   ├── Step 5: compute shares
│   └── All rejection conditions from Section 5
├── check_stop_loss(current_low, stop_price) → bool
└── compute_drawdown_scalar(equity, peak, brake, halt) → (float, str)
```
**Tests:** 30 in tests/test_risk_engine.py
**Acceptance:** `python -m pytest tests/test_risk_engine.py -v` — 30 passed

### SLE-12: Stress Tests (Sonnet) — BLOCKED BY SLE-9
**Deliverables:** 5 stress tests in tests/test_risk_engine.py (or separate file)
- Synthetic crash (10% daily × 5 days)
- All stops trigger simultaneously
- Extreme ATR (50% — penny stock)
- Minimal capital ($2,000)
- Maximum positions (exposure limit)

---

## Milestone 3: Integration & Backtest

**Goal:** backtest_engine.py modified, integration tests pass, Phase 2 backtest runnable.
**Parallelism:** SLE-14 and SLE-15 can start once SLE-13 is partially complete (API defined). SLE-16 runs after SLE-10 and SLE-11 are Done.

### SLE-13: backtest_engine.py Integration (Opus) — BLOCKED BY SLE-10, SLE-11
**Changes to backtest_engine.py:**
1. Add `use_risk_engine: bool = False` to `BacktestEngine.__init__` or `run()`
2. When True:
   - Initialize `RiskState.initial(config.capital.initial_capital)` at backtest start
   - Replace `position_pct = 0.05` with `risk_engine.size_position()` call
   - Add stop-loss check in daily MTM loop: `if low <= stop_price → flag exit`
   - Execute stop exits at next-day open
   - Call `risk_state.update()` after daily MTM
   - Call `risk_state.register_stop()` / `remove_stop()` on entry/exit
3. Record `exit_reason="stop_loss"` and `StopLossEvent` in trade log
4. Update `__init__.py` exports

**Critical invariant:** `use_risk_engine=False` MUST produce byte-identical results to current Phase 1.

### SLE-14: Integration Tests (Codex) — BLOCKED BY SLE-10, SLE-11
**10 integration tests** covering full backtest flows with synthetic data.

### SLE-15: run_phase2.py (Sonnet) — BLOCKED BY SLE-13
**Entry point script** that:
- Loads `results/cached_signals_2024.csv`
- Runs Phase 1 (baseline) and Phase 2 backtests
- Prints comparison table
- Saves results to `results/`

### SLE-16: Cross-Agent Code Review (Gemini + GPT-5.4) — BLOCKED BY SLE-10, SLE-11
- Gemini reviews risk_engine.py (Opus's code)
- GPT-5.4 reviews risk_state.py (Codex's code)
- Both check against PHASE2_SYSTEM_DESIGN.md

---

## Milestone 4: Validation & Documentation

**Goal:** Success criteria met, docs updated, Phase 2 merged to main.
**Parallelism:** SLE-18 can start once SLE-17 produces results. SLE-19 runs last.

### SLE-17: Backtest Validation (Opus) — BLOCKED BY SLE-13, SLE-14, SLE-15
Run Phase 2 backtest, validate all 8 success criteria from Section 1.3.
Create `docs/PHASE2_RESULTS.md`.

### SLE-18: Documentation Updates (Sonnet) — BLOCKED BY SLE-17
Update CLAUDE.md, PROJECT_GUIDE.md, FPPE_TRADING_SYSTEM_DESIGN.md.

### SLE-19: Final Sign-Off (Composer) — BLOCKED BY SLE-16, SLE-17, SLE-18
All agents confirm their work area is complete. "Phase 2 COMPLETE — ready for merge."

---

## Dependency Graph

```
M1 (parallel reviews)
├── SLE-5 (GPT-5.4: math) ─────┐
├── SLE-6 (Opus: integration) ──┤
├── SLE-7 (Sonnet: readiness) ──┼── SLE-9 (Composer: synthesis) ── GATE: Human approval
└── SLE-8 (Gemini: alternatives)┘
                                         │
M2 (parallel implementation)             ▼
├── SLE-10 (Codex: risk_state.py) ──┬── SLE-13 (Opus: integration)
├── SLE-11 (Opus: risk_engine.py) ──┤        │
└── SLE-12 (Sonnet: stress tests)   │   SLE-14 (Codex: integ tests)
                                     │   SLE-15 (Sonnet: run_phase2.py) ← blocked by SLE-13
                                     └── SLE-16 (Gemini+GPT-5.4: review)
M3 (integration)                              │
                                              ▼
M4 (validation)
├── SLE-17 (Opus: backtest validation)
├── SLE-18 (Sonnet: doc updates) ← blocked by SLE-17
└── SLE-19 (Composer: final sign-off) ← blocked by SLE-16,17,18
```

## Agent Workload Summary

| Agent | M1 | M2 | M3 | M4 | Total Issues |
|-------|----|----|----|----|-------------|
| Opus | 1 review | 1 impl | 1 integration | 1 validation | **4** |
| Sonnet | 1 review | 1 impl | 1 impl | 1 docs | **4** |
| Codex | — | 1 impl | 1 impl | — | **2** |
| GPT-5.4 | 1 review | — | 1 review | — | **2** |
| Gemini | 1 review | — | 1 review | — | **2** |
| Composer | 1 synthesis | — | — | 1 sign-off | **2** |

**Critical path:** SLE-9 → SLE-11 → SLE-13 → SLE-17 → SLE-19 (all Opus/Composer)

---

*Phase 2 Task Breakdown v1.0 — March 19, 2026*
*Source of truth: Linear project "FPPE Phase 2: Risk Engine"*
