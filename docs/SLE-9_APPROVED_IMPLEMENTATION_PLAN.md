# SLE-9: APPROVED IMPLEMENTATION PLAN (Composer Synthesis)

## Purpose

This document is the synthesis deliverable for Linear issue **SLE-9** ("Synthesize all reviews into approved implementation plan").
It is written as a ready-to-post Linear comment and source-of-truth handoff for M2.

## Inputs Reviewed

Because the SLE-9 issue thread in this workspace snapshot has no persisted M1 comment bodies, synthesis is based on the recorded M1 review scopes and design artifacts:

1. `docs/PHASE2_TASK_BREAKDOWN.md` (SLE-5/6/7/8 scopes, deliverables, dependencies)
2. `docs/PHASE2_SYSTEM_DESIGN.md` (algorithm spec, edge cases, rejection criteria)
3. Repository state verification:
   - Phase 1 files exist: `trading_system/backtest_engine.py`, `signal_adapter.py`, `run_phase1.py`, `__init__.py`
   - Test inventory currently collects **388 tests** (`python -m pytest tests/ --collect-only -q`)

If any M1 reviewer posted recommendations outside these artifacts, those comments should be appended before final human approval.

---

## Consensus Synthesis (M1)

The following items are treated as consensus and frozen for M2:

1. **Sizing math remains volatility-targeted and stop-anchored**
   - `raw_weight = max_loss_per_trade_pct / stop_distance_pct`
   - `stop_distance_pct = stop_loss_atr_multiple * atr_pct`
   - Risk budget target remains 2% per trade at stop level.

2. **Drawdown control remains a linear brake + hard halt**
   - Normal: DD < brake, scalar = 1.0
   - Brake: brake <= DD < halt, scalar linearly declines to 0
   - Halt: DD >= halt, no new entries

3. **ATR implementation uses existing `ta` dependency**
   - `ta.volatility.AverageTrueRange` with configured lookback
   - Reject insufficient history / non-positive ATR / non-positive price
   - Allow extreme ATR with warning/logging.

4. **Backwards compatibility is mandatory**
   - `use_risk_engine=False` must preserve Phase 1 behavior.
   - Integration must not regress existing P&L accounting invariants.

5. **Architecture for Phase 2 stays simple**
   - Module-level risk functions plus lightweight `RiskState`
   - Shared cross-layer state manager remains deferred to Phase 3+.

6. **Validation burden is explicit**
   - Add 40+ new tests (unit + integration + stress)
   - Existing suite must remain green.

---

## Disagreements / Ambiguities and Resolution

### A) ATR implementation variant (EMA/manual/SMA)
- **Options considered:** manual ATR, SMA ATR, or `ta` ATR (EMA-style).
- **Resolution:** use `ta` ATR in Phase 2 for reliability and lower implementation risk.

### B) Drawdown brake shape (linear vs exponential/sigmoid)
- **Options considered:** nonlinear curves may react differently near halt threshold.
- **Resolution:** keep linear brake for Phase 2 (transparent and directly testable). Nonlinear alternatives are explicitly deferred.

### C) Engine style (stateless functions vs class-based service)
- **Options considered:** class object with encapsulated state vs pure function API + state dataclass.
- **Resolution:** keep stateless function API and external `RiskState` for Phase 2.

### D) Minimum position rule at low adjusted size
- **Ambiguity found:** design text includes both clamp-to-min and "below minimum after DD adjustment" rejection.
- **Resolution:** for safety and consistency with rejection table, **reject** when adjusted size is positive but below minimum after DD scalar (do not upsize to min in that case).

---

## APPROVED IMPLEMENTATION PLAN (for all M2 agents)

### 1) Interface freeze before coding

`trading_system/risk_state.py`:
- `PositionDecision` (frozen dataclass)
- `RiskState` (mutable dataclass with `initial`, `update`, stop register/remove)
- `StopLossEvent` (frozen dataclass)

`trading_system/risk_engine.py`:
- `compute_atr_pct(...) -> float`
- `compute_drawdown_scalar(...) -> tuple[float, str]`
- `check_stop_loss(...) -> bool`
- `size_position(...) -> PositionDecision`

No signature drift without explicit SLE-9 amendment.

### 2) M2 task assignment by agent strength

- **SLE-10 (Codex):** implement `risk_state.py` + `tests/test_risk_state.py`
- **SLE-11 (Opus):** implement `risk_engine.py` + core `tests/test_risk_engine.py`
- **SLE-12 (Sonnet):** implement stress scenarios and edge-case tests (same test file or dedicated file per final structure)

### 3) Acceptance gates for M2 completion

1. SLE-10 tests pass
2. SLE-11 tests pass
3. SLE-12 stress tests pass
4. Existing test suite remains green
5. M3 starts only after SLE-10 + SLE-11 are complete and interfaces are stable

### 4) Non-negotiable invariants

- Stop trigger condition: `low <= stop_price`
- Stop execution price: next-day open (not stop price)
- Halt mode blocks new entries only; no force liquidation in Phase 2
- `use_risk_engine=False` reproduces Phase 1 behavior

---

## Human Approval Gate (required)

**Status:** READY FOR HUMAN APPROVAL  
**Required action:** @Sleep must explicitly reply with approval before any M2 issue moves to In Progress.

Suggested approval phrase:

> Approved for M2. Proceed with SLE-10, SLE-11, and SLE-12 using this SLE-9 plan as the implementation contract.

