# SLE-9 Approved Implementation Plan (Ready-to-Post Linear Comment)

## Scope and evidence used for synthesis

This synthesis is based on the available repository artifacts and issue context in this workspace:

- `docs/PHASE2_SYSTEM_DESIGN.md`
- `docs/PHASE2_TASK_BREAKDOWN.md`
- `AGENTS.md`
- `PROJECT_GUIDE.md`
- Current repository state verification (notably `trading_system/` now includes Phase 1 files)

Note: the SLE-9 issue snapshot in this workspace does not include persisted full comment bodies from SLE-5/6/7/8. Therefore, this plan consolidates explicit review scopes, design decisions, and present repo state into an auditable implementation gate for M2.

---

## APPROVED IMPLEMENTATION PLAN (SLE-9)

### 1) Consensus synthesis (frozen decisions for M2)

1. **Risk sizing formula is adopted as specified**  
   `raw_weight = max_loss_per_trade_pct / stop_distance_pct`, with `stop_distance_pct = stop_loss_atr_multiple * atr_pct`, then drawdown scalar, then clamp to min/max position limits.

2. **Drawdown control remains linear brake + hard halt**  
   Keep linear ramp from brake threshold to halt threshold; halt mode rejects all new entries.

3. **ATR implementation (ratified SLE-29)**  
   ATR% is computed in the risk layer from raw OHLC (`ATR / current_close`), not as an upstream precomputed column. **Shipped implementation:** manual True Range from `High` / `Low` / prior `Close`, then exponentially weighted mean with `alpha = 1/lookback`, `adjust=False`, `min_periods=lookback` (pandas `ewm`) — the usual Wilder-style ATR recipe in vector form. This is functionally correct and review-approved; using `ta.volatility.AverageTrueRange` is optional and would be numeric-parity work, not a correctness fix.

4. **Phase 2 keeps risk engine mostly stateless**  
   Use a lightweight mutable `RiskState` dataclass for drawdown/stop tracking. Full shared state orchestration is deferred to later phases.

5. **Phase 1 compatibility is a non-negotiable invariant**  
   `use_risk_engine=False` must preserve legacy behavior in backtest execution.

6. **Stop-loss fill convention is frozen**  
   Trigger on daily low breach, execute at next-day open (conservative, avoids optimistic fills).

7. **Issue 0.1 status for kickoff**  
   Prior concern about missing Phase 1 files is now resolved in the current repo state (`backtest_engine.py`, `signal_adapter.py`, `run_phase1.py`, `__init__.py` are present), so M2 can proceed.

### 2) Disagreements and resolution status

- **No explicit conflicting reviewer comments are persisted in this SLE-9 snapshot.**  
  Given missing historical comment bodies, no direct reviewer-vs-reviewer conflicts can be quoted verbatim here.

- **Coordinator resolution:** proceed with the documented design baseline above, and require that any newly surfaced disagreements during M2 be posted back to SLE-9 with:
  - both alternatives,
  - impact analysis (correctness, testability, backward compatibility),
  - coordinator recommendation.

### 3) Frozen interfaces for M2 parallel work

#### `trading_system/risk_state.py` (Codex / SLE-10)
- `PositionDecision` (frozen dataclass)
- `RiskState` (mutable dataclass)
- `StopLossEvent` (frozen dataclass)
- `RiskState.initial(starting_equity) -> RiskState`
- `RiskState.update(current_equity, config) -> None`
- `RiskState.register_stop(ticker, stop_price) -> None`
- `RiskState.remove_stop(ticker) -> None`

#### `trading_system/risk_engine.py` (Opus / SLE-11)
- `compute_atr_pct(price_history: pd.DataFrame, lookback: int = 20) -> float`
- `compute_drawdown_scalar(current_equity: float, peak_equity: float, brake_threshold: float, halt_threshold: float) -> tuple[float, str]`
- `check_stop_loss(current_low: float, stop_price: float) -> bool`
- `size_position(...) -> PositionDecision` (signature and behavior aligned to `PHASE2_SYSTEM_DESIGN.md`)

#### `trading_system/backtest_engine.py` integration contract (SLE-13 dependency)
- Backward-compatible feature flag: `use_risk_engine: bool = False`
- Daily loop order preserved: MTM -> risk update -> stop checks -> exits -> entries
- `use_risk_engine=False` path remains behaviorally identical to Phase 1

### 4) M2 task assignment (approved)

- **SLE-10 (`risk_state.py`) -> Agent: Codex**
  - Implement dataclasses and state transitions
  - Add `tests/test_risk_state.py`

- **SLE-11 (`risk_engine.py`) -> Agent: Opus**
  - Implement ATR, sizing, drawdown scalar, and stop trigger helpers
  - Add `tests/test_risk_engine.py`

- **SLE-12 (stress tests) -> Agent: Sonnet**
  - Add stress scenarios for crash / simultaneous stops / extreme ATR / minimal capital / capacity limits

### 5) Acceptance gates to exit M2

1. M2 issue-level tests pass (targeted suites for SLE-10/11/12).
2. Integration path remains open for SLE-13 without API churn.
3. Full regression suite passes on branch:
   - `source venv/bin/activate && python -m pytest tests/ -v`
4. No violation of Phase 1 compatibility invariant for the disabled risk-engine path.

### 6) Required human approval gate

**M2 may begin only after explicit approval from `@Sleep` on SLE-9.**

Use the following approval language in-thread:

> **APPROVAL REQUEST (SLE-9):**  
> Please confirm: "Approved implementation plan for Phase 2 risk engine; M2 (SLE-10/11/12) may move to In Progress."

When approved, coordinator should post:

> **APPROVED by @Sleep** -> M2 issues SLE-10, SLE-11, SLE-12 unblocked and ready to start.
