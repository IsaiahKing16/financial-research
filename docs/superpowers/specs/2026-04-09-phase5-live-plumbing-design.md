# Phase 5: Live Execution Plumbing — Design Spec

**Date:** 2026-04-09
**Author:** Claude + Isaia
**Phase:** 5 of 10 (critical path)
**Prerequisite:** Phase 3 gate passed, Phase 4 complete
**Duration:** 4 weeks estimated
**Roadmap:** `docs/superpowers/plans/2026-03-28-fppe-full-roadmap-v2.md`

---

## Overview

Phase 5 builds the execution layer that bridges trading decisions to broker actions. It creates the broker abstraction, order lifecycle management, position reconciliation, and rewires LiveRunner to use the new infrastructure.

Real broker adapters (IBKR, Alpaca) are deferred to Phase 8 when paper trading begins. This phase focuses on the infrastructure and validates it against the mock broker.

## Scope

| Task | Description | Deliverable |
|------|-------------|-------------|
| 5.1 | BaseBroker ABC (clean break) | `trading_system/broker/base.py` |
| 5.2 | Enhanced MockBroker | `trading_system/broker/mock.py` |
| 5.3 | Order Manager | `trading_system/order_manager.py` |
| 5.4 | Reconciliation | `trading_system/reconciliation.py`, `scripts/reconcile.py` |
| 5.5 | LiveRunner rewire | `pattern_engine/live.py` (modified) |
| 5.6 | Real broker adapter | **DEFERRED** to Phase 8 |

## Key Design Decisions

1. **Clean break from old ABC.** The existing `BaseBrokerAdapter`, `Order`, `OrderResult`, and `MockBrokerAdapter` in `pattern_engine/live.py` are deleted. A single `BaseBroker` ABC lives in `trading_system/broker/base.py`. LiveRunner imports from the new location.

2. **Real broker adapters deferred.** The gate requires mock broker parity only. IBKR/Alpaca adapters cannot be meaningfully tested until Phase 8 (paper trading). Building them now is premature.

3. **Order manager as bridge.** `OrderManager` translates `AllocationDecision` → `Order` (BUY entries) and manages the order state machine. A separate `create_exit_order()` handles SELL exits (stop-loss, max-hold). LiveRunner calls the order manager, not the broker directly.

4. **Reconciliation: reusable core + dual consumption.** A `reconcile()` function in `trading_system/reconciliation.py` is consumed by both a standalone CLI script (`scripts/reconcile.py`) and as a pre-flight check in LiveRunner. Reports discrepancies but never auto-corrects.

5. **LiveRunner is an execution-only runner.** It receives `AllocationDecision` objects (BUY entries) and exit triggers from its caller, translates them to orders via OrderManager, and submits. PM, risk engine, and position sizing happen upstream. The caller (a daily orchestrator, built in Phase 8) is responsible for running the pipeline and passing decisions to LiveRunner. LiveRunner retains its matcher for signal generation but does NOT run PM/risk/sizing.

6. **Extend `OrderStatus`, don't duplicate.** Add `SUBMITTED` and `TIMED_OUT` to the existing `OrderStatus` enum in `contracts/trades.py` rather than creating a parallel `OrderState` enum. The existing `PENDING` semantics shift from "submitted, awaiting fill" to "created, not yet submitted" — audit all usages for compatibility.

7. **Use `capital_allocated` for quantity computation.** `OrderManager.create_order_from_decision()` computes `quantity = decision.capital_allocated / price`, using the dollar amount already computed by the position sizer. More directly traceable than re-deriving from `final_position_pct * equity`.

---

## Section 1: BaseBroker ABC & Schemas

**File:** `trading_system/broker/base.py`

### Schemas

```python
class Order(BaseModel, frozen=True):
    order_id: str               # UUID, assigned by OrderManager
    ticker: str
    side: OrderSide             # reuse from contracts/trades.py
    quantity: float             # shares (fractional OK)
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: Optional[float] = None
    notional: Optional[float] = None  # dollar amount (audit trail)
    timestamp: datetime

class OrderResult(BaseModel, frozen=True):
    order_id: str               # matches Order.order_id
    ticker: str
    status: OrderStatus         # reuse FILLED/PARTIAL/REJECTED/CANCELLED
    filled_quantity: float
    fill_price: float
    latency_ms: float = 0.0
    executed_at: Optional[datetime] = None  # timestamp of broker fill
    error: Optional[str] = None

class BrokerPosition(BaseModel, frozen=True):
    ticker: str
    quantity: float
    avg_cost: float
    current_value: float
    unrealized_pnl: float

class AccountSnapshot(BaseModel, frozen=True):
    total_value: float
    cash: float
    buying_power: float
    positions: tuple[BrokerPosition, ...] = ()
```

### ABC

```python
class BaseBroker(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]: ...

    @abstractmethod
    def get_account(self) -> AccountSnapshot: ...

    @abstractmethod
    def is_connected(self) -> bool: ...
```

### Reuse & Enum Extension

`OrderSide` and `OrderStatus` enums already exist in `trading_system/contracts/trades.py`. No duplication.

**`OrderStatus` extension:** Add two new values to the existing enum:
- `SUBMITTED = "SUBMITTED"` — sent to broker, awaiting response
- `TIMED_OUT = "TIMED_OUT"` — no broker response within deadline

Update the existing `PENDING` docstring from "Submitted, awaiting fill" to "Created, not yet submitted to broker." Audit all existing usages of `OrderStatus.PENDING` in backtest code for semantic compatibility (currently used only in `TradeEvent.backtest_fill()` which creates FILLED events, so PENDING is not used in production paths).

### Package `__init__.py`

`trading_system/broker/__init__.py` re-exports for clean imports:
```python
from .base import BaseBroker, Order, OrderResult, BrokerPosition, AccountSnapshot
from .mock import MockBroker, MockBrokerConfig
```

### Migration

Delete from `pattern_engine/live.py`:
- `BaseBrokerAdapter` class
- Old `Order` dataclass
- Old `OrderResult` dataclass
- `MockBrokerAdapter` class

All replaced by imports from `trading_system/broker/`.

---

## Section 2: Enhanced MockBroker

**File:** `trading_system/broker/mock.py`

### Configuration

```python
class MockBrokerConfig(BaseModel, frozen=True):
    fill_fraction: float = 1.0          # 1.0 = full fill, <1.0 = partial
    latency_ms: float = 0.0            # simulated latency
    slippage_bps: float = 10.0         # adverse price movement on fill
    fail_tickers: frozenset[str] = frozenset()  # always reject these
    initial_cash: float = 10_000.0
    reject_when_insufficient: bool = True  # reject if order > buying power
```

### Implementation

```python
class MockBroker(BaseBroker):
    """In-memory broker for testing. Tracks positions, cash, order history."""

    def __init__(self, config: MockBrokerConfig = MockBrokerConfig()): ...

    # BaseBroker interface
    def submit_order(self, order: Order) -> OrderResult: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_positions(self) -> list[BrokerPosition]: ...
    def get_account(self) -> AccountSnapshot: ...
    def is_connected(self) -> bool: ...  # always True

    # Test introspection (not on ABC)
    @property
    def order_history(self) -> list[tuple[Order, OrderResult]]: ...

    def set_prices(self, prices: dict[str, float]) -> None:
        """Inject current prices for position valuation and fill simulation."""
```

### Behaviors

- **Position tracking:** BUY increases position quantity, SELL decreases. Cash updated accordingly.
- **Slippage:** Fill price = injected price x (1 + slippage_bps/10000) for BUY, x (1 - slippage_bps/10000) for SELL.
- **Partial fills:** `fill_fraction < 1.0` fills that fraction of requested quantity.
- **Fail tickers:** Orders for tickers in `fail_tickers` return `REJECTED` status.
- **Buying power:** If `reject_when_insufficient`, rejects orders exceeding available cash.
- **Price injection via `set_prices()`** — test harness provides prices; no market data dependency.
- **`initial_cash` default:** 10,000 matches `TradingConfig.capital.initial_capital`. The old `MockBrokerAdapter` defaulted to 100,000 — tests migrated from the old mock must specify `initial_cash` explicitly if they relied on the higher default.

---

## Section 3: Order Manager

**File:** `trading_system/order_manager.py`

### Order Lifecycle

```
PENDING → SUBMITTED → FILLED
                    → PARTIAL
                    → REJECTED
                    → TIMED_OUT (→ cancel attempted)
PENDING → CANCELLED (manual cancel before submission)
```

### Schemas

Uses the extended `OrderStatus` enum from `contracts/trades.py` (no separate `OrderState`).

```python
class ManagedOrder(BaseModel, frozen=True):
    order: Order
    status: OrderStatus          # uses extended OrderStatus (PENDING/SUBMITTED/FILLED/etc.)
    result: Optional[OrderResult] = None
    created_at: datetime
    submitted_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
```

### Interface

```python
class OrderManager:
    """Stateful order lifecycle manager. One instance per trading day."""

    def __init__(self, broker: BaseBroker, timeout_seconds: float = 30.0): ...

    def create_order_from_decision(
        self, decision: AllocationDecision, price: float,
    ) -> Order:
        """AllocationDecision -> BUY Order.
        quantity = decision.capital_allocated / price.
        Side is always BUY (AllocationDecisions are entry-only from PM)."""

    def create_exit_order(
        self, ticker: str, quantity: float, price: float,
    ) -> Order:
        """Create a SELL order for an exit (stop-loss, max-hold, drawdown halt).
        Exit orders bypass the PM/AllocationDecision pipeline entirely."""

    def submit(self, order: Order) -> ManagedOrder:
        """Submit to broker. PENDING -> SUBMITTED -> terminal state."""

    def submit_batch(self, orders: list[Order]) -> list[ManagedOrder]:
        """Submit multiple orders sequentially. Returns all results."""

    def cancel(self, order_id: str) -> ManagedOrder:
        """Cancel a pending/submitted order."""

    @property
    def orders(self) -> dict[str, ManagedOrder]:
        """All managed orders keyed by order_id."""

    def summary(self) -> dict[str, int]:
        """Count of orders by status. For logging/diagnostics."""
```

### Design Notes

- **Stateful per day** — create a fresh `OrderManager` each trading day. No cross-day state to serialize.
- **Sequential submission** — `submit_batch` submits one at a time (no parallel broker calls). Simpler, and real brokers rate-limit anyway.
- **Timeout handling** — if broker doesn't respond within `timeout_seconds`, order transitions to `TIMED_OUT` and a cancel is attempted.
- **No retry logic** — rejected orders stay rejected. Retry policy is a Phase 8 concern when we have real failure modes to observe.

---

## Section 4: Reconciliation

### Core Module

**File:** `trading_system/reconciliation.py`

```python
class PositionMismatch(BaseModel, frozen=True):
    ticker: str
    expected_quantity: float    # from PortfolioSnapshot
    actual_quantity: float      # from broker
    delta: float               # actual - expected
    delta_pct: float           # abs(delta / expected) if expected != 0

class ReconciliationResult(BaseModel, frozen=True):
    as_of: datetime
    n_expected: int
    n_actual: int
    mismatches: tuple[PositionMismatch, ...] = ()
    unexpected_positions: tuple[str, ...] = ()  # in broker, not SharedState
    missing_positions: tuple[str, ...] = ()     # in SharedState, not broker
    passed: bool               # True if all within tolerance

def reconcile(
    snapshot: PortfolioSnapshot,
    broker: BaseBroker,
    tolerance_pct: float = 0.05,
) -> ReconciliationResult:
    """Compare PortfolioSnapshot positions against broker positions.

    A position passes if abs(expected_qty - actual_qty) / expected_qty <= tolerance_pct.
    Missing or unexpected positions always fail.
    """
```

### Standalone Script

**File:** `scripts/reconcile.py`

- Loads SharedState from JSON checkpoint file (CLI arg)
- Instantiates broker (mock for now, real adapter later)
- Calls `reconcile()`, prints human-readable report
- Exit code 0 if passed, 1 if mismatches
- `--tolerance` flag to override 5% default

### Pipeline Pre-flight Hook

Called by LiveRunner before processing signals:

```python
# In LiveRunner.run():
result = reconcile(snapshot, self.broker)
if not result.passed:
    log.error("Reconciliation failed", mismatches=result.mismatches)
    # Do not proceed with new orders
```

### Design Notes

- **Tolerance-based** — exact float matching is too brittle. 5% default accommodates rounding and partial fills.
- **Three failure modes:** quantity mismatch (within tolerance = pass), unexpected position (always fail), missing position (always fail).
- **No auto-correction** — reconciliation reports discrepancies but never modifies positions. Human intervention required.
- **SharedState checkpoint format:** The standalone script loads from a JSON file produced by `SharedState.model_dump_json()` (Pydantic v2). The caller specifies the file path via CLI arg.

---

## Section 5: LiveRunner Rewire

**File:** `pattern_engine/live.py` (modified)

### New Constructor

```python
class LiveRunner:
    def __init__(
        self,
        matcher: PatternMatcher,
        shared_state: SharedState,
        broker: BaseBroker,              # was BaseBrokerAdapter
        order_manager: OrderManager,     # NEW - injected via DI
        engine_state: Optional[EngineState] = None,
        reconcile_on_start: bool = True, # NEW - pre-flight toggle
    ): ...
```

### Updated `run()` Signature

```python
def run(
    self,
    entry_decisions: list[AllocationDecision],
    exit_tickers: list[tuple[str, float, float]],  # (ticker, quantity, price)
    snapshot: PortfolioSnapshot,
) -> list[ManagedOrder]:
```

The caller (daily orchestrator, Phase 8) runs the full pipeline upstream:
matcher → PM → risk engine → position sizer → produces AllocationDecisions.
It also determines which positions to exit (stop-loss, max-hold).
LiveRunner receives both lists and executes them.

### Updated `run()` Flow

```
1. Pre-flight reconciliation (if reconcile_on_start=True)
   - calls reconcile(snapshot, self.broker)
   - if failed: return early with no orders, log error
2. Config hash drift check (existing, warn-only)
3. Check shared_state.is_halted (existing)
4. Create exit orders via order_manager.create_exit_order() for each exit_ticker
5. Create entry orders via order_manager.create_order_from_decision() for each entry_decision
6. Submit all orders via order_manager.submit_batch() (exits first, then entries)
7. Return list[ManagedOrder]
```

### What Changes

- Constructor takes `BaseBroker` (new ABC) instead of `BaseBrokerAdapter` (deleted)
- Constructor takes `OrderManager` via DI
- `run()` receives `AllocationDecision` + exit tickers (no longer calls matcher internally)
- `run()` returns `list[ManagedOrder]` instead of `list[OrderResult]`
- Pre-flight reconciliation as opt-in first step
- Hardcoded `notional=1000.0` removed — sizing comes from AllocationDecision via order manager

### What Stays the Same

- Config hash drift warning unchanged
- Halt check unchanged
- LiveRunner still does NOT run PM or risk engine — those are called upstream

### Reconciliation & PortfolioSnapshot

`reconcile()` accepts `PortfolioSnapshot` (from `portfolio_state.py`), but LiveRunner holds `SharedState`. The `run()` method now receives a `PortfolioSnapshot` parameter from its caller. No conversion utility needed — the caller constructs the snapshot (as already done in `run_phase4_walkforward.py`).

### Deletions from `live.py`

- `BaseBrokerAdapter` class
- Old `Order` and `OrderResult` dataclasses
- `MockBrokerAdapter` class

All replaced by imports from `trading_system/broker/`.

---

## Section 6: Gate Criteria & Test Plan

### Gate

| ID | Metric | Verification |
|----|--------|-------------|
| G1 | Mock broker round-trip parity for 100 trades | Replay 100 trades from `results/backtest_trades.csv` through OrderManager -> MockBroker. All 100 fill, quantities match AllocationDecision sizing, cash/positions reconcile at end. |
| G2 | OOB reconciliation passes 30 days | Run reconciliation against MockBroker state after replaying 30 consecutive trading days. Zero mismatches, zero unexpected/missing positions. |
| G3 | Pipeline < 3 min | Time the 100-trade replay end-to-end. Must complete in < 180s. |
| G4 | All tests pass | Full test suite passes: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"` |

### Test Files

```
tests/unit/test_broker_base.py       — ABC contract tests
tests/unit/test_broker_mock.py       — MockBroker: fills, partials, rejections, slippage, position tracking
tests/unit/test_order_manager.py     — AllocationDecision -> Order, state machine, timeout, batch
tests/unit/test_reconciliation.py    — match, mismatch, unexpected, missing, tolerance edge cases
tests/unit/test_live.py              — LiveRunner rewire: reconciliation, order manager, halt check (existing file, modified)
tests/integration/test_phase5_gate.py — 100-trade replay gate (marked @pytest.mark.slow)
```

### Estimated Test Count

~60-80 new tests, bringing total from 743 to ~810-820.

---

## Data Flow

```
                         (upstream - not Phase 5's responsibility)
UnifiedSignal ──> PM.allocate_day() ──> RiskEngine ──> PositionSizer
                                                            │
                                                    AllocationDecision
                                                            │
                              ┌──────────────────────────────┘
                              │
                         (Phase 5 scope)
                              │
                    OrderManager.create_order_from_decision()
                              │
                           Order
                              │
                    OrderManager.submit()
                              │
                    BaseBroker.submit_order()
                              │
                        OrderResult
                              │
                       ManagedOrder (with state tracking)
                              │
                    ┌─────────┴─────────┐
                    │                   │
              LiveRunner returns    Reconciliation
              list[ManagedOrder]    (pre-flight or script)
```

---

## Files Created / Modified

| Action | File |
|--------|------|
| CREATE | `trading_system/broker/__init__.py` |
| CREATE | `trading_system/broker/base.py` |
| CREATE | `trading_system/broker/mock.py` |
| CREATE | `trading_system/order_manager.py` |
| CREATE | `trading_system/reconciliation.py` |
| CREATE | `scripts/reconcile.py` |
| CREATE | `tests/unit/test_broker_base.py` |
| CREATE | `tests/unit/test_broker_mock.py` |
| CREATE | `tests/unit/test_order_manager.py` |
| CREATE | `tests/unit/test_reconciliation.py` |
| CREATE | `tests/integration/test_phase5_gate.py` |
| MODIFY | `pattern_engine/live.py` — delete old ABC/mock, rewire LiveRunner |
| MODIFY | `trading_system/contracts/trades.py` — extend OrderStatus with SUBMITTED, TIMED_OUT |
| MODIFY | `tests/unit/test_live.py` — update for new interfaces |
| MODIFY | `CLAUDE.md` — add broker package + order_manager to codebase section, update test count |

---

## Out of Scope

- Real broker adapters (IBKR, Alpaca) — deferred to Phase 8
- Full daily pipeline orchestrator — Phase 8
- Market data feeds — Phase 8
- Retry/circuit-breaker logic — Phase 8 (need real failure modes first)
- SharedState updates from order results — Phase 8 (orchestrator responsibility)
