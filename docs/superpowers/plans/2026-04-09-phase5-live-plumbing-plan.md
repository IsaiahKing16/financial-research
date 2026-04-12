# Phase 5: Live Execution Plumbing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the execution layer that bridges AllocationDecision → broker orders, with mock broker, order lifecycle management, position reconciliation, and a rewired LiveRunner.

**Architecture:** Clean break from old broker classes in `pattern_engine/live.py`. New `trading_system/broker/` package with Pydantic-frozen schemas, `BaseBroker` ABC, and `MockBroker`. `OrderManager` translates allocation decisions to orders and manages state machine. `reconciliation.py` compares portfolio snapshot vs broker positions. LiveRunner becomes an execution-only runner receiving decisions from its caller.

**Tech Stack:** Python 3.12, Pydantic v2 (frozen models), pytest, ABC pattern

**Spec:** `docs/superpowers/specs/2026-04-09-phase5-live-plumbing-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| CREATE | `trading_system/broker/__init__.py` | Re-exports for clean imports |
| CREATE | `trading_system/broker/base.py` | `Order`, `OrderResult`, `BrokerPosition`, `AccountSnapshot` schemas + `BaseBroker` ABC |
| CREATE | `trading_system/broker/mock.py` | `MockBrokerConfig`, `MockBroker` — in-memory broker for testing |
| CREATE | `trading_system/order_manager.py` | `ManagedOrder`, `OrderManager` — order lifecycle state machine |
| CREATE | `trading_system/reconciliation.py` | `PositionMismatch`, `ReconciliationResult`, `reconcile()` |
| CREATE | `scripts/reconcile.py` | Standalone CLI reconciliation script |
| MODIFY | `trading_system/contracts/trades.py:166-172` | Add `SUBMITTED`, `TIMED_OUT` to `OrderStatus`; update `PENDING` docstring |
| MODIFY | `pattern_engine/live.py` | Delete old classes, rewire `LiveRunner` |
| CREATE | `tests/unit/test_broker_base.py` | ABC contract tests + schema tests |
| CREATE | `tests/unit/test_broker_mock.py` | MockBroker behavioral tests |
| CREATE | `tests/unit/test_order_manager.py` | Order lifecycle, decision→order, batch, timeout |
| CREATE | `tests/unit/test_reconciliation.py` | Match, mismatch, unexpected, missing, tolerance |
| MODIFY | `tests/unit/test_live.py` | Update for new LiveRunner interfaces |
| CREATE | `tests/integration/test_phase5_gate.py` | 100-trade replay gate test |
| MODIFY | `CLAUDE.md` | Add broker package, update test count |

---

## Task 5.1: Extend OrderStatus + BaseBroker ABC & Schemas

### Context
The spec requires extending the existing `OrderStatus` enum (in `trading_system/contracts/trades.py`) with `SUBMITTED` and `TIMED_OUT`, and creating a new `trading_system/broker/` package with Pydantic schemas and a `BaseBroker` ABC. The schemas (`Order`, `OrderResult`, `BrokerPosition`, `AccountSnapshot`) are all frozen Pydantic models.

**Files:**
- Modify: `trading_system/contracts/trades.py:166-172` — extend `OrderStatus`
- Create: `trading_system/broker/__init__.py`
- Create: `trading_system/broker/base.py`
- Test: `tests/unit/test_broker_base.py`

---

- [ ] **Step 1: Write schema and ABC tests**

Create `tests/unit/test_broker_base.py`:

```python
"""Tests for broker schemas and BaseBroker ABC contract."""
import pytest
from datetime import datetime, timezone
from trading_system.contracts.trades import OrderStatus, OrderSide


class TestOrderStatusExtension:
    """Verify SUBMITTED and TIMED_OUT were added to OrderStatus."""

    def test_submitted_exists(self):
        assert OrderStatus.SUBMITTED.value == "SUBMITTED"

    def test_timed_out_exists(self):
        assert OrderStatus.TIMED_OUT.value == "TIMED_OUT"

    def test_pending_still_exists(self):
        assert OrderStatus.PENDING.value == "PENDING"

    def test_all_statuses_present(self):
        names = {s.name for s in OrderStatus}
        assert names == {
            "PENDING", "SUBMITTED", "FILLED", "PARTIAL",
            "CANCELLED", "REJECTED", "TIMED_OUT",
        }


class TestOrderSchema:
    """Test Order frozen model."""

    def test_create_market_order(self):
        from trading_system.broker.base import Order
        o = Order(
            order_id="abc-123",
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10.0,
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert o.order_type == "MARKET"
        assert o.limit_price is None
        assert o.notional is None

    def test_create_limit_order(self):
        from trading_system.broker.base import Order
        o = Order(
            order_id="abc-456",
            ticker="MSFT",
            side=OrderSide.SELL,
            quantity=5.0,
            order_type="LIMIT",
            limit_price=350.0,
            notional=1750.0,
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert o.order_type == "LIMIT"
        assert o.limit_price == 350.0

    def test_order_is_frozen(self):
        from trading_system.broker.base import Order
        o = Order(
            order_id="x", ticker="A", side=OrderSide.BUY,
            quantity=1.0, timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        with pytest.raises(Exception):
            o.quantity = 99.0


class TestOrderResultSchema:
    def test_filled_result(self):
        from trading_system.broker.base import OrderResult
        r = OrderResult(
            order_id="abc-123",
            ticker="AAPL",
            status=OrderStatus.FILLED,
            filled_quantity=10.0,
            fill_price=150.0,
            latency_ms=5.0,
            executed_at=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
        )
        assert r.error is None

    def test_rejected_result(self):
        from trading_system.broker.base import OrderResult
        r = OrderResult(
            order_id="abc-123",
            ticker="AAPL",
            status=OrderStatus.REJECTED,
            filled_quantity=0.0,
            fill_price=0.0,
            error="Insufficient funds",
        )
        assert r.error == "Insufficient funds"


class TestBrokerPositionSchema:
    def test_create(self):
        from trading_system.broker.base import BrokerPosition
        p = BrokerPosition(
            ticker="AAPL",
            quantity=100.0,
            avg_cost=150.0,
            current_value=16000.0,
            unrealized_pnl=1000.0,
        )
        assert p.ticker == "AAPL"


class TestAccountSnapshotSchema:
    def test_create_with_positions(self):
        from trading_system.broker.base import AccountSnapshot, BrokerPosition
        pos = BrokerPosition(
            ticker="AAPL", quantity=10, avg_cost=150,
            current_value=1600, unrealized_pnl=100,
        )
        snap = AccountSnapshot(
            total_value=11600, cash=10000, buying_power=10000,
            positions=(pos,),
        )
        assert len(snap.positions) == 1

    def test_empty_positions_default(self):
        from trading_system.broker.base import AccountSnapshot
        snap = AccountSnapshot(total_value=10000, cash=10000, buying_power=10000)
        assert snap.positions == ()


class TestBaseBrokerABC:
    def test_cannot_instantiate(self):
        from trading_system.broker.base import BaseBroker
        with pytest.raises(TypeError):
            BaseBroker()

    def test_concrete_subclass_works(self):
        from trading_system.broker.base import (
            BaseBroker, Order, OrderResult, BrokerPosition, AccountSnapshot,
        )

        class DummyBroker(BaseBroker):
            def submit_order(self, order: Order) -> OrderResult:
                return OrderResult(
                    order_id=order.order_id, ticker=order.ticker,
                    status=OrderStatus.FILLED, filled_quantity=order.quantity,
                    fill_price=100.0,
                )

            def cancel_order(self, order_id: str) -> bool:
                return True

            def get_positions(self) -> list[BrokerPosition]:
                return []

            def get_account(self) -> AccountSnapshot:
                return AccountSnapshot(
                    total_value=10000, cash=10000, buying_power=10000,
                )

            def is_connected(self) -> bool:
                return True

        broker = DummyBroker()
        assert broker.is_connected()
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_broker_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading_system.broker'`

- [ ] **Step 3: Extend OrderStatus enum**

In `trading_system/contracts/trades.py`, modify `OrderStatus` (lines 166-172):

```python
class OrderStatus(str, Enum):
    """Lifecycle state of an order."""
    PENDING = "PENDING"         # Created, not yet submitted to broker
    SUBMITTED = "SUBMITTED"     # Sent to broker, awaiting response
    FILLED = "FILLED"           # Fully executed
    PARTIAL = "PARTIAL"         # Partially filled (live trading only)
    CANCELLED = "CANCELLED"     # Cancelled before fill
    REJECTED = "REJECTED"       # Broker/system rejected the order
    TIMED_OUT = "TIMED_OUT"     # No broker response within deadline
```

- [ ] **Step 4: Create broker package**

Create `trading_system/broker/__init__.py`:

```python
"""Broker abstraction layer for order execution."""
from .base import BaseBroker, Order, OrderResult, BrokerPosition, AccountSnapshot

__all__ = [
    "BaseBroker",
    "Order",
    "OrderResult",
    "BrokerPosition",
    "AccountSnapshot",
]
```

Create `trading_system/broker/base.py`:

```python
"""BaseBroker ABC and order/position schemas."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from trading_system.contracts.trades import OrderSide, OrderStatus


# ─── Schemas ─────────────────────────────────────────────────────────────────


class Order(BaseModel):
    """Immutable order to submit to a broker."""
    model_config = {"frozen": True}

    order_id: str = Field(description="UUID assigned by OrderManager")
    ticker: str = Field(min_length=1, max_length=10)
    side: OrderSide
    quantity: float = Field(gt=0, description="Shares (fractional OK)")
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: Optional[float] = None
    notional: Optional[float] = Field(
        default=None, description="Dollar amount (audit trail)"
    )
    timestamp: datetime


class OrderResult(BaseModel):
    """Immutable result of a broker order submission."""
    model_config = {"frozen": True}

    order_id: str = Field(description="Matches Order.order_id")
    ticker: str
    status: OrderStatus
    filled_quantity: float = Field(ge=0)
    fill_price: float = Field(ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    executed_at: Optional[datetime] = None
    error: Optional[str] = None


class BrokerPosition(BaseModel):
    """Broker-reported position for a single ticker."""
    model_config = {"frozen": True}

    ticker: str
    quantity: float
    avg_cost: float
    current_value: float
    unrealized_pnl: float


class AccountSnapshot(BaseModel):
    """Broker-reported account summary."""
    model_config = {"frozen": True}

    total_value: float
    cash: float
    buying_power: float
    positions: tuple[BrokerPosition, ...] = ()


# ─── ABC ─────────────────────────────────────────────────────────────────────


class BaseBroker(ABC):
    """Abstract base class for broker adapters."""

    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult:
        """Submit an order and return the fill result."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True if successfully cancelled."""
        ...

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]:
        """Return all open positions from the broker."""
        ...

    @abstractmethod
    def get_account(self) -> AccountSnapshot:
        """Return current account snapshot."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if broker connection is healthy."""
        ...
```

- [ ] **Step 5: Run tests — all should pass**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_broker_base.py -v`
Expected: All 12 tests PASS

- [ ] **Step 6: Run full suite — check no regressions from OrderStatus change**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: 743+ tests pass. The OrderStatus extension is additive — existing code only references PENDING/FILLED/PARTIAL/CANCELLED/REJECTED, so no breakage expected.

- [ ] **Step 7: Commit**

```bash
git add trading_system/contracts/trades.py trading_system/broker/ tests/unit/test_broker_base.py
git commit -m "feat(phase5): T5.1 — BaseBroker ABC, schemas, OrderStatus extension"
```

---

## Task 5.2: Enhanced MockBroker

### Context
`MockBroker` implements `BaseBroker` with in-memory position tracking, configurable slippage, partial fills, fail tickers, and buying power enforcement. It replaces the old `MockBrokerAdapter` from `live.py`. Config via frozen `MockBrokerConfig`. Price injection via `set_prices()`.

**Files:**
- Create: `trading_system/broker/mock.py`
- Modify: `trading_system/broker/__init__.py` — add MockBroker, MockBrokerConfig exports
- Test: `tests/unit/test_broker_mock.py`

---

- [ ] **Step 1: Write MockBroker tests**

Create `tests/unit/test_broker_mock.py`:

```python
"""Tests for MockBroker — in-memory broker for testing."""
import pytest
from datetime import datetime, timezone

from trading_system.contracts.trades import OrderStatus, OrderSide
from trading_system.broker.base import Order, BaseBroker
from trading_system.broker.mock import MockBroker, MockBrokerConfig


def _order(
    ticker: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 10.0,
    order_id: str = "test-001",
) -> Order:
    return Order(
        order_id=order_id,
        ticker=ticker,
        side=side,
        quantity=quantity,
        timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )


class TestMockBrokerBasics:
    def test_is_base_broker(self):
        assert isinstance(MockBroker(), BaseBroker)

    def test_is_connected(self):
        assert MockBroker().is_connected()

    def test_default_cash(self):
        b = MockBroker()
        acct = b.get_account()
        assert acct.cash == 10_000.0
        assert acct.total_value == 10_000.0

    def test_custom_cash(self):
        b = MockBroker(MockBrokerConfig(initial_cash=50_000.0))
        assert b.get_account().cash == 50_000.0

    def test_empty_positions(self):
        assert MockBroker().get_positions() == []


class TestMockBrokerFills:
    def test_full_fill_buy(self):
        b = MockBroker()
        b.set_prices({"AAPL": 150.0})
        r = b.submit_order(_order(quantity=10.0))
        assert r.status == OrderStatus.FILLED
        assert r.filled_quantity == 10.0
        # slippage default 10 bps: 150 * 1.001 = 150.15
        assert r.fill_price == pytest.approx(150.15)

    def test_full_fill_sell(self):
        b = MockBroker()
        b.set_prices({"AAPL": 150.0})
        # Buy first to have a position
        b.submit_order(_order(quantity=10.0, order_id="buy-1"))
        r = b.submit_order(_order(side=OrderSide.SELL, quantity=10.0, order_id="sell-1"))
        assert r.status == OrderStatus.FILLED
        # slippage: 150 * (1 - 0.001) = 149.85
        assert r.fill_price == pytest.approx(149.85)

    def test_partial_fill(self):
        cfg = MockBrokerConfig(fill_fraction=0.5)
        b = MockBroker(cfg)
        b.set_prices({"AAPL": 100.0})
        r = b.submit_order(_order(quantity=10.0))
        assert r.status == OrderStatus.PARTIAL
        assert r.filled_quantity == 5.0

    def test_zero_slippage(self):
        cfg = MockBrokerConfig(slippage_bps=0.0)
        b = MockBroker(cfg)
        b.set_prices({"AAPL": 100.0})
        r = b.submit_order(_order(quantity=1.0))
        assert r.fill_price == 100.0


class TestMockBrokerRejections:
    def test_fail_ticker_rejected(self):
        cfg = MockBrokerConfig(fail_tickers=frozenset({"AAPL"}))
        b = MockBroker(cfg)
        b.set_prices({"AAPL": 100.0})
        r = b.submit_order(_order())
        assert r.status == OrderStatus.REJECTED
        assert r.filled_quantity == 0.0
        assert r.fill_price == 0.0
        assert r.error is not None

    def test_insufficient_funds_rejected(self):
        cfg = MockBrokerConfig(initial_cash=100.0, reject_when_insufficient=True)
        b = MockBroker(cfg)
        b.set_prices({"AAPL": 150.0})
        # 10 shares * 150 = 1500 > 100 cash
        r = b.submit_order(_order(quantity=10.0))
        assert r.status == OrderStatus.REJECTED

    def test_insufficient_funds_allowed_when_disabled(self):
        cfg = MockBrokerConfig(initial_cash=100.0, reject_when_insufficient=False)
        b = MockBroker(cfg)
        b.set_prices({"AAPL": 150.0})
        r = b.submit_order(_order(quantity=10.0))
        assert r.status == OrderStatus.FILLED


class TestMockBrokerPositionTracking:
    def test_buy_creates_position(self):
        b = MockBroker(MockBrokerConfig(slippage_bps=0.0))
        b.set_prices({"AAPL": 100.0})
        b.submit_order(_order(quantity=10.0))
        positions = b.get_positions()
        assert len(positions) == 1
        assert positions[0].ticker == "AAPL"
        assert positions[0].quantity == 10.0

    def test_sell_reduces_position(self):
        b = MockBroker(MockBrokerConfig(slippage_bps=0.0))
        b.set_prices({"AAPL": 100.0})
        b.submit_order(_order(quantity=10.0, order_id="b1"))
        b.submit_order(_order(side=OrderSide.SELL, quantity=3.0, order_id="s1"))
        positions = b.get_positions()
        assert positions[0].quantity == 7.0

    def test_sell_removes_empty_position(self):
        b = MockBroker(MockBrokerConfig(slippage_bps=0.0))
        b.set_prices({"AAPL": 100.0})
        b.submit_order(_order(quantity=10.0, order_id="b1"))
        b.submit_order(_order(side=OrderSide.SELL, quantity=10.0, order_id="s1"))
        assert b.get_positions() == []

    def test_cash_decreases_on_buy(self):
        b = MockBroker(MockBrokerConfig(slippage_bps=0.0, initial_cash=10_000.0))
        b.set_prices({"AAPL": 100.0})
        b.submit_order(_order(quantity=10.0))
        assert b.get_account().cash == pytest.approx(9_000.0)

    def test_cash_increases_on_sell(self):
        b = MockBroker(MockBrokerConfig(slippage_bps=0.0, initial_cash=10_000.0))
        b.set_prices({"AAPL": 100.0})
        b.submit_order(_order(quantity=10.0, order_id="b1"))
        b.submit_order(_order(side=OrderSide.SELL, quantity=10.0, order_id="s1"))
        assert b.get_account().cash == pytest.approx(10_000.0)

    def test_multiple_tickers(self):
        b = MockBroker(MockBrokerConfig(slippage_bps=0.0))
        b.set_prices({"AAPL": 100.0, "MSFT": 200.0})
        b.submit_order(_order(ticker="AAPL", quantity=5.0, order_id="1"))
        b.submit_order(_order(ticker="MSFT", quantity=3.0, order_id="2"))
        positions = b.get_positions()
        tickers = {p.ticker for p in positions}
        assert tickers == {"AAPL", "MSFT"}

    def test_account_total_value(self):
        b = MockBroker(MockBrokerConfig(slippage_bps=0.0, initial_cash=10_000.0))
        b.set_prices({"AAPL": 100.0})
        b.submit_order(_order(quantity=10.0))
        acct = b.get_account()
        # cash=9000, position=10*100=1000, total=10000
        assert acct.total_value == pytest.approx(10_000.0)


class TestMockBrokerIntrospection:
    def test_order_history(self):
        b = MockBroker(MockBrokerConfig(slippage_bps=0.0))
        b.set_prices({"AAPL": 100.0})
        b.submit_order(_order(order_id="o1"))
        b.submit_order(_order(order_id="o2"))
        assert len(b.order_history) == 2
        assert b.order_history[0][0].order_id == "o1"

    def test_latency_recorded(self):
        cfg = MockBrokerConfig(latency_ms=42.0, slippage_bps=0.0)
        b = MockBroker(cfg)
        b.set_prices({"AAPL": 100.0})
        r = b.submit_order(_order())
        assert r.latency_ms == 42.0


class TestMockBrokerCancel:
    def test_cancel_returns_false_unknown(self):
        b = MockBroker()
        assert b.cancel_order("nonexistent") is False
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_broker_mock.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading_system.broker.mock'`

- [ ] **Step 3: Implement MockBroker**

Create `trading_system/broker/mock.py`:

```python
"""MockBroker — in-memory broker for testing."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from trading_system.contracts.trades import OrderSide, OrderStatus
from .base import BaseBroker, Order, OrderResult, BrokerPosition, AccountSnapshot


class MockBrokerConfig(BaseModel):
    """Configuration for MockBroker behavior."""
    model_config = {"frozen": True}

    fill_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=10.0, ge=0.0)
    fail_tickers: frozenset[str] = frozenset()
    initial_cash: float = Field(default=10_000.0, gt=0.0)
    reject_when_insufficient: bool = True


class MockBroker(BaseBroker):
    """In-memory broker for testing. Tracks positions, cash, order history."""

    def __init__(self, config: MockBrokerConfig = MockBrokerConfig()) -> None:
        self._config = config
        self._cash: float = config.initial_cash
        self._positions: dict[str, _MockPosition] = {}
        self._prices: dict[str, float] = {}
        self._history: list[tuple[Order, OrderResult]] = []

    def set_prices(self, prices: dict[str, float]) -> None:
        """Inject current prices for position valuation and fill simulation."""
        self._prices.update(prices)

    # ── BaseBroker interface ──────────────────────────────────────────────

    def submit_order(self, order: Order) -> OrderResult:
        price = self._prices.get(order.ticker, 0.0)

        # Rejection: fail ticker
        if order.ticker in self._config.fail_tickers:
            result = OrderResult(
                order_id=order.order_id,
                ticker=order.ticker,
                status=OrderStatus.REJECTED,
                filled_quantity=0.0,
                fill_price=0.0,
                latency_ms=self._config.latency_ms,
                error=f"Ticker {order.ticker} is in fail list",
            )
            self._history.append((order, result))
            return result

        # Compute fill price with slippage
        slip = self._config.slippage_bps / 10_000
        if order.side == OrderSide.BUY:
            fill_price = price * (1 + slip)
        else:
            fill_price = price * (1 - slip)

        fill_qty = order.quantity * self._config.fill_fraction
        cost = fill_qty * fill_price

        # Rejection: insufficient funds (BUY only)
        if (
            order.side == OrderSide.BUY
            and self._config.reject_when_insufficient
            and cost > self._cash
        ):
            result = OrderResult(
                order_id=order.order_id,
                ticker=order.ticker,
                status=OrderStatus.REJECTED,
                filled_quantity=0.0,
                fill_price=0.0,
                latency_ms=self._config.latency_ms,
                error="Insufficient funds",
            )
            self._history.append((order, result))
            return result

        # Determine status
        if self._config.fill_fraction < 1.0:
            status = OrderStatus.PARTIAL
        else:
            status = OrderStatus.FILLED

        # Update positions and cash
        if order.side == OrderSide.BUY:
            self._cash -= cost
            pos = self._positions.get(order.ticker)
            if pos is None:
                self._positions[order.ticker] = _MockPosition(
                    quantity=fill_qty, avg_cost=fill_price,
                )
            else:
                # Weighted average cost
                total_qty = pos.quantity + fill_qty
                pos.avg_cost = (
                    (pos.avg_cost * pos.quantity + fill_price * fill_qty) / total_qty
                )
                pos.quantity = total_qty
        else:
            self._cash += cost
            pos = self._positions.get(order.ticker)
            if pos is not None:
                pos.quantity -= fill_qty
                if pos.quantity <= 0:
                    del self._positions[order.ticker]

        result = OrderResult(
            order_id=order.order_id,
            ticker=order.ticker,
            status=status,
            filled_quantity=fill_qty,
            fill_price=fill_price,
            latency_ms=self._config.latency_ms,
            executed_at=datetime.now(timezone.utc),
        )
        self._history.append((order, result))
        return result

    def cancel_order(self, order_id: str) -> bool:
        return False  # MockBroker doesn't track pending orders

    def get_positions(self) -> list[BrokerPosition]:
        result = []
        for ticker, pos in self._positions.items():
            price = self._prices.get(ticker, pos.avg_cost)
            current_value = pos.quantity * price
            result.append(BrokerPosition(
                ticker=ticker,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                current_value=current_value,
                unrealized_pnl=current_value - (pos.quantity * pos.avg_cost),
            ))
        return result

    def get_account(self) -> AccountSnapshot:
        positions = self.get_positions()
        invested = sum(p.current_value for p in positions)
        return AccountSnapshot(
            total_value=self._cash + invested,
            cash=self._cash,
            buying_power=self._cash,
            positions=tuple(positions),
        )

    def is_connected(self) -> bool:
        return True

    # ── Test introspection ────────────────────────────────────────────────

    @property
    def order_history(self) -> list[tuple[Order, OrderResult]]:
        return list(self._history)


class _MockPosition:
    """Mutable internal position tracker."""
    __slots__ = ("quantity", "avg_cost")

    def __init__(self, quantity: float, avg_cost: float) -> None:
        self.quantity = quantity
        self.avg_cost = avg_cost
```

- [ ] **Step 4: Update `__init__.py` exports**

Add to `trading_system/broker/__init__.py`:

```python
"""Broker abstraction layer for order execution."""
from .base import BaseBroker, Order, OrderResult, BrokerPosition, AccountSnapshot
from .mock import MockBroker, MockBrokerConfig

__all__ = [
    "BaseBroker",
    "Order",
    "OrderResult",
    "BrokerPosition",
    "AccountSnapshot",
    "MockBroker",
    "MockBrokerConfig",
]
```

- [ ] **Step 5: Run MockBroker tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_broker_mock.py -v`
Expected: All ~22 tests PASS

- [ ] **Step 6: Run full suite**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: All pass (no regressions)

- [ ] **Step 7: Commit**

```bash
git add trading_system/broker/mock.py trading_system/broker/__init__.py tests/unit/test_broker_mock.py
git commit -m "feat(phase5): T5.2 — MockBroker with position tracking, slippage, partial fills"
```

---

## Task 5.3: Order Manager

### Context
`OrderManager` is the bridge between `AllocationDecision` (PM output) and `Order` (broker input). It manages the order state machine: PENDING → SUBMITTED → terminal. One instance per trading day, stateful. `create_order_from_decision()` computes `quantity = capital_allocated / price`. `create_exit_order()` handles SELL exits. `submit()` calls broker and tracks `ManagedOrder`.

**Files:**
- Create: `trading_system/order_manager.py`
- Test: `tests/unit/test_order_manager.py`

---

- [ ] **Step 1: Write OrderManager tests**

Create `tests/unit/test_order_manager.py`:

```python
"""Tests for OrderManager — order lifecycle and decision→order translation."""
import pytest
from datetime import date, datetime, timezone

from trading_system.contracts.trades import OrderStatus, OrderSide
from trading_system.contracts.decisions import AllocationDecision, EvaluatorStatus
from trading_system.broker.base import Order
from trading_system.broker.mock import MockBroker, MockBrokerConfig
from trading_system.order_manager import OrderManager, ManagedOrder


def _decision(
    ticker: str = "AAPL",
    capital: float = 1500.0,
    rank: int = 1,
) -> AllocationDecision:
    return AllocationDecision(
        ticker=ticker,
        signal_date=date(2024, 1, 2),
        final_position_pct=0.05,
        evaluator_status=EvaluatorStatus.GREEN,
        capital_allocated=capital,
        rank_in_queue=rank,
        sector="Technology",
    )


def _broker(cash: float = 100_000.0) -> MockBroker:
    b = MockBroker(MockBrokerConfig(initial_cash=cash, slippage_bps=0.0))
    b.set_prices({"AAPL": 150.0, "MSFT": 300.0, "GOOG": 100.0})
    return b


class TestCreateOrderFromDecision:
    def test_creates_buy_order(self):
        om = OrderManager(broker=_broker())
        order = om.create_order_from_decision(_decision(), price=150.0)
        assert order.side == OrderSide.BUY
        assert order.ticker == "AAPL"
        assert order.quantity == pytest.approx(10.0)  # 1500 / 150

    def test_quantity_is_capital_over_price(self):
        om = OrderManager(broker=_broker())
        order = om.create_order_from_decision(
            _decision(capital=3000.0), price=100.0,
        )
        assert order.quantity == pytest.approx(30.0)

    def test_notional_set_for_audit(self):
        om = OrderManager(broker=_broker())
        order = om.create_order_from_decision(_decision(capital=1500.0), price=150.0)
        assert order.notional == 1500.0

    def test_order_id_is_unique(self):
        om = OrderManager(broker=_broker())
        o1 = om.create_order_from_decision(_decision(), price=150.0)
        o2 = om.create_order_from_decision(_decision(ticker="MSFT"), price=300.0)
        assert o1.order_id != o2.order_id


class TestCreateExitOrder:
    def test_creates_sell_order(self):
        om = OrderManager(broker=_broker())
        order = om.create_exit_order("AAPL", quantity=10.0, price=155.0)
        assert order.side == OrderSide.SELL
        assert order.ticker == "AAPL"
        assert order.quantity == 10.0

    def test_exit_order_id_unique(self):
        om = OrderManager(broker=_broker())
        o1 = om.create_exit_order("AAPL", quantity=10.0, price=155.0)
        o2 = om.create_exit_order("MSFT", quantity=5.0, price=305.0)
        assert o1.order_id != o2.order_id


class TestSubmit:
    def test_submit_returns_managed_order(self):
        om = OrderManager(broker=_broker())
        order = om.create_order_from_decision(_decision(), price=150.0)
        managed = om.submit(order)
        assert isinstance(managed, ManagedOrder)
        assert managed.status == OrderStatus.FILLED
        assert managed.result is not None

    def test_submit_tracks_in_orders_dict(self):
        om = OrderManager(broker=_broker())
        order = om.create_order_from_decision(_decision(), price=150.0)
        om.submit(order)
        assert order.order_id in om.orders

    def test_submit_rejected_order(self):
        broker = MockBroker(MockBrokerConfig(
            fail_tickers=frozenset({"AAPL"}), slippage_bps=0.0,
        ))
        broker.set_prices({"AAPL": 150.0})
        om = OrderManager(broker=broker)
        order = om.create_order_from_decision(_decision(), price=150.0)
        managed = om.submit(order)
        assert managed.status == OrderStatus.REJECTED


class TestSubmitBatch:
    def test_batch_submits_all(self):
        om = OrderManager(broker=_broker())
        orders = [
            om.create_order_from_decision(_decision(ticker="AAPL"), price=150.0),
            om.create_order_from_decision(_decision(ticker="MSFT", rank=2), price=300.0),
        ]
        results = om.submit_batch(orders)
        assert len(results) == 2
        assert all(m.status == OrderStatus.FILLED for m in results)

    def test_batch_one_failure_others_succeed(self):
        broker = MockBroker(MockBrokerConfig(
            fail_tickers=frozenset({"AAPL"}), slippage_bps=0.0,
        ))
        broker.set_prices({"AAPL": 150.0, "MSFT": 300.0})
        om = OrderManager(broker=broker)
        orders = [
            om.create_order_from_decision(_decision(ticker="AAPL"), price=150.0),
            om.create_order_from_decision(_decision(ticker="MSFT", rank=2), price=300.0),
        ]
        results = om.submit_batch(orders)
        assert results[0].status == OrderStatus.REJECTED
        assert results[1].status == OrderStatus.FILLED


class TestSummary:
    def test_summary_counts(self):
        om = OrderManager(broker=_broker())
        order = om.create_order_from_decision(_decision(), price=150.0)
        om.submit(order)
        s = om.summary()
        assert s["FILLED"] == 1

    def test_empty_summary(self):
        om = OrderManager(broker=_broker())
        s = om.summary()
        assert all(v == 0 for v in s.values())
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_order_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading_system.order_manager'`

- [ ] **Step 3: Implement OrderManager**

Create `trading_system/order_manager.py`:

```python
"""OrderManager — order lifecycle management."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from trading_system.contracts.trades import OrderSide, OrderStatus
from trading_system.contracts.decisions import AllocationDecision
from trading_system.broker.base import BaseBroker, Order, OrderResult


class ManagedOrder(BaseModel):
    """Order with lifecycle tracking."""
    model_config = {"frozen": True}

    order: Order
    status: OrderStatus
    result: Optional[OrderResult] = None
    created_at: datetime
    submitted_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class OrderManager:
    """Stateful order lifecycle manager. One instance per trading day."""

    def __init__(
        self,
        broker: BaseBroker,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not isinstance(broker, BaseBroker):
            raise RuntimeError(
                f"broker must be a BaseBroker, got {type(broker).__name__}"
            )
        self._broker = broker
        self._timeout_seconds = timeout_seconds
        self._orders: dict[str, ManagedOrder] = {}

    def create_order_from_decision(
        self,
        decision: AllocationDecision,
        price: float,
    ) -> Order:
        """AllocationDecision -> BUY Order.

        quantity = decision.capital_allocated / price.
        Side is always BUY (AllocationDecisions are entry-only from PM).
        """
        quantity = decision.capital_allocated / price
        return Order(
            order_id=str(uuid.uuid4()),
            ticker=decision.ticker,
            side=OrderSide.BUY,
            quantity=quantity,
            notional=decision.capital_allocated,
            timestamp=datetime.now(timezone.utc),
        )

    def create_exit_order(
        self,
        ticker: str,
        quantity: float,
        price: float,
    ) -> Order:
        """Create a SELL order for an exit (stop-loss, max-hold, drawdown halt)."""
        return Order(
            order_id=str(uuid.uuid4()),
            ticker=ticker,
            side=OrderSide.SELL,
            quantity=quantity,
            notional=quantity * price,
            timestamp=datetime.now(timezone.utc),
        )

    def submit(self, order: Order) -> ManagedOrder:
        """Submit to broker. PENDING -> SUBMITTED -> terminal state."""
        now = datetime.now(timezone.utc)

        # Submit to broker
        result = self._broker.submit_order(order)

        managed = ManagedOrder(
            order=order,
            status=result.status,
            result=result,
            created_at=order.timestamp,
            submitted_at=now,
            resolved_at=datetime.now(timezone.utc),
        )
        self._orders[order.order_id] = managed
        return managed

    def submit_batch(self, orders: list[Order]) -> list[ManagedOrder]:
        """Submit multiple orders sequentially. Returns all results."""
        return [self.submit(order) for order in orders]

    def cancel(self, order_id: str) -> ManagedOrder:
        """Cancel a pending/submitted order."""
        managed = self._orders.get(order_id)
        if managed is None:
            raise RuntimeError(f"Unknown order_id: {order_id}")
        success = self._broker.cancel_order(order_id)
        if success:
            updated = managed.model_copy(update={"status": OrderStatus.CANCELLED})
            self._orders[order_id] = updated
            return updated
        return managed

    @property
    def orders(self) -> dict[str, ManagedOrder]:
        """All managed orders keyed by order_id."""
        return dict(self._orders)

    def summary(self) -> dict[str, int]:
        """Count of orders by status."""
        counts: dict[str, int] = {s.value: 0 for s in OrderStatus}
        for managed in self._orders.values():
            counts[managed.status.value] += 1
        return counts
```

- [ ] **Step 4: Run tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_order_manager.py -v`
Expected: All ~14 tests PASS

- [ ] **Step 5: Run full suite**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add trading_system/order_manager.py tests/unit/test_order_manager.py
git commit -m "feat(phase5): T5.3 — OrderManager with decision-to-order translation"
```

---

## Task 5.4: Reconciliation

### Context
`reconcile()` compares `PortfolioSnapshot` positions (expected) against `BaseBroker` positions (actual). Three failure modes: quantity mismatch beyond tolerance, unexpected positions (in broker, not snapshot), missing positions (in snapshot, not broker). Returns `ReconciliationResult` with `passed` flag. Also a standalone CLI script.

**Files:**
- Create: `trading_system/reconciliation.py`
- Create: `scripts/reconcile.py`
- Test: `tests/unit/test_reconciliation.py`

---

- [ ] **Step 1: Write reconciliation tests**

Create `tests/unit/test_reconciliation.py`:

```python
"""Tests for reconciliation — portfolio vs broker position comparison."""
import pytest
from datetime import date, datetime, timezone

from trading_system.portfolio_state import PortfolioSnapshot, OpenPosition
from trading_system.broker.mock import MockBroker, MockBrokerConfig
from trading_system.broker.base import Order
from trading_system.contracts.trades import OrderSide
from trading_system.reconciliation import reconcile, ReconciliationResult, PositionMismatch


def _snapshot(positions: list[OpenPosition] | None = None) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of_date=date(2024, 1, 2),
        equity=100_000.0,
        cash=90_000.0,
        open_positions=tuple(positions or []),
    )


def _position(ticker: str, pct: float = 0.05) -> OpenPosition:
    return OpenPosition(
        ticker=ticker,
        sector="Technology",
        entry_date=date(2024, 1, 1),
        position_pct=pct,
        entry_price=100.0,
    )


def _broker_with_positions(
    holdings: dict[str, float],
    cash: float = 90_000.0,
) -> MockBroker:
    """Create a MockBroker with specific position quantities."""
    b = MockBroker(MockBrokerConfig(initial_cash=cash, slippage_bps=0.0))
    prices = {t: 100.0 for t in holdings}
    b.set_prices(prices)
    for ticker, qty in holdings.items():
        if qty > 0:
            b.submit_order(Order(
                order_id=f"setup-{ticker}",
                ticker=ticker,
                side=OrderSide.BUY,
                quantity=qty,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ))
    return b


class TestReconcileMatch:
    def test_empty_both(self):
        snap = _snapshot()
        broker = MockBroker(MockBrokerConfig(initial_cash=90_000.0))
        result = reconcile(snap, broker)
        assert result.passed is True
        assert result.mismatches == ()

    def test_matching_positions(self):
        # Snapshot expects 50 shares of AAPL (5% of 100k equity / 100 price)
        snap = _snapshot([_position("AAPL", pct=0.05)])
        broker = _broker_with_positions({"AAPL": 50.0})
        result = reconcile(snap, broker)
        assert result.passed is True

    def test_within_tolerance(self):
        # Expect 50, actual 51 → 2% diff, within 5% tolerance
        snap = _snapshot([_position("AAPL", pct=0.05)])
        broker = _broker_with_positions({"AAPL": 51.0})
        result = reconcile(snap, broker, tolerance_pct=0.05)
        assert result.passed is True


class TestReconcileMismatch:
    def test_quantity_mismatch(self):
        # Expect 50, actual 40 → 20% diff, beyond 5% tolerance
        snap = _snapshot([_position("AAPL", pct=0.05)])
        broker = _broker_with_positions({"AAPL": 40.0})
        result = reconcile(snap, broker)
        assert result.passed is False
        assert len(result.mismatches) == 1
        assert result.mismatches[0].ticker == "AAPL"

    def test_unexpected_position(self):
        # Broker has MSFT, snapshot does not
        snap = _snapshot()
        broker = _broker_with_positions({"MSFT": 10.0})
        result = reconcile(snap, broker)
        assert result.passed is False
        assert "MSFT" in result.unexpected_positions

    def test_missing_position(self):
        # Snapshot expects AAPL, broker has nothing
        snap = _snapshot([_position("AAPL")])
        broker = MockBroker(MockBrokerConfig(initial_cash=90_000.0))
        result = reconcile(snap, broker)
        assert result.passed is False
        assert "AAPL" in result.missing_positions


class TestReconcileTolerance:
    def test_exact_boundary_passes(self):
        # Exactly at 5% tolerance → should pass
        snap = _snapshot([_position("AAPL", pct=0.05)])
        broker = _broker_with_positions({"AAPL": 47.5})  # 5% below 50
        result = reconcile(snap, broker, tolerance_pct=0.05)
        assert result.passed is True

    def test_just_beyond_boundary_fails(self):
        snap = _snapshot([_position("AAPL", pct=0.05)])
        broker = _broker_with_positions({"AAPL": 47.0})  # 6% below 50
        result = reconcile(snap, broker, tolerance_pct=0.05)
        assert result.passed is False

    def test_custom_tolerance(self):
        snap = _snapshot([_position("AAPL", pct=0.05)])
        broker = _broker_with_positions({"AAPL": 40.0})  # 20% off
        result = reconcile(snap, broker, tolerance_pct=0.25)
        assert result.passed is True


class TestReconciliationResult:
    def test_counts(self):
        snap = _snapshot([_position("AAPL"), _position("MSFT")])
        broker = _broker_with_positions({"AAPL": 50.0, "MSFT": 50.0})
        result = reconcile(snap, broker)
        assert result.n_expected == 2
        assert result.n_actual == 2
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_reconciliation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading_system.reconciliation'`

- [ ] **Step 3: Implement reconciliation module**

Create `trading_system/reconciliation.py`:

```python
"""Position reconciliation — compare portfolio snapshot vs broker state."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel

from trading_system.portfolio_state import PortfolioSnapshot
from trading_system.broker.base import BaseBroker


class PositionMismatch(BaseModel):
    """A single position quantity discrepancy."""
    model_config = {"frozen": True}

    ticker: str
    expected_quantity: float
    actual_quantity: float
    delta: float
    delta_pct: float


class ReconciliationResult(BaseModel):
    """Result of comparing portfolio snapshot vs broker positions."""
    model_config = {"frozen": True}

    as_of: datetime
    n_expected: int
    n_actual: int
    mismatches: tuple[PositionMismatch, ...] = ()
    unexpected_positions: tuple[str, ...] = ()
    missing_positions: tuple[str, ...] = ()
    passed: bool


def reconcile(
    snapshot: PortfolioSnapshot,
    broker: BaseBroker,
    tolerance_pct: float = 0.05,
) -> ReconciliationResult:
    """Compare PortfolioSnapshot positions against broker positions.

    A position passes if abs(expected - actual) / expected <= tolerance_pct.
    Missing or unexpected positions always fail.
    """
    # Build expected quantities from snapshot.
    # Known limitation: derives shares as position_pct * equity / entry_price,
    # which is an approximation. In Phase 8 with real slippage/partial fills,
    # OpenPosition should gain a `quantity` field for exact tracking.
    # For Phase 5 (mock broker, deterministic fills), this approximation is exact.
    expected: dict[str, float] = {}
    for pos in snapshot.open_positions:
        expected[pos.ticker] = (pos.position_pct * snapshot.equity) / pos.entry_price

    # Build actual quantities from broker
    broker_positions = broker.get_positions()
    actual: dict[str, float] = {p.ticker: p.quantity for p in broker_positions}

    expected_tickers = set(expected)
    actual_tickers = set(actual)

    missing = tuple(sorted(expected_tickers - actual_tickers))
    unexpected = tuple(sorted(actual_tickers - expected_tickers))

    mismatches: list[PositionMismatch] = []
    for ticker in expected_tickers & actual_tickers:
        exp_qty = expected[ticker]
        act_qty = actual[ticker]
        delta = act_qty - exp_qty
        delta_pct = abs(delta / exp_qty) if exp_qty != 0 else float("inf")
        if delta_pct > tolerance_pct:
            mismatches.append(PositionMismatch(
                ticker=ticker,
                expected_quantity=exp_qty,
                actual_quantity=act_qty,
                delta=delta,
                delta_pct=delta_pct,
            ))

    passed = len(mismatches) == 0 and len(missing) == 0 and len(unexpected) == 0

    return ReconciliationResult(
        as_of=datetime.now(timezone.utc),
        n_expected=len(expected),
        n_actual=len(actual),
        mismatches=tuple(mismatches),
        unexpected_positions=unexpected,
        missing_positions=missing,
        passed=passed,
    )
```

- [ ] **Step 4: Run tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_reconciliation.py -v`
Expected: All ~10 tests PASS

- [ ] **Step 5: Create standalone reconciliation script**

Create `scripts/reconcile.py`:

```python
"""Standalone reconciliation CLI — compare SharedState checkpoint vs broker."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from trading_system.contracts.state import SharedState
from trading_system.portfolio_state import PortfolioSnapshot, OpenPosition
from trading_system.broker.mock import MockBroker, MockBrokerConfig
from trading_system.reconciliation import reconcile


def _snapshot_from_shared_state(state: SharedState) -> PortfolioSnapshot:
    """Build PortfolioSnapshot from SharedState for reconciliation.

    NOTE: position_pct and entry_price are placeholders. This script produces
    meaningful results only when SharedState carries actual sizing data
    (Phase 8 orchestrator responsibility). For Phase 5, use the reconcile()
    function directly with a real PortfolioSnapshot built from walk-forward output.
    """
    positions = []
    for ticker in state.positions.open_tickers:
        sector = state.positions.ticker_sectors.get(ticker, "Unknown")
        positions.append(OpenPosition(
            ticker=ticker,
            sector=sector,
            entry_date=state.trading_date,
            position_pct=0.05,  # Placeholder — Phase 8 will provide actual sizing
            entry_price=100.0,  # Placeholder — Phase 8 will provide actual price
        ))
    return PortfolioSnapshot(
        as_of_date=state.trading_date,
        equity=state.total_equity,
        cash=state.equity.cash,
        open_positions=tuple(positions),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile positions vs broker")
    parser.add_argument("checkpoint", help="Path to SharedState JSON checkpoint")
    parser.add_argument(
        "--tolerance", type=float, default=0.05,
        help="Tolerance percentage (default: 0.05 = 5%%)",
    )
    args = parser.parse_args()

    with open(args.checkpoint, "r") as f:
        data = json.load(f)
    state = SharedState.model_validate(data)

    snapshot = _snapshot_from_shared_state(state)
    broker = MockBroker()  # Real broker in Phase 8

    result = reconcile(snapshot, broker, tolerance_pct=args.tolerance)

    print(f"Reconciliation as of {result.as_of}")
    print(f"Expected positions: {result.n_expected}")
    print(f"Actual positions:   {result.n_actual}")

    if result.mismatches:
        print("\nMISMATCHES:")
        for m in result.mismatches:
            print(f"  {m.ticker}: expected={m.expected_quantity:.2f}, "
                  f"actual={m.actual_quantity:.2f}, delta={m.delta:+.2f} "
                  f"({m.delta_pct:.1%})")

    if result.missing_positions:
        print(f"\nMISSING (in snapshot, not broker): {', '.join(result.missing_positions)}")

    if result.unexpected_positions:
        print(f"\nUNEXPECTED (in broker, not snapshot): {', '.join(result.unexpected_positions)}")

    if result.passed:
        print("\nRESULT: PASS")
    else:
        print("\nRESULT: FAIL")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run full suite**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add trading_system/reconciliation.py scripts/reconcile.py tests/unit/test_reconciliation.py
git commit -m "feat(phase5): T5.4 — reconciliation module + CLI script"
```

---

## Task 5.5: LiveRunner Rewire

### Context
This is the most complex task. Delete old `BaseBrokerAdapter`, `Order`, `OrderResult`, `MockBrokerAdapter` from `pattern_engine/live.py`. Rewire `LiveRunner` to:
- Take `BaseBroker` + `OrderManager` via DI
- Accept `entry_decisions: list[AllocationDecision]` + `exit_tickers` + `PortfolioSnapshot` in `run()`
- Run pre-flight reconciliation
- Create and submit orders via OrderManager
- Return `list[ManagedOrder]`

All 17 existing tests in `test_live.py` must be rewritten for the new interface.

**Files:**
- Modify: `pattern_engine/live.py` — delete old classes, rewire LiveRunner
- Modify: `tests/unit/test_live.py` — rewrite for new interfaces

---

- [ ] **Step 1: Read current `live.py` and `test_live.py` in full**

Read `pattern_engine/live.py` (full file) and `tests/unit/test_live.py` (full file) to understand all code paths before modifying.

- [ ] **Step 2: Write new LiveRunner tests**

Rewrite `tests/unit/test_live.py` entirely. The new tests should cover:

1. Constructor validation (matcher type check, broker type check)
2. Halt check — `is_halted=True` returns empty, no orders
3. Reconciliation pre-flight — fails → returns empty
4. Reconciliation pre-flight — passes → proceeds
5. Reconciliation opt-out (`reconcile_on_start=False`)
6. Entry decisions → BUY orders via OrderManager
7. Exit tickers → SELL orders via OrderManager
8. Exits submitted before entries
9. Empty decisions → empty results
10. Config drift warning (logs but doesn't halt)

```python
"""Tests for LiveRunner — rewired for Phase 5."""
import logging
import pytest
from datetime import date, datetime, timezone
from unittest.mock import patch

from pattern_engine.matcher import PatternMatcher
from pattern_engine.config import EngineConfig
from pattern_engine.contracts.state import EngineState
from trading_system.contracts.state import SharedState
from trading_system.contracts.decisions import AllocationDecision, EvaluatorStatus
from trading_system.contracts.trades import OrderStatus, OrderSide
from trading_system.portfolio_state import PortfolioSnapshot, OpenPosition
from trading_system.broker.mock import MockBroker, MockBrokerConfig
from trading_system.order_manager import OrderManager, ManagedOrder
from pattern_engine.live import LiveRunner


def _decision(ticker: str = "AAPL", capital: float = 1500.0, rank: int = 1):
    return AllocationDecision(
        ticker=ticker,
        signal_date=date(2024, 1, 2),
        final_position_pct=0.05,
        evaluator_status=EvaluatorStatus.GREEN,
        capital_allocated=capital,
        rank_in_queue=rank,
        sector="Technology",
    )


def _snapshot(positions=None):
    return PortfolioSnapshot(
        as_of_date=date(2024, 1, 2),
        equity=100_000.0,
        cash=90_000.0,
        open_positions=tuple(positions or []),
    )


def _make_runner(
    cash: float = 100_000.0,
    reconcile_on_start: bool = False,
    prices: dict[str, float] | None = None,
):
    """Create a LiveRunner with MockBroker for testing."""
    import numpy as np
    import pandas as pd

    config = EngineConfig()
    matcher = PatternMatcher(config)

    # Fit matcher with minimal synthetic data
    n = 200
    rng = np.random.default_rng(42)
    cols = [f"feat_{i}" for i in range(8)]
    train = pd.DataFrame(rng.standard_normal((n, 8)), columns=cols)
    train["fwd_7d_up"] = rng.integers(0, 2, size=n)
    train["ticker"] = "TRAIN"
    train["date"] = pd.date_range("2020-01-01", periods=n)
    matcher.fit(train)

    state = SharedState.initial(cash, date(2024, 1, 2))
    broker = MockBroker(MockBrokerConfig(initial_cash=cash, slippage_bps=0.0))
    if prices:
        broker.set_prices(prices)
    else:
        broker.set_prices({"AAPL": 150.0, "MSFT": 300.0, "GOOG": 100.0})
    om = OrderManager(broker=broker)

    runner = LiveRunner(
        matcher=matcher,
        shared_state=state,
        broker=broker,
        order_manager=om,
        reconcile_on_start=reconcile_on_start,
    )
    return runner, broker, om


class TestLiveRunnerConstructor:
    def test_invalid_matcher_raises(self):
        broker = MockBroker()
        om = OrderManager(broker=broker)
        state = SharedState.initial(100_000, date(2024, 1, 2))
        with pytest.raises(RuntimeError):
            LiveRunner(
                matcher="not_a_matcher",
                shared_state=state,
                broker=broker,
                order_manager=om,
            )

    def test_invalid_broker_raises(self):
        import numpy as np
        import pandas as pd
        config = EngineConfig()
        m = PatternMatcher(config)
        n = 50
        rng = np.random.default_rng(42)
        cols = [f"feat_{i}" for i in range(8)]
        train = pd.DataFrame(rng.standard_normal((n, 8)), columns=cols)
        train["fwd_7d_up"] = rng.integers(0, 2, size=n)
        train["ticker"] = "TRAIN"
        train["date"] = pd.date_range("2020-01-01", periods=n)
        m.fit(train)

        state = SharedState.initial(100_000, date(2024, 1, 2))
        with pytest.raises(RuntimeError):
            LiveRunner(
                matcher=m,
                shared_state=state,
                broker="not_a_broker",
                order_manager=OrderManager(broker=MockBroker()),
            )


class TestLiveRunnerHalt:
    def test_halted_returns_empty(self):
        runner, broker, om = _make_runner()
        halted = runner._shared_state.model_copy(
            update={"equity": runner._shared_state.equity.model_copy(
                update={"total_equity": 0.0}
            )}
        )
        # Force halt via command
        from trading_system.contracts.state import SystemCommand
        halted = runner._shared_state.model_copy(
            update={"command_queue": (SystemCommand.HALT,)}
        )
        runner._shared_state = halted
        results = runner.run(
            entry_decisions=[_decision()],
            exit_tickers=[],
            snapshot=_snapshot(),
            prices={"AAPL": 150.0},
        )
        assert results == []


class TestLiveRunnerReconciliation:
    def test_reconcile_failure_returns_empty(self):
        runner, broker, om = _make_runner(reconcile_on_start=True)
        # Snapshot expects AAPL position but broker has none → fail
        snap = _snapshot([OpenPosition(
            ticker="AAPL", sector="Tech", entry_date=date(2024, 1, 1),
            position_pct=0.05, entry_price=100.0,
        )])
        results = runner.run(
            entry_decisions=[_decision()],
            exit_tickers=[],
            snapshot=snap,
            prices={"AAPL": 150.0},
        )
        assert results == []

    def test_reconcile_off_proceeds(self):
        runner, broker, om = _make_runner(reconcile_on_start=False)
        snap = _snapshot()
        results = runner.run(
            entry_decisions=[_decision()],
            exit_tickers=[],
            snapshot=snap,
            prices={"AAPL": 150.0},
        )
        assert len(results) == 1
        assert results[0].status == OrderStatus.FILLED


class TestLiveRunnerOrders:
    def test_entry_decision_creates_buy(self):
        runner, broker, om = _make_runner()
        results = runner.run(
            entry_decisions=[_decision()],
            exit_tickers=[],
            snapshot=_snapshot(),
            prices={"AAPL": 150.0},
        )
        assert len(results) == 1
        assert results[0].order.side == OrderSide.BUY
        assert results[0].order.ticker == "AAPL"
        assert results[0].order.quantity == pytest.approx(10.0)  # 1500/150

    def test_exit_ticker_creates_sell(self):
        runner, broker, om = _make_runner()
        # First buy AAPL so broker has position
        from trading_system.broker.base import Order as BrokerOrder
        broker.submit_order(BrokerOrder(
            order_id="setup", ticker="AAPL", side=OrderSide.BUY,
            quantity=10.0, timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ))
        results = runner.run(
            entry_decisions=[],
            exit_tickers=[("AAPL", 10.0, 155.0)],
            snapshot=_snapshot(),
        )
        assert len(results) == 1
        assert results[0].order.side == OrderSide.SELL

    def test_exits_before_entries(self):
        runner, broker, om = _make_runner()
        # Buy AAPL for exit
        from trading_system.broker.base import Order as BrokerOrder
        broker.submit_order(BrokerOrder(
            order_id="setup", ticker="AAPL", side=OrderSide.BUY,
            quantity=10.0, timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ))
        results = runner.run(
            entry_decisions=[_decision(ticker="MSFT", capital=3000.0)],
            exit_tickers=[("AAPL", 10.0, 155.0)],
            snapshot=_snapshot(),
            prices={"MSFT": 300.0},
        )
        assert len(results) == 2
        # Exit (SELL) should be first
        assert results[0].order.side == OrderSide.SELL
        assert results[1].order.side == OrderSide.BUY

    def test_empty_decisions_empty_results(self):
        runner, broker, om = _make_runner()
        results = runner.run(
            entry_decisions=[],
            exit_tickers=[],
            snapshot=_snapshot(),
        )
        assert results == []

    def test_multiple_entries(self):
        runner, broker, om = _make_runner()
        results = runner.run(
            entry_decisions=[
                _decision(ticker="AAPL", rank=1),
                _decision(ticker="MSFT", capital=3000.0, rank=2),
            ],
            exit_tickers=[],
            snapshot=_snapshot(),
            prices={"AAPL": 150.0, "MSFT": 300.0},
        )
        assert len(results) == 2
        assert all(m.status == OrderStatus.FILLED for m in results)


class TestLiveRunnerConfigDrift:
    def test_config_drift_logs_warning(self, caplog):
        runner, broker, om = _make_runner()
        # Set mismatched engine state
        engine_state = EngineState.from_fitted(
            scaler=runner._matcher._scaler,
            matcher=runner._matcher,
            feature_cols=runner._matcher._feature_cols,
            config=EngineConfig(),
            feature_set_name="test",
        )
        # Tamper the hash
        runner._engine_state = engine_state.model_copy(
            update={"config_hash": "0" * 64}
        )
        with caplog.at_level(logging.WARNING):
            runner.run(
                entry_decisions=[_decision()],
                exit_tickers=[],
                snapshot=_snapshot(),
                prices={"AAPL": 150.0},
            )
        assert any("drift" in r.message.lower() or "mismatch" in r.message.lower()
                    for r in caplog.records)
```

Note: The import `from trading_system.broker.base import Order as BrokerOrder` is needed in one test. The existing `Order` import alias should be checked during implementation.

- [ ] **Step 3: Run new tests — expect failures from old LiveRunner interface**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_live.py -v`
Expected: Multiple failures — LiveRunner constructor doesn't accept `order_manager` yet

- [ ] **Step 4: Rewrite LiveRunner in `live.py`**

Delete the old `Order`, `OrderResult`, `BaseBrokerAdapter`, `MockBrokerAdapter` classes from `live.py`. Rewrite `LiveRunner`:

```python
"""LiveRunner — execution-only runner for live trading."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from pattern_engine.matcher import PatternMatcher
from pattern_engine.contracts.state import EngineState
from trading_system.contracts.state import SharedState
from trading_system.contracts.decisions import AllocationDecision
from trading_system.contracts.trades import OrderSide
from trading_system.portfolio_state import PortfolioSnapshot
from trading_system.broker.base import BaseBroker
from trading_system.order_manager import OrderManager, ManagedOrder
from trading_system.reconciliation import reconcile

log = logging.getLogger(__name__)


class LiveRunner:
    """Execution-only runner. Receives decisions, submits orders via OrderManager.

    The caller (daily orchestrator, Phase 8) runs the full pipeline upstream:
    matcher → PM → risk engine → position sizer → AllocationDecisions.
    LiveRunner receives decisions and exit tickers, then executes them.
    """

    def __init__(
        self,
        matcher: PatternMatcher,
        shared_state: SharedState,
        broker: BaseBroker,
        order_manager: OrderManager,
        engine_state: Optional[EngineState] = None,
        reconcile_on_start: bool = True,
    ) -> None:
        if not isinstance(matcher, PatternMatcher):
            raise RuntimeError(
                f"matcher must be a PatternMatcher, got {type(matcher).__name__}"
            )
        if not isinstance(broker, BaseBroker):
            raise RuntimeError(
                f"broker must be a BaseBroker, got {type(broker).__name__}"
            )
        self._matcher = matcher
        self._shared_state = shared_state
        self._broker = broker
        self._order_manager = order_manager
        self._engine_state = engine_state
        self._reconcile_on_start = reconcile_on_start

    def run(
        self,
        entry_decisions: list[AllocationDecision],
        exit_tickers: list[tuple[str, float, float]],
        snapshot: PortfolioSnapshot,
        prices: dict[str, float] | None = None,
    ) -> list[ManagedOrder]:
        """Execute entry and exit orders.

        Args:
            entry_decisions: BUY allocations from upstream pipeline.
            exit_tickers: List of (ticker, quantity, price) for exits.
            snapshot: Current portfolio snapshot for reconciliation.
            prices: Current prices by ticker (required for entry orders).

        Returns:
            List of ManagedOrder results (exits first, then entries).
        """
        # 1. Pre-flight reconciliation (spec flow step 1)
        if self._reconcile_on_start:
            recon_result = reconcile(snapshot, self._broker)
            if not recon_result.passed:
                log.error(
                    "Reconciliation failed: %d mismatches, %d missing, %d unexpected",
                    len(recon_result.mismatches),
                    len(recon_result.missing_positions),
                    len(recon_result.unexpected_positions),
                )
                return []

        # 2. Config hash drift check (warn-only)
        if self._engine_state is not None:
            current_hash = hashlib.sha256(
                json.dumps(self._matcher._config.model_dump(), sort_keys=True).encode()
            ).hexdigest()
            if current_hash != self._engine_state.config_hash:
                log.warning(
                    "Config drift detected: engine_state hash=%s, current=%s",
                    self._engine_state.config_hash,
                    current_hash,
                )

        # 3. Halt check
        if self._shared_state.is_halted:
            log.warning("SharedState is halted — skipping all orders")
            return []

        # 4. Create exit orders
        exit_orders = [
            self._order_manager.create_exit_order(ticker, quantity, price)
            for ticker, quantity, price in exit_tickers
        ]

        # 5. Create entry orders (prices dict required for new positions)
        entry_orders = []
        for decision in entry_decisions:
            price = (prices or {}).get(decision.ticker)
            if price is not None and price > 0:
                entry_orders.append(
                    self._order_manager.create_order_from_decision(decision, price)
                )
            else:
                log.warning("No price for %s — skipping entry order", decision.ticker)

        # 6. Submit all orders (exits first, then entries)
        all_orders = exit_orders + entry_orders
        if not all_orders:
            return []

        return self._order_manager.submit_batch(all_orders)
```

- [ ] **Step 5: Run LiveRunner tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_live.py -v`
Expected: All new tests PASS

- [ ] **Step 6: Run full suite — check for regressions**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: All pass. Any other test importing old `Order`/`OrderResult`/`BaseBrokerAdapter`/`MockBrokerAdapter` from `live.py` will need import path updates.

- [ ] **Step 7: Commit**

```bash
git add pattern_engine/live.py tests/unit/test_live.py
git commit -m "feat(phase5): T5.5 — LiveRunner rewire with OrderManager + reconciliation"
```

---

## Task 5.6: Integration Gate Test

### Context
The gate test replays 100 trades from `results/backtest_trades.csv` through OrderManager → MockBroker. Verifies: all 100 fill, quantities match sizing, cash/positions reconcile. Marked `@pytest.mark.slow`.

**Files:**
- Create: `tests/integration/test_phase5_gate.py`

---

- [ ] **Step 1: Write gate test**

Create `tests/integration/test_phase5_gate.py`:

```python
"""Phase 5 gate test — 100-trade replay through OrderManager → MockBroker."""
import csv
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from trading_system.contracts.trades import OrderStatus, OrderSide
from trading_system.contracts.decisions import AllocationDecision, EvaluatorStatus
from trading_system.broker.mock import MockBroker, MockBrokerConfig
from trading_system.order_manager import OrderManager
from trading_system.portfolio_state import PortfolioSnapshot
from trading_system.reconciliation import reconcile


TRADES_FILE = Path("results/backtest_trades.csv")


def _load_trades(n: int = 100) -> list[dict]:
    """Load first n trades from backtest results."""
    if not TRADES_FILE.exists():
        pytest.skip(f"{TRADES_FILE} not found")
    trades = []
    with open(TRADES_FILE, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            trades.append(row)
    if len(trades) < n:
        pytest.skip(f"Only {len(trades)} trades available, need {n}")
    return trades


@pytest.mark.slow
class TestPhase5Gate:
    """G1: Mock broker round-trip parity for 100 trades."""

    def test_g1_100_trade_replay(self):
        trades = _load_trades(100)
        broker = MockBroker(MockBrokerConfig(
            initial_cash=100_000.0,
            slippage_bps=0.0,
            fill_fraction=1.0,
        ))
        om = OrderManager(broker=broker)

        # Set up prices from trade data
        prices = {}
        for t in trades:
            ticker = t.get("ticker", t.get("Ticker", ""))
            price = float(t.get("entry_price", t.get("Entry Price", "100.0")))
            prices[ticker] = price
        broker.set_prices(prices)

        filled_count = 0
        for t in trades:
            ticker = t.get("ticker", t.get("Ticker", ""))
            price = float(t.get("entry_price", t.get("Entry Price", "100.0")))
            # Use 5% of initial capital as allocation
            capital = 5000.0

            decision = AllocationDecision(
                ticker=ticker,
                signal_date=date(2024, 1, 2),
                final_position_pct=0.05,
                evaluator_status=EvaluatorStatus.GREEN,
                capital_allocated=capital,
                rank_in_queue=1,
                sector=t.get("sector", t.get("Sector", "Unknown")),
            )
            order = om.create_order_from_decision(decision, price)
            managed = om.submit(order)

            if managed.status == OrderStatus.FILLED:
                filled_count += 1
                # Verify quantity matches sizing
                expected_qty = capital / price
                assert managed.result.filled_quantity == pytest.approx(expected_qty)

        assert filled_count == 100, f"Expected 100 fills, got {filled_count}"

        # Verify order manager tracked everything
        summary = om.summary()
        assert summary["FILLED"] == 100

    def test_g2_30_day_reconciliation(self):
        """G2: Reconciliation passes after 30 consecutive trading days."""
        from trading_system.portfolio_state import PortfolioSnapshot, OpenPosition
        trades = _load_trades(100)

        broker = MockBroker(MockBrokerConfig(
            initial_cash=1_000_000.0,
            slippage_bps=0.0,
            fill_fraction=1.0,
        ))
        om = OrderManager(broker=broker)

        # Set up prices
        prices = {}
        for t in trades:
            ticker = t.get("ticker", t.get("Ticker", ""))
            price = float(t.get("entry_price", t.get("Entry Price", "100.0")))
            prices[ticker] = price
        broker.set_prices(prices)

        # Simulate 30 days, submitting trades and reconciling each day
        trades_per_day = max(1, len(trades) // 30)
        expected_positions: dict[str, tuple[float, float]] = {}  # ticker -> (qty, price)

        for day in range(30):
            day_start = day * trades_per_day
            day_end = min(day_start + trades_per_day, len(trades))
            day_trades = trades[day_start:day_end]

            for t in day_trades:
                ticker = t.get("ticker", t.get("Ticker", ""))
                price = float(t.get("entry_price", t.get("Entry Price", "100.0")))
                capital = 5000.0
                qty = capital / price

                decision = AllocationDecision(
                    ticker=ticker,
                    signal_date=date(2024, 1, 2),
                    final_position_pct=0.05,
                    evaluator_status=EvaluatorStatus.GREEN,
                    capital_allocated=capital,
                    rank_in_queue=1,
                    sector=t.get("sector", t.get("Sector", "Unknown")),
                )
                order = om.create_order_from_decision(decision, price)
                managed = om.submit(order)
                if managed.status == OrderStatus.FILLED:
                    old_qty = expected_positions.get(ticker, (0.0, price))[0]
                    expected_positions[ticker] = (old_qty + qty, price)

            # Build snapshot from expected positions and reconcile
            open_positions = []
            equity = broker.get_account().total_value
            for ticker, (qty, price) in expected_positions.items():
                pct = (qty * price) / equity if equity > 0 else 0.0
                open_positions.append(OpenPosition(
                    ticker=ticker, sector="Unknown",
                    entry_date=date(2024, 1, 2),
                    position_pct=pct, entry_price=price,
                ))
            snap = PortfolioSnapshot(
                as_of_date=date(2024, 1, 2),
                equity=equity,
                cash=broker.get_account().cash,
                open_positions=tuple(open_positions),
            )
            result = reconcile(snap, broker)
            assert result.passed, (
                f"Day {day+1}: {len(result.mismatches)} mismatches, "
                f"{len(result.missing_positions)} missing, "
                f"{len(result.unexpected_positions)} unexpected"
            )

    def test_g3_pipeline_under_3_min(self):
        """G3: Pipeline < 3 min."""
        trades = _load_trades(100)
        broker = MockBroker(MockBrokerConfig(
            initial_cash=1_000_000.0,
            slippage_bps=0.0,
        ))
        om = OrderManager(broker=broker)

        prices = {}
        for t in trades:
            ticker = t.get("ticker", t.get("Ticker", ""))
            price = float(t.get("entry_price", t.get("Entry Price", "100.0")))
            prices[ticker] = price
        broker.set_prices(prices)

        start = time.perf_counter()
        for t in trades:
            ticker = t.get("ticker", t.get("Ticker", ""))
            price = float(t.get("entry_price", t.get("Entry Price", "100.0")))
            decision = AllocationDecision(
                ticker=ticker,
                signal_date=date(2024, 1, 2),
                final_position_pct=0.05,
                evaluator_status=EvaluatorStatus.GREEN,
                capital_allocated=5000.0,
                rank_in_queue=1,
                sector=t.get("sector", t.get("Sector", "Unknown")),
            )
            order = om.create_order_from_decision(decision, price)
            om.submit(order)
        elapsed = time.perf_counter() - start

        assert elapsed < 180, f"Pipeline took {elapsed:.1f}s, limit is 180s"
```

- [ ] **Step 2: Run gate test**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/integration/test_phase5_gate.py -v`
Expected: PASS (or skip if `results/backtest_trades.csv` not available)

- [ ] **Step 3: Run full suite including slow tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q`
Expected: All pass (743 existing + ~60-80 new = ~810+ total)

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_phase5_gate.py
git commit -m "feat(phase5): T5.6 — integration gate test (100-trade replay)"
```

---

## Task 5.7: CLAUDE.md Update + Final Verification

### Context
Update CLAUDE.md to reflect new broker package, order_manager, and updated test count.

**Files:**
- Modify: `CLAUDE.md`

---

- [ ] **Step 1: Run full suite and count tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Record the exact test count.

- [ ] **Step 2: Update CLAUDE.md**

Add to the `trading_system/` section in Codebases:
```
  - `broker/`: BaseBroker ABC, Order/OrderResult schemas, MockBroker (Phase 5)
  - `order_manager.py`: OrderManager — AllocationDecision→Order lifecycle (Phase 5)
  - `reconciliation.py`: Position reconciliation vs broker (Phase 5)
```

Update the test count in Critical Rules from `743` to the actual count.

Update Current Phase to reflect Phase 5 status.

- [ ] **Step 3: Run full suite one final time**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Phase 5 broker/order_manager/reconciliation"
```

---

## Dependency Graph

```
T5.1 (BaseBroker ABC + schemas)
  │
  ├──→ T5.2 (MockBroker) ──→ T5.3 (OrderManager) ──→ T5.5 (LiveRunner rewire)
  │                              │                         │
  │                              └──→ T5.4 (Reconciliation)┘
  │                                                        │
  │                                                        ├──→ T5.6 (Gate test)
  │                                                        │
  └────────────────────────────────────────────────────────→ T5.7 (CLAUDE.md)
```

Tasks must be executed in order: T5.1 → T5.2 → T5.3 → T5.4 → T5.5 → T5.6 → T5.7.
