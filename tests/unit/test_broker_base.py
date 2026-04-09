"""Tests for broker schemas and BaseBroker ABC contract."""
import pytest
from datetime import datetime, timezone
from trading_system.contracts.trades import OrderStatus, OrderSide


class TestOrderStatusExtension:
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
        from pydantic import ValidationError
        from trading_system.broker.base import Order
        o = Order(
            order_id="x", ticker="A", side=OrderSide.BUY,
            quantity=1.0, timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        with pytest.raises((ValidationError, TypeError)):
            o.quantity = 99.0

    def test_limit_order_requires_price(self):
        from pydantic import ValidationError
        from trading_system.broker.base import Order
        with pytest.raises(ValidationError):
            Order(
                order_id="x", ticker="AAPL", side=OrderSide.BUY,
                quantity=1.0, order_type="LIMIT", limit_price=None,
                timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            )


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
                return AccountSnapshot(total_value=10000, cash=10000, buying_power=10000)
            def is_connected(self) -> bool:
                return True

        broker = DummyBroker()
        assert broker.is_connected()
