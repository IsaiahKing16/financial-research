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
