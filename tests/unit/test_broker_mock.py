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
