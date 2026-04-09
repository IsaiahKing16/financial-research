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

    def test_sell_without_position_rejected(self):
        b = MockBroker(MockBrokerConfig(slippage_bps=0.0))
        b.set_prices({"AAPL": 100.0})
        r = b.submit_order(_order(side=OrderSide.SELL, quantity=5.0))
        assert r.status == OrderStatus.REJECTED
        assert r.error is not None
        assert b.get_account().cash == pytest.approx(10_000.0)  # cash unchanged

    def test_unpriced_ticker_raises(self):
        b = MockBroker()
        with pytest.raises(RuntimeError, match="no price set"):
            b.submit_order(_order(ticker="TSLA"))


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

    def test_avg_cost_weighted_average(self):
        b = MockBroker(MockBrokerConfig(slippage_bps=0.0))
        b.set_prices({"AAPL": 100.0})
        b.submit_order(_order(quantity=10.0, order_id="b1"))
        b.set_prices({"AAPL": 120.0})
        b.submit_order(_order(quantity=10.0, order_id="b2"))
        pos = b.get_positions()[0]
        assert pos.avg_cost == pytest.approx(110.0)
        assert pos.quantity == pytest.approx(20.0)

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


class TestMockBrokerOverSellGuard:
    """Tests for oversell clamping (Codex P1 fix)."""

    def test_sell_more_than_held_clamps_to_position(self):
        """Hold 10 shares, try to sell 20 → fills 10, PARTIAL status."""
        from trading_system.broker.mock import MockBroker, MockBrokerConfig
        from trading_system.broker.base import Order
        from trading_system.contracts.trades import OrderStatus, OrderSide
        from datetime import datetime, timezone

        broker = MockBroker(MockBrokerConfig(
            initial_cash=100_000.0, slippage_bps=0.0, fill_fraction=1.0,
        ))
        broker.set_prices({"AAPL": 150.0})

        buy = Order(
            order_id="buy-1", ticker="AAPL", side=OrderSide.BUY,
            quantity=10.0, timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        broker.submit_order(buy)

        sell = Order(
            order_id="sell-1", ticker="AAPL", side=OrderSide.SELL,
            quantity=20.0, timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        result = broker.submit_order(sell)

        assert result.filled_quantity == 10.0, (
            f"Should clamp to held quantity, got {result.filled_quantity}"
        )
        assert result.status == OrderStatus.PARTIAL

    def test_sell_clamped_cash_is_correct(self):
        """Hold 5 @ $100, sell 10 → cash increases by $500 (5 shares), not $1000."""
        from trading_system.broker.mock import MockBroker, MockBrokerConfig
        from trading_system.broker.base import Order
        from trading_system.contracts.trades import OrderSide
        from datetime import datetime, timezone
        import pytest

        broker = MockBroker(MockBrokerConfig(
            initial_cash=100_000.0, slippage_bps=0.0, fill_fraction=1.0,
        ))
        broker.set_prices({"TEST": 100.0})

        buy = Order(
            order_id="buy-1", ticker="TEST", side=OrderSide.BUY,
            quantity=5.0, timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        broker.submit_order(buy)
        cash_after_buy = broker.get_account().cash

        sell = Order(
            order_id="sell-1", ticker="TEST", side=OrderSide.SELL,
            quantity=10.0, timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        broker.submit_order(sell)
        cash_after_sell = broker.get_account().cash

        assert cash_after_sell == pytest.approx(cash_after_buy + 500.0), (
            f"Expected cash increase of $500, got {cash_after_sell - cash_after_buy}"
        )

    def test_sell_clamp_removes_position(self):
        """After clamped sell of full position, ticker no longer in positions."""
        from trading_system.broker.mock import MockBroker, MockBrokerConfig
        from trading_system.broker.base import Order
        from trading_system.contracts.trades import OrderSide
        from datetime import datetime, timezone

        broker = MockBroker(MockBrokerConfig(
            initial_cash=100_000.0, slippage_bps=0.0, fill_fraction=1.0,
        ))
        broker.set_prices({"XYZ": 50.0})

        buy = Order(
            order_id="buy-1", ticker="XYZ", side=OrderSide.BUY,
            quantity=10.0, timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        broker.submit_order(buy)
        assert any(p.ticker == "XYZ" for p in broker.get_positions())

        sell = Order(
            order_id="sell-1", ticker="XYZ", side=OrderSide.SELL,
            quantity=999.0,
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        broker.submit_order(sell)
        assert not any(p.ticker == "XYZ" for p in broker.get_positions()), (
            "Position should be fully closed after clamped sell"
        )


class TestMockBrokerZeroFillGuard:
    """Tests for zero-fill ghost position prevention (Codex P2 fix)."""

    def test_zero_fill_fraction_buy_no_ghost_position(self):
        """fill_fraction=0.0 → BUY creates NO position entry."""
        from trading_system.broker.mock import MockBroker, MockBrokerConfig
        from trading_system.broker.base import Order
        from trading_system.contracts.trades import OrderStatus, OrderSide
        from datetime import datetime, timezone

        broker = MockBroker(MockBrokerConfig(
            initial_cash=100_000.0, slippage_bps=0.0, fill_fraction=0.0,
        ))
        broker.set_prices({"GHOST": 100.0})

        buy = Order(
            order_id="buy-1", ticker="GHOST", side=OrderSide.BUY,
            quantity=10.0, timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        result = broker.submit_order(buy)

        assert result.status == OrderStatus.REJECTED
        assert result.filled_quantity == 0.0
        assert not any(p.ticker == "GHOST" for p in broker.get_positions()), (
            "Zero-fill should not create a ghost position"
        )

    def test_zero_fill_fraction_buy_cash_unchanged(self):
        """fill_fraction=0.0 → cash should not change."""
        from trading_system.broker.mock import MockBroker, MockBrokerConfig
        from trading_system.broker.base import Order
        from trading_system.contracts.trades import OrderSide
        from datetime import datetime, timezone
        import pytest

        broker = MockBroker(MockBrokerConfig(
            initial_cash=50_000.0, slippage_bps=0.0, fill_fraction=0.0,
        ))
        broker.set_prices({"GHOST": 200.0})

        buy = Order(
            order_id="buy-1", ticker="GHOST", side=OrderSide.BUY,
            quantity=100.0, timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        broker.submit_order(buy)

        assert broker.get_account().cash == pytest.approx(50_000.0)
