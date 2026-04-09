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

    @property
    def order_history(self) -> list[tuple[Order, OrderResult]]:
        return list(self._history)


class _MockPosition:
    """Mutable internal position tracker."""
    __slots__ = ("quantity", "avg_cost")

    def __init__(self, quantity: float, avg_cost: float) -> None:
        self.quantity = quantity
        self.avg_cost = avg_cost
