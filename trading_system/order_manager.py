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
        if price <= 0.0:
            raise RuntimeError(
                f"create_order_from_decision: price must be > 0, got {price} for {decision.ticker}"
            )
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
        """Submit to broker. PENDING -> terminal state."""
        now = datetime.now(timezone.utc)
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
