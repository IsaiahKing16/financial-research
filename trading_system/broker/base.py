"""BaseBroker ABC and order/position schemas."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from trading_system.contracts.trades import OrderSide, OrderStatus


class Order(BaseModel):
    """Immutable order to submit to a broker."""
    model_config = {"frozen": True}

    order_id: str = Field(description="UUID assigned by OrderManager")
    ticker: str = Field(min_length=1, max_length=10)
    side: OrderSide
    quantity: float = Field(gt=0, description="Shares (fractional OK)")
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: Optional[float] = None
    notional: Optional[float] = Field(default=None, description="Dollar amount (audit trail)")
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


class BaseBroker(ABC):
    """Abstract base class for broker adapters."""

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
