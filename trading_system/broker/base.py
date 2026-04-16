"""BaseBroker ABC and order/position schemas."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from trading_system.contracts.trades import OrderSide, OrderStatus


class Order(BaseModel):
    """Immutable order to submit to a broker."""
    model_config = {"frozen": True}

    order_id: str = Field(description="UUID assigned by OrderManager")
    ticker: str = Field(min_length=1, max_length=10)
    side: OrderSide
    quantity: float = Field(gt=0, description="Shares (fractional OK)")
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: float | None = None
    notional: float | None = Field(default=None, description="Dollar amount (audit trail)")
    timestamp: datetime

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase: got '{v}'")
        return v

    @model_validator(mode="after")
    def _validate_limit_price(self) -> Order:
        if self.order_type == "LIMIT" and self.limit_price is None:
            raise ValueError("limit_price is required when order_type='LIMIT'")
        if self.order_type == "MARKET" and self.limit_price is not None:
            raise ValueError("limit_price must be None for MARKET orders")
        return self


class OrderResult(BaseModel):
    """Immutable result of a broker order submission."""
    model_config = {"frozen": True}

    order_id: str = Field(description="Matches Order.order_id")
    ticker: str
    status: OrderStatus
    filled_quantity: float = Field(ge=0)
    fill_price: float = Field(ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    executed_at: datetime | None = None
    error: str | None = None

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase: got '{v}'")
        return v


class BrokerPosition(BaseModel):
    """Broker-reported position for a single ticker."""
    model_config = {"frozen": True}

    ticker: str
    quantity: float
    avg_cost: float
    current_value: float
    unrealized_pnl: float

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase: got '{v}'")
        return v


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
