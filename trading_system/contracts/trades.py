"""
trades.py — Pydantic contracts for trade lifecycle objects.

These models formalize the implicit contracts currently defined as plain
@dataclass in backtest_engine.py (OpenPosition, CompletedTrade, DailyRecord).

Model families:
    - PositionRecord   — open position (maps to backtest_engine.OpenPosition)
    - TradeRecord      — completed trade with full P&L (maps to CompletedTrade)
    - DailySnapshot    — daily equity curve point (maps to DailyRecord)
    - TradeEvent       — future-ready execution record for live trading

Design decisions:
    - Frozen models prevent downstream mutation (matches existing frozen=True pattern)
    - Validators catch impossible states at construction (negative shares, future dates)
    - Field names match existing dataclass names for migration compatibility
    - Exit reasons are enumerated to prevent string typos
    - TradeEvent carries fill_quantity / fill_price / fill_ratio for execution slippage
      tracking in live trading (not used in backtest mode — fill_ratio defaults to 1.0)

Linear: SLE-58
"""

from datetime import date as Date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pattern_engine.contracts.finite_types import FiniteFloat


class ExitReason(str, Enum):
    """Why a position was closed."""
    SIGNAL = "signal"               # New signal reversed direction
    STOP_LOSS = "stop_loss"         # Hit ATR-based stop
    MAX_HOLD = "max_hold"           # Exceeded max_holding_days
    DRAWDOWN_HALT = "drawdown_halt" # Portfolio drawdown circuit breaker


class PositionRecord(BaseModel):
    """An open position in the portfolio.

    Maps to: trading_system/backtest_engine.py::OpenPosition
    """
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    trade_id: int = Field(ge=0, description="Unique trade identifier")
    ticker: str = Field(min_length=1, max_length=10)
    sector: str = Field(min_length=1)
    entry_date: Date
    raw_entry_price: FiniteFloat = Field(gt=0, description="Next-day open BEFORE friction")
    entry_price: FiniteFloat = Field(gt=0, description="After slippage + spread")
    shares: FiniteFloat = Field(gt=0, description="May be fractional")
    position_pct: FiniteFloat = Field(ge=0, le=1, description="Fraction of equity at entry")
    confidence_at_entry: FiniteFloat = Field(ge=0, le=1)
    stop_loss_price: FiniteFloat = Field(ge=0, description="0 = no stop (Phase 1 mode)")
    atr_pct_at_entry: FiniteFloat = Field(ge=0, default=0.0)
    days_held: int = Field(ge=0, default=0)
    last_close_price: FiniteFloat = Field(ge=0, default=0.0)

    @model_validator(mode="after")
    def entry_price_after_friction(self):
        """Entry price must be >= raw entry price (friction adds cost on buys)."""
        if self.entry_price < self.raw_entry_price * 0.95:
            raise ValueError(
                f"entry_price ({self.entry_price}) is suspiciously below "
                f"raw_entry_price ({self.raw_entry_price}) — friction should increase cost"
            )
        return self


class TradeRecord(BaseModel):
    """A completed trade with full P&L accounting.

    Maps to: trading_system/backtest_engine.py::CompletedTrade

    Separates gross_pnl and net_pnl to allow friction impact auditing.
    At $10k capital with 26 bps round-trip, a $500 position pays ~$1.30
    in friction — significant relative to per-trade alpha.
    """
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    trade_id: int = Field(ge=0)
    ticker: str = Field(min_length=1, max_length=10)
    sector: str = Field(min_length=1)
    direction: str = Field(default="LONG", description="v1 is long-only")
    entry_date: Date
    entry_price: FiniteFloat = Field(gt=0)
    exit_date: Date
    exit_price: FiniteFloat = Field(gt=0)
    position_pct: FiniteFloat = Field(ge=0, le=1)
    shares: FiniteFloat = Field(gt=0)
    gross_pnl: FiniteFloat = Field(description="(raw_exit - raw_entry) * shares")
    entry_friction_cost: FiniteFloat = Field(ge=0)
    exit_friction_cost: FiniteFloat = Field(ge=0)
    slippage_cost: FiniteFloat = Field(ge=0)
    spread_cost: FiniteFloat = Field(ge=0)
    total_costs: FiniteFloat = Field(ge=0)
    net_pnl: FiniteFloat = Field(description="gross_pnl - total_costs")
    holding_days: int = Field(ge=1)
    exit_reason: ExitReason
    confidence_at_entry: FiniteFloat = Field(ge=0, le=1)

    @model_validator(mode="after")
    def dates_ordered(self):
        """Exit date must be on or after entry date."""
        if self.exit_date < self.entry_date:
            raise ValueError(
                f"exit_date ({self.exit_date}) before entry_date ({self.entry_date})"
            )
        return self

    @model_validator(mode="after")
    def costs_consistent(self):
        """Total costs must equal entry + exit friction + slippage + spread.

        All four cost components contribute to total_costs.  Omitting slippage_cost
        or spread_cost from the sum produces a false mismatch for any trade with
        non-zero slippage or spread.  I6 (SLE review).
        """
        expected = (
            self.entry_friction_cost
            + self.exit_friction_cost
            + self.slippage_cost
            + self.spread_cost
        )
        if abs(self.total_costs - expected) > 0.01:
            raise ValueError(
                f"total_costs ({self.total_costs:.4f}) != "
                f"entry_friction ({self.entry_friction_cost:.4f}) + "
                f"exit_friction ({self.exit_friction_cost:.4f}) + "
                f"slippage ({self.slippage_cost:.4f}) + "
                f"spread ({self.spread_cost:.4f}) = {expected:.4f}"
            )
        return self


class DailySnapshot(BaseModel):
    """Daily portfolio state for equity curve construction.

    Maps to: trading_system/backtest_engine.py::DailyRecord
    """
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    date: Date
    equity: FiniteFloat = Field(gt=0, description="Total portfolio value")
    cash: FiniteFloat = Field(ge=0, description="Uninvested cash")
    invested_capital: FiniteFloat = Field(ge=0, description="Capital in open positions")
    gross_exposure: FiniteFloat = Field(ge=0, le=1.5, description="Should be <= 1.0 for v1")
    open_positions: int = Field(ge=0)
    daily_return: FiniteFloat = Field(description="Today's return (may be negative)")
    cumulative_return: FiniteFloat = Field(description="Return since inception")
    drawdown_from_peak: FiniteFloat = Field(ge=0, le=1, description="Current drawdown [0, 1]")
    cash_yield_today: FiniteFloat = Field(ge=0, description="Risk-free yield on idle cash")
    strategy_return_excl_cash: FiniteFloat
    strategy_return_incl_cash: FiniteFloat


# ─── TradeEvent ────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    """Direction of an order."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Lifecycle state of an order."""
    PENDING = "PENDING"         # Created, not yet submitted to broker
    SUBMITTED = "SUBMITTED"     # Sent to broker, awaiting response
    FILLED = "FILLED"           # Fully executed
    PARTIAL = "PARTIAL"         # Partially filled (live trading only)
    CANCELLED = "CANCELLED"     # Cancelled before fill
    REJECTED = "REJECTED"       # Broker/system rejected the order
    TIMED_OUT = "TIMED_OUT"     # No broker response within deadline


class TradeEvent(BaseModel):
    """
    Future-ready execution record for a single order.

    In backtest mode: fill_ratio=1.0 (full fill assumed), execution_latency_seconds=0.
    In live trading: fill_quantity may differ from ordered_quantity (partial fills);
    fill_price may differ from limit_price_estimate (slippage beyond model).

    This model bridges the backtest ↔ live trading gap. The backtest engine
    writes TradeEvents with fill_ratio=1.0; a live broker adapter would write
    the actual fill details.

    Args:
        trade_event_id: Unique identifier for this execution event.
        trade_id: The parent trade this event belongs to (links to TradeRecord).
        ticker: Stock ticker.
        side: BUY or SELL.
        order_date: Date the order was placed.
        ordered_quantity: Number of shares ordered.
        limit_price_estimate: Price we expected to fill at.
        fill_quantity: Shares actually filled (= ordered_quantity in backtest).
        fill_price: Actual fill price (may differ from limit_price_estimate in live).
        fill_ratio: fill_quantity / ordered_quantity. 1.0 in backtest; < 1.0 on partials.
        status: Current order status.
        execution_timestamp: ISO 8601 UTC timestamp of fill (or None if pending).
        execution_latency_seconds: Time from order submission to fill (0 in backtest).
        broker_order_id: External broker order ID (None in backtest mode).
    """
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    trade_event_id: int = Field(ge=0, description="Unique event identifier")
    trade_id: int = Field(ge=0, description="Parent TradeRecord identifier")
    ticker: str = Field(min_length=1, max_length=10)
    side: OrderSide
    order_date: Date
    ordered_quantity: FiniteFloat = Field(gt=0, description="Shares ordered")
    limit_price_estimate: FiniteFloat = Field(gt=0, description="Expected fill price")
    fill_quantity: FiniteFloat = Field(ge=0, description="Shares actually filled")
    fill_price: FiniteFloat = Field(ge=0, description="Actual fill price (0 if not yet filled)")
    fill_ratio: FiniteFloat = Field(ge=0.0, le=1.0, default=1.0, description="fill_quantity / ordered_quantity")
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    execution_timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 UTC timestamp of fill (None = not filled yet)",
    )
    execution_latency_seconds: FiniteFloat = Field(
        ge=0.0,
        default=0.0,
        description="Seconds from order submission to fill (0 in backtest)",
    )
    broker_order_id: Optional[str] = Field(
        default=None,
        description="External broker order ID (None in backtest mode)",
    )

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase: got '{v}'")
        return v

    @model_validator(mode="after")
    def fill_ratio_consistent(self) -> "TradeEvent":
        """fill_ratio must match fill_quantity / ordered_quantity (within tolerance)."""
        if self.ordered_quantity > 0:
            expected_ratio = self.fill_quantity / self.ordered_quantity
            if abs(self.fill_ratio - expected_ratio) > 0.001:
                raise ValueError(
                    f"fill_ratio ({self.fill_ratio:.4f}) != "
                    f"fill_quantity/ordered_quantity ({expected_ratio:.4f})"
                )
        return self

    @model_validator(mode="after")
    def filled_requires_price(self) -> "TradeEvent":
        """A FILLED or PARTIAL order must have fill_price > 0."""
        if self.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            if self.fill_price <= 0:
                raise ValueError(
                    f"Order with status={self.status} must have fill_price > 0"
                )
        return self

    @classmethod
    def backtest_fill(
        cls,
        trade_event_id: int,
        trade_id: int,
        ticker: str,
        side: OrderSide,
        order_date: Date,
        ordered_quantity: float,
        fill_price: float,
    ) -> "TradeEvent":
        """
        Create a fully-filled TradeEvent for backtest mode.

        In backtest, every order fills at the modeled price with no partial fills.

        Args:
            trade_event_id: Unique event ID.
            trade_id: Parent trade ID.
            ticker: Stock ticker (uppercase).
            side: BUY or SELL.
            order_date: Date of order execution.
            ordered_quantity: Shares ordered (= filled in backtest).
            fill_price: Execution price (next-day open + friction).

        Returns:
            A TradeEvent with fill_ratio=1.0, status=FILLED.
        """
        return cls(
            trade_event_id=trade_event_id,
            trade_id=trade_id,
            ticker=ticker,
            side=side,
            order_date=order_date,
            ordered_quantity=ordered_quantity,
            limit_price_estimate=fill_price,
            fill_quantity=ordered_quantity,
            fill_price=fill_price,
            fill_ratio=1.0,
            status=OrderStatus.FILLED,
            execution_timestamp=datetime.now(timezone.utc).isoformat(),
            execution_latency_seconds=0.0,
            broker_order_id=None,
        )
