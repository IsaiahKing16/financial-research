"""
state.py — SharedState inter-layer bus for the trading system.

SharedState is the single object that flows through all trading system layers:
  pattern_engine → risk_engine → portfolio_manager → backtest_engine

Rather than passing individual values through function parameters (which breaks
every caller when a new field is added), all layers read from and write to a
single SharedState. Because it is frozen (Pydantic v2), "writes" produce a new
instance via model_copy(update={...}) — the functional update pattern.

Sub-state design:
  - EquityState:     Capital accounting (total equity, cash, invested)
  - PositionsState:  Open position registry
  - RiskState:       Risk engine parameters and current exposure
  - PortfolioState:  Portfolio-level metrics and signal queue
  - EvaluatorState:  Strategy evaluator health metrics

Command queue:
  - SystemCommand enum: HALT, REDUCE_EXPOSURE, RESUME — allows evaluator
    to signal intent to portfolio_manager without direct coupling

Linear: SLE-58
"""

from __future__ import annotations

from datetime import date as Date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pattern_engine.contracts.finite_types import FiniteFloat
from trading_system.contracts.decisions import (
    EvaluatorSnapshot,
    EvaluatorStatus,
)

# ─── SystemCommand ─────────────────────────────────────────────────────────────

class SystemCommand(str, Enum):
    """
    Commands the evaluator can issue to the portfolio manager.

    These travel on the SharedState.command_queue and are consumed by
    the portfolio manager on each trading day.

    HALT:             Stop all new entries. Evaluator is RED.
    REDUCE_EXPOSURE:  Scale all new position sizes by 80%. Evaluator is YELLOW.
    RESUME:           Return to normal trading. Evaluator returned to GREEN.
    """
    HALT = "HALT"
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"
    RESUME = "RESUME"


# ─── Sub-State Models ──────────────────────────────────────────────────────────

class EquityState(BaseModel):
    """Current capital accounting."""
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    total_equity: FiniteFloat = Field(gt=0, description="Total portfolio value (cash + positions)")
    cash: FiniteFloat = Field(ge=0, description="Uninvested cash balance")
    invested_capital: FiniteFloat = Field(ge=0, description="Capital in open positions")
    peak_equity: FiniteFloat = Field(gt=0, description="All-time high equity (for drawdown calc)")
    inception_equity: FiniteFloat = Field(gt=0, description="Starting equity (for total return calc)")

    @model_validator(mode="after")
    def equity_components_consistent(self) -> EquityState:
        """total_equity should approximately equal cash + invested_capital."""
        diff = abs(self.total_equity - (self.cash + self.invested_capital))
        if diff > 1.0:  # Allow $1 rounding tolerance
            raise ValueError(
                f"total_equity ({self.total_equity:.2f}) != "
                f"cash ({self.cash:.2f}) + invested ({self.invested_capital:.2f})"
            )
        return self

    @model_validator(mode="after")
    def peak_ge_equity(self) -> EquityState:
        """Peak equity must be >= current equity (peaks only move up)."""
        if self.peak_equity < self.total_equity - 0.01:  # $0.01 tolerance for float arithmetic
            raise ValueError(
                f"peak_equity ({self.peak_equity:.2f}) < "
                f"total_equity ({self.total_equity:.2f})"
            )
        return self

    @property
    def drawdown_from_peak(self) -> float:
        """Current drawdown from peak equity [0, 1]."""
        return max(0.0, 1.0 - self.total_equity / self.peak_equity)

    @property
    def total_return(self) -> float:
        """Total return since inception [fractional]."""
        return self.total_equity / self.inception_equity - 1.0

    @property
    def gross_exposure(self) -> float:
        """Fraction of equity in open positions [0, 1]."""
        return self.invested_capital / self.total_equity


class PositionsState(BaseModel):
    """Registry of open positions."""
    model_config = {"frozen": True}

    # Maps ticker → (trade_id, entry_date, entry_price, position_pct, days_held)
    # Using a tuple list instead of nested model for JSON serializability
    open_tickers: list[str] = Field(
        default_factory=list,
        description="Tickers currently held",
    )
    n_open: int = Field(ge=0, default=0, description="Count of open positions")
    max_positions: int = Field(ge=1, default=10, description="Hard cap on concurrent positions")

    # Sector exposure: ticker -> sector (for concentration checks)
    ticker_sectors: dict[str, str] = Field(
        default_factory=dict,
        description="Ticker → sector mapping for open positions",
    )

    @model_validator(mode="after")
    def n_open_consistent(self) -> PositionsState:
        """n_open must match len(open_tickers)."""
        if self.n_open != len(self.open_tickers):
            raise ValueError(
                f"n_open ({self.n_open}) != len(open_tickers) ({len(self.open_tickers)})"
            )
        return self

    @property
    def is_at_capacity(self) -> bool:
        """True if portfolio is at the maximum position count."""
        return self.n_open >= self.max_positions

    def sector_exposure(self, sector: str) -> int:
        """Number of open positions in the given sector."""
        return sum(1 for s in self.ticker_sectors.values() if s == sector)


class RiskState(BaseModel):
    """Risk engine parameters and current risk metrics."""
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    stop_loss_atr_multiple: float = Field(gt=0.0, default=3.0, description="ATR stop multiplier")
    max_holding_days: int = Field(ge=1, default=14)
    confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.65)
    max_sector_concentration: int = Field(ge=1, default=3, description="Max positions per sector")
    current_atr_estimates: dict[str, float] = Field(
        default_factory=dict,
        description="Ticker → ATR estimate (% of price) at last update",
    )


class PortfolioState(BaseModel):
    """Portfolio-level metrics and pending signal queue."""
    model_config = {"frozen": True}

    trading_date: Date = Field(description="Current trading day")
    trading_day_count: int = Field(ge=0, default=0, description="Days since inception")
    pending_signals_count: int = Field(ge=0, default=0, description="Signals queued for evaluation")
    trades_executed_today: int = Field(ge=0, default=0)
    trades_rejected_today: int = Field(ge=0, default=0)


class EvaluatorState(BaseModel):
    """Strategy evaluator health metrics."""
    model_config = {"frozen": True}

    latest_snapshot: EvaluatorSnapshot | None = Field(
        default=None,
        description="Most recent evaluator assessment",
    )
    consecutive_underperformance_days: int = Field(
        ge=0, default=0,
        description="Days since last GREEN status",
    )

    @property
    def current_status(self) -> EvaluatorStatus:
        """Current evaluator status (GREEN if no snapshot yet)."""
        if self.latest_snapshot is None:
            return EvaluatorStatus.GREEN
        return self.latest_snapshot.status

    @property
    def is_halted(self) -> bool:
        return self.current_status == EvaluatorStatus.RED


# ─── SharedState ──────────────────────────────────────────────────────────────

class SharedState(BaseModel):
    """
    Inter-layer communication bus for the trading system.

    This single object flows through all trading system layers. Each layer
    receives SharedState, reads what it needs, and may return an updated
    SharedState (via model_copy(update={...})).

    The frozen invariant means no layer can mutate the state in-place.
    All "updates" create a new SharedState instance, making the state
    evolution explicit and auditable.

    Args:
        equity: Current capital accounting.
        positions: Open position registry.
        risk: Risk engine parameters.
        portfolio: Portfolio-level metrics.
        evaluator: Strategy evaluator health.
        command_queue: Commands from evaluator → portfolio manager.
                       Consumed (cleared) by portfolio manager each day.

    Usage:
        # Update equity after a trade closes:
        new_state = state.model_copy(update={
            "equity": state.equity.model_copy(update={"cash": new_cash, ...})
        })
    """
    model_config = {"frozen": True}

    equity: EquityState
    positions: PositionsState
    risk: RiskState
    portfolio: PortfolioState
    evaluator: EvaluatorState
    command_queue: list[SystemCommand] = Field(
        default_factory=list,
        description="Pending commands from evaluator to portfolio manager",
    )

    # ── Convenience Properties ─────────────────────────────────────────────────

    @property
    def trading_date(self) -> Date:
        """Current trading date (delegates to PortfolioState)."""
        return self.portfolio.trading_date

    @property
    def total_equity(self) -> float:
        """Current total equity."""
        return self.equity.total_equity

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak [0, 1]."""
        return self.equity.drawdown_from_peak

    @property
    def is_halted(self) -> bool:
        """True if evaluator RED or HALT command is pending."""
        return (
            self.evaluator.is_halted
            or SystemCommand.HALT in self.command_queue
        )

    @property
    def exposure_multiplier(self) -> float:
        """Current position size multiplier from evaluator."""
        return self.evaluator.current_status.exposure_multiplier

    def has_capacity(self) -> bool:
        """True if portfolio can accept a new position."""
        return not self.positions.is_at_capacity and not self.is_halted

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def initial(
        cls,
        starting_equity: float,
        trading_date: Date,
        max_positions: int = 10,
        stop_loss_atr_multiple: float = 3.0,
        max_holding_days: int = 14,
        confidence_threshold: float = 0.65,
        max_sector_concentration: int = 3,
    ) -> SharedState:
        """
        Create the initial SharedState at system start.

        Args:
            starting_equity: Initial capital in dollars.
            trading_date: First trading day.
            max_positions: Maximum concurrent open positions.
            stop_loss_atr_multiple: ATR stop multiplier (locked setting: 3.0).
            max_holding_days: Max days to hold a position (locked setting: 14).
            confidence_threshold: Minimum confidence to trade (locked: 0.65).
            max_sector_concentration: Max positions per sector.

        Returns:
            A fully initialized SharedState.
        """
        if starting_equity <= 0:
            raise RuntimeError(f"starting_equity must be positive; got {starting_equity}")

        return cls(
            equity=EquityState(
                total_equity=starting_equity,
                cash=starting_equity,
                invested_capital=0.0,
                peak_equity=starting_equity,
                inception_equity=starting_equity,
            ),
            positions=PositionsState(
                open_tickers=[],
                n_open=0,
                max_positions=max_positions,
                ticker_sectors={},
            ),
            risk=RiskState(
                stop_loss_atr_multiple=stop_loss_atr_multiple,
                max_holding_days=max_holding_days,
                confidence_threshold=confidence_threshold,
                max_sector_concentration=max_sector_concentration,
            ),
            portfolio=PortfolioState(
                trading_date=trading_date,
                trading_day_count=0,
            ),
            evaluator=EvaluatorState(),
            command_queue=[],
        )
