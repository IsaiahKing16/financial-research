"""
risk_state.py — Phase 2 portfolio risk state and decision records.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class PositionDecision:
    """Output of a position sizing request."""

    approved: bool
    ticker: str
    position_pct: float
    shares: float
    dollar_amount: float
    stop_price: float
    stop_distance_pct: float
    atr_pct: float
    drawdown_scalar: float
    raw_weight: float
    rejection_reason: Optional[str]

    def __post_init__(self) -> None:
        """Validate decision integrity."""
        if self.approved and self.shares <= 0:
            raise ValueError("Approved decision must have positive shares")
        if self.approved and self.stop_price <= 0:
            raise ValueError("Approved decision must have positive stop_price")
        if not 0.0 <= self.drawdown_scalar <= 1.0:
            raise ValueError(
                f"drawdown_scalar must be in [0, 1], got {self.drawdown_scalar}"
            )


@dataclass(frozen=True)
class StopLossEvent:
    """Audit record created when a stop-loss is triggered."""

    ticker: str
    trigger_date: str
    stop_price: float
    trigger_low: float
    entry_price: float
    exit_price: float
    gap_through: bool
    atr_at_entry: float


@dataclass
class RiskState:
    """Mutable portfolio-level risk state updated through the backtest lifecycle."""

    peak_equity: float
    current_equity: float
    current_drawdown: float
    drawdown_mode: str
    sizing_scalar: float
    active_stops: Dict[str, float] = field(default_factory=dict)
    daily_atr_cache: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def initial(cls, starting_equity: float) -> "RiskState":
        """Create initial state at backtest start."""
        return cls(
            peak_equity=starting_equity,
            current_equity=starting_equity,
            current_drawdown=0.0,
            drawdown_mode="normal",
            sizing_scalar=1.0,
            active_stops={},
            daily_atr_cache={},
        )

    def update(self, current_equity: float, brake_threshold: float, halt_threshold: float) -> None:
        """Update drawdown metrics after mark-to-market."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        self.current_equity = current_equity
        if self.peak_equity > 0:
            self.current_drawdown = max(0.0, 1.0 - (current_equity / self.peak_equity))
        else:
            self.current_drawdown = 0.0

        if self.current_drawdown >= halt_threshold:
            self.drawdown_mode = "halt"
            self.sizing_scalar = 0.0
            return

        if self.current_drawdown >= brake_threshold:
            self.drawdown_mode = "brake"
            span = halt_threshold - brake_threshold
            self.sizing_scalar = (
                (halt_threshold - self.current_drawdown) / span if span > 0 else 0.0
            )
            self.sizing_scalar = min(1.0, max(0.0, self.sizing_scalar))
            return

        self.drawdown_mode = "normal"
        self.sizing_scalar = 1.0

    def register_stop(self, ticker: str, stop_price: float) -> None:
        """Track active stop-loss for an open position."""
        self.active_stops[ticker] = stop_price

    def remove_stop(self, ticker: str) -> None:
        """Remove active stop when position closes."""
        self.active_stops.pop(ticker, None)
