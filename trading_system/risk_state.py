"""
risk_state.py — Portfolio risk tracking dataclasses for Phase 2.

Design: docs/PHASE2_SYSTEM_DESIGN.md Section 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol


class SupportsDrawdownThresholds(Protocol):
    """Minimal surface for drawdown brake/halt configuration.

    Satisfied by ``RiskConfig`` and test doubles — keeps this module free of
    heavy imports.
    """

    drawdown_brake_threshold: float
    drawdown_halt_threshold: float


@dataclass(frozen=True)
class PositionDecision:
    """Output of the risk engine's position sizing calculation.

    Immutable — computed once per trade candidate, never modified.
    """

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
    """Record of a stop-loss trigger for audit trail.

    Created when intraday low <= stop_price. The actual exit occurs at
    next-day open (not at the stop price) per design doc Section 4.2.
    """

    ticker: str
    trigger_date: str
    stop_price: float
    trigger_low: float
    entry_price: float
    exit_price: float
    gap_through: bool
    atr_at_entry: float

    def __post_init__(self) -> None:
        """Validate event fields."""
        if not self.ticker:
            raise ValueError("ticker must be non-empty")
        if not self.trigger_date:
            raise ValueError("trigger_date must be non-empty")


@dataclass
class RiskState:
    """Mutable state tracking portfolio-level risk metrics.

    Updated daily by backtest_engine after mark-to-market.
    Read by risk_engine when sizing new positions.
    """

    peak_equity: float
    current_equity: float
    current_drawdown: float
    drawdown_mode: str
    sizing_scalar: float
    active_stops: Dict[str, float] = field(default_factory=dict)
    daily_atr_cache: Dict[str, float] = field(default_factory=dict)

    def update(self, current_equity: float, config: SupportsDrawdownThresholds) -> None:
        """Recompute drawdown state after daily mark-to-market.

        Args:
            current_equity: Portfolio equity after today's mark-to-market.
            config: Risk configuration for brake/halt thresholds.
        """
        brake = config.drawdown_brake_threshold
        halt = config.drawdown_halt_threshold
        if brake >= halt:
            raise ValueError(
                f"drawdown_brake_threshold ({brake}) must be < "
                f"drawdown_halt_threshold ({halt})"
            )

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        self.current_equity = current_equity

        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0

        if self.current_drawdown >= halt:
            self.drawdown_mode = "halt"
            self.sizing_scalar = 0.0
        elif self.current_drawdown >= brake:
            self.drawdown_mode = "brake"
            self.sizing_scalar = (halt - self.current_drawdown) / (halt - brake)
        else:
            self.drawdown_mode = "normal"
            self.sizing_scalar = 1.0

    def register_stop(self, ticker: str, stop_price: float) -> None:
        """Record a new stop-loss for an open position."""
        self.active_stops[ticker] = stop_price

    def remove_stop(self, ticker: str) -> None:
        """Remove stop when position is closed."""
        self.active_stops.pop(ticker, None)

    @classmethod
    def initial(cls, starting_equity: float) -> RiskState:
        """Create initial risk state at start of backtest."""
        return cls(
            peak_equity=starting_equity,
            current_equity=starting_equity,
            current_drawdown=0.0,
            drawdown_mode="normal",
            sizing_scalar=1.0,
            active_stops={},
            daily_atr_cache={},
        )
