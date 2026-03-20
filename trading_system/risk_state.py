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
    imports beyond the standard library and ``typing``.
    """

    drawdown_brake_threshold: float
    drawdown_halt_threshold: float


@dataclass(frozen=True)
class PositionDecision:
    """Output of the risk engine's position sizing calculation.

    Immutable — computed once per trade candidate, never modified.

    Args:
        approved: True if trade is permitted.
        ticker: Ticker symbol.
        position_pct: Position as fraction of equity (0.0–1.0).
        shares: Share count (fractional allowed).
        dollar_amount: Position dollar value.
        stop_price: ATR-based stop-loss price.
        stop_distance_pct: Distance from entry to stop as a fraction.
        atr_pct: 20-day ATR as a fraction of current price.
        drawdown_scalar: Drawdown adjustment applied (0.0–1.0).
        raw_weight: Pre-clamp, pre-drawdown weight.
        rejection_reason: None if approved; explanation if rejected.
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


@dataclass
class RiskState:
    """Mutable state tracking portfolio-level risk metrics.

    Updated daily by backtest_engine after mark-to-market.
    Read by risk_engine when sizing new positions.

    NOT frozen — this is mutable state that changes daily.

    Args:
        peak_equity: High-water mark.
        current_equity: As of last mark-to-market.
        current_drawdown: (peak - current) / peak, always >= 0 when peak > 0.
        drawdown_mode: One of ``"normal"``, ``"brake"``, or ``"halt"``.
        sizing_scalar: 0.0–1.0 based on drawdown position.
        active_stops: Ticker to stop price for open positions.
        daily_atr_cache: Ticker to most recent ATR% (refreshed daily).
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
        """Record a new stop-loss for an open position.

        Args:
            ticker: Ticker symbol.
            stop_price: Stop price for the position.
        """
        self.active_stops[ticker] = stop_price

    def remove_stop(self, ticker: str) -> None:
        """Remove stop when position is closed.

        Args:
            ticker: Ticker symbol to remove; missing tickers are ignored.
        """
        self.active_stops.pop(ticker, None)

    @classmethod
    def initial(cls, starting_equity: float) -> RiskState:
        """Create initial risk state at start of backtest.

        Args:
            starting_equity: Portfolio equity at backtest start.

        Returns:
            ``RiskState`` with zero drawdown and full sizing scalar.
        """
        return cls(
            peak_equity=starting_equity,
            current_equity=starting_equity,
            current_drawdown=0.0,
            drawdown_mode="normal",
            sizing_scalar=1.0,
            active_stops={},
            daily_atr_cache={},
        )


@dataclass(frozen=True)
class StopLossEvent:
    """Record of a stop-loss trigger for audit trail.

    Created when intraday low <= stop_price. The actual exit occurs at
    next-day open (not at the stop price) per design doc Section 4.2.

    Args:
        ticker: Ticker symbol.
        trigger_date: Date the low breached the stop (ISO string).
        stop_price: The stop level that was breached.
        trigger_low: The intraday low that triggered the stop.
        entry_price: Original entry price.
        exit_price: Next-day open (actual execution).
        gap_through: True if trigger_low < stop_price (gap-down).
        atr_at_entry: ATR% when position was opened.
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
