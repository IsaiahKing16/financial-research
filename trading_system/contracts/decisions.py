"""
decisions.py — Pydantic contracts for trading system decision objects.

Three contract families:
  1. EvaluatorStatus — RED/YELLOW/GREEN health signal from the evaluator layer
  2. PositionDecision — Risk engine output (size, stop, approval) for a candidate signal
  3. AllocationDecision — Portfolio manager output (final allocation per ticker)

These replace the implicit plain-dict / ad-hoc dataclass patterns currently
crossing the risk_engine → portfolio_manager → backtest_engine boundaries.

Design decisions:
  - EvaluatorStatus is an enum, not a bool — YELLOW is a distinct third state
    meaning "reduce caution, not halt" (20% exposure reduction vs 100% halt)
  - PositionDecision is the output of the risk layer; it may REJECT a signal
    (approved=False) with a reason — the portfolio manager must check this
  - AllocationDecision is the final sizing used by the execution layer
  - All models are frozen (immutable once created)

Linear: SLE-58
"""

from __future__ import annotations

from datetime import date as Date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pattern_engine.contracts.finite_types import FiniteFloat

# ─── EvaluatorStatus ──────────────────────────────────────────────────────────

class EvaluatorStatus(str, Enum):
    """
    Health status from the strategy evaluator.

    Determines whether the system can trade at full, reduced, or zero capacity.

    GREEN:  All metrics within expected bounds. Trade normally.
    YELLOW: Performance degraded but within tolerance. Reduce exposure by 20%.
            Trigger: 30-day Sharpe < 0.5 (but > 0) or 90-day drawdown 10–15%.
    RED:    Performance below minimum threshold. Halt all new entries.
            Trigger: 90-day Sharpe < 0 OR portfolio drawdown > 15%.

    Note: RED does not force existing position liquidation. It only blocks new entries.
    """
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

    @property
    def exposure_multiplier(self) -> float:
        """Factor to apply to target position sizes under this status."""
        if self == EvaluatorStatus.GREEN:
            return 1.0
        elif self == EvaluatorStatus.YELLOW:
            return 0.8   # 20% reduction
        else:  # RED
            return 0.0   # No new entries


class EvaluatorSnapshot(BaseModel):
    """
    Point-in-time evaluator assessment.

    Carries the status, the metrics that drove it, and the evaluation date
    so the portfolio manager can log decisions with full context.

    Args:
        evaluation_date: Date this assessment was computed.
        status: GREEN / YELLOW / RED decision.
        sharpe_30d: 30-day rolling Sharpe ratio (None if insufficient history).
        sharpe_90d: 90-day rolling Sharpe ratio (None if insufficient history).
        drawdown_from_peak: Current portfolio drawdown from all-time high [0, 1].
        days_in_market: Trading days since system start.
        reason: Human-readable explanation of the status decision.
    """
    model_config = ConfigDict(frozen=True)

    evaluation_date: Date
    status: EvaluatorStatus
    sharpe_30d: float | None = Field(default=None, description="30d Sharpe (None = insufficient history)")
    sharpe_90d: float | None = Field(default=None, description="90d Sharpe (None = insufficient history)")
    drawdown_from_peak: float = Field(ge=0.0, le=1.0, description="Current drawdown [0, 1]")
    days_in_market: int = Field(ge=0, description="Trading days since system start")
    reason: str = Field(min_length=1, description="Explanation of status decision")

    @property
    def exposure_multiplier(self) -> float:
        """Pass through to EvaluatorStatus.exposure_multiplier."""
        return self.status.exposure_multiplier

    @property
    def is_halted(self) -> bool:
        """True if system is in RED state (no new entries allowed)."""
        return self.status == EvaluatorStatus.RED


# ─── PositionDecision ─────────────────────────────────────────────────────────

class RejectionReason(str, Enum):
    """Why the risk engine rejected a candidate signal."""
    MAX_POSITIONS = "max_positions"           # Portfolio at position cap
    SECTOR_CONCENTRATION = "sector_concentration"  # Sector exposure limit hit
    EVALUATOR_RED = "evaluator_red"           # Evaluator in RED state
    LOW_CONFIDENCE = "low_confidence"         # Below confidence_threshold
    INSUFFICIENT_MATCHES = "insufficient_matches"  # Too few KNN analogues
    CAPITAL_EXHAUSTED = "capital_exhausted"   # No free cash
    ALREADY_HELD = "already_held"             # Already in position for this ticker


class PositionDecision(BaseModel):
    """
    Risk engine output for a single candidate signal.

    The risk engine evaluates each SignalRecord and produces a PositionDecision
    that either approves the trade (with sizing and stop parameters) or rejects
    it (with a reason). The portfolio manager must check `approved` before acting.

    Args:
        ticker: Stock ticker.
        signal_date: Date the signal was generated.
        approved: True if the risk engine approves entry.
        rejection_reason: Set only when approved=False.
        position_pct: Target position size as fraction of equity (0 if rejected).
        target_shares: Number of shares to buy at next-day open (0 if rejected).
        entry_price_estimate: Expected next-day open price for pre-trade sizing.
        stop_loss_price: ATR-based stop price (0 = no stop).
        atr_pct: ATR as % of price at decision time.
        max_holding_days: Position age limit in trading days.
        confidence: Signal confidence (passed through from SignalRecord).
        sector: Sector classification (passed through from SignalRecord).
    """
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    ticker: str = Field(min_length=1, max_length=10)
    signal_date: Date
    approved: bool = Field(description="True = enter; False = skip")
    rejection_reason: RejectionReason | None = Field(
        default=None,
        description="Set only when approved=False",
    )
    position_pct: FiniteFloat = Field(ge=0.0, le=1.0, description="Target fraction of equity")
    target_shares: FiniteFloat = Field(ge=0.0, description="Shares to buy (may be fractional)")
    entry_price_estimate: FiniteFloat = Field(ge=0.0, description="Estimated entry price")
    stop_loss_price: FiniteFloat = Field(ge=0.0, description="ATR stop price (0 = no stop)")
    atr_pct: FiniteFloat = Field(ge=0.0, description="ATR as % of price")
    max_holding_days: int = Field(ge=1, default=14)
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    sector: str = Field(min_length=1)

    @model_validator(mode="after")
    def rejection_requires_reason(self) -> PositionDecision:
        """A rejected decision must have a rejection_reason."""
        if not self.approved and self.rejection_reason is None:
            raise ValueError(
                "PositionDecision with approved=False must set rejection_reason"
            )
        if self.approved and self.rejection_reason is not None:
            raise ValueError(
                "PositionDecision with approved=True must not set rejection_reason"
            )
        return self

    @model_validator(mode="after")
    def approved_requires_sizing(self) -> PositionDecision:
        """An approved decision must have non-zero sizing."""
        if self.approved and self.position_pct == 0.0:
            raise ValueError(
                "PositionDecision with approved=True must have position_pct > 0"
            )
        return self

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase: got '{v}'")
        return v


# ─── AllocationDecision ───────────────────────────────────────────────────────

class AllocationDecision(BaseModel):
    """
    Portfolio manager output: the final allocation applied to a signal.

    This is a richer version of the existing `AllocationDecision` in
    `trading_system/portfolio_state.py`. It replaces that frozen dataclass
    with a Pydantic model and adds evaluator context.

    Args:
        ticker: Stock ticker.
        signal_date: Signal date (not necessarily the entry date).
        final_position_pct: Actual position size after all adjustments.
                            May be smaller than PositionDecision.position_pct if
                            evaluator status reduced exposure.
        evaluator_status: EvaluatorStatus at decision time (for audit trail).
        capital_allocated: Dollar amount allocated (equity * final_position_pct).
        rank_in_queue: Position in the signal priority queue (1 = highest priority).
        sector: Sector classification.
        adjusted_for_evaluator: True if position was scaled down due to YELLOW status.
    """
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    ticker: str = Field(min_length=1, max_length=10)
    signal_date: Date
    final_position_pct: FiniteFloat = Field(ge=0.0, le=1.0)
    evaluator_status: EvaluatorStatus
    capital_allocated: FiniteFloat = Field(ge=0.0, description="Dollar amount committed")
    rank_in_queue: int = Field(ge=1, description="Priority rank in signal queue")
    sector: str = Field(min_length=1)
    adjusted_for_evaluator: bool = Field(
        default=False,
        description="True if position was scaled due to YELLOW evaluator status",
    )

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase: got '{v}'")
        return v
