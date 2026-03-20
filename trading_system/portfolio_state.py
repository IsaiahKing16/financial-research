"""
portfolio_state.py — Layer 3 Data Structures

Pure dataclasses for the portfolio manager. No business logic.
All frozen (immutable) except where noted.

Design doc: FPPE_TRADING_SYSTEM_DESIGN.md v0.3, Section 4.4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankedSignal:
    """A BUY signal annotated with its rank position and score.

    Args:
        ticker: Stock symbol (e.g., "AAPL").
        confidence: Calibrated probability from FPPE, [0.0, 1.0].
        sector: Sector classification from SECTOR_MAP.
        rank: 1-based position in the ranked list (1 = highest priority).
        rank_score: Scoring value used for ranking. v1: equals confidence.
        date: Signal generation date.
        raw_metadata: Preserved from UnifiedSignal for audit trail.
    """

    ticker: str
    confidence: float
    sector: str
    rank: int
    rank_score: float
    date: pd.Timestamp
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("ticker must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be [0, 1], got {self.confidence}"
            )
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if not 0.0 <= self.rank_score <= 1.0:
            raise ValueError(
                f"rank_score must be [0, 1], got {self.rank_score}"
            )


@dataclass(frozen=True)
class AllocationDecision:
    """Result of portfolio-level allocation check for one signal.

    Args:
        ticker: Stock symbol.
        approved: Whether the signal passed all portfolio constraints.
        rank: Original rank position from ranking step.
        confidence: Signal confidence (preserved for rejection logging).
        sector: Signal sector.
        rejection_reason: Human-readable reason if not approved, None if approved.
        rejection_layer: Always "portfolio" for portfolio manager rejections.
    """

    ticker: str
    approved: bool
    rank: int
    confidence: float
    sector: str
    rejection_reason: Optional[str] = None
    rejection_layer: str = "portfolio"

    def __post_init__(self) -> None:
        if self.approved and self.rejection_reason is not None:
            raise ValueError(
                "Approved decision must not have a rejection_reason"
            )
        if not self.approved and self.rejection_reason is None:
            raise ValueError(
                "Rejected decision must have a rejection_reason"
            )


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Read-only snapshot of current portfolio state for allocation decisions.

    This is constructed by backtest_engine at the start of each day's
    BUY signal processing and passed to portfolio_manager functions.

    Args:
        open_tickers: Set of currently held ticker symbols.
        sector_position_counts: Sector name -> count of open positions in that sector.
        cooldowns: Ticker -> {until_date: pd.Timestamp, last_confidence: float}.
        cooldown_reentry_margin: Config value for reentry_confidence_margin.
    """

    open_tickers: frozenset[str]
    sector_position_counts: dict[str, int] = field(default_factory=dict)
    cooldowns: dict[str, dict[str, Any]] = field(default_factory=dict)
    cooldown_reentry_margin: float = 0.05
