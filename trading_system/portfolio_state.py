"""
trading_system/portfolio_state.py — Phase 4 Portfolio Manager data schemas.

Pydantic v2 frozen models consumed by trading_system/portfolio_manager.py.
No business logic — schemas and computed properties only. All models are
immutable (model_config = {"frozen": True}) following the Phase 2/3 pattern.

Types:
    OpenPosition       — one currently-held position
    PortfolioSnapshot  — read-only snapshot of portfolio state on a given day
    RankedSignal       — a BUY candidate with its rank in the day's queue
    PMRejection        — structured reason a ranked signal failed portfolio checks
    AllocationResult   — final per-signal verdict (approved + optional rejection)

Design (see plan §3):
    - Distinct from trading_system/contracts/decisions.py::AllocationDecision
      which is a post-sizing output. These types are PRE-sizing.
    - `PMRejection.reason` is a Literal with 5 values so downstream histograms
      (T4.3) can't be polluted by free-form strings.
    - `PortfolioSnapshot.open_positions` is a tuple (hashable, order-stable)
      so Pydantic can freeze the model.

Plan: docs/superpowers/plans/2026-04-08-phase4-portfolio-manager-plan.md
"""
from __future__ import annotations

from datetime import date as Date
from typing import Dict, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from pattern_engine.contracts.finite_types import FiniteFloat


# ── OpenPosition ──────────────────────────────────────────────────────────────

class OpenPosition(BaseModel):
    """One currently-held position for portfolio-state purposes.

    This is a lightweight mirror of the trade row — only the fields the
    Portfolio Manager needs to reason about sector concentration and
    already-held checks. It is NOT a replacement for the trade record.
    """
    model_config = {"frozen": True}

    ticker: str = Field(min_length=1, max_length=10)
    sector: str = Field(min_length=1)
    entry_date: Date
    position_pct: FiniteFloat = Field(ge=0.0, le=1.0)
    entry_price: FiniteFloat = Field(gt=0.0)

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase; got '{v}'")
        return v


# ── PortfolioSnapshot ─────────────────────────────────────────────────────────

class PortfolioSnapshot(BaseModel):
    """Read-only snapshot of portfolio state on one trading day.

    Constructed by the walk-forward script at the start of each day before
    running the PM. Must be reconstructed (not mutated) after any approved
    allocation within the day — the `allocate_day` function does this
    internally via a running snapshot.

    Computed properties (sector_counts, sector_exposure_pct, cash_pct) are
    derived from open_positions + equity/cash on every access.
    """
    model_config = {"frozen": True}

    as_of_date: Date
    equity: FiniteFloat = Field(gt=0.0, description="Total portfolio equity (cash + positions)")
    cash: FiniteFloat = Field(ge=0.0, description="Available cash")
    open_positions: Tuple[OpenPosition, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def cash_must_not_exceed_equity(self) -> "PortfolioSnapshot":
        if self.cash > self.equity:
            raise ValueError(
                f"cash ({self.cash}) must not exceed equity ({self.equity})"
            )
        return self

    @property
    def n_open_positions(self) -> int:
        return len(self.open_positions)

    @property
    def sector_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in self.open_positions:
            counts[p.sector] = counts.get(p.sector, 0) + 1
        return counts

    @property
    def sector_exposure_pct(self) -> Dict[str, float]:
        exposure: Dict[str, float] = {}
        for p in self.open_positions:
            exposure[p.sector] = exposure.get(p.sector, 0.0) + p.position_pct
        return exposure

    @property
    def cash_pct(self) -> float:
        return self.cash / self.equity if self.equity > 0 else 0.0

    def contains_ticker(self, ticker: str) -> bool:
        return any(p.ticker == ticker for p in self.open_positions)


# ── RankedSignal ──────────────────────────────────────────────────────────────

class RankedSignal(BaseModel):
    """A BUY candidate with its 1-based rank in the day's priority queue.

    v1 ranking: sort by confidence descending, ticker ascending (deterministic
    tie-break). v2+ may introduce composite scores; the schema is stable.
    """
    model_config = {"frozen": True}

    ticker: str = Field(min_length=1, max_length=10)
    sector: str = Field(min_length=1)
    signal_date: Date
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1, description="1-based rank; 1 = highest priority")

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase; got '{v}'")
        return v


# ── PMRejection ───────────────────────────────────────────────────────────────

PMRejectionReason = Literal[
    "already_held",
    "cooldown",
    "sector_count_limit",
    "sector_pct_limit",
    "insufficient_capital",
]


class PMRejection(BaseModel):
    """Structured reason the PM rejected a ranked signal.

    The `reason` field is a Literal so T4.3 histograms are bounded. The
    `detail` field is a free-form string for audit context (e.g.,
    'sector Technology at 2/3 positions', 'cash 0.02 < min 0.02').
    """
    model_config = {"frozen": True}

    ticker: str = Field(min_length=1, max_length=10)
    sector: str = Field(min_length=1)
    signal_date: Date
    confidence: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    reason: PMRejectionReason
    detail: str = Field(min_length=1)

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase; got '{v}'")
        return v


# ── AllocationResult ──────────────────────────────────────────────────────────

class AllocationResult(BaseModel):
    """Per-signal verdict from allocate_day().

    Exactly one of the following is true:
      - approved=True and rejection=None
      - approved=False and rejection is not None

    The walk-forward script filters the returned list to approved=True to
    get the pass-through set for the risk engine.
    """
    model_config = {"frozen": True}

    ticker: str = Field(min_length=1, max_length=10)
    sector: str = Field(min_length=1)
    signal_date: Date
    confidence: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    approved: bool
    rejection: Optional[PMRejection] = None

    @model_validator(mode="after")
    def approval_state_consistency(self) -> "AllocationResult":
        if self.approved and self.rejection is not None:
            raise ValueError(
                "approved=True must not have a rejection attached"
            )
        if not self.approved and self.rejection is None:
            raise ValueError(
                "approved=False must have a rejection attached"
            )
        return self

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase; got '{v}'")
        return v
