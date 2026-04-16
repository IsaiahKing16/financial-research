"""Position reconciliation — compare portfolio snapshot vs broker state."""
from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from trading_system.broker.base import BaseBroker
from trading_system.portfolio_state import PortfolioSnapshot


class PositionMismatch(BaseModel):
    """A single position quantity discrepancy."""
    model_config = {"frozen": True}

    ticker: str
    expected_quantity: float
    actual_quantity: float
    delta: float
    delta_pct: float


class ReconciliationResult(BaseModel):
    """Result of comparing portfolio snapshot vs broker positions."""
    model_config = {"frozen": True}

    as_of: datetime
    n_expected: int
    n_actual: int
    mismatches: tuple[PositionMismatch, ...] = ()
    unexpected_positions: tuple[str, ...] = ()
    missing_positions: tuple[str, ...] = ()
    passed: bool


def reconcile(
    snapshot: PortfolioSnapshot,
    broker: BaseBroker,
    tolerance_pct: float = 0.05,
) -> ReconciliationResult:
    """Compare PortfolioSnapshot positions against broker positions.

    A position passes if abs(expected - actual) / expected <= tolerance_pct.
    Missing or unexpected positions always fail.

    Known limitation: derives shares as position_pct * equity / entry_price,
    which is an approximation. In Phase 8 with real slippage/partial fills,
    OpenPosition should gain a `quantity` field for exact tracking.
    For Phase 5 (mock broker, deterministic fills), this approximation is exact.
    """
    # Build expected quantities from snapshot
    expected: dict[str, float] = {}
    for pos in snapshot.open_positions:
        expected[pos.ticker] = (pos.position_pct * snapshot.equity) / pos.entry_price

    # Build actual quantities from broker
    broker_positions = broker.get_positions()
    actual: dict[str, float] = {p.ticker: p.quantity for p in broker_positions}

    expected_tickers = set(expected)
    actual_tickers = set(actual)

    missing = tuple(sorted(expected_tickers - actual_tickers))
    unexpected = tuple(sorted(actual_tickers - expected_tickers))

    mismatches: list[PositionMismatch] = []
    for ticker in expected_tickers & actual_tickers:
        exp_qty = expected[ticker]
        act_qty = actual[ticker]
        delta = act_qty - exp_qty
        delta_pct = abs(delta / exp_qty) if exp_qty != 0 else float("inf")
        if delta_pct > tolerance_pct:
            mismatches.append(PositionMismatch(
                ticker=ticker,
                expected_quantity=exp_qty,
                actual_quantity=act_qty,
                delta=delta,
                delta_pct=delta_pct,
            ))

    passed = len(mismatches) == 0 and len(missing) == 0 and len(unexpected) == 0

    return ReconciliationResult(
        as_of=datetime.now(UTC),
        n_expected=len(expected),
        n_actual=len(actual),
        mismatches=tuple(mismatches),
        unexpected_positions=unexpected,
        missing_positions=missing,
        passed=passed,
    )
