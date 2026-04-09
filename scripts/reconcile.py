"""Standalone reconciliation CLI — compare SharedState checkpoint vs broker."""
from __future__ import annotations

import argparse
import json
import sys

from trading_system.contracts.state import SharedState
from trading_system.portfolio_state import PortfolioSnapshot, OpenPosition
from trading_system.broker.mock import MockBroker, MockBrokerConfig
from trading_system.reconciliation import reconcile


def _snapshot_from_shared_state(state: SharedState) -> PortfolioSnapshot:
    """Build PortfolioSnapshot from SharedState for reconciliation.

    NOTE: position_pct and entry_price are placeholders. This script produces
    meaningful results only when SharedState carries actual sizing data
    (Phase 8 orchestrator responsibility). For Phase 5, use the reconcile()
    function directly with a real PortfolioSnapshot.
    """
    positions = []
    for ticker in state.positions.open_tickers:
        sector = state.positions.ticker_sectors.get(ticker, "Unknown")
        positions.append(OpenPosition(
            ticker=ticker,
            sector=sector,
            entry_date=state.trading_date,
            position_pct=0.05,  # Placeholder — Phase 8 will provide actual sizing
            entry_price=100.0,  # Placeholder — Phase 8 will provide actual price
        ))
    return PortfolioSnapshot(
        as_of_date=state.trading_date,
        equity=state.total_equity,
        cash=state.equity.cash,
        open_positions=tuple(positions),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile positions vs broker")
    parser.add_argument("checkpoint", help="Path to SharedState JSON checkpoint")
    parser.add_argument(
        "--tolerance", type=float, default=0.05,
        help="Tolerance percentage (default: 0.05 = 5%%)",
    )
    args = parser.parse_args()

    with open(args.checkpoint, "r") as f:
        data = json.load(f)
    state = SharedState.model_validate(data)

    snapshot = _snapshot_from_shared_state(state)
    broker = MockBroker()  # Real broker in Phase 8

    result = reconcile(snapshot, broker, tolerance_pct=args.tolerance)

    print(f"Reconciliation as of {result.as_of}")
    print(f"Expected positions: {result.n_expected}")
    print(f"Actual positions:   {result.n_actual}")

    if result.mismatches:
        print("\nMISMATCHES:")
        for m in result.mismatches:
            print(f"  {m.ticker}: expected={m.expected_quantity:.2f}, "
                  f"actual={m.actual_quantity:.2f}, delta={m.delta:+.2f} "
                  f"({m.delta_pct:.1%})")

    if result.missing_positions:
        print(f"\nMISSING (in snapshot, not broker): {', '.join(result.missing_positions)}")

    if result.unexpected_positions:
        print(f"\nUNEXPECTED (in broker, not snapshot): {', '.join(result.unexpected_positions)}")

    if result.passed:
        print("\nRESULT: PASS")
    else:
        print("\nRESULT: FAIL")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
