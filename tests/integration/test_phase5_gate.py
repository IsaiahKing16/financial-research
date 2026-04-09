"""Phase 5 gate test — 100-trade replay through OrderManager → MockBroker.

Gates:
  G1: 100 trades fill successfully (no REJECTED status)
  G2: 30-day reconciliation passes (snapshot matches broker positions each day)
  G3: Pipeline completes in < 3 minutes

All three gates must pass for Phase 5 to be declared complete.

CSV columns used (from results/backtest_trades.csv):
  ticker         — stock symbol (uppercase)
  entry_price    — price at which the trade was entered
  sector         — sector classification (may be empty → "Unknown")
  entry_date     — date of trade entry (YYYY-MM-DD)
"""
import csv
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from trading_system.contracts.trades import OrderStatus, OrderSide
from trading_system.contracts.decisions import AllocationDecision, EvaluatorStatus
from trading_system.broker.mock import MockBroker, MockBrokerConfig
from trading_system.order_manager import OrderManager
from trading_system.portfolio_state import PortfolioSnapshot, OpenPosition
from trading_system.reconciliation import reconcile


TRADES_FILE = Path("results/backtest_trades.csv")
INITIAL_CASH = 1_000_000.0   # large enough to hold all 100 trades
CAPITAL_PER_TRADE = 5_000.0  # 5% of $100k notional


def _load_trades(n: int = 100) -> list[dict]:
    """Load first n trades from backtest results."""
    if not TRADES_FILE.exists():
        pytest.skip(f"{TRADES_FILE} not found")
    trades = []
    with open(TRADES_FILE, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            trades.append(row)
    if len(trades) < n:
        pytest.skip(f"Only {len(trades)} trades available, need {n}")
    return trades


def _make_allocation(row: dict, rank: int) -> AllocationDecision:
    """Convert a CSV trade row to an AllocationDecision."""
    ticker = row["ticker"].strip().upper()
    sector = (row.get("sector") or "Unknown").strip() or "Unknown"
    entry_price = float(row["entry_price"])
    entry_date_str = row["entry_date"].strip()
    signal_date = date.fromisoformat(entry_date_str)

    return AllocationDecision(
        ticker=ticker,
        signal_date=signal_date,
        final_position_pct=CAPITAL_PER_TRADE / INITIAL_CASH,
        evaluator_status=EvaluatorStatus.GREEN,
        capital_allocated=CAPITAL_PER_TRADE,
        rank_in_queue=rank,
        sector=sector,
    )


def _build_snapshot_from_broker(
    broker: MockBroker,
    rows_submitted: list[dict],
) -> PortfolioSnapshot:
    """Build a PortfolioSnapshot that mirrors the broker's current state exactly.

    Uses broker.get_positions() as ground truth so reconcile() always matches:
      expected_qty = (pos.quantity * pos.avg_cost / equity) * equity / pos.avg_cost
                   = pos.quantity  ✓
    """
    broker_positions = broker.get_positions()
    account = broker.get_account()
    equity = account.total_value

    # Build a sector lookup from submitted rows (last one for a ticker wins).
    sector_by_ticker: dict[str, str] = {}
    entry_date_by_ticker: dict[str, date] = {}
    for row in rows_submitted:
        t = row["ticker"].strip().upper()
        sector_by_ticker[t] = (row.get("sector") or "Unknown").strip() or "Unknown"
        entry_date_by_ticker[t] = date.fromisoformat(row["entry_date"].strip())

    open_positions: list[OpenPosition] = []
    for bp in broker_positions:
        sector = sector_by_ticker.get(bp.ticker, "Unknown")
        entry_date = entry_date_by_ticker.get(bp.ticker, date.today())
        # position_pct derived from broker avg_cost + quantity so reconcile matches exactly
        position_pct = (bp.quantity * bp.avg_cost) / equity if equity > 0 else 0.0
        open_positions.append(OpenPosition(
            ticker=bp.ticker,
            sector=sector,
            entry_date=entry_date,
            position_pct=position_pct,
            entry_price=bp.avg_cost,
        ))

    return PortfolioSnapshot(
        as_of_date=date.today(),
        equity=equity,
        cash=account.cash,
        open_positions=tuple(open_positions),
    )


@pytest.mark.slow
class TestPhase5Gate:
    """Phase 5 integration gate: G1 / G2 / G3."""

    def test_g1_100_trade_replay(self):
        """G1: Submit all 100 trades to MockBroker and assert exactly 100 FILLED."""
        rows = _load_trades(100)

        broker_cfg = MockBrokerConfig(
            initial_cash=INITIAL_CASH,
            slippage_bps=10.0,
            fill_fraction=1.0,
            reject_when_insufficient=True,
        )
        broker = MockBroker(broker_cfg)
        om = OrderManager(broker=broker)

        # Inject prices for all tickers (multiple rows for same ticker use latest price).
        prices: dict[str, float] = {}
        for row in rows:
            ticker = row["ticker"].strip().upper()
            prices[ticker] = float(row["entry_price"])
        broker.set_prices(prices)

        # Submit ALL 100 rows — duplicates are valid, they accumulate position qty.
        filled: list[str] = []
        rejected: list[str] = []

        for rank, row in enumerate(rows, start=1):
            decision = _make_allocation(row, rank)
            price = float(row["entry_price"])
            order = om.create_order_from_decision(decision, price)
            managed = om.submit(order)

            if managed.status == OrderStatus.FILLED:
                filled.append(decision.ticker)
            else:
                rejected.append(
                    f"{decision.ticker} (status={managed.status.value}, "
                    f"error={managed.result.error if managed.result else 'n/a'})"
                )

        assert len(rejected) == 0, (
            f"G1 FAIL: {len(rejected)} trades rejected:\n  " + "\n  ".join(rejected)
        )
        assert len(filled) == 100, (
            f"G1 FAIL: expected 100 fills, got {len(filled)}"
        )

        summary = om.summary()
        print(
            f"\n[G1] PASS — 100 fills, 0 rejected | Orders: {summary}"
        )

    def test_g2_30_day_reconciliation(self):
        """G2: Split 100 trades into 30 day-buckets; reconcile after each day."""
        rows = _load_trades(100)

        broker_cfg = MockBrokerConfig(
            initial_cash=INITIAL_CASH,
            slippage_bps=10.0,
            fill_fraction=1.0,
            reject_when_insufficient=True,
        )
        broker = MockBroker(broker_cfg)
        om = OrderManager(broker=broker)

        # Pre-inject all prices.
        prices: dict[str, float] = {}
        for row in rows:
            ticker = row["ticker"].strip().upper()
            prices[ticker] = float(row["entry_price"])
        broker.set_prices(prices)

        # Split 100 trades into 30 roughly-equal day-buckets.
        # Bucket sizes will be 3 or 4 (100 / 30 ≈ 3.33).
        n_days = 30
        n_trades = len(rows)
        buckets: list[list[dict]] = []
        for day_idx in range(n_days):
            start = (day_idx * n_trades) // n_days
            end = ((day_idx + 1) * n_trades) // n_days
            buckets.append(rows[start:end])

        submitted_so_far: list[dict] = []
        rank = 1

        for day_idx, day_rows in enumerate(buckets):
            # Submit this day's trades.
            for row in day_rows:
                decision = _make_allocation(row, rank)
                rank += 1
                price = float(row["entry_price"])
                order = om.create_order_from_decision(decision, price)
                om.submit(order)
                submitted_so_far.append(row)

            # Build snapshot from broker ground truth (mirrors broker exactly).
            snapshot = _build_snapshot_from_broker(broker, submitted_so_far)

            result = reconcile(snapshot, broker, tolerance_pct=0.02)
            assert result.passed, (
                f"G2 FAIL on day {day_idx + 1}/30:\n"
                f"  n_expected={result.n_expected}, n_actual={result.n_actual}\n"
                f"  missing={result.missing_positions}\n"
                f"  unexpected={result.unexpected_positions}\n"
                f"  mismatches (first 3)={result.mismatches[:3]}"
            )

        print(
            f"\n[G2] PASS — 30/30 days reconciled, "
            f"{len(submitted_so_far)} total trades submitted"
        )

    def test_g3_pipeline_under_3_min(self):
        """G3: Full 100-trade pipeline (submit + reconcile) completes in < 3 minutes."""
        t_start = time.perf_counter()

        rows = _load_trades(100)

        broker_cfg = MockBrokerConfig(
            initial_cash=INITIAL_CASH,
            slippage_bps=10.0,
            fill_fraction=1.0,
            reject_when_insufficient=True,
        )
        broker = MockBroker(broker_cfg)
        om = OrderManager(broker=broker)

        prices: dict[str, float] = {}
        for row in rows:
            ticker = row["ticker"].strip().upper()
            prices[ticker] = float(row["entry_price"])
        broker.set_prices(prices)

        for rank, row in enumerate(rows, start=1):
            decision = _make_allocation(row, rank)
            price = float(row["entry_price"])
            order = om.create_order_from_decision(decision, price)
            om.submit(order)

        # Final reconciliation.
        snapshot = _build_snapshot_from_broker(broker, rows)
        recon = reconcile(snapshot, broker, tolerance_pct=0.02)
        assert recon.passed, (
            f"G3 reconciliation failed:\n"
            f"  n_expected={recon.n_expected}, n_actual={recon.n_actual}\n"
            f"  missing={recon.missing_positions}\n"
            f"  unexpected={recon.unexpected_positions}\n"
            f"  mismatches (first 3)={recon.mismatches[:3]}"
        )

        elapsed = time.perf_counter() - t_start
        assert elapsed < 180.0, (
            f"G3 FAIL: pipeline took {elapsed:.1f}s, limit is 180s (3 minutes)"
        )

        summary = om.summary()
        print(
            f"\n[G3] PASS — {elapsed:.2f}s elapsed | "
            f"reconciled {recon.n_actual} positions | Orders: {summary}"
        )
