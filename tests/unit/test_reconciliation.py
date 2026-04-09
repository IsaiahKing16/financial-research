"""Tests for reconciliation — portfolio vs broker position comparison."""
import pytest
from datetime import date, datetime, timezone

from trading_system.portfolio_state import PortfolioSnapshot, OpenPosition
from trading_system.broker.mock import MockBroker, MockBrokerConfig
from trading_system.broker.base import Order
from trading_system.contracts.trades import OrderSide
from trading_system.reconciliation import reconcile, ReconciliationResult, PositionMismatch


def _snapshot(positions: list[OpenPosition] | None = None) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of_date=date(2024, 1, 2),
        equity=100_000.0,
        cash=90_000.0,
        open_positions=tuple(positions or []),
    )


def _position(ticker: str, pct: float = 0.05) -> OpenPosition:
    return OpenPosition(
        ticker=ticker,
        sector="Technology",
        entry_date=date(2024, 1, 1),
        position_pct=pct,
        entry_price=100.0,
    )


def _broker_with_positions(
    holdings: dict[str, float],
    cash: float = 90_000.0,
) -> MockBroker:
    """Create a MockBroker with specific position quantities."""
    b = MockBroker(MockBrokerConfig(initial_cash=cash, slippage_bps=0.0))
    prices = {t: 100.0 for t in holdings}
    b.set_prices(prices)
    for ticker, qty in holdings.items():
        if qty > 0:
            b.submit_order(Order(
                order_id=f"setup-{ticker}",
                ticker=ticker,
                side=OrderSide.BUY,
                quantity=qty,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ))
    return b


class TestReconcileMatch:
    def test_empty_both(self):
        snap = _snapshot()
        broker = MockBroker(MockBrokerConfig(initial_cash=90_000.0))
        result = reconcile(snap, broker)
        assert result.passed is True
        assert result.mismatches == ()

    def test_matching_positions(self):
        # Snapshot expects 50 shares of AAPL (5% of 100k equity / 100 price)
        snap = _snapshot([_position("AAPL", pct=0.05)])
        broker = _broker_with_positions({"AAPL": 50.0})
        result = reconcile(snap, broker)
        assert result.passed is True

    def test_within_tolerance(self):
        # Expect 50, actual 51 → 2% diff, within 5% tolerance
        snap = _snapshot([_position("AAPL", pct=0.05)])
        broker = _broker_with_positions({"AAPL": 51.0})
        result = reconcile(snap, broker, tolerance_pct=0.05)
        assert result.passed is True


class TestReconcileMismatch:
    def test_quantity_mismatch(self):
        # Expect 50, actual 40 → 20% diff, beyond 5% tolerance
        snap = _snapshot([_position("AAPL", pct=0.05)])
        broker = _broker_with_positions({"AAPL": 40.0})
        result = reconcile(snap, broker)
        assert result.passed is False
        assert len(result.mismatches) == 1
        assert result.mismatches[0].ticker == "AAPL"

    def test_unexpected_position(self):
        # Broker has MSFT, snapshot does not
        snap = _snapshot()
        broker = _broker_with_positions({"MSFT": 10.0})
        result = reconcile(snap, broker)
        assert result.passed is False
        assert "MSFT" in result.unexpected_positions

    def test_missing_position(self):
        # Snapshot expects AAPL, broker has nothing
        snap = _snapshot([_position("AAPL")])
        broker = MockBroker(MockBrokerConfig(initial_cash=90_000.0))
        result = reconcile(snap, broker)
        assert result.passed is False
        assert "AAPL" in result.missing_positions


class TestReconcileTolerance:
    def test_exact_boundary_passes(self):
        # Exactly at 5% tolerance → should pass
        snap = _snapshot([_position("AAPL", pct=0.05)])
        broker = _broker_with_positions({"AAPL": 47.5})  # 5% below 50
        result = reconcile(snap, broker, tolerance_pct=0.05)
        assert result.passed is True

    def test_just_beyond_boundary_fails(self):
        snap = _snapshot([_position("AAPL", pct=0.05)])
        broker = _broker_with_positions({"AAPL": 47.0})  # 6% below 50
        result = reconcile(snap, broker, tolerance_pct=0.05)
        assert result.passed is False

    def test_custom_tolerance(self):
        snap = _snapshot([_position("AAPL", pct=0.05)])
        broker = _broker_with_positions({"AAPL": 40.0})  # 20% off
        result = reconcile(snap, broker, tolerance_pct=0.25)
        assert result.passed is True


class TestReconciliationResult:
    def test_counts(self):
        snap = _snapshot([_position("AAPL"), _position("MSFT")])
        broker = _broker_with_positions({"AAPL": 50.0, "MSFT": 50.0})
        result = reconcile(snap, broker)
        assert result.n_expected == 2
        assert result.n_actual == 2


class TestReconcileCLIGuard:
    def test_cli_script_raises_not_configured(self):
        """reconcile.py CLI should refuse to run with placeholder values."""
        import subprocess
        import sys
        from pathlib import Path

        project_root = Path(__file__).parents[2]
        script = project_root / "scripts" / "reconcile.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(project_root)},
        )
        assert result.returncode != 0
        assert "placeholder" in result.stderr.lower() or "not configured" in result.stderr.lower()
