"""
test_contracts.py — Unit tests for Pydantic contract models.

Tests that:
    - Valid data constructs successfully
    - Invalid data raises ValidationError
    - Frozen models prevent mutation
    - Validators catch impossible states

Linear: SLE-57
"""

import pytest
from datetime import date
from pydantic import ValidationError

from rebuild_phase_3z.fppe.pattern_engine.contracts.signals import (
    SignalRecord,
    SignalDirection,
    SignalSource,
)
from rebuild_phase_3z.fppe.trading_system.contracts.trades import (
    TradeRecord,
    PositionRecord,
    DailySnapshot,
    ExitReason,
)


# ─── SignalRecord ────────────────────────────────────────────────

class TestSignalRecord:
    """Tests for the SignalRecord Pydantic model."""

    def _valid_signal(self, **overrides):
        """Factory for valid signal data."""
        defaults = {
            "date": date(2024, 1, 2),
            "ticker": "AAPL",
            "signal": SignalDirection.BUY,
            "confidence": 0.72,
            "signal_source": SignalSource.KNN,
            "sector": "Technology",
            "n_matches": 50,
            "raw_prob": 0.68,
            "mean_7d_return": 1.23,
        }
        defaults.update(overrides)
        return SignalRecord(**defaults)

    def test_valid_construction(self):
        s = self._valid_signal()
        assert s.ticker == "AAPL"
        assert s.confidence == 0.72
        assert s.signal == SignalDirection.BUY

    def test_frozen_immutability(self):
        s = self._valid_signal()
        with pytest.raises(ValidationError):
            s.confidence = 0.99

    def test_confidence_bounds(self):
        self._valid_signal(confidence=0.0)   # lower bound OK
        self._valid_signal(confidence=1.0)   # upper bound OK
        with pytest.raises(ValidationError):
            self._valid_signal(confidence=-0.1)
        with pytest.raises(ValidationError):
            self._valid_signal(confidence=1.1)

    def test_ticker_must_be_uppercase(self):
        with pytest.raises(ValidationError, match="uppercase"):
            self._valid_signal(ticker="aapl")

    def test_ticker_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            self._valid_signal(ticker="")

    def test_n_matches_non_negative(self):
        self._valid_signal(n_matches=0)  # OK
        with pytest.raises(ValidationError):
            self._valid_signal(n_matches=-1)

    def test_n_matches_unreasonably_high(self):
        with pytest.raises(ValidationError, match="unreasonably high"):
            self._valid_signal(n_matches=100_000)

    def test_signal_direction_enum(self):
        for direction in [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]:
            s = self._valid_signal(signal=direction)
            assert s.signal == direction

    def test_invalid_signal_direction(self):
        with pytest.raises(ValidationError):
            self._valid_signal(signal="MAYBE")

    def test_string_signal_coercion(self):
        """SignalDirection is a str enum — raw strings should work."""
        s = self._valid_signal(signal="BUY")
        assert s.signal == SignalDirection.BUY


# ─── PositionRecord ─────────────────────────────────────────────

class TestPositionRecord:
    """Tests for the PositionRecord Pydantic model."""

    def _valid_position(self, **overrides):
        defaults = {
            "trade_id": 1,
            "ticker": "AAPL",
            "sector": "Technology",
            "entry_date": date(2024, 1, 3),
            "raw_entry_price": 185.50,
            "entry_price": 185.69,  # After 10 bps slippage
            "shares": 2.7,
            "position_pct": 0.05,
            "confidence_at_entry": 0.72,
            "stop_loss_price": 179.20,
            "atr_pct_at_entry": 0.02,
        }
        defaults.update(overrides)
        return PositionRecord(**defaults)

    def test_valid_construction(self):
        p = self._valid_position()
        assert p.ticker == "AAPL"
        assert p.shares == 2.7

    def test_shares_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._valid_position(shares=0)
        with pytest.raises(ValidationError):
            self._valid_position(shares=-1)

    def test_entry_price_sanity(self):
        """Entry price should not be wildly below raw (friction adds cost)."""
        with pytest.raises(ValidationError, match="suspiciously below"):
            self._valid_position(
                raw_entry_price=100.0,
                entry_price=90.0,  # 10% below — impossible with ~13 bps friction
            )

    def test_frozen(self):
        p = self._valid_position()
        with pytest.raises(ValidationError):
            p.shares = 10.0


# ─── TradeRecord ─────────────────────────────────────────────────

class TestTradeRecord:
    """Tests for the TradeRecord Pydantic model."""

    def _valid_trade(self, **overrides):
        defaults = {
            "trade_id": 1,
            "ticker": "AAPL",
            "sector": "Technology",
            "direction": "LONG",
            "entry_date": date(2024, 1, 3),
            "entry_price": 185.69,
            "exit_date": date(2024, 1, 17),
            "exit_price": 190.50,
            "position_pct": 0.05,
            "shares": 2.7,
            "gross_pnl": 12.99,
            "entry_friction_cost": 0.65,
            "exit_friction_cost": 0.67,
            "slippage_cost": 0.88,
            "spread_cost": 0.44,
            # total_costs = entry(0.65) + exit(0.67) + slippage(0.88) + spread(0.44) = 2.64
            "total_costs": 2.64,
            "net_pnl": 10.35,  # gross_pnl(12.99) - total_costs(2.64)
            "holding_days": 10,
            "exit_reason": ExitReason.MAX_HOLD,
            "confidence_at_entry": 0.72,
        }
        defaults.update(overrides)
        return TradeRecord(**defaults)

    def test_valid_construction(self):
        t = self._valid_trade()
        assert t.net_pnl == 10.35
        assert t.exit_reason == ExitReason.MAX_HOLD

    def test_exit_before_entry_rejected(self):
        with pytest.raises(ValidationError, match="before entry_date"):
            self._valid_trade(
                entry_date=date(2024, 1, 17),
                exit_date=date(2024, 1, 3),
            )

    def test_costs_consistency(self):
        """total_costs must equal entry + exit friction + slippage + spread."""
        with pytest.raises(ValidationError, match="total_costs"):
            self._valid_trade(
                entry_friction_cost=1.00,
                exit_friction_cost=1.00,
                # slippage_cost=0.88, spread_cost=0.44 from defaults
                total_costs=5.00,  # Should be 1.00+1.00+0.88+0.44 = 3.32
            )

    def test_holding_days_minimum(self):
        with pytest.raises(ValidationError):
            self._valid_trade(holding_days=0)

    def test_exit_reason_enum(self):
        for reason in ExitReason:
            t = self._valid_trade(exit_reason=reason)
            assert t.exit_reason == reason

    def test_string_exit_reason(self):
        """ExitReason is a str enum — raw strings should work."""
        t = self._valid_trade(exit_reason="stop_loss")
        assert t.exit_reason == ExitReason.STOP_LOSS


# ─── DailySnapshot ──────────────────────────────────────────────

class TestDailySnapshot:
    """Tests for the DailySnapshot Pydantic model."""

    def _valid_snapshot(self, **overrides):
        defaults = {
            "date": date(2024, 1, 3),
            "equity": 10050.00,
            "cash": 9550.00,
            "invested_capital": 500.00,
            "gross_exposure": 0.05,
            "open_positions": 1,
            "daily_return": 0.005,
            "cumulative_return": 0.005,
            "drawdown_from_peak": 0.0,
            "cash_yield_today": 0.12,
            "strategy_return_excl_cash": 0.005,
            "strategy_return_incl_cash": 0.0051,
        }
        defaults.update(overrides)
        return DailySnapshot(**defaults)

    def test_valid_construction(self):
        d = self._valid_snapshot()
        assert d.equity == 10050.00
        assert d.open_positions == 1

    def test_equity_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._valid_snapshot(equity=0)
        with pytest.raises(ValidationError):
            self._valid_snapshot(equity=-100)

    def test_drawdown_bounds(self):
        self._valid_snapshot(drawdown_from_peak=0.0)   # No drawdown
        self._valid_snapshot(drawdown_from_peak=0.20)  # 20% drawdown
        with pytest.raises(ValidationError):
            self._valid_snapshot(drawdown_from_peak=-0.1)
        with pytest.raises(ValidationError):
            self._valid_snapshot(drawdown_from_peak=1.5)
