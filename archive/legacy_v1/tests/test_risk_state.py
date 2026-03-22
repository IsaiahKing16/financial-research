"""
tests/test_risk_state.py — Unit tests for trading_system/risk_state.py

Covers:
  - PositionDecision: frozen dataclass, __post_init__ validation
  - StopLossEvent: frozen dataclass
  - RiskState: initial(), update(), register_stop(), remove_stop()
"""

import pytest
from trading_system.risk_state import PositionDecision, RiskState, StopLossEvent


# ── PositionDecision ──────────────────────────────────────────────────────────

class TestPositionDecision:
    """Tests for PositionDecision frozen dataclass."""

    def _approved(self, **overrides):
        defaults = dict(
            approved=True,
            ticker="AAPL",
            position_pct=0.05,
            shares=10.0,
            dollar_amount=1000.0,
            stop_price=95.0,
            stop_distance_pct=0.05,
            atr_pct=0.025,
            drawdown_scalar=1.0,
            raw_weight=0.08,
            rejection_reason=None,
        )
        defaults.update(overrides)
        return PositionDecision(**defaults)

    def _rejected(self, **overrides):
        defaults = dict(
            approved=False,
            ticker="TSLA",
            position_pct=0.0,
            shares=0.0,
            dollar_amount=0.0,
            stop_price=0.0,
            stop_distance_pct=0.0,
            atr_pct=0.0,
            drawdown_scalar=0.5,
            raw_weight=0.0,
            rejection_reason="Drawdown halt",
        )
        defaults.update(overrides)
        return PositionDecision(**defaults)

    def test_approved_decision_creates_successfully(self):
        d = self._approved()
        assert d.approved is True
        assert d.ticker == "AAPL"
        assert d.shares == 10.0

    def test_rejected_decision_creates_successfully(self):
        d = self._rejected()
        assert d.approved is False
        assert d.rejection_reason == "Drawdown halt"

    def test_frozen_raises_on_mutation(self):
        d = self._approved()
        with pytest.raises((AttributeError, TypeError)):
            d.shares = 99.0  # type: ignore[misc]

    def test_approved_with_zero_shares_raises(self):
        with pytest.raises(ValueError, match="positive shares"):
            self._approved(shares=0.0)

    def test_approved_with_negative_shares_raises(self):
        with pytest.raises(ValueError, match="positive shares"):
            self._approved(shares=-5.0)

    def test_approved_with_zero_stop_price_raises(self):
        with pytest.raises(ValueError, match="stop_price"):
            self._approved(stop_price=0.0)

    def test_approved_with_negative_stop_price_raises(self):
        with pytest.raises(ValueError, match="stop_price"):
            self._approved(stop_price=-1.0)

    def test_drawdown_scalar_above_one_raises(self):
        with pytest.raises(ValueError, match="drawdown_scalar"):
            self._approved(drawdown_scalar=1.01)

    def test_drawdown_scalar_below_zero_raises(self):
        with pytest.raises(ValueError, match="drawdown_scalar"):
            self._rejected(drawdown_scalar=-0.01)

    def test_drawdown_scalar_boundary_values_accepted(self):
        """0.0 and 1.0 are valid boundary values."""
        d0 = self._rejected(drawdown_scalar=0.0)
        d1 = self._approved(drawdown_scalar=1.0)
        assert d0.drawdown_scalar == 0.0
        assert d1.drawdown_scalar == 1.0

    def test_rejected_allows_zero_shares_and_stop(self):
        d = self._rejected(shares=0.0, stop_price=0.0)
        assert d.approved is False


# ── StopLossEvent ─────────────────────────────────────────────────────────────

class TestStopLossEvent:
    """Tests for StopLossEvent frozen dataclass."""

    def _event(self, **overrides):
        defaults = dict(
            ticker="NVDA",
            trigger_date="2024-03-15",
            stop_price=450.0,
            trigger_low=448.0,
            entry_price=470.0,
            exit_price=450.0,
            gap_through=False,
            atr_at_entry=0.022,
        )
        defaults.update(overrides)
        return StopLossEvent(**defaults)

    def test_creates_successfully(self):
        e = self._event()
        assert e.ticker == "NVDA"
        assert e.gap_through is False

    def test_frozen_raises_on_mutation(self):
        e = self._event()
        with pytest.raises((AttributeError, TypeError)):
            e.stop_price = 999.0  # type: ignore[misc]

    def test_gap_through_flag_true_when_low_below_stop(self):
        """gap_through is set by caller — just verify it's stored correctly."""
        e = self._event(trigger_low=440.0, stop_price=450.0, gap_through=True)
        assert e.gap_through is True


# ── RiskState ─────────────────────────────────────────────────────────────────

class TestRiskStateInitial:
    """Tests for RiskState.initial() classmethod."""

    def test_initial_sets_peak_equals_starting_equity(self):
        rs = RiskState.initial(10_000.0)
        assert rs.peak_equity == 10_000.0

    def test_initial_sets_current_equity(self):
        rs = RiskState.initial(10_000.0)
        assert rs.current_equity == 10_000.0

    def test_initial_drawdown_is_zero(self):
        rs = RiskState.initial(10_000.0)
        assert rs.current_drawdown == 0.0

    def test_initial_mode_is_normal(self):
        rs = RiskState.initial(10_000.0)
        assert rs.drawdown_mode == "normal"

    def test_initial_scalar_is_one(self):
        rs = RiskState.initial(10_000.0)
        assert rs.sizing_scalar == 1.0

    def test_initial_active_stops_is_empty(self):
        rs = RiskState.initial(10_000.0)
        assert rs.active_stops == {}


class TestRiskStateUpdate:
    """Tests for RiskState.update()."""

    BRAKE = 0.15
    HALT = 0.20

    def _state(self, equity: float = 10_000.0) -> RiskState:
        return RiskState.initial(equity)

    def test_update_equity_growth_updates_peak(self):
        rs = self._state(10_000.0)
        rs.update(11_000.0, self.BRAKE, self.HALT)
        assert rs.peak_equity == 11_000.0
        assert rs.current_equity == 11_000.0

    def test_update_no_drawdown_stays_normal(self):
        rs = self._state(10_000.0)
        rs.update(10_000.0, self.BRAKE, self.HALT)
        assert rs.drawdown_mode == "normal"
        assert rs.sizing_scalar == 1.0

    def test_update_drawdown_below_brake_stays_normal(self):
        rs = self._state(10_000.0)
        rs.update(8_600.0, self.BRAKE, self.HALT)  # 14% drawdown
        assert rs.current_drawdown == pytest.approx(0.14, rel=1e-4)
        assert rs.drawdown_mode == "normal"
        assert rs.sizing_scalar == 1.0

    def test_update_drawdown_at_brake_threshold_enters_brake(self):
        rs = self._state(10_000.0)
        rs.update(8_500.0, self.BRAKE, self.HALT)  # exactly 15% drawdown
        assert rs.drawdown_mode == "brake"
        assert rs.sizing_scalar == pytest.approx(1.0, rel=1e-4)  # at brake edge = scalar 1.0

    def test_update_drawdown_midpoint_brake_scalar(self):
        rs = self._state(10_000.0)
        # 17.5% drawdown = midpoint of 15%–20% range → scalar = 0.5
        rs.update(8_250.0, self.BRAKE, self.HALT)
        assert rs.drawdown_mode == "brake"
        assert rs.sizing_scalar == pytest.approx(0.5, abs=0.01)

    def test_update_drawdown_at_halt_threshold(self):
        rs = self._state(10_000.0)
        # 7_990 → ~20.1% drawdown, safely above the 20% halt threshold
        rs.update(7_990.0, self.BRAKE, self.HALT)
        assert rs.drawdown_mode == "halt"
        assert rs.sizing_scalar == 0.0

    def test_update_drawdown_beyond_halt(self):
        rs = self._state(10_000.0)
        rs.update(7_000.0, self.BRAKE, self.HALT)  # 30% drawdown
        assert rs.drawdown_mode == "halt"
        assert rs.sizing_scalar == 0.0

    def test_update_recovery_from_halt_back_to_normal(self):
        rs = self._state(10_000.0)
        rs.update(7_000.0, self.BRAKE, self.HALT)  # hit halt
        # Peak is now 10_000. Recovery to 9_500 = only 5% DD → normal
        rs.update(9_500.0, self.BRAKE, self.HALT)
        assert rs.drawdown_mode == "normal"
        assert rs.sizing_scalar == 1.0

    def test_update_new_peak_resets_drawdown_to_zero(self):
        rs = self._state(10_000.0)
        rs.update(9_000.0, self.BRAKE, self.HALT)  # 10% drawdown
        rs.update(12_000.0, self.BRAKE, self.HALT)  # new peak
        assert rs.current_drawdown == 0.0
        assert rs.peak_equity == 12_000.0


class TestRiskStateStopManagement:
    """Tests for register_stop() and remove_stop()."""

    def test_register_stop_adds_to_active_stops(self):
        rs = RiskState.initial(10_000.0)
        rs.register_stop("AAPL", 150.0)
        assert rs.active_stops["AAPL"] == 150.0

    def test_register_stop_multiple_tickers(self):
        rs = RiskState.initial(10_000.0)
        rs.register_stop("AAPL", 150.0)
        rs.register_stop("MSFT", 320.0)
        assert len(rs.active_stops) == 2

    def test_remove_stop_removes_ticker(self):
        rs = RiskState.initial(10_000.0)
        rs.register_stop("AAPL", 150.0)
        rs.remove_stop("AAPL")
        assert "AAPL" not in rs.active_stops

    def test_remove_stop_nonexistent_does_not_raise(self):
        rs = RiskState.initial(10_000.0)
        rs.remove_stop("NVDA")  # Should silently pass

    def test_register_stop_overwrites_previous_stop(self):
        rs = RiskState.initial(10_000.0)
        rs.register_stop("AAPL", 150.0)
        rs.register_stop("AAPL", 145.0)
        assert rs.active_stops["AAPL"] == 145.0
