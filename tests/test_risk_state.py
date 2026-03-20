"""Unit tests for trading_system.risk_state — PHASE2_SYSTEM_DESIGN.md Section 8.2."""

from types import SimpleNamespace

import pytest

from trading_system.config import RiskConfig
from trading_system.risk_state import (
    PositionDecision,
    RiskState,
    StopLossEvent,
)


def test_initial_state() -> None:
    rs = RiskState.initial(10_000.0)
    assert rs.peak_equity == 10_000.0
    assert rs.current_equity == 10_000.0
    assert rs.current_drawdown == 0.0
    assert rs.drawdown_mode == "normal"
    assert rs.sizing_scalar == 1.0
    assert rs.active_stops == {}
    assert rs.daily_atr_cache == {}


def test_update_new_peak() -> None:
    rs = RiskState.initial(100_000.0)
    cfg = RiskConfig()
    rs.update(105_000.0, cfg)
    assert rs.peak_equity == 105_000.0
    assert rs.current_equity == 105_000.0
    assert rs.current_drawdown == 0.0
    assert rs.drawdown_mode == "normal"


def test_update_drawdown() -> None:
    rs = RiskState.initial(100_000.0)
    cfg = RiskConfig()
    rs.update(90_000.0, cfg)
    assert rs.peak_equity == 100_000.0
    assert pytest.approx(rs.current_drawdown) == 0.10
    assert rs.drawdown_mode == "normal"


def test_update_mode_transitions() -> None:
    rs = RiskState.initial(100_000.0)
    cfg = RiskConfig(
        drawdown_brake_threshold=0.15,
        drawdown_halt_threshold=0.20,
    )
    rs.update(100_000.0, cfg)
    assert rs.drawdown_mode == "normal"

    rs.update(85_000.0, cfg)
    assert rs.drawdown_mode == "brake"
    assert pytest.approx(rs.sizing_scalar) == 1.0

    rs.update(80_000.0, cfg)
    assert rs.drawdown_mode == "halt"
    assert rs.sizing_scalar == 0.0

    rs.update(75_000.0, cfg)
    assert rs.drawdown_mode == "halt"

    rs.update(86_000.0, cfg)
    assert rs.drawdown_mode == "normal"
    assert rs.sizing_scalar == 1.0


def test_register_and_remove_stop() -> None:
    rs = RiskState.initial(1.0)
    rs.register_stop("SPY", 400.0)
    assert rs.active_stops == {"SPY": 400.0}
    rs.remove_stop("SPY")
    assert rs.active_stops == {}
    rs.remove_stop("MISSING")
    assert rs.active_stops == {}


def test_atr_cache() -> None:
    rs = RiskState.initial(10_000.0)
    rs.daily_atr_cache["AAPL"] = 0.03
    rs.daily_atr_cache["MSFT"] = 0.025
    assert rs.daily_atr_cache["AAPL"] == 0.03
    assert len(rs.daily_atr_cache) == 2


def test_multiple_stops() -> None:
    rs = RiskState.initial(1.0)
    rs.register_stop("A", 10.0)
    rs.register_stop("B", 20.0)
    rs.register_stop("C", 30.0)
    assert set(rs.active_stops.keys()) == {"A", "B", "C"}
    rs.remove_stop("B")
    assert rs.active_stops == {"A": 10.0, "C": 30.0}


def test_zero_peak_equity() -> None:
    rs = RiskState.initial(0.0)
    cfg = RiskConfig()
    assert rs.current_drawdown == 0.0
    rs.update(1_000.0, cfg)
    assert rs.peak_equity == 1_000.0
    assert rs.current_equity == 1_000.0
    assert rs.current_drawdown == 0.0


def test_drawdown_scalar_boundary_values() -> None:
    rs = RiskState.initial(100_000.0)
    cfg = RiskConfig(drawdown_brake_threshold=0.15, drawdown_halt_threshold=0.20)

    rs.update(85_000.0, cfg)
    assert rs.drawdown_mode == "brake"
    assert pytest.approx(rs.sizing_scalar) == 1.0

    rs = RiskState.initial(100_000.0)
    rs.update(80_000.0, cfg)
    assert rs.drawdown_mode == "halt"
    assert rs.sizing_scalar == 0.0

    # Midpoint drawdown: 17.5% (halfway between brake 15% and halt 20%)
    rs = RiskState.initial(100_000.0)
    rs.update(82_500.0, cfg)
    assert rs.drawdown_mode == "brake"
    assert pytest.approx(rs.sizing_scalar) == 0.5


def test_state_immutability_where_expected() -> None:
    d = PositionDecision(
        approved=True,
        ticker="X",
        position_pct=0.05,
        shares=10.0,
        dollar_amount=500.0,
        stop_price=40.0,
        stop_distance_pct=0.06,
        atr_pct=0.03,
        drawdown_scalar=1.0,
        raw_weight=0.05,
        rejection_reason=None,
    )
    with pytest.raises(AttributeError):
        d.approved = False  # type: ignore[misc]

    ev = StopLossEvent(
        ticker="X",
        trigger_date="2024-01-02",
        stop_price=40.0,
        trigger_low=39.0,
        entry_price=45.0,
        exit_price=41.0,
        gap_through=False,
        atr_at_entry=0.03,
    )
    with pytest.raises(AttributeError):
        ev.ticker = "Y"  # type: ignore[misc]

    rs = RiskState.initial(100.0)
    rs.peak_equity = 200.0
    assert rs.peak_equity == 200.0


def test_position_decision_validation() -> None:
    with pytest.raises(ValueError, match="positive shares"):
        PositionDecision(
            approved=True,
            ticker="X",
            position_pct=0.0,
            shares=0.0,
            dollar_amount=0.0,
            stop_price=1.0,
            stop_distance_pct=0.1,
            atr_pct=0.05,
            drawdown_scalar=1.0,
            raw_weight=0.0,
            rejection_reason="x",
        )
    with pytest.raises(ValueError, match="drawdown_scalar"):
        PositionDecision(
            approved=False,
            ticker="X",
            position_pct=0.0,
            shares=0.0,
            dollar_amount=0.0,
            stop_price=0.0,
            stop_distance_pct=0.0,
            atr_pct=0.0,
            drawdown_scalar=1.5,
            raw_weight=0.0,
            rejection_reason="halt",
        )


def test_stop_loss_event_validation() -> None:
    with pytest.raises(ValueError, match="ticker"):
        StopLossEvent(
            ticker="",
            trigger_date="2024-01-01",
            stop_price=1.0,
            trigger_low=0.5,
            entry_price=2.0,
            exit_price=1.0,
            gap_through=True,
            atr_at_entry=0.02,
        )


def test_update_invalid_brake_halt_order() -> None:
    rs = RiskState.initial(100_000.0)
    bad = SimpleNamespace(drawdown_brake_threshold=0.20, drawdown_halt_threshold=0.15)
    with pytest.raises(ValueError, match="must be <"):
        rs.update(90_000.0, bad)
