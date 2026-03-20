"""Stress tests for Phase 2 risk engine (Section 8.4).

These tests intentionally use synthetic data only and target the public
risk-engine API expected by PHASE2_SYSTEM_DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
import pytest

from trading_system.config import DEFAULT_CONFIG

# Phase 2 modules may be developed in parallel; skip cleanly until available.
risk_state_mod = pytest.importorskip("trading_system.risk_state")
risk_engine_mod = pytest.importorskip("trading_system.risk_engine")

RiskState = risk_state_mod.RiskState
size_position = risk_engine_mod.size_position
check_stop_loss = risk_engine_mod.check_stop_loss
compute_drawdown_scalar = risk_engine_mod.compute_drawdown_scalar


@dataclass
class _OpenPos:
    """Minimal open-position shape for exposure/sector checks."""

    sector: str
    position_pct: float


def _make_price_history(
    start: str,
    periods: int,
    start_price: float,
    daily_return: float = 0.0,
    range_frac: float = 0.01,
) -> pd.DataFrame:
    """Build synthetic OHLCV history with controllable volatility envelope."""
    dates = pd.bdate_range(start=start, periods=periods)
    rows = []
    price = float(start_price)
    for d in dates:
        open_p = price
        close_p = open_p * (1.0 + daily_return)
        high_p = max(open_p, close_p) * (1.0 + range_frac)
        low_p = min(open_p, close_p) * (1.0 - range_frac)
        rows.append(
            {
                "Date": d,
                "Open": open_p,
                "High": high_p,
                "Low": low_p,
                "Close": close_p,
                "Volume": 1_000_000,
            }
        )
        price = close_p
    return pd.DataFrame(rows)


def _make_extreme_atr_history(start: str, periods: int, price: float = 1.0) -> pd.DataFrame:
    """Build history with ~50% intraday range to emulate penny-stock behavior."""
    dates = pd.bdate_range(start=start, periods=periods)
    rows = []
    for d in dates:
        rows.append(
            {
                "Date": d,
                "Open": price,
                "High": price * 1.50,
                "Low": price * 0.50,
                "Close": price,
                "Volume": 2_000_000,
            }
        )
    return pd.DataFrame(rows)


def test_synthetic_crash_scenario() -> None:
    """10% daily losses over 5 days should transition normal -> brake -> halt."""
    equity = 10_000.0
    brake = DEFAULT_CONFIG.risk.drawdown_brake_threshold
    halt = DEFAULT_CONFIG.risk.drawdown_halt_threshold
    peak = equity

    modes = []
    scalars = []
    for _ in range(5):
        equity *= 0.90
        scalar, mode = compute_drawdown_scalar(
            current_equity=equity,
            peak_equity=peak,
            brake_threshold=brake,
            halt_threshold=halt,
        )
        modes.append(mode)
        scalars.append(scalar)

    assert modes[0] == "normal"  # 10% DD
    assert modes[1] == "brake"  # 19% DD
    assert "halt" in modes[2:]  # >= 20% DD reached by day 3
    assert scalars[0] == pytest.approx(1.0, abs=1e-9)
    assert scalars[-1] == pytest.approx(0.0, abs=1e-9)


def test_all_stops_trigger_same_day() -> None:
    """If the same intraday low breaches all stops, every stop check must trigger."""
    stop_prices = [95.0, 92.5, 88.0, 73.0, 41.0]
    crash_low = 40.0

    triggered = [check_stop_loss(current_low=crash_low, stop_price=s) for s in stop_prices]
    assert all(triggered)


def test_extreme_atr() -> None:
    """ATR%=~50% should produce either rejection or a very small approved position."""
    price_history = _make_extreme_atr_history(start="2024-01-02", periods=40, price=1.0)
    risk_state = RiskState.initial(10_000.0)

    decision = size_position(
        ticker="PENNY",
        entry_price=1.0,
        current_equity=10_000.0,
        price_history=price_history,
        risk_state=risk_state,
        config=DEFAULT_CONFIG.risk,
        position_limits=DEFAULT_CONFIG.position_limits,
        sector_map={"PENNY": "MicroCap"},
        open_positions={},
    )

    if decision.approved:
        assert decision.position_pct <= DEFAULT_CONFIG.position_limits.min_position_pct + 1e-9
    else:
        assert decision.rejection_reason


def test_minimal_capital() -> None:
    """$2,000 equity should still produce coherent min-position sizing behavior."""
    position_limits = DEFAULT_CONFIG.position_limits
    low_vol_history = _make_price_history(
        start="2024-01-02",
        periods=40,
        start_price=100.0,
        daily_return=0.001,
        range_frac=0.005,
    )
    risk_state = RiskState.initial(2_000.0)

    decision = size_position(
        ticker="SMALL",
        entry_price=100.0,
        current_equity=2_000.0,
        price_history=low_vol_history,
        risk_state=risk_state,
        config=DEFAULT_CONFIG.risk,
        position_limits=position_limits,
        sector_map={"SMALL": "Utilities"},
        open_positions={},
    )

    assert decision.approved
    assert decision.shares > 0
    assert decision.dollar_amount >= 2_000.0 * position_limits.min_position_pct


def test_maximum_positions() -> None:
    """When gross exposure is full, an additional trade should be rejected."""
    risk_cfg = DEFAULT_CONFIG.risk
    pos_limits = DEFAULT_CONFIG.position_limits
    price_history = _make_price_history(
        start="2024-01-02",
        periods=40,
        start_price=100.0,
        daily_return=0.0,
        range_frac=0.01,
    )
    risk_state = RiskState.initial(10_000.0)

    # Fill to 100% gross exposure using ten 10% positions.
    open_positions: Dict[str, Any] = {
        f"T{i}": _OpenPos(sector="Tech", position_pct=0.10) for i in range(10)
    }

    decision = size_position(
        ticker="NEXT",
        entry_price=100.0,
        current_equity=10_000.0,
        price_history=price_history,
        risk_state=risk_state,
        config=risk_cfg,
        position_limits=pos_limits,
        sector_map={"NEXT": "HealthCare"},
        open_positions=open_positions,
    )

    assert not decision.approved
    assert decision.rejection_reason
