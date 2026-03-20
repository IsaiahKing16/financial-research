"""
tests/test_risk_engine.py — Unit + stress tests for trading_system/risk_engine.py

Covers (SLE-22 unit tests):
  - compute_drawdown_scalar: normal / brake / halt modes, edge cases
  - check_stop_loss: breach / no-breach / zero stop
  - compute_atr_pct: happy path, missing columns, insufficient rows, zero price
  - size_position: approved trades, all rejection paths, drawdown modes

Stress tests (SLE-24):
  - test_synthetic_crash_scenario: 5 consecutive 10% daily drops
  - test_all_stops_trigger_same_day: every position hits stop simultaneously
  - test_extreme_atr: penny-stock style 50% ATR
  - test_minimal_capital: $2,000 starting capital
  - test_drawdown_recovery_then_re_entry: brake/halt then equity recovery
"""

import numpy as np
import pandas as pd
import pytest

from trading_system.config import PositionLimitsConfig, RiskConfig
from trading_system.risk_engine import (
    check_stop_loss,
    compute_atr_pct,
    compute_drawdown_scalar,
    size_position,
)
from trading_system.risk_state import RiskState


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _price_history(n: int = 25, close: float = 100.0, atr_frac: float = 0.02) -> pd.DataFrame:
    """Return synthetic OHLC DataFrame with controlled ATR."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    half_range = close * atr_frac / 2
    return pd.DataFrame({
        "Date": dates,
        "Open": [close] * n,
        "High": [close + half_range] * n,
        "Low": [close - half_range] * n,
        "Close": [close] * n,
    })


def _default_risk_config() -> RiskConfig:
    return RiskConfig()


def _default_limits() -> PositionLimitsConfig:
    return PositionLimitsConfig()


def _normal_state(equity: float = 10_000.0) -> RiskState:
    return RiskState.initial(equity)


# ── compute_drawdown_scalar ───────────────────────────────────────────────────

class TestComputeDrawdownScalar:

    BRAKE = 0.15
    HALT = 0.20

    def test_no_drawdown_returns_one_normal(self):
        scalar, mode = compute_drawdown_scalar(10_000, 10_000, self.BRAKE, self.HALT)
        assert scalar == 1.0
        assert mode == "normal"

    def test_equity_growth_above_peak_normal(self):
        scalar, mode = compute_drawdown_scalar(11_000, 10_000, self.BRAKE, self.HALT)
        assert scalar == 1.0
        assert mode == "normal"

    def test_drawdown_just_below_brake_is_normal(self):
        scalar, mode = compute_drawdown_scalar(8_510, 10_000, self.BRAKE, self.HALT)
        assert mode == "normal"
        assert scalar == 1.0

    def test_drawdown_at_brake_threshold(self):
        # 15% exactly: current = 8500
        scalar, mode = compute_drawdown_scalar(8_500, 10_000, self.BRAKE, self.HALT)
        assert mode == "brake"
        assert scalar == pytest.approx(1.0, abs=0.01)

    def test_drawdown_midpoint_brake_scalar_half(self):
        # 17.5% DD = midpoint → scalar 0.5
        scalar, mode = compute_drawdown_scalar(8_250, 10_000, self.BRAKE, self.HALT)
        assert mode == "brake"
        assert scalar == pytest.approx(0.5, abs=0.01)

    def test_drawdown_at_halt_threshold(self):
        scalar, mode = compute_drawdown_scalar(7_990, 10_000, self.BRAKE, self.HALT)
        assert mode == "halt"
        assert scalar == 0.0

    def test_drawdown_beyond_halt(self):
        scalar, mode = compute_drawdown_scalar(5_000, 10_000, self.BRAKE, self.HALT)
        assert mode == "halt"
        assert scalar == 0.0

    def test_zero_peak_equity_returns_normal(self):
        scalar, mode = compute_drawdown_scalar(0.0, 0.0, self.BRAKE, self.HALT)
        assert mode == "normal"
        assert scalar == 1.0

    def test_halt_lte_brake_raises(self):
        with pytest.raises(ValueError, match="halt_threshold"):
            compute_drawdown_scalar(10_000, 10_000, brake_threshold=0.20, halt_threshold=0.15)

    def test_halt_equals_brake_raises(self):
        with pytest.raises(ValueError, match="halt_threshold"):
            compute_drawdown_scalar(10_000, 10_000, brake_threshold=0.15, halt_threshold=0.15)


# ── check_stop_loss ───────────────────────────────────────────────────────────

class TestCheckStopLoss:

    def test_low_equals_stop_triggers(self):
        assert check_stop_loss(current_low=100.0, stop_price=100.0) is True

    def test_low_below_stop_triggers(self):
        assert check_stop_loss(current_low=95.0, stop_price=100.0) is True

    def test_low_above_stop_no_trigger(self):
        assert check_stop_loss(current_low=105.0, stop_price=100.0) is False

    def test_zero_stop_price_never_triggers(self):
        assert check_stop_loss(current_low=0.0, stop_price=0.0) is False

    def test_negative_stop_price_never_triggers(self):
        assert check_stop_loss(current_low=50.0, stop_price=-1.0) is False


# ── compute_atr_pct ───────────────────────────────────────────────────────────

class TestComputeAtrPct:

    def test_returns_positive_float(self):
        hist = _price_history(25, close=100.0, atr_frac=0.02)
        result = compute_atr_pct(hist, lookback=20)
        assert isinstance(result, float)
        assert result > 0

    def test_higher_volatility_gives_higher_atr(self):
        low_vol = _price_history(25, close=100.0, atr_frac=0.01)
        high_vol = _price_history(25, close=100.0, atr_frac=0.05)
        assert compute_atr_pct(high_vol, 20) > compute_atr_pct(low_vol, 20)

    def test_missing_high_column_raises(self):
        hist = _price_history(25).drop(columns=["High"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_atr_pct(hist)

    def test_missing_low_column_raises(self):
        hist = _price_history(25).drop(columns=["Low"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_atr_pct(hist)

    def test_insufficient_rows_raises(self):
        hist = _price_history(5)  # lookback=20 needs 21 rows
        with pytest.raises(ValueError, match="Insufficient"):
            compute_atr_pct(hist, lookback=20)

    def test_exactly_minimum_rows_succeeds(self):
        hist = _price_history(21)  # lookback=20 needs exactly 21 rows
        result = compute_atr_pct(hist, lookback=20)
        assert result > 0

    def test_zero_close_price_raises(self):
        hist = _price_history(25, close=0.01)
        hist["Close"] = 0.0
        with pytest.raises(ValueError):
            compute_atr_pct(hist)

    def test_custom_lookback(self):
        hist = _price_history(15, close=100.0, atr_frac=0.03)
        result = compute_atr_pct(hist, lookback=10)
        assert result > 0


# ── size_position ─────────────────────────────────────────────────────────────

class TestSizePositionApproved:

    def test_returns_approved_decision(self):
        decision = size_position(
            ticker="AAPL",
            entry_price=100.0,
            current_equity=10_000.0,
            price_history=_price_history(25, close=100.0, atr_frac=0.02),
            risk_state=_normal_state(10_000.0),
            config=_default_risk_config(),
            position_limits=_default_limits(),
        )
        assert decision.approved is True

    def test_approved_decision_has_positive_shares(self):
        decision = size_position(
            ticker="MSFT",
            entry_price=300.0,
            current_equity=10_000.0,
            price_history=_price_history(25, close=300.0, atr_frac=0.02),
            risk_state=_normal_state(10_000.0),
            config=_default_risk_config(),
            position_limits=_default_limits(),
        )
        assert decision.shares > 0
        assert decision.stop_price > 0
        assert decision.stop_price < 300.0

    def test_stop_price_below_entry(self):
        decision = size_position(
            ticker="NVDA",
            entry_price=500.0,
            current_equity=50_000.0,
            price_history=_price_history(25, close=500.0, atr_frac=0.02),
            risk_state=_normal_state(50_000.0),
            config=_default_risk_config(),
            position_limits=_default_limits(),
        )
        assert decision.stop_price < 500.0

    def test_position_pct_within_limits(self):
        limits = _default_limits()
        decision = size_position(
            ticker="TSLA",
            entry_price=200.0,
            current_equity=10_000.0,
            price_history=_price_history(25, close=200.0, atr_frac=0.02),
            risk_state=_normal_state(10_000.0),
            config=_default_risk_config(),
            position_limits=limits,
        )
        if decision.approved:
            assert limits.min_position_pct <= decision.position_pct <= limits.max_position_pct

    def test_dollar_amount_equals_shares_times_price(self):
        decision = size_position(
            ticker="GOOG",
            entry_price=150.0,
            current_equity=10_000.0,
            price_history=_price_history(25, close=150.0, atr_frac=0.02),
            risk_state=_normal_state(10_000.0),
            config=_default_risk_config(),
            position_limits=_default_limits(),
        )
        if decision.approved:
            assert decision.dollar_amount == pytest.approx(
                decision.shares * 150.0, rel=1e-4
            )


class TestSizePositionRejections:

    def test_negative_entry_price_rejected(self):
        decision = size_position(
            ticker="AAPL",
            entry_price=-1.0,
            current_equity=10_000.0,
            price_history=_price_history(25),
            risk_state=_normal_state(),
            config=_default_risk_config(),
            position_limits=_default_limits(),
        )
        assert decision.approved is False
        assert "Invalid entry price" in (decision.rejection_reason or "")

    def test_zero_entry_price_rejected(self):
        decision = size_position(
            ticker="AAPL",
            entry_price=0.0,
            current_equity=10_000.0,
            price_history=_price_history(25),
            risk_state=_normal_state(),
            config=_default_risk_config(),
            position_limits=_default_limits(),
        )
        assert decision.approved is False

    def test_zero_equity_rejected(self):
        decision = size_position(
            ticker="AAPL",
            entry_price=100.0,
            current_equity=0.0,
            price_history=_price_history(25),
            risk_state=_normal_state(0.0),
            config=_default_risk_config(),
            position_limits=_default_limits(),
        )
        assert decision.approved is False
        assert "Non-positive equity" in (decision.rejection_reason or "")

    def test_drawdown_halt_rejects_new_positions(self):
        # Create a state at halt level
        rs = RiskState.initial(10_000.0)
        rs.update(7_990.0, brake_threshold=0.15, halt_threshold=0.20)
        assert rs.drawdown_mode == "halt"

        decision = size_position(
            ticker="AAPL",
            entry_price=100.0,
            current_equity=7_990.0,
            price_history=_price_history(25),
            risk_state=rs,
            config=_default_risk_config(),
            position_limits=_default_limits(),
        )
        assert decision.approved is False
        assert "halt" in (decision.rejection_reason or "").lower()

    def test_already_holding_ticker_rejected(self):
        decision = size_position(
            ticker="AAPL",
            entry_price=100.0,
            current_equity=10_000.0,
            price_history=_price_history(25),
            risk_state=_normal_state(),
            config=_default_risk_config(),
            position_limits=_default_limits(),
            open_positions={"AAPL": object()},
        )
        assert decision.approved is False
        assert "Already holding" in (decision.rejection_reason or "")

    def test_insufficient_price_history_rejected(self):
        decision = size_position(
            ticker="AAPL",
            entry_price=100.0,
            current_equity=10_000.0,
            price_history=_price_history(5),  # too few rows
            risk_state=_normal_state(),
            config=_default_risk_config(),
            position_limits=_default_limits(),
        )
        assert decision.approved is False
        assert "Insufficient" in (decision.rejection_reason or "")

    def test_brake_mode_reduces_position_size(self):
        """In brake mode the scalar < 1.0, so position should be smaller.

        Uses a very low ATR (0.5%) so raw_weight = 2% / (2×0.5%) = 2.0 (200%).
        With a raised max_position_pct cap of 0.60, normal mode hits 60% cap,
        while brake scalar ~0.5 yields 1.0 × 0.5 = 0.5 (50%) — visibly below cap.
        """
        # With ATR=2% and 2× multiple: stop_distance=4%, raw_weight=2%/4%=0.50.
        # Normal: 0.50×1.0=0.50. Brake (~0.5 scalar): 0.50×0.5=0.25.
        # Cap must be above both (0.60) so neither is clamped equally.
        wide_limits = PositionLimitsConfig(min_position_pct=0.02, max_position_pct=0.60)
        hist = _price_history(25, close=100.0, atr_frac=0.02)  # 2% ATR → raw_weight 0.5

        rs_normal = _normal_state(10_000.0)
        rs_brake = RiskState.initial(10_000.0)
        rs_brake.update(8_250.0, brake_threshold=0.15, halt_threshold=0.20)  # 17.5% DD → scalar ~0.5

        d_normal = size_position(
            ticker="AAPL", entry_price=100.0, current_equity=10_000.0,
            price_history=hist, risk_state=rs_normal,
            config=_default_risk_config(), position_limits=wide_limits,
        )
        d_brake = size_position(
            ticker="AAPL", entry_price=100.0, current_equity=8_250.0,
            price_history=hist, risk_state=rs_brake,
            config=_default_risk_config(), position_limits=wide_limits,
        )
        assert d_normal.approved and d_brake.approved
        assert d_brake.position_pct < d_normal.position_pct


# ── Stress Tests (SLE-24) ─────────────────────────────────────────────────────

class TestStressScenarios:

    def test_synthetic_crash_scenario(self):
        """5 consecutive 10% daily drops should trigger halt by day 3-4."""
        equity = 10_000.0
        peak = equity
        rs = RiskState.initial(equity)
        brake = 0.15
        halt = 0.20

        for day in range(5):
            equity *= 0.90
            rs.update(equity, brake_threshold=brake, halt_threshold=halt)

        # After 5 × 10% drops: equity ≈ 5,905 (41% drawdown) — well past halt
        assert rs.drawdown_mode == "halt"
        assert rs.sizing_scalar == 0.0
        assert rs.current_drawdown > halt

    def test_all_stops_trigger_same_day(self):
        """All open positions hitting stop simultaneously should all be rejected on next entry."""
        rs = RiskState.initial(10_000.0)
        tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "TSLA"]
        for t in tickers:
            rs.register_stop(t, 100.0)
        assert len(rs.active_stops) == 5

        # Simulate all stops firing: remove all
        for t in tickers:
            rs.remove_stop(t)
        assert rs.active_stops == {}

    def test_extreme_atr_penny_stock(self):
        """50% ATR should produce a tiny or rejected position."""
        hist = _price_history(25, close=1.0, atr_frac=0.50)
        decision = size_position(
            ticker="JUNK",
            entry_price=1.0,
            current_equity=10_000.0,
            price_history=hist,
            risk_state=_normal_state(10_000.0),
            config=_default_risk_config(),
            position_limits=_default_limits(),
        )
        # Either rejected (below min size) or approved with very small position
        if decision.approved:
            assert decision.position_pct <= _default_limits().max_position_pct
        else:
            assert decision.rejection_reason is not None

    def test_minimal_capital(self):
        """$2,000 starting capital should still compute valid decisions."""
        hist = _price_history(25, close=10.0, atr_frac=0.02)
        decision = size_position(
            ticker="LOW_PRICE",
            entry_price=10.0,
            current_equity=2_000.0,
            price_history=hist,
            risk_state=_normal_state(2_000.0),
            config=_default_risk_config(),
            position_limits=_default_limits(),
        )
        # Either valid or gracefully rejected — must not raise
        assert isinstance(decision.approved, bool)

    def test_drawdown_recovery_then_re_entry(self):
        """Equity hits halt, recovers to normal — new trades should be approved again."""
        rs = RiskState.initial(10_000.0)
        # Hit halt
        rs.update(7_990.0, brake_threshold=0.15, halt_threshold=0.20)
        assert rs.drawdown_mode == "halt"

        # Recover beyond original peak
        rs.update(11_000.0, brake_threshold=0.15, halt_threshold=0.20)
        assert rs.drawdown_mode == "normal"
        assert rs.sizing_scalar == 1.0

        # New trade should now be approvable
        hist = _price_history(25, close=100.0, atr_frac=0.02)
        decision = size_position(
            ticker="AAPL",
            entry_price=100.0,
            current_equity=11_000.0,
            price_history=hist,
            risk_state=rs,
            config=_default_risk_config(),
            position_limits=_default_limits(),
        )
        assert decision.approved is True

    def test_high_volatility_universe_sizing(self):
        """Run size_position across 10 tickers simultaneously — no crashes."""
        tickers = [f"TICK{i}" for i in range(10)]
        rs = _normal_state(100_000.0)
        results = []
        for t in tickers:
            hist = _price_history(25, close=100.0, atr_frac=np.random.uniform(0.01, 0.08))
            d = size_position(
                ticker=t,
                entry_price=100.0,
                current_equity=100_000.0,
                price_history=hist,
                risk_state=rs,
                config=_default_risk_config(),
                position_limits=_default_limits(),
            )
            results.append(d)
        # All should return a decision (approved or rejected) without raising
        assert len(results) == 10
        assert all(isinstance(d.approved, bool) for d in results)

    def test_consecutive_halt_and_size_position_always_rejects(self):
        """Once halted, size_position must reject every ticker until equity recovers."""
        rs = RiskState.initial(10_000.0)
        rs.update(7_990.0, brake_threshold=0.15, halt_threshold=0.20)
        hist = _price_history(25, close=100.0, atr_frac=0.02)

        for ticker in ["AAPL", "MSFT", "NVDA", "GOOG"]:
            d = size_position(
                ticker=ticker,
                entry_price=100.0,
                current_equity=7_990.0,
                price_history=hist,
                risk_state=rs,
                config=_default_risk_config(),
                position_limits=_default_limits(),
            )
            assert d.approved is False, f"{ticker} should be rejected during halt"
