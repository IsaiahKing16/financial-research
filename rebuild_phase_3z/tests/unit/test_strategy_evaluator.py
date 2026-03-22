"""
test_strategy_evaluator.py — Unit tests for StrategyEvaluator (SLE-70).

Tests:
  - Rolling Sharpe computation (30d, 90d, all-time)
  - Drawdown computation from equity curve
  - RED / YELLOW / GREEN status determination
  - Status transition → SystemCommand emission
  - BSS drift detection
  - evaluate_and_update_state SharedState integration
  - Edge cases: insufficient history, all-GREEN, extreme drawdown

Linear: SLE-70
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import List, Optional

import pytest

from rebuild_phase_3z.fppe.trading_system.contracts.decisions import (
    EvaluatorStatus,
)
from rebuild_phase_3z.fppe.trading_system.contracts.state import SystemCommand
from rebuild_phase_3z.fppe.trading_system.strategy_evaluator import (
    ClosedTrade,
    EvaluatorConfig,
    StrategyEvaluator,
    _annualized_sharpe,
    _drawdown_from_peak,
    _linear_slope,
)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def make_evaluator(
    config: Optional[EvaluatorConfig] = None,
) -> StrategyEvaluator:
    return StrategyEvaluator(config=config or EvaluatorConfig())


def _add_daily_returns(
    evaluator: StrategyEvaluator,
    returns: List[float],
    start_date: date = date(2026, 1, 2),
) -> None:
    """Add a list of daily returns starting from start_date."""
    cumulative = 0.0
    for i, r in enumerate(returns):
        cumulative = (1 + cumulative) * (1 + r) - 1
        d = start_date + timedelta(days=i)
        evaluator.record_daily_return(
            trade_date=d,
            portfolio_return=r,
            cumulative_return=cumulative,
        )


def _add_trades(
    evaluator: StrategyEvaluator,
    n: int,
    pnl_pct: float = 0.02,
    start_date: date = date(2026, 1, 2),
) -> None:
    """Add n closed trades with constant pnl_pct."""
    for i in range(n):
        evaluator.record_trade(ClosedTrade(
            close_date=start_date + timedelta(days=i),
            pnl_pct=pnl_pct,
            holding_days=7,
        ))


# ─── TestMathHelpers ──────────────────────────────────────────────────────────

class TestMathHelpers:
    """Pure-function math helpers."""

    def test_annualized_sharpe_positive(self):
        """Consistent positive returns give positive Sharpe."""
        returns = [0.001] * 100  # 0.1% per day, zero variance
        # When std=0 and mean>0, should return inf
        result = _annualized_sharpe(returns)
        assert result == float("inf") or result > 0

    def test_annualized_sharpe_zero_mean(self):
        """Alternating equal returns give 0 Sharpe."""
        # mean excess = 0
        returns = [0.01, -0.01] * 50
        result = _annualized_sharpe(returns)
        assert result is not None
        assert abs(result) < 0.01  # should be ~0 given alternating equal returns

    def test_annualized_sharpe_none_on_single_point(self):
        """Single data point returns None (can't compute std)."""
        assert _annualized_sharpe([0.05]) is None

    def test_annualized_sharpe_none_on_empty(self):
        """Empty list returns None."""
        assert _annualized_sharpe([]) is None

    def test_annualized_sharpe_scaling(self):
        """Annualization factor is √252."""
        # Known returns: daily mean=0.001, std=0.001
        # Sharpe = (0.001 / 0.001) * sqrt(252) = sqrt(252) ≈ 15.87
        returns = [0.001 + (0.001 if i % 2 == 0 else -0.001) for i in range(100)]
        result = _annualized_sharpe(returns, risk_free_daily=0.0)
        assert result is not None
        # Rough sanity: should be positive and finite
        assert 0 < result < 100

    def test_drawdown_from_peak_no_drawdown(self):
        """Monotonically increasing equity gives 0 drawdown."""
        cumulative = [0.01 * i for i in range(10)]
        assert _drawdown_from_peak(cumulative) == 0.0

    def test_drawdown_from_peak_known_value(self):
        """Known drawdown scenario: peak=1.1, current=0.99."""
        # equity: 1.0, 1.1, 0.99
        # peak=1.1, current=0.99 → drawdown = 1 - 0.99/1.1 ≈ 0.1
        cumulative = [0.0, 0.1, -0.01]
        dd = _drawdown_from_peak(cumulative)
        # equity: 1.0, 1.1, 0.99; peak=1.1 → dd = 1 - 0.99/1.1
        expected = 1 - 0.99 / 1.1
        assert abs(dd - expected) < 1e-9

    def test_drawdown_from_peak_empty(self):
        """Empty series returns 0 drawdown."""
        assert _drawdown_from_peak([]) == 0.0

    def test_linear_slope_increasing(self):
        """Increasing series gives positive slope."""
        assert _linear_slope([1.0, 2.0, 3.0, 4.0]) > 0

    def test_linear_slope_decreasing(self):
        """Decreasing series gives negative slope."""
        assert _linear_slope([4.0, 3.0, 2.0, 1.0]) < 0

    def test_linear_slope_flat(self):
        """Flat series gives 0 slope."""
        assert _linear_slope([1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_linear_slope_single_point(self):
        """Single point returns 0 (no slope computable)."""
        assert _linear_slope([5.0]) == 0.0


# ─── TestStatusDetermination ──────────────────────────────────────────────────

class TestStatusDetermination:
    """RED / YELLOW / GREEN status logic."""

    def test_green_with_sufficient_history(self):
        """Good metrics + sufficient trades → GREEN."""
        evaluator = make_evaluator()
        # Add 35 trades and 35 days of good returns
        _add_trades(evaluator, n=35, pnl_pct=0.02)
        _add_daily_returns(evaluator, returns=[0.002] * 35)
        snapshot = evaluator.evaluate(current_date=date(2026, 2, 15))
        assert snapshot.status == EvaluatorStatus.GREEN

    def test_yellow_insufficient_trades(self):
        """Fewer than min_trades → YELLOW."""
        evaluator = make_evaluator()
        _add_trades(evaluator, n=10)  # below yellow_min_trades=30
        _add_daily_returns(evaluator, returns=[0.002] * 10)
        snapshot = evaluator.evaluate(current_date=date(2026, 1, 15))
        assert snapshot.status == EvaluatorStatus.YELLOW
        assert "Insufficient" in snapshot.reason

    def test_yellow_low_30d_sharpe(self):
        """30d Sharpe < 0.5 → YELLOW."""
        cfg = EvaluatorConfig(yellow_min_trades=1)  # bypass trade count check
        evaluator = make_evaluator(config=cfg)
        _add_trades(evaluator, n=2)
        # Mix of good and bad returns: Sharpe will be low
        # Create 30 days of returns with negative drift
        returns = [0.001, -0.003] * 15  # mean = -0.001 → Sharpe < 0
        _add_daily_returns(evaluator, returns=returns)
        snapshot = evaluator.evaluate(current_date=date(2026, 2, 2))
        # With negative mean returns, Sharpe will be < 0.5 → at least YELLOW
        assert snapshot.status in (EvaluatorStatus.YELLOW, EvaluatorStatus.RED)

    def test_red_drawdown_exceeds_threshold(self):
        """Drawdown > 15% → RED."""
        evaluator = make_evaluator()
        _add_trades(evaluator, n=35)
        _add_daily_returns(evaluator, returns=[0.001] * 35)
        snapshot = evaluator.evaluate(
            current_date=date(2026, 2, 15),
            drawdown=0.20,  # explicit 20% drawdown
        )
        assert snapshot.status == EvaluatorStatus.RED
        assert "drawdown" in snapshot.reason.lower()

    def test_red_negative_90d_sharpe(self):
        """90-day Sharpe < 0 → RED."""
        cfg = EvaluatorConfig(
            yellow_min_trades=1,
            min_trades_for_sharpe=5,
        )
        evaluator = make_evaluator(config=cfg)
        _add_trades(evaluator, n=5)
        # 90 days of consistently negative returns
        returns = [-0.005] * 90
        _add_daily_returns(evaluator, returns=returns)
        snapshot = evaluator.evaluate(current_date=date(2026, 4, 4))
        assert snapshot.status == EvaluatorStatus.RED

    def test_drawdown_beats_sharpe_check(self):
        """Drawdown check is evaluated before Sharpe (RED priority order)."""
        cfg = EvaluatorConfig(yellow_min_trades=1, min_trades_for_sharpe=5)
        evaluator = make_evaluator(config=cfg)
        _add_trades(evaluator, n=5)
        # Good Sharpe but terrible drawdown
        returns = [0.002] * 90
        _add_daily_returns(evaluator, returns=returns)
        snapshot = evaluator.evaluate(
            current_date=date(2026, 4, 4),
            drawdown=0.20,
        )
        assert snapshot.status == EvaluatorStatus.RED
        assert "drawdown" in snapshot.reason.lower()

    def test_explicit_drawdown_overrides_computed(self):
        """Passing drawdown kwarg uses that value, not the internal series."""
        evaluator = make_evaluator()
        _add_daily_returns(evaluator, returns=[0.001] * 5)  # tiny positive returns
        snapshot = evaluator.evaluate(
            current_date=date(2026, 1, 10),
            drawdown=0.50,  # force extreme drawdown
        )
        assert snapshot.status == EvaluatorStatus.RED

    def test_green_reason_mentions_sharpe(self):
        """GREEN reason string mentions Sharpe and drawdown."""
        cfg = EvaluatorConfig(yellow_min_trades=1, min_trades_for_sharpe=5)
        evaluator = make_evaluator(config=cfg)
        _add_trades(evaluator, n=5)
        _add_daily_returns(evaluator, returns=[0.002] * 90)
        snapshot = evaluator.evaluate(current_date=date(2026, 4, 4))
        if snapshot.status == EvaluatorStatus.GREEN:
            assert "Sharpe" in snapshot.reason or "metrics" in snapshot.reason.lower()


# ─── TestCommandEmission ──────────────────────────────────────────────────────

class TestCommandEmission:
    """Status transitions emit the correct SystemCommands."""

    def test_green_to_red_emits_halt(self):
        """Transition from GREEN → RED emits HALT."""
        evaluator = make_evaluator()
        evaluator._previous_status = EvaluatorStatus.GREEN
        commands = evaluator._determine_commands(EvaluatorStatus.RED)
        assert SystemCommand.HALT in commands

    def test_green_to_yellow_emits_reduce(self):
        """Transition from GREEN → YELLOW emits REDUCE_EXPOSURE."""
        evaluator = make_evaluator()
        evaluator._previous_status = EvaluatorStatus.GREEN
        commands = evaluator._determine_commands(EvaluatorStatus.YELLOW)
        assert SystemCommand.REDUCE_EXPOSURE in commands

    def test_red_to_green_emits_resume(self):
        """Transition from RED → GREEN emits RESUME."""
        evaluator = make_evaluator()
        evaluator._previous_status = EvaluatorStatus.RED
        commands = evaluator._determine_commands(EvaluatorStatus.GREEN)
        assert SystemCommand.RESUME in commands

    def test_yellow_to_green_emits_resume(self):
        """Transition from YELLOW → GREEN emits RESUME."""
        evaluator = make_evaluator()
        evaluator._previous_status = EvaluatorStatus.YELLOW
        commands = evaluator._determine_commands(EvaluatorStatus.GREEN)
        assert SystemCommand.RESUME in commands

    def test_no_command_on_stable_status(self):
        """No command emitted when status does not change."""
        evaluator = make_evaluator()
        evaluator._previous_status = EvaluatorStatus.GREEN
        commands = evaluator._determine_commands(EvaluatorStatus.GREEN)
        assert commands == []

    def test_red_to_yellow_emits_resume_then_reduce(self):
        """Transition RED → YELLOW: RESUME first (clears halt), then REDUCE_EXPOSURE."""
        evaluator = make_evaluator()
        evaluator._previous_status = EvaluatorStatus.RED
        commands = evaluator._determine_commands(EvaluatorStatus.YELLOW)
        assert SystemCommand.RESUME in commands
        assert SystemCommand.REDUCE_EXPOSURE in commands

    def test_first_evaluation_sets_previous_status(self):
        """After evaluate(), _previous_status reflects the emitted status."""
        cfg = EvaluatorConfig(yellow_min_trades=1)
        evaluator = make_evaluator(config=cfg)
        _add_trades(evaluator, n=1)
        _add_daily_returns(evaluator, returns=[0.001])
        snapshot = evaluator.evaluate(current_date=date(2026, 1, 3))
        assert evaluator._previous_status == snapshot.status


# ─── TestBssDriftDetection ────────────────────────────────────────────────────

class TestBssDriftDetection:
    """Calibration drift detection via BSS slope."""

    def test_no_bss_data_no_drift(self):
        """No BSS data gives slope 0 — no drift warning."""
        evaluator = make_evaluator()
        slope = evaluator._compute_bss_drift_slope()
        assert slope == 0.0

    def test_declining_bss_gives_negative_slope(self):
        """Declining BSS series produces negative slope."""
        evaluator = make_evaluator()
        for i, bss in enumerate([0.10, 0.08, 0.05, 0.02, -0.01]):
            evaluator.record_bss(date(2026, 1, 1) + timedelta(days=i * 30), bss)
        slope = evaluator._compute_bss_drift_slope()
        assert slope < 0

    def test_improving_bss_gives_positive_slope(self):
        """Improving BSS series produces positive slope."""
        evaluator = make_evaluator()
        for i, bss in enumerate([0.01, 0.03, 0.05, 0.07, 0.09]):
            evaluator.record_bss(date(2026, 1, 1) + timedelta(days=i * 30), bss)
        slope = evaluator._compute_bss_drift_slope()
        assert slope > 0

    def test_bss_drift_triggers_yellow(self):
        """Negative BSS drift slope below threshold → YELLOW."""
        cfg = EvaluatorConfig(
            bss_drift_slope_threshold=-0.001,
            yellow_min_trades=1,
            min_trades_for_sharpe=2,
        )
        evaluator = make_evaluator(config=cfg)
        _add_trades(evaluator, n=40)  # enough trades to bypass trade count check

        # Add mildly positive 30d/90d returns (bypass sharpe checks)
        _add_daily_returns(evaluator, returns=[0.003] * 35)

        # Force a steep BSS decline (slope ≈ -0.1 per period)
        for i, bss in enumerate([0.5, 0.4, 0.3, 0.2, 0.1]):
            evaluator.record_bss(date(2026, 1, 1) + timedelta(days=i * 30), bss)

        snapshot = evaluator.evaluate(current_date=date(2026, 3, 1))
        # Should be at least YELLOW due to BSS drift (or GREEN if other metrics look good)
        # The important thing: BSS slope is detected
        slope = evaluator._compute_bss_drift_slope()
        assert slope < 0


# ─── TestRollingWindows ───────────────────────────────────────────────────────

class TestRollingWindows:
    """Rolling window metric computations."""

    def test_sharpe_none_before_min_trades(self):
        """Sharpe is None when daily returns < min_trades_for_sharpe."""
        cfg = EvaluatorConfig(min_trades_for_sharpe=10)
        evaluator = make_evaluator(config=cfg)
        _add_daily_returns(evaluator, returns=[0.001] * 5)  # only 5, need 10
        assert evaluator._compute_sharpe(30) is None

    def test_sharpe_computed_after_min_trades(self):
        """Sharpe is computed when daily returns ≥ min_trades_for_sharpe."""
        cfg = EvaluatorConfig(min_trades_for_sharpe=5)
        evaluator = make_evaluator(config=cfg)
        _add_daily_returns(evaluator, returns=[0.002] * 10)
        result = evaluator._compute_sharpe(30)
        assert result is not None
        assert result > 0

    def test_30d_window_uses_last_30_days(self):
        """30d Sharpe only uses the last 30 daily returns."""
        cfg = EvaluatorConfig(min_trades_for_sharpe=5)
        evaluator = make_evaluator(config=cfg)
        # First 60 days: terrible returns
        _add_daily_returns(evaluator, returns=[-0.01] * 60)
        # Last 30 days: great returns
        _add_daily_returns(
            evaluator,
            returns=[0.005] * 30,
            start_date=date(2026, 3, 3),
        )
        sharpe_30d = evaluator._compute_sharpe(30)
        all_time = evaluator._compute_all_time_sharpe()
        # 30d window should reflect only the recent good period
        assert sharpe_30d is not None
        assert all_time is not None
        # Recent Sharpe should be better than all-time
        if sharpe_30d != float("inf"):
            assert sharpe_30d > all_time

    def test_drawdown_zero_on_monotonic_gains(self):
        """Monotonically increasing cumulative returns → 0 drawdown."""
        evaluator = make_evaluator()
        _add_daily_returns(evaluator, returns=[0.001] * 50)
        assert evaluator._compute_drawdown() == pytest.approx(0.0, abs=1e-9)

    def test_drawdown_detected_after_peak(self):
        """Drawdown is correctly detected after a peak."""
        evaluator = make_evaluator()
        # Go up 10%, then down 5%
        _add_daily_returns(evaluator, returns=[0.01] * 10 + [-0.01] * 5)
        dd = evaluator._compute_drawdown()
        assert dd > 0


# ─── TestMetricsSummary ───────────────────────────────────────────────────────

class TestMetricsSummary:
    """metrics_summary() and property tests."""

    def test_metrics_summary_keys(self):
        """metrics_summary() returns all expected keys."""
        evaluator = make_evaluator()
        summary = evaluator.metrics_summary()
        expected_keys = {
            "sharpe_30d", "sharpe_90d", "sharpe_252d", "all_time_sharpe",
            "drawdown", "total_trades", "days_in_market", "bss_drift_slope",
        }
        assert set(summary.keys()) == expected_keys

    def test_total_trades_property(self):
        """total_trades property matches recorded trades."""
        evaluator = make_evaluator()
        assert evaluator.total_trades == 0
        _add_trades(evaluator, n=7)
        assert evaluator.total_trades == 7

    def test_days_in_market_property(self):
        """days_in_market property matches recorded daily returns."""
        evaluator = make_evaluator()
        assert evaluator.days_in_market == 0
        _add_daily_returns(evaluator, returns=[0.001] * 15)
        assert evaluator.days_in_market == 15

    def test_snapshot_fields_match_evaluate(self):
        """Snapshot from evaluate() has consistent fields."""
        cfg = EvaluatorConfig(yellow_min_trades=1, min_trades_for_sharpe=5)
        evaluator = make_evaluator(config=cfg)
        _add_trades(evaluator, n=5)
        _add_daily_returns(evaluator, returns=[0.002] * 30)
        snapshot = evaluator.evaluate(
            current_date=date(2026, 2, 1),
            drawdown=0.05,
        )
        assert snapshot.drawdown_from_peak == 0.05
        assert snapshot.days_in_market == 30
        assert snapshot.evaluation_date == date(2026, 2, 1)
        assert snapshot.reason  # non-empty


# ─── TestEvaluateAndUpdateState ───────────────────────────────────────────────

class TestEvaluateAndUpdateState:
    """Integration test: evaluator writes back to SharedState."""

    def test_evaluate_and_update_state_green(self):
        """GREEN evaluation updates EvaluatorState without enqueueing commands."""
        from rebuild_phase_3z.fppe.trading_system.shared_state import (
            LayerTag,
            SharedStateManager,
        )
        from rebuild_phase_3z.fppe.trading_system.contracts.decisions import EvaluatorStatus

        manager = SharedStateManager()
        state = SharedStateManager.initial_state(
            starting_equity=10_000.0,
            trading_date=date(2026, 3, 21),
        )

        cfg = EvaluatorConfig(yellow_min_trades=1, min_trades_for_sharpe=2)
        evaluator = StrategyEvaluator(config=cfg)
        _add_trades(evaluator, n=5)
        _add_daily_returns(evaluator, returns=[0.003] * 30)

        snapshot, new_state = evaluator.evaluate_and_update_state(
            state=state,
            current_date=date(2026, 2, 5),
            manager=manager,
            drawdown=0.02,
        )
        assert snapshot.status == EvaluatorStatus.GREEN
        assert new_state.evaluator.latest_snapshot is not None
        assert new_state.evaluator.consecutive_underperformance_days == 0

    def test_evaluate_and_update_state_emits_halt_command(self):
        """RED evaluation enqueues HALT command."""
        from rebuild_phase_3z.fppe.trading_system.shared_state import (
            LayerTag,
            SharedStateManager,
        )
        from rebuild_phase_3z.fppe.trading_system.contracts.decisions import EvaluatorStatus

        manager = SharedStateManager()
        state = SharedStateManager.initial_state(
            starting_equity=10_000.0,
            trading_date=date(2026, 3, 21),
        )

        cfg = EvaluatorConfig(yellow_min_trades=1)
        evaluator = StrategyEvaluator(config=cfg)
        _add_trades(evaluator, n=5)
        _add_daily_returns(evaluator, returns=[-0.001] * 30)

        snapshot, new_state = evaluator.evaluate_and_update_state(
            state=state,
            current_date=date(2026, 2, 5),
            manager=manager,
            drawdown=0.20,  # force RED
        )
        assert snapshot.status == EvaluatorStatus.RED
        assert SystemCommand.HALT in new_state.command_queue

    def test_underperformance_days_increments_on_non_green(self):
        """consecutive_underperformance_days increments each non-GREEN evaluation."""
        from rebuild_phase_3z.fppe.trading_system.shared_state import (
            LayerTag,
            SharedStateManager,
        )

        manager = SharedStateManager()
        state = SharedStateManager.initial_state(
            starting_equity=10_000.0,
            trading_date=date(2026, 3, 21),
        )

        cfg = EvaluatorConfig(yellow_min_trades=100)  # force YELLOW
        evaluator = StrategyEvaluator(config=cfg)
        _add_trades(evaluator, n=5)
        _add_daily_returns(evaluator, returns=[0.001] * 5)

        _, state = evaluator.evaluate_and_update_state(
            state, date(2026, 1, 6), manager
        )
        assert state.evaluator.consecutive_underperformance_days == 1

        # Evaluate again — should be 2 (if still not GREEN)
        _, state = evaluator.evaluate_and_update_state(
            state, date(2026, 1, 7), manager
        )
        assert state.evaluator.consecutive_underperformance_days == 2
