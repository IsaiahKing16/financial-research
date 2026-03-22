"""
test_trading_contracts.py — Unit tests for SLE-58 trading system contracts.

Tests:
  - EvaluatorStatus enum and exposure_multiplier property
  - EvaluatorSnapshot construction and properties
  - PositionDecision: approval logic, rejection_reason constraints
  - AllocationDecision: construction, validator
  - SharedState: initial factory, properties, command queue
  - TradeEvent: backtest_fill factory, fill_ratio validator
  - EquityState: component consistency validator

Linear: SLE-58
"""

from datetime import date

import pytest

from trading_system.contracts.decisions import (
    AllocationDecision,
    EvaluatorSnapshot,
    EvaluatorStatus,
    PositionDecision,
    RejectionReason,
)
from trading_system.contracts.state import (
    EquityState,
    EvaluatorState,
    PortfolioState,
    PositionsState,
    RiskState,
    SharedState,
    SystemCommand,
)
from trading_system.contracts.trades import (
    OrderSide,
    OrderStatus,
    TradeEvent,
)


# ─── EvaluatorStatus ──────────────────────────────────────────────────────────

class TestEvaluatorStatus:

    def test_exposure_multipliers(self):
        assert EvaluatorStatus.GREEN.exposure_multiplier == 1.0
        assert EvaluatorStatus.YELLOW.exposure_multiplier == 0.8
        assert EvaluatorStatus.RED.exposure_multiplier == 0.0

    def test_string_values(self):
        assert EvaluatorStatus.GREEN == "GREEN"
        assert EvaluatorStatus.RED == "RED"


class TestEvaluatorSnapshot:

    def _make_snapshot(self, status=EvaluatorStatus.GREEN) -> EvaluatorSnapshot:
        return EvaluatorSnapshot(
            evaluation_date=date(2024, 6, 1),
            status=status,
            sharpe_30d=1.2,
            sharpe_90d=1.5,
            drawdown_from_peak=0.02,
            days_in_market=180,
            reason="All metrics within bounds",
        )

    def test_green_not_halted(self):
        snap = self._make_snapshot(EvaluatorStatus.GREEN)
        assert not snap.is_halted
        assert snap.exposure_multiplier == 1.0

    def test_red_is_halted(self):
        snap = self._make_snapshot(EvaluatorStatus.RED)
        assert snap.is_halted
        assert snap.exposure_multiplier == 0.0

    def test_yellow_not_halted_but_reduced(self):
        snap = self._make_snapshot(EvaluatorStatus.YELLOW)
        assert not snap.is_halted
        assert snap.exposure_multiplier == 0.8

    def test_drawdown_out_of_range_raises(self):
        with pytest.raises(ValueError):
            EvaluatorSnapshot(
                evaluation_date=date(2024, 6, 1),
                status=EvaluatorStatus.GREEN,
                drawdown_from_peak=1.5,  # > 1.0
                days_in_market=100,
                reason="test",
            )

    def test_none_sharpe_allowed(self):
        snap = EvaluatorSnapshot(
            evaluation_date=date(2024, 1, 2),
            status=EvaluatorStatus.GREEN,
            sharpe_30d=None,  # Insufficient history
            drawdown_from_peak=0.0,
            days_in_market=5,
            reason="Insufficient history for Sharpe",
        )
        assert snap.sharpe_30d is None


# ─── PositionDecision ─────────────────────────────────────────────────────────

class TestPositionDecision:

    def _approved(self) -> PositionDecision:
        return PositionDecision(
            ticker="AAPL",
            signal_date=date(2024, 1, 2),
            approved=True,
            rejection_reason=None,
            position_pct=0.05,
            target_shares=10.0,
            entry_price_estimate=185.0,
            stop_loss_price=175.0,
            atr_pct=0.015,
            confidence=0.72,
            sector="Technology",
        )

    def _rejected(self, reason=RejectionReason.MAX_POSITIONS) -> PositionDecision:
        return PositionDecision(
            ticker="MSFT",
            signal_date=date(2024, 1, 2),
            approved=False,
            rejection_reason=reason,
            position_pct=0.0,
            target_shares=0.0,
            entry_price_estimate=0.0,
            stop_loss_price=0.0,
            atr_pct=0.0,
            confidence=0.70,
            sector="Technology",
        )

    def test_approved_construction(self):
        d = self._approved()
        assert d.approved is True
        assert d.rejection_reason is None

    def test_rejected_construction(self):
        d = self._rejected()
        assert d.approved is False
        assert d.rejection_reason == RejectionReason.MAX_POSITIONS

    def test_approved_without_reason_ok(self):
        # Passes
        self._approved()

    def test_rejected_without_reason_raises(self):
        with pytest.raises(ValueError, match="rejection_reason"):
            PositionDecision(
                ticker="AAPL",
                signal_date=date(2024, 1, 2),
                approved=False,
                rejection_reason=None,  # Must be set when rejected
                position_pct=0.0,
                target_shares=0.0,
                entry_price_estimate=0.0,
                stop_loss_price=0.0,
                atr_pct=0.0,
                confidence=0.7,
                sector="Technology",
            )

    def test_approved_with_reason_raises(self):
        with pytest.raises(ValueError, match="rejection_reason"):
            PositionDecision(
                ticker="AAPL",
                signal_date=date(2024, 1, 2),
                approved=True,
                rejection_reason=RejectionReason.LOW_CONFIDENCE,  # Should not be set
                position_pct=0.05,
                target_shares=10.0,
                entry_price_estimate=185.0,
                stop_loss_price=175.0,
                atr_pct=0.015,
                confidence=0.7,
                sector="Technology",
            )

    def test_approved_requires_nonzero_sizing(self):
        with pytest.raises(ValueError, match="position_pct"):
            PositionDecision(
                ticker="AAPL",
                signal_date=date(2024, 1, 2),
                approved=True,
                rejection_reason=None,
                position_pct=0.0,  # Zero sizing on approved trade
                target_shares=0.0,
                entry_price_estimate=185.0,
                stop_loss_price=175.0,
                atr_pct=0.015,
                confidence=0.7,
                sector="Technology",
            )

    def test_lowercase_ticker_raises(self):
        with pytest.raises(ValueError, match="uppercase"):
            PositionDecision(
                ticker="aapl",
                signal_date=date(2024, 1, 2),
                approved=False,
                rejection_reason=RejectionReason.MAX_POSITIONS,
                position_pct=0.0, target_shares=0.0,
                entry_price_estimate=0.0, stop_loss_price=0.0,
                atr_pct=0.0, confidence=0.7, sector="Technology",
            )

    def test_frozen(self):
        d = self._approved()
        with pytest.raises(Exception):
            d.position_pct = 0.1  # type: ignore


# ─── AllocationDecision ───────────────────────────────────────────────────────

class TestAllocationDecision:

    def test_valid_construction(self):
        d = AllocationDecision(
            ticker="NVDA",
            signal_date=date(2024, 3, 15),
            final_position_pct=0.04,
            evaluator_status=EvaluatorStatus.YELLOW,
            capital_allocated=400.0,
            rank_in_queue=2,
            sector="Technology",
            adjusted_for_evaluator=True,
        )
        assert d.adjusted_for_evaluator is True

    def test_frozen(self):
        d = AllocationDecision(
            ticker="NVDA", signal_date=date(2024, 3, 15),
            final_position_pct=0.04, evaluator_status=EvaluatorStatus.GREEN,
            capital_allocated=400.0, rank_in_queue=1, sector="Technology",
        )
        with pytest.raises(Exception):
            d.final_position_pct = 0.1  # type: ignore


# ─── SharedState ──────────────────────────────────────────────────────────────

class TestSharedState:

    def test_initial_factory(self):
        state = SharedState.initial(
            starting_equity=10_000.0,
            trading_date=date(2024, 1, 2),
        )
        assert state.total_equity == 10_000.0
        assert state.equity.cash == 10_000.0
        assert state.equity.invested_capital == 0.0
        assert state.positions.n_open == 0
        assert state.command_queue == []

    def test_initial_not_halted(self):
        state = SharedState.initial(10_000.0, date(2024, 1, 2))
        assert not state.is_halted
        assert state.exposure_multiplier == 1.0

    def test_has_capacity(self):
        state = SharedState.initial(10_000.0, date(2024, 1, 2))
        assert state.has_capacity()

    def test_negative_equity_raises(self):
        with pytest.raises(RuntimeError, match="positive"):
            SharedState.initial(-1000.0, date(2024, 1, 2))

    def test_trading_date_property(self):
        state = SharedState.initial(10_000.0, date(2024, 6, 15))
        assert state.trading_date == date(2024, 6, 15)

    def test_drawdown_property(self):
        state = SharedState.initial(10_000.0, date(2024, 1, 2))
        assert state.drawdown == 0.0

    def test_halt_command_makes_halted(self):
        state = SharedState.initial(10_000.0, date(2024, 1, 2))
        state_with_halt = state.model_copy(update={"command_queue": [SystemCommand.HALT]})
        assert state_with_halt.is_halted

    def test_frozen(self):
        state = SharedState.initial(10_000.0, date(2024, 1, 2))
        with pytest.raises(Exception):
            state.command_queue = []  # type: ignore

    def test_model_copy_update(self):
        """model_copy(update={...}) pattern creates a new instance."""
        state = SharedState.initial(10_000.0, date(2024, 1, 2))
        new_equity = state.equity.model_copy(update={
            "cash": 9_000.0,
            "invested_capital": 1_000.0,
        })
        new_state = state.model_copy(update={"equity": new_equity})
        assert new_state.equity.cash == 9_000.0
        assert state.equity.cash == 10_000.0  # Original unchanged


# ─── EquityState ──────────────────────────────────────────────────────────────

class TestEquityState:

    def test_component_inconsistency_raises(self):
        with pytest.raises(ValueError, match="total_equity"):
            EquityState(
                total_equity=10_000.0,
                cash=5_000.0,
                invested_capital=6_000.0,  # 5000 + 6000 = 11000 ≠ 10000
                peak_equity=10_000.0,
                inception_equity=10_000.0,
            )

    def test_peak_below_equity_raises(self):
        with pytest.raises(ValueError, match="peak_equity"):
            EquityState(
                total_equity=10_000.0,
                cash=10_000.0,
                invested_capital=0.0,
                peak_equity=9_000.0,  # Peak below current — impossible
                inception_equity=8_000.0,
            )

    def test_drawdown_zero_at_peak(self):
        eq = EquityState(
            total_equity=10_000.0,
            cash=10_000.0,
            invested_capital=0.0,
            peak_equity=10_000.0,
            inception_equity=10_000.0,
        )
        assert eq.drawdown_from_peak == 0.0

    def test_drawdown_nonzero(self):
        eq = EquityState(
            total_equity=9_000.0,
            cash=9_000.0,
            invested_capital=0.0,
            peak_equity=10_000.0,
            inception_equity=10_000.0,
        )
        assert abs(eq.drawdown_from_peak - 0.1) < 1e-10


# ─── TradeEvent ───────────────────────────────────────────────────────────────

class TestTradeEvent:

    def test_backtest_fill_factory(self):
        event = TradeEvent.backtest_fill(
            trade_event_id=1,
            trade_id=0,
            ticker="AAPL",
            side=OrderSide.BUY,
            order_date=date(2024, 1, 3),
            ordered_quantity=10.0,
            fill_price=185.5,
        )
        assert event.fill_ratio == 1.0
        assert event.status == OrderStatus.FILLED
        assert event.fill_quantity == 10.0
        assert event.execution_latency_seconds == 0.0
        assert event.broker_order_id is None

    def test_fill_ratio_mismatch_raises(self):
        with pytest.raises(ValueError, match="fill_ratio"):
            TradeEvent(
                trade_event_id=1,
                trade_id=0,
                ticker="AAPL",
                side=OrderSide.BUY,
                order_date=date(2024, 1, 3),
                ordered_quantity=10.0,
                limit_price_estimate=185.0,
                fill_quantity=10.0,
                fill_price=185.5,
                fill_ratio=0.5,  # Wrong: fill_quantity/ordered_quantity = 1.0
                status=OrderStatus.FILLED,
            )

    def test_filled_without_price_raises(self):
        with pytest.raises(ValueError, match="fill_price"):
            TradeEvent(
                trade_event_id=1,
                trade_id=0,
                ticker="AAPL",
                side=OrderSide.SELL,
                order_date=date(2024, 1, 3),
                ordered_quantity=10.0,
                limit_price_estimate=185.0,
                fill_quantity=10.0,
                fill_price=0.0,  # Not filled but status says FILLED
                fill_ratio=1.0,
                status=OrderStatus.FILLED,
            )

    def test_frozen(self):
        event = TradeEvent.backtest_fill(1, 0, "AAPL", OrderSide.BUY, date(2024, 1, 3), 10.0, 185.0)
        with pytest.raises(Exception):
            event.fill_price = 200.0  # type: ignore

    def test_lowercase_ticker_raises(self):
        with pytest.raises(ValueError, match="uppercase"):
            TradeEvent.backtest_fill(1, 0, "aapl", OrderSide.BUY, date(2024, 1, 3), 5.0, 185.0)
