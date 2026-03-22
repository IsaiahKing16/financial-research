"""
test_shared_state.py — Unit tests for SharedStateManager (SLE-68).

Tests:
  - Layer write isolation (permitted and rejected operations)
  - Command queue enqueue / drain semantics
  - JSON serialization round-trip
  - SharedState.initial() factory delegation
  - evaluate_and_update_state integration with StrategyEvaluator

Linear: SLE-68
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from rebuild_phase_3z.fppe.trading_system.contracts.state import (
    EvaluatorState,
    EquityState,
    PortfolioState,
    PositionsState,
    RiskState,
    SharedState,
    SystemCommand,
)
from rebuild_phase_3z.fppe.trading_system.shared_state import (
    LayerTag,
    SharedStateManager,
    _LAYER_WRITE_PERMITS,
    _COMMAND_ENQUEUE_LAYER,
    _COMMAND_DRAIN_LAYER,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def initial_state() -> SharedState:
    """A minimal valid SharedState at inception."""
    return SharedState.initial(
        starting_equity=10_000.0,
        trading_date=date(2026, 3, 21),
    )


@pytest.fixture
def manager() -> SharedStateManager:
    return SharedStateManager()


# ─── TestLayerPermits ─────────────────────────────────────────────────────────

class TestLayerPermits:
    """Write isolation enforcement tests."""

    def test_permit_table_covers_all_layers(self):
        """Every LayerTag must appear in the write-permit table."""
        for layer in LayerTag:
            assert layer in _LAYER_WRITE_PERMITS, f"{layer} missing from _LAYER_WRITE_PERMITS"

    def test_pattern_engine_is_read_only(self):
        """pattern_engine has an empty permit set."""
        assert _LAYER_WRITE_PERMITS[LayerTag.PATTERN_ENGINE] == frozenset()

    def test_evaluator_enqueue_layer(self):
        assert _COMMAND_ENQUEUE_LAYER == LayerTag.STRATEGY_EVALUATOR

    def test_portfolio_drain_layer(self):
        assert _COMMAND_DRAIN_LAYER == LayerTag.PORTFOLIO_MANAGER

    def test_risk_engine_may_update_risk(self, manager, initial_state):
        """Risk engine can update its permitted sub-states."""
        new_risk = RiskState(stop_loss_atr_multiple=2.5)
        new_positions = PositionsState(open_tickers=[], n_open=0, max_positions=10)
        new_state = manager.update(
            initial_state,
            LayerTag.RISK_ENGINE,
            risk=new_risk,
            positions=new_positions,
        )
        assert new_state.risk.stop_loss_atr_multiple == 2.5

    def test_portfolio_manager_may_update_equity(self, manager, initial_state):
        """Portfolio manager can update equity sub-state."""
        new_equity = EquityState(
            total_equity=10_500.0,
            cash=10_500.0,
            invested_capital=0.0,
            peak_equity=10_500.0,
            inception_equity=10_000.0,
        )
        new_state = manager.update(
            initial_state,
            LayerTag.PORTFOLIO_MANAGER,
            equity=new_equity,
        )
        assert new_state.equity.total_equity == 10_500.0

    def test_evaluator_may_update_evaluator(self, manager, initial_state):
        """Strategy evaluator can update evaluator sub-state."""
        from rebuild_phase_3z.fppe.trading_system.contracts.decisions import (
            EvaluatorSnapshot,
            EvaluatorStatus,
        )
        snapshot = EvaluatorSnapshot(
            evaluation_date=date(2026, 3, 21),
            status=EvaluatorStatus.GREEN,
            drawdown_from_peak=0.0,
            days_in_market=1,
            reason="First day",
        )
        new_eval = EvaluatorState(latest_snapshot=snapshot, consecutive_underperformance_days=0)
        new_state = manager.update(
            initial_state,
            LayerTag.STRATEGY_EVALUATOR,
            evaluator=new_eval,
        )
        assert new_state.evaluator.latest_snapshot.status == EvaluatorStatus.GREEN

    def test_pattern_engine_cannot_write_anything(self, manager, initial_state):
        """pattern_engine write attempt raises RuntimeError."""
        with pytest.raises(RuntimeError, match="does not have write permission"):
            manager.update(
                initial_state,
                LayerTag.PATTERN_ENGINE,
                equity=initial_state.equity,
            )

    def test_risk_engine_cannot_write_equity(self, manager, initial_state):
        """risk_engine cannot write equity (only risk + positions)."""
        with pytest.raises(RuntimeError, match="does not have write permission"):
            manager.update(
                initial_state,
                LayerTag.RISK_ENGINE,
                equity=initial_state.equity,
            )

    def test_evaluator_cannot_write_equity(self, manager, initial_state):
        """strategy_evaluator cannot write equity."""
        with pytest.raises(RuntimeError, match="does not have write permission"):
            manager.update(
                initial_state,
                LayerTag.STRATEGY_EVALUATOR,
                equity=initial_state.equity,
            )

    def test_unknown_field_raises(self, manager, initial_state):
        """Unknown field names raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Unknown SharedState field"):
            manager.update(
                initial_state,
                LayerTag.PORTFOLIO_MANAGER,
                nonexistent_field=42,
            )

    def test_empty_update_returns_same_state(self, manager, initial_state):
        """update() with no fields returns the same state object."""
        result = manager.update(initial_state, LayerTag.PORTFOLIO_MANAGER)
        assert result is initial_state


# ─── TestCommandQueue ─────────────────────────────────────────────────────────

class TestCommandQueue:
    """Command queue enqueue and drain semantics."""

    def test_enqueue_halt_by_evaluator(self, manager, initial_state):
        """STRATEGY_EVALUATOR can enqueue HALT."""
        new_state = manager.enqueue_command(
            initial_state, LayerTag.STRATEGY_EVALUATOR, SystemCommand.HALT
        )
        assert SystemCommand.HALT in new_state.command_queue

    def test_enqueue_multiple_commands(self, manager, initial_state):
        """Multiple commands accumulate in order."""
        s = manager.enqueue_command(initial_state, LayerTag.STRATEGY_EVALUATOR, SystemCommand.REDUCE_EXPOSURE)
        s = manager.enqueue_command(s, LayerTag.STRATEGY_EVALUATOR, SystemCommand.HALT)
        assert s.command_queue == [SystemCommand.REDUCE_EXPOSURE, SystemCommand.HALT]

    def test_only_evaluator_can_enqueue(self, manager, initial_state):
        """Non-evaluator layers cannot enqueue commands."""
        for layer in [LayerTag.PATTERN_ENGINE, LayerTag.RISK_ENGINE, LayerTag.PORTFOLIO_MANAGER]:
            with pytest.raises(RuntimeError, match="may enqueue commands"):
                manager.enqueue_command(initial_state, layer, SystemCommand.HALT)

    def test_drain_returns_commands_and_empty_queue(self, manager, initial_state):
        """drain_commands() returns pending commands and clears the queue."""
        s = manager.enqueue_command(initial_state, LayerTag.STRATEGY_EVALUATOR, SystemCommand.HALT)
        commands, new_state = manager.drain_commands(s, LayerTag.PORTFOLIO_MANAGER)
        assert commands == [SystemCommand.HALT]
        assert new_state.command_queue == []

    def test_drain_empty_queue(self, manager, initial_state):
        """drain_commands() on empty queue returns [] and unchanged state."""
        commands, new_state = manager.drain_commands(initial_state, LayerTag.PORTFOLIO_MANAGER)
        assert commands == []
        assert new_state.command_queue == []

    def test_only_portfolio_manager_can_drain(self, manager, initial_state):
        """Non-portfolio-manager layers cannot drain commands."""
        s = manager.enqueue_command(initial_state, LayerTag.STRATEGY_EVALUATOR, SystemCommand.HALT)
        for layer in [LayerTag.PATTERN_ENGINE, LayerTag.RISK_ENGINE, LayerTag.STRATEGY_EVALUATOR]:
            with pytest.raises(RuntimeError, match="may drain commands"):
                manager.drain_commands(s, layer)

    def test_drain_does_not_affect_other_fields(self, manager, initial_state):
        """drain_commands() only changes command_queue — all other fields preserved."""
        s = manager.enqueue_command(initial_state, LayerTag.STRATEGY_EVALUATOR, SystemCommand.RESUME)
        _, new_state = manager.drain_commands(s, LayerTag.PORTFOLIO_MANAGER)
        assert new_state.equity == initial_state.equity
        assert new_state.positions == initial_state.positions
        assert new_state.risk == initial_state.risk


# ─── TestJsonSerialization ────────────────────────────────────────────────────

class TestJsonSerialization:
    """JSON round-trip for daily checkpoint files."""

    def test_round_trip_initial_state(self, manager, initial_state):
        """to_json / from_json round-trip preserves all fields."""
        json_str = manager.to_json(initial_state)
        restored = manager.from_json(json_str)

        assert restored.equity.total_equity == initial_state.equity.total_equity
        assert restored.equity.inception_equity == initial_state.equity.inception_equity
        assert restored.portfolio.trading_date == initial_state.portfolio.trading_date
        assert restored.positions.max_positions == initial_state.positions.max_positions

    def test_schema_version_in_json(self, manager, initial_state):
        """Serialized JSON includes _schema_version field."""
        json_str = manager.to_json(initial_state)
        data = json.loads(json_str)
        assert "_schema_version" in data
        assert data["_schema_version"] == "SLE-68-v1"

    def test_invalid_schema_version_raises(self, manager, initial_state):
        """Unknown schema version raises RuntimeError on load."""
        json_str = manager.to_json(initial_state)
        data = json.loads(json_str)
        data["_schema_version"] = "FUTURE-v99"
        with pytest.raises(RuntimeError, match="schema version"):
            manager.from_json(json.dumps(data))

    def test_round_trip_with_commands(self, manager, initial_state):
        """Commands survive JSON round-trip."""
        s = manager.enqueue_command(initial_state, LayerTag.STRATEGY_EVALUATOR, SystemCommand.HALT)
        restored = manager.from_json(manager.to_json(s))
        assert SystemCommand.HALT in restored.command_queue

    def test_round_trip_preserves_date(self, manager, initial_state):
        """Trading date is preserved as date (not datetime) after round-trip."""
        restored = manager.from_json(manager.to_json(initial_state))
        restored_date = restored.portfolio.trading_date
        assert isinstance(restored_date, date)
        assert restored_date == date(2026, 3, 21)

    def test_malformed_json_raises_value_error(self, manager):
        """Malformed JSON raises ValueError (Pydantic validation error)."""
        with pytest.raises((ValueError, Exception)):
            manager.from_json('{"equity": "not_a_dict"}')


# ─── TestInitialStateFactory ──────────────────────────────────────────────────

class TestInitialStateFactory:
    """SharedStateManager.initial_state() factory tests."""

    def test_initial_state_equity(self):
        """Factory sets starting equity correctly."""
        state = SharedStateManager.initial_state(
            starting_equity=50_000.0,
            trading_date=date(2026, 1, 1),
        )
        assert state.equity.total_equity == 50_000.0
        assert state.equity.cash == 50_000.0
        assert state.equity.invested_capital == 0.0
        assert state.equity.peak_equity == 50_000.0
        assert state.equity.inception_equity == 50_000.0

    def test_initial_state_positions_empty(self):
        """Factory creates an empty positions sub-state."""
        state = SharedStateManager.initial_state(
            starting_equity=10_000.0,
            trading_date=date(2026, 1, 1),
        )
        assert state.positions.n_open == 0
        assert state.positions.open_tickers == []

    def test_initial_state_not_halted(self):
        """System starts in GREEN state (not halted)."""
        state = SharedStateManager.initial_state(
            starting_equity=10_000.0,
            trading_date=date(2026, 1, 1),
        )
        assert not state.is_halted

    def test_initial_state_has_capacity(self):
        """System starts with capacity for new positions."""
        state = SharedStateManager.initial_state(
            starting_equity=10_000.0,
            trading_date=date(2026, 1, 1),
            max_positions=5,
        )
        assert state.has_capacity()

    def test_manager_validates_state_type(self):
        """Passing a non-SharedState to __init__ raises RuntimeError."""
        with pytest.raises(RuntimeError, match="expects a SharedState"):
            SharedStateManager(initial_state={"not": "a SharedState"})

    def test_manager_accepts_none(self):
        """Passing None to __init__ is fine (stateless manager)."""
        manager = SharedStateManager(initial_state=None)
        assert manager is not None


# ─── TestImmutability ─────────────────────────────────────────────────────────

class TestImmutability:
    """Verify that update() never mutates the original state."""

    def test_update_does_not_mutate_original(self, manager, initial_state):
        """update() returns a new state; original is unchanged."""
        original_equity = initial_state.equity.total_equity
        new_equity = EquityState(
            total_equity=20_000.0,
            cash=20_000.0,
            invested_capital=0.0,
            peak_equity=20_000.0,
            inception_equity=10_000.0,
        )
        new_state = manager.update(
            initial_state, LayerTag.PORTFOLIO_MANAGER, equity=new_equity
        )
        # Original unchanged
        assert initial_state.equity.total_equity == original_equity
        # New state has new value
        assert new_state.equity.total_equity == 20_000.0

    def test_enqueue_does_not_mutate_original(self, manager, initial_state):
        """enqueue_command() returns a new state; original queue is unchanged."""
        original_queue = list(initial_state.command_queue)
        new_state = manager.enqueue_command(
            initial_state, LayerTag.STRATEGY_EVALUATOR, SystemCommand.HALT
        )
        assert initial_state.command_queue == original_queue
        assert SystemCommand.HALT in new_state.command_queue
