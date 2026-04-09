"""Tests for LiveRunner — rewired for Phase 5."""
import logging
import pytest
from datetime import date, datetime, timezone
from unittest.mock import patch

from pattern_engine.matcher import PatternMatcher
from pattern_engine.config import EngineConfig
from pattern_engine.contracts.state import EngineState
from trading_system.contracts.state import SharedState
from trading_system.contracts.decisions import AllocationDecision, EvaluatorStatus
from trading_system.contracts.trades import OrderStatus, OrderSide
from trading_system.portfolio_state import PortfolioSnapshot, OpenPosition
from trading_system.broker.mock import MockBroker, MockBrokerConfig
from trading_system.order_manager import OrderManager, ManagedOrder
from pattern_engine.live import LiveRunner


def _decision(ticker: str = "AAPL", capital: float = 1500.0, rank: int = 1):
    return AllocationDecision(
        ticker=ticker,
        signal_date=date(2024, 1, 2),
        final_position_pct=0.05,
        evaluator_status=EvaluatorStatus.GREEN,
        capital_allocated=capital,
        rank_in_queue=rank,
        sector="Technology",
    )


def _snapshot(positions=None):
    return PortfolioSnapshot(
        as_of_date=date(2024, 1, 2),
        equity=100_000.0,
        cash=90_000.0,
        open_positions=tuple(positions or []),
    )


def _make_runner(
    cash: float = 100_000.0,
    reconcile_on_start: bool = False,
    prices: dict[str, float] | None = None,
):
    """Create a LiveRunner with MockBroker for testing."""
    import numpy as np
    import pandas as pd

    config = EngineConfig()
    matcher = PatternMatcher(config)

    # Fit matcher with minimal synthetic data
    n = 200
    rng = np.random.default_rng(42)
    cols = [f"feat_{i}" for i in range(8)]
    train = pd.DataFrame(rng.standard_normal((n, 8)), columns=cols)
    train["fwd_7d_up"] = rng.integers(0, 2, size=n)
    train["Ticker"] = "TRAIN"
    train["Date"] = pd.date_range("2020-01-01", periods=n)
    matcher.fit(train, cols)

    state = SharedState.initial(cash, date(2024, 1, 2))
    broker = MockBroker(MockBrokerConfig(initial_cash=cash, slippage_bps=0.0))
    if prices:
        broker.set_prices(prices)
    else:
        broker.set_prices({"AAPL": 150.0, "MSFT": 300.0, "GOOG": 100.0})
    om = OrderManager(broker=broker)

    runner = LiveRunner(
        matcher=matcher,
        shared_state=state,
        broker=broker,
        order_manager=om,
        reconcile_on_start=reconcile_on_start,
    )
    return runner, broker, om


class TestLiveRunnerConstructor:
    def test_invalid_matcher_raises(self):
        broker = MockBroker()
        om = OrderManager(broker=broker)
        state = SharedState.initial(100_000, date(2024, 1, 2))
        with pytest.raises(RuntimeError):
            LiveRunner(
                matcher="not_a_matcher",
                shared_state=state,
                broker=broker,
                order_manager=om,
            )

    def test_invalid_broker_raises(self):
        import numpy as np
        import pandas as pd
        config = EngineConfig()
        m = PatternMatcher(config)
        n = 50
        rng = np.random.default_rng(42)
        cols = [f"feat_{i}" for i in range(8)]
        train = pd.DataFrame(rng.standard_normal((n, 8)), columns=cols)
        train["fwd_7d_up"] = rng.integers(0, 2, size=n)
        train["Ticker"] = "TRAIN"
        train["Date"] = pd.date_range("2020-01-01", periods=n)
        m.fit(train, cols)

        state = SharedState.initial(100_000, date(2024, 1, 2))
        with pytest.raises(RuntimeError):
            LiveRunner(
                matcher=m,
                shared_state=state,
                broker="not_a_broker",
                order_manager=OrderManager(broker=MockBroker()),
            )


class TestLiveRunnerHalt:
    def test_halted_returns_empty(self):
        runner, broker, om = _make_runner()
        from trading_system.contracts.state import SystemCommand
        halted = runner._shared_state.model_copy(
            update={"command_queue": (SystemCommand.HALT,)}
        )
        runner._shared_state = halted
        results = runner.run(
            entry_decisions=[_decision()],
            exit_tickers=[],
            snapshot=_snapshot(),
            prices={"AAPL": 150.0},
        )
        assert results == []


class TestLiveRunnerReconciliation:
    def test_reconcile_failure_returns_empty(self):
        runner, broker, om = _make_runner(reconcile_on_start=True)
        # Snapshot expects AAPL position but broker has none → fail
        snap = _snapshot([OpenPosition(
            ticker="AAPL", sector="Tech", entry_date=date(2024, 1, 1),
            position_pct=0.05, entry_price=100.0,
        )])
        results = runner.run(
            entry_decisions=[_decision()],
            exit_tickers=[],
            snapshot=snap,
            prices={"AAPL": 150.0},
        )
        assert results == []

    def test_reconcile_off_proceeds(self):
        runner, broker, om = _make_runner(reconcile_on_start=False)
        snap = _snapshot()
        results = runner.run(
            entry_decisions=[_decision()],
            exit_tickers=[],
            snapshot=snap,
            prices={"AAPL": 150.0},
        )
        assert len(results) == 1
        assert results[0].status == OrderStatus.FILLED

    def test_reconcile_on_passes_with_clean_state(self):
        runner, broker, om = _make_runner(reconcile_on_start=True)
        # Empty snapshot + empty broker → reconcile passes → orders proceed
        snap = _snapshot()  # no open positions
        results = runner.run(
            entry_decisions=[_decision()],
            exit_tickers=[],
            snapshot=snap,
            prices={"AAPL": 150.0},
        )
        assert len(results) == 1
        assert results[0].status == OrderStatus.FILLED


class TestLiveRunnerOrders:
    def test_entry_decision_creates_buy(self):
        runner, broker, om = _make_runner()
        results = runner.run(
            entry_decisions=[_decision()],
            exit_tickers=[],
            snapshot=_snapshot(),
            prices={"AAPL": 150.0},
        )
        assert len(results) == 1
        assert results[0].order.side == OrderSide.BUY
        assert results[0].order.ticker == "AAPL"
        assert results[0].order.quantity == pytest.approx(10.0)  # 1500/150

    def test_exit_ticker_creates_sell(self):
        runner, broker, om = _make_runner()
        # First buy AAPL so broker has position
        from trading_system.broker.base import Order as BrokerOrder
        broker.submit_order(BrokerOrder(
            order_id="setup", ticker="AAPL", side=OrderSide.BUY,
            quantity=10.0, timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ))
        results = runner.run(
            entry_decisions=[],
            exit_tickers=[("AAPL", 10.0, 155.0)],
            snapshot=_snapshot(),
        )
        assert len(results) == 1
        assert results[0].order.side == OrderSide.SELL

    def test_exits_before_entries(self):
        runner, broker, om = _make_runner()
        # Buy AAPL for exit
        from trading_system.broker.base import Order as BrokerOrder
        broker.submit_order(BrokerOrder(
            order_id="setup", ticker="AAPL", side=OrderSide.BUY,
            quantity=10.0, timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ))
        results = runner.run(
            entry_decisions=[_decision(ticker="MSFT", capital=3000.0)],
            exit_tickers=[("AAPL", 10.0, 155.0)],
            snapshot=_snapshot(),
            prices={"MSFT": 300.0},
        )
        assert len(results) == 2
        # Exit (SELL) should be first
        assert results[0].order.side == OrderSide.SELL
        assert results[1].order.side == OrderSide.BUY

    def test_empty_decisions_empty_results(self):
        runner, broker, om = _make_runner()
        results = runner.run(
            entry_decisions=[],
            exit_tickers=[],
            snapshot=_snapshot(),
        )
        assert results == []

    def test_multiple_entries(self):
        runner, broker, om = _make_runner()
        results = runner.run(
            entry_decisions=[
                _decision(ticker="AAPL", rank=1),
                _decision(ticker="MSFT", capital=3000.0, rank=2),
            ],
            exit_tickers=[],
            snapshot=_snapshot(),
            prices={"AAPL": 150.0, "MSFT": 300.0},
        )
        assert len(results) == 2
        assert all(m.status == OrderStatus.FILLED for m in results)


class TestLiveRunnerConfigDrift:
    def test_config_drift_logs_warning(self, caplog):
        runner, broker, om = _make_runner()
        # Build EngineState directly — from_fitted requires a BaseMatcher backend,
        # not a PatternMatcher. We use the underlying backend from _matcher._backend.
        backend = runner._matcher._backend
        scaler = runner._matcher._scaler
        feature_cols = runner._matcher._feature_cols
        engine_state = EngineState.from_fitted(
            scaler=scaler,
            matcher=backend,
            feature_cols=feature_cols,
            config=EngineConfig(),
            feature_set_name="test",
        )
        # Tamper the hash
        runner._engine_state = engine_state.model_copy(
            update={"config_hash": "0" * 64}
        )
        with caplog.at_level(logging.WARNING):
            runner.run(
                entry_decisions=[_decision()],
                exit_tickers=[],
                snapshot=_snapshot(),
                prices={"AAPL": 150.0},
            )
        assert any("drift" in r.message.lower() or "mismatch" in r.message.lower()
                    for r in caplog.records)
