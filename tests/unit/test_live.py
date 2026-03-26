"""
test_live.py — Unit tests for LiveRunner, MockBrokerAdapter, Order, OrderResult.

Tests:
  - LiveRunner state management and broker integration
  - MockBrokerAdapter: latency, partial fills, per-ticker failures, order tracking
  - Halted-system guard (is_halted skips all broker calls)
  - Config drift warning (logs warning but does not halt execution)
  - HOLD signals do not generate orders

Uses synthetic data only — no yfinance or network calls.

Linear: M9
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from pattern_engine.features import RETURNS_ONLY_COLS
from pattern_engine.live import (
    BaseBrokerAdapter,
    LiveRunner,
    MockBrokerAdapter,
    Order,
    OrderResult,
)
from pattern_engine.matcher import PatternMatcher
from trading_system.contracts.state import SharedState, SystemCommand


# ─── Feature columns (locked: returns_only, 8 features) ─────────────────────

FEATURE_COLS = RETURNS_ONLY_COLS  # ["ret_1d", "ret_3d", ..., "ret_90d"]
N_FEATURES = len(FEATURE_COLS)


# ─── Minimal config shim (mirrors WalkForwardConfig from scripts/run_walkforward) ──

class _MinimalConfig:
    """Minimal EngineConfig-like object for testing PatternMatcher."""
    top_k = 50
    max_distance = 1.1019
    distance_weighting = "uniform"
    feature_weights: dict = {}
    batch_size = 256
    confidence_threshold = 0.65
    agreement_spread = 0.05
    min_matches = 5
    exclude_same_ticker = True
    same_sector_only = False
    regime_filter = False
    regime_fallback = False
    projection_horizon = "fwd_7d_up"
    calibration_method = "none"  # skip Platt for fast unit tests
    cal_max_samples = 100_000
    use_hnsw = False             # BallTree — deterministic for tests
    use_sax_filter = False
    use_wfa_rerank = False
    use_ib_compression = False
    journal_top_n = 0
    use_sector_conviction = False
    use_momentum_filter = False
    use_sentiment_veto = False
    sector_conviction_lift = 0.005
    momentum_min_outperformance = 0.015


# ─── Synthetic data builders ─────────────────────────────────────────────────

def _make_train_db(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Build a minimal training DataFrame with feature columns and targets."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2015-01-01", periods=n, freq="B"),
            "Ticker": rng.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], size=n),
            "fwd_7d_up": rng.randint(0, 2, size=n).astype(float),
            "fwd_7d": rng.randn(n) * 2.0,
        }
    )
    for col in FEATURE_COLS:
        df[col] = rng.randn(n)
    return df


def _make_val_db(n: int = 10, seed: int = 99) -> pd.DataFrame:
    """Build a minimal val DataFrame for querying."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-02", periods=n, freq="B"),
            "Ticker": rng.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], size=n),
            "fwd_7d_up": rng.randint(0, 2, size=n).astype(float),
            "fwd_7d": rng.randn(n) * 2.0,
        }
    )
    for col in FEATURE_COLS:
        df[col] = rng.randn(n)
    return df


# ─── TestLiveRunner ───────────────────────────────────────────────────────────

class TestLiveRunner:
    """Tests for LiveRunner state management and broker integration."""

    def _make_matcher_and_db(self):
        """Build a minimal fitted PatternMatcher and val_db for testing."""
        config = _MinimalConfig()
        matcher = PatternMatcher(config)
        train_db = _make_train_db(n=200, seed=42)
        matcher.fit(train_db, FEATURE_COLS)
        val_db = _make_val_db(n=10, seed=99)
        return matcher, val_db

    def _make_shared_state(self) -> SharedState:
        """Build a minimal SharedState using SharedState.initial()."""
        return SharedState.initial(
            starting_equity=100_000.0,
            trading_date=date(2024, 1, 2),
        )

    # ── Core integration tests ────────────────────────────────────────────────

    def test_run_returns_state_and_results(self):
        """run() returns (SharedState, list[OrderResult])."""
        matcher, val_db = self._make_matcher_and_db()
        state = self._make_shared_state()
        broker = MockBrokerAdapter()
        runner = LiveRunner(matcher=matcher, shared_state=state, broker=broker)

        result_state, order_results = runner.run(val_db)

        assert isinstance(result_state, SharedState)
        assert isinstance(order_results, list)
        for r in order_results:
            assert isinstance(r, OrderResult)

    def test_run_returns_original_shared_state(self):
        """run() returns the original SharedState unchanged (Phase 4 deferred)."""
        matcher, val_db = self._make_matcher_and_db()
        state = self._make_shared_state()
        broker = MockBrokerAdapter()
        runner = LiveRunner(matcher=matcher, shared_state=state, broker=broker)

        result_state, _ = runner.run(val_db)

        # Frozen Pydantic — identity or equality with original
        assert result_state is state or result_state == state

    def test_halted_state_skips_all_orders(self):
        """If shared_state.is_halted, run() returns empty order_results list."""
        matcher, val_db = self._make_matcher_and_db()
        # Build a halted SharedState by putting HALT in the command_queue
        base_state = self._make_shared_state()
        halted_state = base_state.model_copy(
            update={"command_queue": [SystemCommand.HALT]}
        )
        assert halted_state.is_halted

        broker = MockBrokerAdapter()
        runner = LiveRunner(
            matcher=matcher, shared_state=halted_state, broker=broker
        )
        _, order_results = runner.run(val_db)

        assert order_results == []
        assert len(broker.submitted_orders) == 0

    def test_buy_signals_generate_orders(self):
        """BUY signals result in broker.submit_order() calls (no crash)."""
        # Use a large training set to increase chance of BUY signals; we
        # assert no exception and that any resulting orders are well-formed.
        config = _MinimalConfig()
        # Lower confidence threshold to increase BUY signal count in synthetic data
        config.confidence_threshold = 0.51
        config.agreement_spread = 0.0
        config.min_matches = 1

        matcher = PatternMatcher(config)
        train_db = _make_train_db(n=500, seed=42)
        matcher.fit(train_db, FEATURE_COLS)
        val_db = _make_val_db(n=20, seed=7)

        state = self._make_shared_state()
        broker = MockBrokerAdapter()
        runner = LiveRunner(matcher=matcher, shared_state=state, broker=broker)

        result_state, order_results = runner.run(val_db)

        # No exception raised — pipeline completed successfully
        assert isinstance(result_state, SharedState)
        assert isinstance(order_results, list)
        # Every submitted order has the correct structure
        for order in broker.submitted_orders:
            assert isinstance(order, Order)
            assert order.direction in ("BUY", "SELL")
            assert order.notional == 1000.0
            assert order.ticker == order.ticker.upper()

    def test_hold_signals_do_not_generate_orders(self):
        """HOLD signals should not result in broker calls."""
        # Use a high confidence threshold so all signals become HOLD
        config = _MinimalConfig()
        config.confidence_threshold = 0.999  # near-impossible to satisfy
        config.agreement_spread = 0.999      # also near-impossible

        matcher = PatternMatcher(config)
        train_db = _make_train_db(n=200, seed=42)
        matcher.fit(train_db, FEATURE_COLS)
        val_db = _make_val_db(n=10, seed=99)

        state = self._make_shared_state()
        broker = MockBrokerAdapter()
        runner = LiveRunner(matcher=matcher, shared_state=state, broker=broker)

        _, order_results = runner.run(val_db)

        # With impossibly high thresholds, all signals should be HOLD
        assert len(broker.submitted_orders) == 0
        assert order_results == []

    def test_engine_state_config_drift_logs_warning(self, caplog):
        """Mismatched config_hash logs a warning but does not halt execution."""
        matcher, val_db = self._make_matcher_and_db()
        state = self._make_shared_state()
        broker = MockBrokerAdapter()

        # Build an EngineState with a deliberately different config_hash
        # (a valid SHA-256 hex string that won't match the matcher's config)
        different_hash = hashlib.sha256(b"different_config").hexdigest()
        fit_timestamp = datetime.now(timezone.utc).isoformat()
        from pattern_engine.contracts.state import EngineState
        engine_state = EngineState(
            feature_cols=FEATURE_COLS,
            scaler_mean=[0.0] * N_FEATURES,
            scaler_scale=[1.0] * N_FEATURES,
            n_samples=200,
            matcher_backend="balltree",
            matcher_params={
                "backend": "balltree",
                "n_neighbors": 50,
                "metric": "euclidean",
            },
            config_hash=different_hash,
            fit_timestamp=fit_timestamp,
            feature_set_name="returns_only",
        )

        runner = LiveRunner(
            matcher=matcher,
            shared_state=state,
            broker=broker,
            engine_state=engine_state,
        )

        with caplog.at_level(logging.WARNING, logger="pattern_engine.live"):
            result_state, order_results = runner.run(val_db)

        # Execution must complete — no exception raised
        assert isinstance(result_state, SharedState)
        assert isinstance(order_results, list)
        # Warning must have been logged
        assert any("config_hash" in record.message for record in caplog.records)

    # ── MockBrokerAdapter unit tests ──────────────────────────────────────────

    def test_partial_fill_recorded_in_results(self):
        """MockBrokerAdapter with fill_fraction=0.5 returns filled_fraction=0.5."""
        broker = MockBrokerAdapter(fill_fraction=0.5)
        result = broker.submit_order(Order("AAPL", "BUY", 1000.0))
        assert result.filled_fraction == 0.5
        assert result.error is None

    def test_broker_latency_recorded(self):
        """MockBrokerAdapter records latency_ms in OrderResult."""
        broker = MockBrokerAdapter(latency_ms=10.0)
        result = broker.submit_order(Order("AAPL", "BUY", 1000.0))
        assert result.latency_ms == 10.0

    def test_failed_ticker_returns_error_result(self):
        """fail_tickers causes OrderResult with filled_fraction=0 and error set."""
        broker = MockBrokerAdapter(fail_tickers={"AAPL"})
        result = broker.submit_order(Order("AAPL", "BUY", 1000.0))
        assert result.filled_fraction == 0.0
        assert result.error is not None
        assert "AAPL" in result.error

    def test_failed_ticker_does_not_affect_other_orders(self):
        """Broker failure on one ticker still processes others."""
        broker = MockBrokerAdapter(fail_tickers={"AAPL"})
        r1 = broker.submit_order(Order("AAPL", "BUY", 1000.0))
        r2 = broker.submit_order(Order("MSFT", "BUY", 1000.0))
        assert r1.filled_fraction == 0.0
        assert r2.filled_fraction == 1.0
        assert r2.error is None

    def test_mock_broker_records_submitted_orders(self):
        """MockBrokerAdapter.submitted_orders tracks all calls for assertions."""
        broker = MockBrokerAdapter()
        broker.submit_order(Order("AAPL", "BUY", 1000.0))
        broker.submit_order(Order("MSFT", "SELL", 500.0))
        assert len(broker.submitted_orders) == 2
        assert broker.submitted_orders[0].ticker == "AAPL"
        assert broker.submitted_orders[1].ticker == "MSFT"

    def test_full_fill_returns_fill_price(self):
        """Default MockBrokerAdapter returns fill_price=100.0 on success."""
        broker = MockBrokerAdapter()
        result = broker.submit_order(Order("NVDA", "BUY", 2000.0))
        assert result.fill_price == 100.0
        assert result.filled_fraction == 1.0
        assert result.error is None

    def test_rejected_fill_price_is_zero(self):
        """Rejected orders (fail_tickers) return fill_price=0.0."""
        broker = MockBrokerAdapter(fail_tickers={"TSLA"})
        result = broker.submit_order(Order("TSLA", "BUY", 1000.0))
        assert result.fill_price == 0.0
        assert result.filled_fraction == 0.0

    def test_get_account_value(self):
        """MockBrokerAdapter returns the configured account value."""
        broker = MockBrokerAdapter(account_value=250_000.0)
        assert broker.get_account_value() == 250_000.0

    def test_mock_broker_is_base_adapter(self):
        """MockBrokerAdapter satisfies the BaseBrokerAdapter ABC."""
        assert issubclass(MockBrokerAdapter, BaseBrokerAdapter)

    def test_order_result_ticker_matches_order(self):
        """OrderResult.ticker matches the Order.ticker that was submitted."""
        broker = MockBrokerAdapter()
        result = broker.submit_order(Order("GOOGL", "SELL", 750.0))
        assert result.ticker == "GOOGL"

    def test_rejected_order_ticker_matches(self):
        """Rejected OrderResult.ticker still matches the Order.ticker."""
        broker = MockBrokerAdapter(fail_tickers={"AMZN"})
        result = broker.submit_order(Order("AMZN", "BUY", 500.0))
        assert result.ticker == "AMZN"

    # ── Constructor guard tests ───────────────────────────────────────────────

    def test_invalid_matcher_raises_runtime_error(self):
        """LiveRunner rejects non-PatternMatcher as matcher."""
        state = self._make_shared_state()
        broker = MockBrokerAdapter()
        with pytest.raises(RuntimeError, match="PatternMatcher"):
            LiveRunner(
                matcher="not_a_matcher",  # type: ignore
                shared_state=state,
                broker=broker,
            )

    def test_invalid_broker_raises_runtime_error(self):
        """LiveRunner rejects non-BaseBrokerAdapter as broker."""
        matcher, _ = self._make_matcher_and_db()
        state = self._make_shared_state()
        with pytest.raises(RuntimeError, match="BaseBrokerAdapter"):
            LiveRunner(
                matcher=matcher,
                shared_state=state,
                broker="not_a_broker",  # type: ignore
            )
