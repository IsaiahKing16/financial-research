"""
live.py — EOD live signal generation and order submission runner.

Orchestrates the nightly 4:00 PM execution pipeline:
  1. Validate EngineState config hash hasn't drifted since overnight fit
  2. Query PatternMatcher on today's val_db
  3. Apply signal filters (SentimentVetoFilter if enabled)
  4. Submit orders via broker adapter
  5. Return updated SharedState with new positions

Design: dependency injection (PatternMatcher + broker injected, not hardcoded)
so that tests can swap in a MockBrokerAdapter without real API keys.

Linear: M9 (data ingestion scale-up)
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from pattern_engine.contracts.state import EngineState
from pattern_engine.matcher import PatternMatcher
from trading_system.contracts.state import SharedState

logger = logging.getLogger(__name__)


# ─── Order / OrderResult ──────────────────────────────────────────────────────

@dataclass
class Order:
    """A single order to submit to the broker.

    Args:
        ticker: Stock ticker symbol (uppercase).
        direction: "BUY" or "SELL".
        notional: Dollar amount (pre-position-sizing).
    """
    ticker: str
    direction: str       # "BUY" or "SELL"
    notional: float      # dollar amount (pre-position-sizing)


@dataclass
class OrderResult:
    """Result returned by the broker after submitting an order.

    Args:
        ticker: Stock ticker symbol.
        filled_fraction: Fraction of order filled (1.0 = full, 0.0 = rejected).
        fill_price: Execution price (0.0 if rejected).
        latency_ms: Round-trip broker latency in milliseconds.
        error: None if successful; error message string if failed.
    """
    ticker: str
    filled_fraction: float  # 1.0 = full fill, 0.5 = half fill, 0.0 = rejected
    fill_price: float       # execution price (0.0 if rejected)
    latency_ms: float       # round-trip broker latency observed
    error: Optional[str]    # None if successful, error message if failed


# ─── BaseBrokerAdapter ABC ────────────────────────────────────────────────────

class BaseBrokerAdapter(ABC):
    """Abstract broker interface. Inject a concrete implementation into LiveRunner.

    All implementations must return OrderResult even on failure — never raise.
    """

    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult:
        """Submit a single order. Must return OrderResult even on failure."""

    @abstractmethod
    def get_account_value(self) -> float:
        """Return current total account value in dollars."""


# ─── MockBrokerAdapter ────────────────────────────────────────────────────────

class MockBrokerAdapter(BaseBrokerAdapter):
    """In-memory broker for testing. Simulates latency and partial fills.

    Args:
        fill_fraction: Fraction of each order filled (1.0 = full, 0.5 = half).
        latency_ms: Simulated round-trip latency in milliseconds.
        fail_tickers: Set of ticker symbols that will be rejected.
        account_value: Simulated account value.

    Note: latency_ms causes a real blocking time.sleep() call. Keep test
    values small (≤ 50ms) to avoid slowing the test suite.
    """

    def __init__(
        self,
        fill_fraction: float = 1.0,
        latency_ms: float = 0.0,
        fail_tickers: Optional[set] = None,
        account_value: float = 100_000.0,
    ) -> None:
        self.fill_fraction = fill_fraction
        self.latency_ms = latency_ms
        self.fail_tickers = fail_tickers or set()
        self._account_value = account_value
        self.submitted_orders: list[Order] = []  # for test assertions

    def submit_order(self, order: Order) -> OrderResult:
        import time
        self.submitted_orders.append(order)
        time.sleep(self.latency_ms / 1000.0)
        if order.ticker in self.fail_tickers:
            return OrderResult(
                ticker=order.ticker,
                filled_fraction=0.0,
                fill_price=0.0,
                latency_ms=self.latency_ms,
                error=f"Broker rejected {order.ticker}",
            )
        return OrderResult(
            ticker=order.ticker,
            filled_fraction=self.fill_fraction,
            fill_price=100.0,  # arbitrary sentinel — no real price relationship
            latency_ms=self.latency_ms,
            error=None,
        )

    def get_account_value(self) -> float:
        return self._account_value


# ─── LiveRunner ───────────────────────────────────────────────────────────────

class LiveRunner:
    """EOD live signal generation and order submission runner.

    Executes the nightly pipeline:
      1. (Optional) Verify EngineState config_hash hasn't drifted — warn, don't halt.
      2. Check shared_state.is_halted — skip all orders if True.
      3. Query matcher on val_db to get BUY/SELL/HOLD signals.
      4. Build and submit an Order for each BUY or SELL signal.
      5. Return (shared_state, order_results).

    SharedState update is deferred to Phase 4 PortfolioManager (out of scope
    here). This runner returns the original shared_state unchanged plus the
    list of OrderResult objects.

    Args:
        matcher: A fitted PatternMatcher instance.
        shared_state: Current SharedState (frozen Pydantic bus).
        broker: A BaseBrokerAdapter implementation (real or mock).
        engine_state: Optional EngineState checkpoint used to verify config
                      hash hasn't drifted since the overnight fit. If None,
                      the check is skipped entirely.
    """

    def __init__(
        self,
        matcher: PatternMatcher,
        shared_state: SharedState,
        broker: BaseBrokerAdapter,
        engine_state: Optional[EngineState] = None,
    ) -> None:
        if not isinstance(matcher, PatternMatcher):
            raise RuntimeError(
                f"matcher must be a PatternMatcher instance; got {type(matcher)}"
            )
        if not isinstance(shared_state, SharedState):
            raise RuntimeError(
                f"shared_state must be a SharedState instance; got {type(shared_state)}"
            )
        if not isinstance(broker, BaseBrokerAdapter):
            raise RuntimeError(
                f"broker must be a BaseBrokerAdapter instance; got {type(broker)}"
            )
        self._matcher = matcher
        self._shared_state = shared_state
        self._broker = broker
        self._engine_state = engine_state

    def run(
        self,
        val_db: pd.DataFrame,
    ) -> tuple[SharedState, list[OrderResult]]:
        """Execute the EOD pipeline. Returns (updated_state, order_results).

        Pipeline:
          1. If engine_state provided, verify config_hash hasn't drifted
             (log warning if so — don't halt).
          2. Check shared_state.is_halted — if True, skip all orders, return
             state unchanged and an empty list.
          3. Query matcher on val_db (confidence signals only: BUY/SELL,
             not HOLD).
          4. For each BUY/SELL signal: build Order(ticker, direction,
             notional=1000.0).
          5. Submit each Order via broker.submit_order().
          6. Return (shared_state, order_results).
             Note: SharedState update is deferred to Phase 4 PortfolioManager
                   (out of scope here — just return original state + results).

        Args:
            val_db: Query DataFrame with feature columns + Ticker + Date.
                    The matcher uses its own internally stored feature columns.

        Returns:
            (shared_state, order_results) — original state plus broker results.

        Raises:
            RuntimeError: If the matcher has not been fitted (raised by
                          PatternMatcher.query() internally).
        """
        # Step 1 — config drift check (warn only)
        if self._engine_state is not None:
            # We cannot reconstruct the original EngineConfig here without
            # full context, so we compare the stored hash directly against
            # the matcher's current config object when possible.
            try:
                config = self._matcher.config
                config_dict = (
                    config.model_dump()
                    if hasattr(config, "model_dump")
                    else vars(config)
                )
                current_hash = hashlib.sha256(
                    json.dumps(config_dict, sort_keys=True, default=str).encode()
                ).hexdigest()
                if current_hash != self._engine_state.config_hash:
                    logger.warning(
                        "LiveRunner: config_hash drift detected. "
                        "Stored hash=%s, current hash=%s. "
                        "Matcher may have been reconfigured since last fit. "
                        "Continuing execution.",
                        self._engine_state.config_hash[:16],
                        current_hash[:16],
                    )
            except Exception as exc:
                logger.warning(
                    "LiveRunner: could not verify config_hash: %s. "
                    "Continuing execution.",
                    exc,
                )

        # Step 2 — halt guard
        if self._shared_state.is_halted:
            logger.warning(
                "LiveRunner: system is halted (evaluator RED or HALT in queue). "
                "Skipping all order submission."
            )
            return self._shared_state, []

        # Step 3 — query matcher
        # query() raises RuntimeError("Call fit() before query()") if not fitted.
        _probs, signals_arr, _reasons, _n_matches, _mean_ret, _ensemble = (
            self._matcher.query(val_db, verbose=0)
        )

        # Step 4 & 5 — build orders and submit
        tickers = val_db["Ticker"].values
        order_results: list[OrderResult] = []

        for i, signal in enumerate(signals_arr):
            if signal not in ("BUY", "SELL"):
                # HOLD — skip
                continue
            ticker = str(tickers[i])
            order = Order(
                ticker=ticker,
                direction=signal,
                notional=1000.0,
            )
            result = self._broker.submit_order(order)
            order_results.append(result)

        # Step 6 — return original state + results
        # SharedState update deferred to Phase 4 PortfolioManager
        return self._shared_state, order_results
