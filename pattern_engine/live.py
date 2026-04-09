"""LiveRunner — execution-only runner for live trading."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from pattern_engine.matcher import PatternMatcher
from pattern_engine.contracts.state import EngineState
from trading_system.contracts.state import SharedState
from trading_system.contracts.decisions import AllocationDecision
from trading_system.portfolio_state import PortfolioSnapshot
from trading_system.broker.base import BaseBroker
from trading_system.order_manager import OrderManager, ManagedOrder
from trading_system.reconciliation import reconcile

log = logging.getLogger(__name__)


class LiveRunner:
    """Execution-only runner. Receives decisions, submits orders via OrderManager.

    The caller (daily orchestrator, Phase 8) runs the full pipeline upstream:
    matcher → PM → risk engine → position sizer → AllocationDecisions.
    LiveRunner receives decisions and exit tickers, then executes them.
    """

    def __init__(
        self,
        matcher: PatternMatcher,
        shared_state: SharedState,
        broker: BaseBroker,
        order_manager: OrderManager,
        engine_state: Optional[EngineState] = None,
        reconcile_on_start: bool = True,
    ) -> None:
        if not isinstance(matcher, PatternMatcher):
            raise RuntimeError(
                f"matcher must be a PatternMatcher, got {type(matcher).__name__}"
            )
        if not isinstance(broker, BaseBroker):
            raise RuntimeError(
                f"broker must be a BaseBroker, got {type(broker).__name__}"
            )
        self._matcher = matcher
        self._shared_state = shared_state
        self._broker = broker
        self._order_manager = order_manager
        self._engine_state = engine_state
        self._reconcile_on_start = reconcile_on_start

    def run(
        self,
        entry_decisions: list[AllocationDecision],
        exit_tickers: list[tuple[str, float, float]],
        snapshot: PortfolioSnapshot,
        prices: dict[str, float] | None = None,
    ) -> list[ManagedOrder]:
        """Execute entry and exit orders.

        Args:
            entry_decisions: BUY allocations from upstream pipeline.
            exit_tickers: List of (ticker, quantity, price) for exits.
            snapshot: Current portfolio snapshot for reconciliation.
            prices: Current prices by ticker (required for entry orders).

        Returns:
            List of ManagedOrder results (exits first, then entries).
        """
        # 1. Pre-flight reconciliation
        if self._reconcile_on_start:
            recon_result = reconcile(snapshot, self._broker)
            if not recon_result.passed:
                log.error(
                    "Reconciliation failed: %d mismatches, %d missing, %d unexpected",
                    len(recon_result.mismatches),
                    len(recon_result.missing_positions),
                    len(recon_result.unexpected_positions),
                )
                return []

        # 2. Config hash drift check (warn-only)
        if self._engine_state is not None:
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
                log.warning(
                    "Config drift detected: engine_state hash=%s, current=%s",
                    self._engine_state.config_hash,
                    current_hash,
                )

        # 3. Halt check
        if self._shared_state.is_halted:
            log.warning("SharedState is halted — skipping all orders")
            return []

        # 4. Create exit orders
        exit_orders = [
            self._order_manager.create_exit_order(ticker, quantity, price)
            for ticker, quantity, price in exit_tickers
        ]

        # 5. Create entry orders
        entry_orders = []
        for decision in entry_decisions:
            price = (prices or {}).get(decision.ticker)
            if price is not None and price > 0:
                entry_orders.append(
                    self._order_manager.create_order_from_decision(decision, price)
                )
            else:
                log.warning("No price for %s — skipping entry order", decision.ticker)

        # 6. Submit all orders (exits first, then entries)
        all_orders = exit_orders + entry_orders
        if not all_orders:
            return []

        return self._order_manager.submit_batch(all_orders)
