"""
shared_state.py — Operational SharedState layer isolation and persistence.

Wraps the SharedState Pydantic model (contracts/state.py) with three
operational concerns the contracts layer intentionally omits:

  1. Layer write isolation  — Each trading layer may only update its own
     sub-states. Violations raise RuntimeError immediately (fail-fast).

  2. JSON serialization    — Pydantic v2 model_dump_json / model_validate_json
     with date → ISO-8601 string handling for daily checkpoint files.

  3. Command queue helpers — Enqueue (evaluator) and drain (portfolio manager)
     without accidentally clearing fields that shouldn't change.

Write permits:
  LayerTag.PATTERN_ENGINE      → read-only  (produces signals; never writes state)
  LayerTag.RISK_ENGINE         → "risk", "positions"
  LayerTag.PORTFOLIO_MANAGER   → "equity", "positions", "portfolio"
  LayerTag.STRATEGY_EVALUATOR  → "evaluator"

Command queue:
  Only STRATEGY_EVALUATOR may enqueue commands.
  Only PORTFOLIO_MANAGER may drain the command queue.
  This is enforced separately from the general field-permit map.

Usage:
    manager = SharedStateManager(initial_state)

    # Evaluator updates its sub-state and enqueues a command:
    state = manager.update(LayerTag.STRATEGY_EVALUATOR, evaluator=new_eval)
    state = manager.enqueue_command(state, LayerTag.STRATEGY_EVALUATOR, SystemCommand.HALT)

    # Portfolio manager drains commands each day:
    commands, state = manager.drain_commands(state, LayerTag.PORTFOLIO_MANAGER)

    # Serialize for daily checkpoint:
    json_str = manager.to_json(state)
    restored = manager.from_json(json_str)

Linear: SLE-68
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Dict, FrozenSet, List, Tuple

from rebuild_phase_3z.fppe.trading_system.contracts.state import (
    SharedState,
    SystemCommand,
    EquityState,
    PositionsState,
    RiskState,
    PortfolioState,
    EvaluatorState,
)


# ─── LayerTag ──────────────────────────────────────────────────────────────────

class LayerTag(str, Enum):
    """
    Identity tag for the trading system's four operational layers.

    Used by SharedStateManager to enforce write isolation — each layer
    may only modify the sub-states it owns.

    Pattern engine is listed but has no write permit: it produces signals
    and reads SharedState for config, but never mutates it.
    """
    PATTERN_ENGINE = "pattern_engine"
    RISK_ENGINE = "risk_engine"
    PORTFOLIO_MANAGER = "portfolio_manager"
    STRATEGY_EVALUATOR = "strategy_evaluator"


# ─── Write-permit table ────────────────────────────────────────────────────────

# Maps layer → frozenset of SharedState field names it may write.
# These are the only fields each layer is allowed to pass in an update() call.
_LAYER_WRITE_PERMITS: Dict[LayerTag, FrozenSet[str]] = {
    LayerTag.PATTERN_ENGINE: frozenset(),                          # read-only
    LayerTag.RISK_ENGINE: frozenset({"risk", "positions"}),
    LayerTag.PORTFOLIO_MANAGER: frozenset({"equity", "positions", "portfolio"}),
    LayerTag.STRATEGY_EVALUATOR: frozenset({"evaluator"}),
}

# Only STRATEGY_EVALUATOR may enqueue; only PORTFOLIO_MANAGER may drain.
_COMMAND_ENQUEUE_LAYER = LayerTag.STRATEGY_EVALUATOR
_COMMAND_DRAIN_LAYER = LayerTag.PORTFOLIO_MANAGER


# ─── SharedStateManager ────────────────────────────────────────────────────────

class SharedStateManager:
    """
    Layer isolation and persistence wrapper for SharedState.

    SharedStateManager enforces the contract that each trading layer may
    only write to its designated sub-states. It also provides JSON
    serialization and command-queue helpers.

    The manager itself is stateless — it does NOT hold a current SharedState.
    All methods accept a SharedState and return a new one (pure functions).
    This makes the state evolution explicit and testable.

    Args:
        initial_state: Optional starting SharedState for validation checks.
                       Does not hold a reference after __init__.

    Raises:
        RuntimeError: If a layer attempts to write a field it does not own.
    """

    def __init__(self, initial_state: SharedState | None = None) -> None:
        # initial_state is accepted for API symmetry but not stored —
        # the manager is stateless; callers own the current state.
        if initial_state is not None and not isinstance(initial_state, SharedState):
            raise RuntimeError(
                f"SharedStateManager expects a SharedState; got {type(initial_state)}"
            )

    # ── Write isolation ────────────────────────────────────────────────────────

    def update(
        self,
        state: SharedState,
        layer: LayerTag,
        **fields,
    ) -> SharedState:
        """
        Apply a partial update to SharedState with layer write-isolation enforcement.

        Args:
            state: Current SharedState.
            layer: The layer requesting the update (must match write permits).
            **fields: Sub-state fields to update (e.g. equity=new_equity_state).
                      Each key must be a top-level SharedState field name.

        Returns:
            A new SharedState with the requested fields replaced.

        Raises:
            RuntimeError: If the layer does not have a write permit for any
                          requested field, or if `fields` contains unknown
                          SharedState field names.
        """
        if not fields:
            return state

        # Validate field names are real SharedState fields (check first for clearer errors)
        valid_state_fields = SharedState.model_fields
        unknown_fields = set(fields) - set(valid_state_fields)
        if unknown_fields:
            raise RuntimeError(
                f"Unknown SharedState field(s): {sorted(unknown_fields)}. "
                f"Valid fields: {sorted(valid_state_fields)}"
            )

        # Enforce layer write permits
        permitted = _LAYER_WRITE_PERMITS[layer]
        invalid_fields = set(fields) - permitted

        if invalid_fields:
            raise RuntimeError(
                f"Layer '{layer.value}' does not have write permission for "
                f"field(s): {sorted(invalid_fields)}. "
                f"Permitted fields: {sorted(permitted) or '(read-only)'}."
            )

        return state.model_copy(update=fields)

    # ── Command queue helpers ──────────────────────────────────────────────────

    def enqueue_command(
        self,
        state: SharedState,
        layer: LayerTag,
        command: SystemCommand,
    ) -> SharedState:
        """
        Add a command to the SharedState command queue.

        Only STRATEGY_EVALUATOR may call this.

        Args:
            state: Current SharedState.
            layer: Must be LayerTag.STRATEGY_EVALUATOR.
            command: The SystemCommand to enqueue.

        Returns:
            New SharedState with the command appended to command_queue.

        Raises:
            RuntimeError: If called by a layer other than STRATEGY_EVALUATOR.
        """
        if layer != _COMMAND_ENQUEUE_LAYER:
            raise RuntimeError(
                f"Only '{_COMMAND_ENQUEUE_LAYER.value}' may enqueue commands; "
                f"called by '{layer.value}'."
            )
        new_queue = list(state.command_queue) + [command]
        return state.model_copy(update={"command_queue": new_queue})

    def drain_commands(
        self,
        state: SharedState,
        layer: LayerTag,
    ) -> Tuple[List[SystemCommand], SharedState]:
        """
        Remove all pending commands from the queue and return them.

        Only PORTFOLIO_MANAGER may call this (it consumes commands each day).

        Args:
            state: Current SharedState.
            layer: Must be LayerTag.PORTFOLIO_MANAGER.

        Returns:
            (commands, new_state): commands is the list that was pending;
            new_state has an empty command_queue.

        Raises:
            RuntimeError: If called by a layer other than PORTFOLIO_MANAGER.
        """
        if layer != _COMMAND_DRAIN_LAYER:
            raise RuntimeError(
                f"Only '{_COMMAND_DRAIN_LAYER.value}' may drain commands; "
                f"called by '{layer.value}'."
            )
        commands = list(state.command_queue)
        new_state = state.model_copy(update={"command_queue": []})
        return commands, new_state

    # ── JSON serialization ─────────────────────────────────────────────────────

    def to_json(self, state: SharedState) -> str:
        """
        Serialize SharedState to a JSON string for daily checkpoints.

        Uses Pydantic v2's model_dump_json() which handles datetime.date →
        ISO-8601 string (YYYY-MM-DD) automatically.

        Args:
            state: SharedState to serialize.

        Returns:
            JSON string with schema_version field injected for forward compat.
        """
        data = json.loads(state.model_dump_json())
        data["_schema_version"] = "SLE-68-v1"
        return json.dumps(data, indent=2)

    def from_json(self, json_str: str) -> SharedState:
        """
        Deserialize a SharedState from a JSON checkpoint string.

        Handles forward-compatible schema_version field (stripped before
        Pydantic validation — unknown fields would cause a validation error).

        Args:
            json_str: JSON string produced by to_json().

        Returns:
            Reconstructed SharedState.

        Raises:
            ValueError: If the JSON is malformed or fails Pydantic validation.
            RuntimeError: If the schema_version is from an incompatible future version.
        """
        data = json.loads(json_str)
        schema_version = data.pop("_schema_version", "SLE-68-v1")

        # Guard against loading state written by a future incompatible schema
        supported = {"SLE-68-v1"}
        if schema_version not in supported:
            raise RuntimeError(
                f"SharedState checkpoint schema version '{schema_version}' is not "
                f"supported by this codebase. Supported: {supported}."
            )

        return SharedState.model_validate(data)

    # ── Convenience factory ────────────────────────────────────────────────────

    @staticmethod
    def initial_state(
        starting_equity: float,
        trading_date,
        max_positions: int = 10,
        stop_loss_atr_multiple: float = 3.0,
        max_holding_days: int = 14,
        confidence_threshold: float = 0.65,
        max_sector_concentration: int = 3,
    ) -> SharedState:
        """
        Create the initial SharedState at system start.

        Delegates to SharedState.initial() factory — provided here so callers
        only need to import shared_state.py, not contracts/state.py directly.

        Args:
            starting_equity: Initial capital in dollars.
            trading_date: First trading day (datetime.date).
            max_positions: Maximum concurrent open positions.
            stop_loss_atr_multiple: ATR stop multiplier (locked setting: 3.0).
            max_holding_days: Max days to hold a position (locked setting: 14).
            confidence_threshold: Minimum confidence to trade (locked: 0.65).
            max_sector_concentration: Max positions per sector.

        Returns:
            Fully initialized SharedState.
        """
        return SharedState.initial(
            starting_equity=starting_equity,
            trading_date=trading_date,
            max_positions=max_positions,
            stop_loss_atr_multiple=stop_loss_atr_multiple,
            max_holding_days=max_holding_days,
            confidence_threshold=confidence_threshold,
            max_sector_concentration=max_sector_concentration,
        )
