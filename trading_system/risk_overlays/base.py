"""
base.py — BaseRiskOverlay ABC for all research risk pilots (SLE-74, SLE-75).

Every risk overlay follows the same contract:
  1. update(current_date, **market_data) — ingest the latest market snapshot.
  2. get_signal_multiplier() → float in [0, 1] — throttle factor applied to
     signal confidence before thresholding.  1.0 = no throttle, 0.0 = block.
  3. reset() — reset all accumulated state.

Signal integration (Phase 3 contract):
    final_position_pct = sizing.position_pct × overlay.get_signal_multiplier()
    Overlays multiply POSITION SIZE in the risk_engine orchestrator, not
    signal confidence.  Reason: Half-Kelly position_sizer already incorporates
    confidence into position size, so re-throttling confidence would
    double-count the same input.

    Phase 1/2 historical contract (deprecated):
        effective_confidence = raw_confidence × overlay.get_signal_multiplier()

Feature flag pattern:
    All overlays are guarded by a flag in ResearchFlagsConfig:
        if cfg.research_flags.use_liquidity_congestion_gate:
            overlay = LiquidityCongestionGate(...)
    Default=False means production behavior is unchanged.

Linear: SLE-74, SLE-75
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any


class BaseRiskOverlay(ABC):
    """Abstract base class for pluggable risk overlays.

    All methods must be implemented by concrete subclasses.  The overlay
    is stateful — it accumulates data across multiple `update()` calls.
    """

    @abstractmethod
    def update(self, current_date: date, **market_data: Any) -> None:
        """Ingest the latest market snapshot.

        Implementations should update any internal state (rolling buffers,
        accumulators, regime trackers) based on the provided data.

        Args:
            current_date: Trading date for this snapshot.
            **market_data: Keyword arguments with market data relevant to
                           this overlay (e.g., volume, atr, regime_label).
                           Unrecognised keys must be silently ignored so
                           overlays compose safely.
        """

    @abstractmethod
    def _compute_multiplier(self) -> float:
        """Subclass hook: compute the raw multiplier.

        Must return a float in [0.0, 1.0]. Values outside this range are
        considered category errors and will be rejected by the base class
        (see `get_signal_multiplier`). Subclasses should not clamp — if
        you need to clamp, do so explicitly inside this method, but
        returning out-of-range values is a bug worth surfacing.

        Returns:
            Raw multiplier float. Validated by base class before return.
        """

    def get_signal_multiplier(self) -> float:
        """Return the current signal throttle factor, validated to [0.0, 1.0].

        Template method: delegates to subclass `_compute_multiplier`, then
        guards the post-condition that the returned value is in [0.0, 1.0].
        A value outside this range raises `RuntimeError` — amplification
        (>1) or negative sizing is always a bug, never graceful degradation.

        Returns:
            float in [0.0, 1.0]:
                1.0 — no throttle; signals are unchanged.
                0.0 — full block; all signals become HOLD.
                (0, 1) — partial throttle; position size is reduced.

        Raises:
            RuntimeError: if the subclass returns a value outside [0.0, 1.0].
        """
        value = self._compute_multiplier()
        if not (0.0 <= value <= 1.0):
            raise RuntimeError(
                f"{type(self).__name__}._compute_multiplier() returned "
                f"{value}, must be in [0.0, 1.0]"
            )
        return value

    @abstractmethod
    def reset(self) -> None:
        """Reset all accumulated state to the initial (no-signal) condition."""

    @property
    def name(self) -> str:
        """Human-readable overlay name (defaults to class name)."""
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{self.name}(multiplier={self.get_signal_multiplier():.3f})"
