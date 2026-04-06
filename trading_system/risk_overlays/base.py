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
    def get_signal_multiplier(self) -> float:
        """Return the current signal throttle factor.

        Returns:
            float in [0.0, 1.0]:
                1.0 — no throttle; signals are unchanged.
                0.0 — full block; all signals become HOLD.
                (0, 1) — partial throttle; confidence is reduced.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all accumulated state to the initial (no-signal) condition."""

    @property
    def name(self) -> str:
        """Human-readable overlay name (defaults to class name)."""
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{self.name}(multiplier={self.get_signal_multiplier():.3f})"
