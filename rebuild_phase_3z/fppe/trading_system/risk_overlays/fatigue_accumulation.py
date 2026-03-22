"""
fatigue_accumulation.py — Miner-style fatigue accumulation overlay (SLE-75).

Models the idea that extended trends "exhaust" themselves — the longer a
regime persists, the greater the risk of a sudden reversal.  Inspired by
research in mining / structural engineering where cumulative stress
accumulates during sustained load and resets when the load is removed.

Mechanism:
    - `fatigue_score` in [0, 1] accumulates monotonically during a continuous
      regime (same regime_label across consecutive updates).
    - Accumulation is exponential-approach: each period adds a fraction of the
      remaining headroom → score approaches 1 asymptotically.
    - A regime transition (regime_label change) resets the score to 0.0.
    - signal_multiplier = 1 - fatigue_score (full multiplier when fresh,
      zero when fully fatigued).

Typical usage:
    overlay = FatigueAccumulationOverlay(decay_rate=0.15)
    overlay.update(today, regime_label="BULL")
    # ... many BULL days later ...
    multiplier = overlay.get_signal_multiplier()  # approaches 0.0
    overlay.update(today, regime_label="BEAR")   # reset → multiplier=1.0

Feature flag:
    ResearchFlagsConfig.use_fatigue_accumulation = False (default)

Linear: SLE-75
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from rebuild_phase_3z.fppe.trading_system.risk_overlays.base import BaseRiskOverlay


class FatigueAccumulationOverlay(BaseRiskOverlay):
    """Regime fatigue accumulator — throttles signals in extended trends.

    Fatigue accumulates monotonically during each continuous regime period
    and resets instantly on any regime transition.

    Args:
        decay_rate:        Per-period increment as a fraction of remaining
                           headroom (0 < decay_rate < 1).  Higher values
                           saturate faster.  Default 0.15 → ~50% fatigued
                           after ~4 periods, ~90% after ~14 periods.
        min_multiplier:    Floor on the signal multiplier even at full fatigue.
                           Default 0.0 = full block allowed.
        reset_on_neutral:  If True, a regime_label of None or "NEUTRAL" also
                           triggers a reset.  Default True.
    """

    def __init__(
        self,
        decay_rate: float = 0.15,
        min_multiplier: float = 0.0,
        reset_on_neutral: bool = True,
    ) -> None:
        if not (0 < decay_rate < 1):
            raise ValueError(f"decay_rate must be in (0, 1), got {decay_rate}")
        if not (0.0 <= min_multiplier <= 1.0):
            raise ValueError(f"min_multiplier must be in [0, 1], got {min_multiplier}")

        self.decay_rate = decay_rate
        self.min_multiplier = min_multiplier
        self.reset_on_neutral = reset_on_neutral

        self._fatigue_score: float = 0.0
        self._current_regime: Optional[str] = None
        self._regime_duration: int = 0
        self._last_date: Optional[date] = None

    @property
    def fatigue_score(self) -> float:
        """Current fatigue level in [0, 1].  0 = fresh, 1 = fully fatigued."""
        return self._fatigue_score

    @property
    def regime_duration(self) -> int:
        """Number of consecutive periods in the current regime."""
        return self._regime_duration

    @property
    def current_regime(self) -> Optional[str]:
        """Active regime label, or None if not yet set."""
        return self._current_regime

    def update(self, current_date: date, **market_data: Any) -> None:
        """Ingest the latest regime observation.

        Recognised keyword arguments:
            regime_label: str | None — current regime identifier
                          (e.g. "BULL", "BEAR", "NEUTRAL", None).

        Args:
            current_date: Trading date.
            **market_data: Market data kwargs; unrecognised keys are ignored.
        """
        regime = market_data.get("regime_label", None)
        self._last_date = current_date

        # Determine if this is a regime transition
        is_neutral = (regime is None or str(regime).upper() in {"NEUTRAL", "NONE", ""})
        should_reset = (
            (regime != self._current_regime)
            or (self.reset_on_neutral and is_neutral)
        )

        if should_reset:
            self._fatigue_score = 0.0
            # Count the first period of the new regime as period 1, not 0.
            # regime_duration represents "consecutive periods in current regime".
            self._regime_duration = 1
            # When reset_on_neutral=False, store the actual neutral label so
            # NEUTRAL→NEUTRAL sequences can accumulate (not always treated as
            # a new regime change).  When reset_on_neutral=True, store None so
            # any subsequent NEUTRAL call still resets.
            if is_neutral and not self.reset_on_neutral:
                self._current_regime = regime   # e.g. "NEUTRAL"
            else:
                self._current_regime = regime if not is_neutral else None
        else:
            # Same regime — accumulate fatigue (exponential approach to 1)
            remaining = 1.0 - self._fatigue_score
            self._fatigue_score += self.decay_rate * remaining
            self._fatigue_score = min(1.0, self._fatigue_score)
            self._regime_duration += 1

    def get_signal_multiplier(self) -> float:
        """Return 1 - fatigue_score, floored at min_multiplier.

        Returns:
            float in [min_multiplier, 1.0]:
                1.0 — no fatigue (fresh regime or just reset).
                ~0.0 — severely fatigued (extended regime, near saturation).
        """
        raw = 1.0 - self._fatigue_score
        return max(self.min_multiplier, raw)

    def reset(self) -> None:
        """Reset all accumulated state to the initial (fresh) condition."""
        self._fatigue_score = 0.0
        self._current_regime = None
        self._regime_duration = 0
        self._last_date = None
