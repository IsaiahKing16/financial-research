"""
liquidity_congestion.py — MFD-style liquidity congestion gate (SLE-74).

Throttles signal generation when market microstructure indicates congestion
risk.  Inspired by Market Fragmentation Dynamics (MFD) research that tracks
order-book imbalance and ATR-relative spread widening as proxies for periods
of adverse execution conditions.

Congestion detection (proxy-based, works with EOD data only):
    - Tracks a rolling window of ATR (Average True Range) normalised to price.
    - When ATR/price exceeds a threshold (market volatility spike → wide
      spreads), the gate activates and reduces the signal multiplier.
    - A cooldown window prevents rapid oscillation between active/inactive.

The gate is intentionally conservative:
    - Default threshold=0.03 (~3% of price ATR) triggers on high-volatility days.
    - Default cooldown_periods=3 prevents whipsaw.
    - Full block (multiplier=0.0) only when congestion_level >= block_threshold.
    - Partial throttle (multiplier in (0, 1)) during marginal congestion.

Feature flag:
    ResearchFlagsConfig.use_liquidity_congestion_gate = False (default)

Linear: SLE-74
"""

from __future__ import annotations

from collections import deque
from datetime import date
from typing import Any, Deque, Optional

from trading_system.risk_overlays.base import BaseRiskOverlay


class LiquidityCongestionGate(BaseRiskOverlay):
    """Market liquidity congestion gate based on ATR/price ratio.

    Monitors the ratio of ATR to closing price over a rolling window.
    When the rolling mean ratio exceeds `congestion_threshold`, the gate
    throttles signal confidence by reducing the multiplier toward 0.

    Args:
        window:                Rolling window for ATR/price ratio (bars).
        congestion_threshold:  ATR/price ratio above which gate activates.
        block_threshold:       ATR/price ratio above which gate fully blocks
                               (multiplier = 0.0).  Must be > congestion_threshold.
        cooldown_periods:      Minimum bars the gate stays active after triggering,
                               preventing rapid oscillation.
    """

    def __init__(
        self,
        window: int = 10,
        congestion_threshold: float = 0.025,
        block_threshold: float = 0.05,
        cooldown_periods: int = 3,
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if congestion_threshold <= 0:
            raise ValueError("congestion_threshold must be > 0")
        if block_threshold <= congestion_threshold:
            raise ValueError("block_threshold must be > congestion_threshold")
        if cooldown_periods < 0:
            raise ValueError("cooldown_periods must be >= 0")

        self.window = window
        self.congestion_threshold = congestion_threshold
        self.block_threshold = block_threshold
        self.cooldown_periods = cooldown_periods

        self._atr_price_ratios: Deque[float] = deque(maxlen=window)
        self._cooldown_remaining: int = 0
        self._last_date: Optional[date] = None
        self._congestion_level: float = 0.0   # rolling mean of ATR/price

    @property
    def congestion_level(self) -> float:
        """Current rolling mean ATR/price ratio."""
        return self._congestion_level

    @property
    def is_congested(self) -> bool:
        """True if the gate is currently active (congestion detected)."""
        return (
            self._congestion_level >= self.congestion_threshold
            or self._cooldown_remaining > 0
        )

    def update(self, current_date: date, **market_data: Any) -> None:
        """Ingest the latest market snapshot.

        Recognised keyword arguments:
            atr:   float — Average True Range (same unit as price).
            close: float — Closing price.
            atr_price_ratio: float — pre-computed ATR/price (overrides atr+close).

        Unrecognised keys are silently ignored.

        Args:
            current_date: Trading date.
            **market_data: Market data kwargs (see above).
        """
        # Compute ATR/price ratio
        if "atr_price_ratio" in market_data:
            ratio = float(market_data["atr_price_ratio"])
        elif "atr" in market_data and "close" in market_data:
            close = float(market_data["close"])
            atr = float(market_data["atr"])
            if close <= 0:
                ratio = 0.0
            else:
                ratio = atr / close
        else:
            # No data provided — maintain current state without adding a sample
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
            return

        self._atr_price_ratios.append(ratio)
        self._last_date = current_date

        # Update rolling congestion level
        if self._atr_price_ratios:
            self._congestion_level = sum(self._atr_price_ratios) / len(self._atr_price_ratios)
        else:
            self._congestion_level = 0.0

        # Check for new congestion trigger
        if self._congestion_level >= self.congestion_threshold:
            # Extend cooldown
            self._cooldown_remaining = max(self._cooldown_remaining, self.cooldown_periods)
        elif self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    def _compute_multiplier(self) -> float:
        """Return throttle factor based on current congestion level.

        Template-method hook (see BaseRiskOverlay.get_signal_multiplier).

        Returns:
            1.0  — no congestion; signals pass through unchanged.
            (0, 1) — partial congestion; confidence linearly interpolated.
            0.0  — severe congestion (level >= block_threshold); full block.
        """
        level = self._congestion_level

        if level < self.congestion_threshold and self._cooldown_remaining == 0:
            return 1.0

        if level >= self.block_threshold:
            return 0.0

        # Linear interpolation in [congestion_threshold, block_threshold]
        span = self.block_threshold - self.congestion_threshold
        progress = (level - self.congestion_threshold) / span
        # progress=0 → multiplier=1; progress=1 → multiplier=0
        # During cooldown with level below threshold: partial throttle at 0.5
        if level < self.congestion_threshold:
            # In cooldown but level recovered — partial throttle that decays
            # proportionally as remaining cooldown shrinks toward zero.
            # progress ∈ (0, 0.5] → multiplier ∈ [0.5, 1.0).
            progress = 0.5 * (self._cooldown_remaining / max(self.cooldown_periods, 1))
        return max(0.0, 1.0 - progress)

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._atr_price_ratios.clear()
        self._cooldown_remaining = 0
        self._last_date = None
        self._congestion_level = 0.0
