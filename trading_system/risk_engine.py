"""
risk_engine.py — Phase 3: Risk Engine Integration (thin orchestrator).

Stateless functions composing Phase 2 sizing with drawdown brake and risk
overlays.  No state held — the caller (walk-forward script or live trader)
wires the pieces together each trading day.

Components:
  - compute_atr_pct        : atr_14 / close → ATR as fraction of price
  - drawdown_brake_scalar  : linear scalar 1.0 → 0.0 over [warn, halt] DD
  - AdjustedSizing         : result dataclass
  - apply_risk_adjustments : composes sizing × dd_scalar × overlay_multiplier

Phase 3 contract evolution:
  Risk overlays (BaseRiskOverlay subclasses) multiply POSITION SIZE in this
  orchestrator, not signal confidence as the original ABC docstring stated.
  Reason: Half-Kelly already incorporates confidence into position size, so
  re-throttling confidence would double-count the same input.

Spec: docs/superpowers/specs/2026-04-06-phase3-risk-engine-integration-design.md
Linear: Phase 3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from trading_system.position_sizer import SizingResult
from trading_system.risk_overlays.base import BaseRiskOverlay


def compute_atr_pct(atr_14: float, close: float) -> float:
    """ATR as a fraction of price.

    Computes the standard ATR/price ratio used for stop distance and
    volatility-aware position sizing.  Both inputs come from the 52T
    volnorm dataset (`atr_14` and `Close` columns).

    Args:
        atr_14: 14-period Average True Range in dollar terms.  Must be > 0.
        close:  Closing price.  Must be > 0.

    Returns:
        atr_14 / close — a positive float, typically in [0.005, 0.05] for
        liquid US equities.

    Raises:
        RuntimeError: if `atr_14 <= 0` (no volatility data) or `close <= 0`.

    Caller contract:
        Wrap calls in try/except RuntimeError and skip the trade with a
        logged reason.  Loud failure at source is preferred over silent
        rejection cascading through downstream layers.
    """
    if close <= 0:
        raise RuntimeError(f"close must be > 0, got {close}")
    if atr_14 <= 0:
        raise RuntimeError(
            f"atr_14 must be > 0, got {atr_14}; ticker has no volatility data"
        )
    return atr_14 / close
