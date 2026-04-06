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


def drawdown_brake_scalar(
    drawdown: float,
    warn: float = 0.15,
    halt: float = 0.20,
) -> float:
    """Linear drawdown brake scalar.

    Returns a scalar in [0.0, 1.0] used to throttle position sizing as
    portfolio drawdown approaches the halt threshold.

    Schedule:
        drawdown < warn          → 1.0  (full sizing, no brake)
        warn ≤ drawdown < halt   → linear interpolation 1.0 → 0.0
        drawdown ≥ halt          → 0.0  (halt; no new positions)

    Defaults match the roadmap spec: warn=15%, halt=20%.

    Args:
        drawdown: Current drawdown from peak equity, in [0, 1].  Negative
            values (impossible in practice) are treated as zero.
        warn: Drawdown threshold where the brake starts engaging.
        halt: Drawdown threshold where the brake reaches zero.

    Returns:
        Scalar in [0.0, 1.0] to multiply against `position_pct`.

    Raises:
        RuntimeError: if `warn >= halt`.
    """
    if warn >= halt:
        raise RuntimeError(
            f"warn must be < halt, got warn={warn}, halt={halt}"
        )
    if drawdown <= warn:
        return 1.0
    if drawdown >= halt:
        return 0.0
    # Linear interpolation: at warn → 1.0, at halt → 0.0
    span = halt - warn
    return max(0.0, 1.0 - (drawdown - warn) / span)
