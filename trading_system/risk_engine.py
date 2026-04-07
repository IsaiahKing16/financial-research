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
        drawdown ≤ warn          → 1.0  (full sizing, no brake)
        warn  < drawdown <  halt → linear interpolation 1.0 → 0.0
        drawdown ≥ halt          → 0.0  (halt; no new positions)

    Boundary contract (symmetric-inclusive):
        At exactly `warn`, the brake has NOT engaged yet → returns 1.0.
        At exactly `halt`, the brake is fully engaged     → returns 0.0.
        Both boundaries are inclusive on the "safe" side — i.e. you get
        full sizing at warn and full halt at halt. Do not change this
        without updating TestDrawdownBoundaryContract in test_risk_engine.py.

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


# ─── Result dataclass ─────────────────────────────────────────────────────────

# Tolerance threshold for "blocked" detection on `final_position_pct`.
# Phase 3 contract: overlays should return exactly 0.0 to block, but redesigned
# overlays (e.g. SLE-75 fatigue) may saturate to ~1e-13. Anything below this
# bound is treated as an effective block so audit/reporting stays correct.
_BLOCKED_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class AdjustedSizing:
    """Output of apply_risk_adjustments.

    Attrs:
        original: The pre-adjustment SizingResult from position_sizer.
        final_position_pct: Position size after DD brake and overlays.
        dd_scalar: Drawdown brake scalar in [0, 1].  NaN if sizing was
            rejected (not computed in that branch).
        overlay_multiplier: Product of all overlay multipliers in [0, 1].
            NaN if sizing was rejected.
        blocked: True if the caller cannot open a new position. Detected via
            `final_position_pct < _BLOCKED_TOLERANCE` (not exact `== 0.0`)
            so overlays that saturate to ~1e-13 are treated as blocked.
        block_reason: Structured reason string when blocked, else None.
            Formats:
              - "sizing_rejected:<original_reason>"
              - "dd_brake:<drawdown>"
              - "overlay:<OverlayClassName>"
              - "position_below_threshold"  (fallback: collapsed w/o attribution)
    """
    original: SizingResult
    final_position_pct: float
    dd_scalar: float
    overlay_multiplier: float
    blocked: bool
    block_reason: Optional[str]


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def apply_risk_adjustments(
    sizing: SizingResult,
    drawdown: float,
    overlays: Optional[List[BaseRiskOverlay]] = None,
    dd_warn: float = 0.15,
    dd_halt: float = 0.20,
) -> AdjustedSizing:
    """Compose Phase 2 sizing with DD brake and risk overlays.

    Pipeline:
      1. If sizing was rejected → propagate as blocked.
      2. Compute DD brake scalar from current drawdown.
      3. Compute overlay multiplier as product of all overlay
         signal multipliers (1.0 if no overlays).
      4. final = sizing.position_pct × dd_scalar × overlay_multiplier.
      5. If final == 0 → blocked, with structured reason indicating
         which component caused the block.

    Block reason priority (first match wins):
      sizing_rejected → dd_brake → overlay:<name>

    Phase 3 contract: overlays multiply POSITION SIZE in this orchestrator,
    not signal confidence as the original BaseRiskOverlay docstring stated.
    Half-Kelly already incorporates confidence; re-throttling it would
    double-count.

    Args:
        sizing: SizingResult from position_sizer.size_position().
        drawdown: Current portfolio drawdown from peak [0, 1].
        overlays: Optional list of BaseRiskOverlay instances.  Each must
            have already been updated for the current trading day.
        dd_warn: Drawdown brake warn threshold (default 0.15).
        dd_halt: Drawdown brake halt threshold (default 0.20).

    Returns:
        AdjustedSizing with composed final_position_pct and diagnostic fields.

    Raises:
        RuntimeError: if drawdown > 1.0 (catches upstream data bugs).
    """
    if drawdown > 1.0:
        raise RuntimeError(f"drawdown must be <= 1.0, got {drawdown}")

    # Case 1: Phase 2 rejected the sizing — propagate
    if not sizing.approved:
        return AdjustedSizing(
            original=sizing,
            final_position_pct=0.0,
            dd_scalar=float("nan"),
            overlay_multiplier=float("nan"),
            blocked=True,
            block_reason=f"sizing_rejected:{sizing.rejection_reason}",
        )

    # Case 1b: approved=True but position_pct is malformed (<= 0).
    # This is a category error from upstream sizing; log it with a
    # structured reason so the caller's blocked_log captures the
    # audit trail instead of silently producing final_position_pct=0.
    if sizing.position_pct <= 0.0:
        return AdjustedSizing(
            original=sizing,
            final_position_pct=0.0,
            dd_scalar=float("nan"),
            overlay_multiplier=float("nan"),
            blocked=True,
            block_reason="invalid_original_pct",
        )

    # Case 2: compute scalars
    dd_scalar = drawdown_brake_scalar(drawdown, warn=dd_warn, halt=dd_halt)

    overlay_multiplier = 1.0
    blocking_overlay: Optional[BaseRiskOverlay] = None
    if overlays:
        for overlay in overlays:
            m = overlay.get_signal_multiplier()
            overlay_multiplier *= m
            if m == 0.0 and blocking_overlay is None:
                blocking_overlay = overlay

    final_position_pct = sizing.position_pct * dd_scalar * overlay_multiplier

    # Determine block status + structured reason (priority order).
    # Tolerance-based detection catches redesigned overlays that saturate to
    # ~1e-13 instead of returning exactly 0.0 (see _BLOCKED_TOLERANCE note).
    if final_position_pct < _BLOCKED_TOLERANCE:
        if dd_scalar == 0.0:
            reason = f"dd_brake:{drawdown}"
        elif blocking_overlay is not None:
            reason = f"overlay:{type(blocking_overlay).__name__}"
        else:
            # Fallback: neither DD halt nor an overlay returned exactly 0.0,
            # yet the composed position collapsed below the tolerance
            # (e.g. an overlay returned a microscopic positive value).
            reason = "position_below_threshold"
        return AdjustedSizing(
            original=sizing,
            final_position_pct=final_position_pct,
            dd_scalar=dd_scalar,
            overlay_multiplier=overlay_multiplier,
            blocked=True,
            block_reason=reason,
        )

    return AdjustedSizing(
        original=sizing,
        final_position_pct=final_position_pct,
        dd_scalar=dd_scalar,
        overlay_multiplier=overlay_multiplier,
        blocked=False,
        block_reason=None,
    )
