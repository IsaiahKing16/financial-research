"""
position_sizer.py — Phase 2: Half-Kelly Position Sizing

Computes position size as:
    position_pct = atr_weight × half_kelly
    position_pct = clamp(position_pct, min_pct, max_pct)

where:
    atr_weight  = max_loss_per_trade_pct / stop_distance
    stop_distance = stop_loss_atr_multiple × flat_atr_pct  (Phase 2 constant)
    kelly_fraction = (p·b − q) / b  (standard Kelly criterion)
    half_kelly  = 0.5 × kelly_fraction

Phase 2 contract: flat_atr_pct is a configured constant; Phase 3 swaps in
real per-trade ATR (compute_atr_pct from risk_engine) without changing this
interface.  Only `SizingConfig.flat_atr_pct` needs to change.

Regime note: regime=hold signals (SPY ret_90d < +0.05) already carry signal=HOLD
from the pattern engine; they never reach size_position().  Only BUY signals
are sized.

Walk-forward provenance:
    b_ratio default (1.1811) derived from results/backtest_trades.csv,
    278 trades, 2024 fold.  Recompute per fold for walk-forward validation.

Linear: Phase 2, Task T2.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import icontract


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SizingConfig:
    """Parameters for Half-Kelly position sizing.

    Args:
        max_loss_per_trade_pct: Maximum equity fraction to risk per trade.
            Drives the ATR weight: atr_weight = max_loss / stop_distance.
        stop_loss_atr_multiple: Stop distance in ATR units (locked from CLAUDE.md
            ATR sweep, provenance: results/atr_sweep_results.tsv).
        flat_atr_pct: Phase 2 flat ATR% constant.  Represents a typical stock's
            daily ATR as a fraction of price.  Phase 3 replaces this with
            per-trade ATR computed from OHLCV history.
        kelly_multiplier: Fraction of full Kelly to apply.  0.5 = Half-Kelly.
        min_position_pct: Floor on position size (fraction of equity).
        max_position_pct: Cap on position size (fraction of equity).
    """
    max_loss_per_trade_pct: float = 0.02
    stop_loss_atr_multiple: float = 3.0
    flat_atr_pct: float = 0.02
    kelly_multiplier: float = 0.5
    min_position_pct: float = 0.02
    max_position_pct: float = 0.10

    def __post_init__(self) -> None:
        if not (0 < self.max_loss_per_trade_pct < 1):
            raise RuntimeError(
                f"max_loss_per_trade_pct must be in (0, 1), got {self.max_loss_per_trade_pct}"
            )
        if self.stop_loss_atr_multiple <= 0:
            raise RuntimeError(
                f"stop_loss_atr_multiple must be > 0, got {self.stop_loss_atr_multiple}"
            )
        if not (0 < self.flat_atr_pct < 1):
            raise RuntimeError(
                f"flat_atr_pct must be in (0, 1), got {self.flat_atr_pct}"
            )
        if not (0 < self.kelly_multiplier <= 1):
            raise RuntimeError(
                f"kelly_multiplier must be in (0, 1], got {self.kelly_multiplier}"
            )
        if not (0 < self.min_position_pct < self.max_position_pct <= 1):
            raise RuntimeError(
                f"Need 0 < min_position_pct < max_position_pct <= 1, "
                f"got [{self.min_position_pct}, {self.max_position_pct}]"
            )


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SizingResult:
    """Output of size_position().

    Attrs:
        approved: True if a position size was computed; False if rejected.
        position_pct: Fraction of current equity to allocate.  0.0 if rejected.
        kelly_fraction: Full (pre-multiplier) Kelly fraction.  May be negative.
        scaled_kelly: kelly_fraction × kelly_multiplier (half-Kelly by default).
        atr_weight: max_loss_per_trade_pct / stop_distance.
        rejection_reason: Human-readable reason if approved=False; None otherwise.
    """
    approved: bool
    position_pct: float
    kelly_fraction: float
    scaled_kelly: float
    atr_weight: float
    rejection_reason: Optional[str]


# ─── Core functions ───────────────────────────────────────────────────────────

def compute_kelly_fraction(p: float, b: float) -> float:
    """Standard Kelly criterion fraction.

    Args:
        p: Win probability ∈ (0, 1).  Use calibrated Platt confidence.
        b: Win/loss ratio = avg_win / avg_loss.  Must be > 0.

    Returns:
        Kelly fraction = (p·b − q) / b where q = 1 − p.
        Negative values indicate negative edge (do not trade).

    Raises:
        RuntimeError: if p or b are out of valid range.
    """
    if not (0 < p < 1):
        raise RuntimeError(f"Win probability p must be in (0, 1), got {p}")
    if b <= 0:
        raise RuntimeError(f"Win/loss ratio b must be > 0, got {b}")
    q = 1.0 - p
    return (p * b - q) / b


@icontract.require(
    lambda confidence: math.isfinite(confidence),
    "confidence must be finite (not NaN/inf).",
)
@icontract.require(
    lambda b_ratio: math.isfinite(b_ratio),
    "b_ratio must be finite (not NaN/inf).",
)
@icontract.require(
    lambda atr_pct: atr_pct is None or math.isfinite(atr_pct),
    "atr_pct must be finite when provided (not NaN/inf).",
)
@icontract.ensure(
    lambda result: not result.approved or math.isfinite(result.position_pct),
    "approved result must have finite position_pct.",
)
def size_position(
    confidence: float,
    b_ratio: float,
    config: Optional[SizingConfig] = None,
    atr_pct: Optional[float] = None,
) -> SizingResult:
    """Compute Half-Kelly position size for a BUY signal.

    Sizing formula:
        effective_atr = atr_pct if atr_pct is not None else config.flat_atr_pct
        stop_distance = config.stop_loss_atr_multiple × effective_atr
        atr_weight    = config.max_loss_per_trade_pct / stop_distance
        kelly         = compute_kelly_fraction(confidence, b_ratio)
        scaled_kelly  = config.kelly_multiplier × kelly
        position_pct  = atr_weight × scaled_kelly
        position_pct  = clamp(position_pct, min_position_pct, max_position_pct)

    Phase 3 ATR override:
        When `atr_pct` is provided, it replaces `config.flat_atr_pct` for this
        single call.  Use this with `risk_engine.compute_atr_pct(atr_14, close)`
        to apply real per-ticker ATR sizing.  Phase 2 callers that pass
        `atr_pct=None` (or omit it) get unchanged behavior.

    Rejection conditions:
        - Kelly fraction ≤ 0 → negative or zero edge; no position taken.
        - confidence ≤ 0 or ≥ 1 → invalid probability input.
        - b_ratio ≤ 0 → invalid win/loss ratio.
        - atr_pct provided but not in (0, 1) → invalid ATR.

    Args:
        confidence: Calibrated Platt win probability from FPPE signal.
        b_ratio: Historical avg_win / avg_loss ratio.
        config: SizingConfig instance; defaults to SizingConfig() if None.
        atr_pct: Optional Phase 3 override for per-ticker ATR (fraction of price).
            When None, the config's `flat_atr_pct` constant is used.

    Returns:
        SizingResult with approved=True and position_pct ∈ [min, max] on success,
        or approved=False with rejection_reason on failure.
    """
    if config is None:
        config = SizingConfig()

    # Input validation
    if not (0 < confidence < 1):
        return SizingResult(
            approved=False,
            position_pct=0.0,
            kelly_fraction=float("nan"),
            scaled_kelly=float("nan"),
            atr_weight=float("nan"),
            rejection_reason=f"confidence must be in (0, 1), got {confidence}",
        )
    if b_ratio <= 0:
        return SizingResult(
            approved=False,
            position_pct=0.0,
            kelly_fraction=float("nan"),
            scaled_kelly=float("nan"),
            atr_weight=float("nan"),
            rejection_reason=f"b_ratio must be > 0, got {b_ratio}",
        )
    if atr_pct is not None and not (0 < atr_pct < 1):
        return SizingResult(
            approved=False,
            position_pct=0.0,
            kelly_fraction=float("nan"),
            scaled_kelly=float("nan"),
            atr_weight=float("nan"),
            rejection_reason=f"atr_pct must be in (0, 1) when provided, got {atr_pct}",
        )

    # Step 1: ATR weight (Phase 3: per-trade override; Phase 2: flat constant)
    effective_atr = atr_pct if atr_pct is not None else config.flat_atr_pct
    stop_distance = config.stop_loss_atr_multiple * effective_atr
    atr_weight = config.max_loss_per_trade_pct / stop_distance

    # Step 2: Kelly
    kelly = compute_kelly_fraction(confidence, b_ratio)

    # Reject if edge is non-positive
    if kelly <= 0:
        return SizingResult(
            approved=False,
            position_pct=0.0,
            kelly_fraction=kelly,
            scaled_kelly=0.0,
            atr_weight=atr_weight,
            rejection_reason=f"Kelly fraction non-positive ({kelly:.6f}): edge does not support a position",
        )

    # Step 3: Scale Kelly
    scaled_kelly = config.kelly_multiplier * kelly

    # Step 4: Combined size + clamp
    raw_pct = atr_weight * scaled_kelly
    position_pct = max(config.min_position_pct, min(config.max_position_pct, raw_pct))

    return SizingResult(
        approved=True,
        position_pct=position_pct,
        kelly_fraction=kelly,
        scaled_kelly=scaled_kelly,
        atr_weight=atr_weight,
        rejection_reason=None,
    )
