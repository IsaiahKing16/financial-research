"""
risk_engine.py — Phase 2 position sizing and stop-loss helpers.
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .config import PositionLimitsConfig, RiskConfig
from .risk_state import PositionDecision, RiskState


def compute_drawdown_scalar(
    current_equity: float,
    peak_equity: float,
    brake_threshold: float,
    halt_threshold: float,
) -> Tuple[float, str]:
    """Compute the drawdown sizing scalar and current mode."""
    if halt_threshold <= brake_threshold:
        raise ValueError(
            f"halt_threshold ({halt_threshold}) must be > brake_threshold ({brake_threshold})"
        )
    if peak_equity <= 0:
        return 1.0, "normal"

    drawdown = max(0.0, 1.0 - (current_equity / peak_equity))
    if drawdown >= halt_threshold:
        return 0.0, "halt"
    if drawdown >= brake_threshold:
        scalar = (halt_threshold - drawdown) / (halt_threshold - brake_threshold)
        return max(0.0, min(1.0, scalar)), "brake"
    return 1.0, "normal"


def check_stop_loss(current_low: float, stop_price: float) -> bool:
    """Return True when intraday low breaches stop level."""
    if stop_price <= 0:
        return False
    return current_low <= stop_price


def compute_atr_pct(price_history: pd.DataFrame, lookback: int = 20) -> float:
    """Compute ATR as a fraction of the latest close price."""
    required_columns = {"High", "Low", "Close"}
    missing = required_columns.difference(price_history.columns)
    if missing:
        raise ValueError(f"price_history missing required columns: {sorted(missing)}")

    required_rows = lookback + 1
    if len(price_history) < required_rows:
        raise ValueError(
            f"Insufficient history: {len(price_history)} rows < {required_rows}"
        )

    history = price_history.copy()
    for col in ("High", "Low", "Close"):
        history[col] = pd.to_numeric(history[col], errors="coerce")
    history = history.dropna(subset=["High", "Low", "Close"])
    if len(history) < required_rows:
        raise ValueError(
            f"Insufficient history: {len(history)} rows < {required_rows}"
        )

    prev_close = history["Close"].shift(1)
    true_range = pd.concat(
        [
            (history["High"] - history["Low"]),
            (history["High"] - prev_close).abs(),
            (history["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / lookback, adjust=False, min_periods=lookback).mean()

    atr_value = float(atr.iloc[-1])
    current_close = float(history["Close"].iloc[-1])
    if current_close <= 0:
        raise ValueError("Current close price must be > 0")
    atr_pct = atr_value / current_close
    if atr_pct <= 0:
        raise ValueError("Zero ATR — likely data issue")
    return atr_pct


def _rejected_decision(
    ticker: str,
    drawdown_scalar: float,
    raw_weight: float,
    reason: str,
) -> PositionDecision:
    return PositionDecision(
        approved=False,
        ticker=ticker,
        position_pct=0.0,
        shares=0.0,
        dollar_amount=0.0,
        stop_price=0.0,
        stop_distance_pct=0.0,
        atr_pct=0.0,
        drawdown_scalar=drawdown_scalar,
        raw_weight=raw_weight,
        rejection_reason=reason,
    )


def size_position(
    ticker: str,
    entry_price: float,
    current_equity: float,
    price_history: pd.DataFrame,
    risk_state: RiskState,
    config: RiskConfig,
    position_limits: PositionLimitsConfig,
    sector_map: Optional[Dict[str, str]] = None,
    open_positions: Optional[Dict[str, Any]] = None,
    fractional_shares: bool = True,
) -> PositionDecision:
    """Compute position size and stop-loss for a long trade candidate."""
    del sector_map  # Reserved for Phase 3 cross-module checks.

    if entry_price <= 0:
        return _rejected_decision(
            ticker=ticker,
            drawdown_scalar=0.0,
            raw_weight=0.0,
            reason=f"Invalid entry price for {ticker}: {entry_price}",
        )
    if current_equity <= 0:
        return _rejected_decision(
            ticker=ticker,
            drawdown_scalar=0.0,
            raw_weight=0.0,
            reason=f"Non-positive equity: {current_equity}",
        )
    if open_positions is not None and ticker in open_positions:
        return _rejected_decision(
            ticker=ticker,
            drawdown_scalar=risk_state.sizing_scalar,
            raw_weight=0.0,
            reason=f"Already holding {ticker}",
        )

    drawdown_scalar, mode = compute_drawdown_scalar(
        current_equity=current_equity,
        peak_equity=risk_state.peak_equity,
        brake_threshold=config.drawdown_brake_threshold,
        halt_threshold=config.drawdown_halt_threshold,
    )
    if mode == "halt":
        current_dd = max(0.0, 1.0 - (current_equity / risk_state.peak_equity))
        return _rejected_decision(
            ticker=ticker,
            drawdown_scalar=0.0,
            raw_weight=0.0,
            reason=(
                f"Drawdown halt: {current_dd:.1%} ≥ "
                f"{config.drawdown_halt_threshold:.1%}"
            ),
        )

    try:
        atr_pct = compute_atr_pct(
            price_history=price_history,
            lookback=config.volatility_lookback,
        )
    except ValueError as exc:
        return _rejected_decision(
            ticker=ticker,
            drawdown_scalar=drawdown_scalar,
            raw_weight=0.0,
            reason=str(exc),
        )

    stop_distance_pct = config.stop_loss_atr_multiple * atr_pct
    if stop_distance_pct <= 0:
        return _rejected_decision(
            ticker=ticker,
            drawdown_scalar=drawdown_scalar,
            raw_weight=0.0,
            reason=f"Invalid stop distance for {ticker}: {stop_distance_pct}",
        )

    stop_price = entry_price * (1.0 - stop_distance_pct)
    if stop_price <= 0:
        return _rejected_decision(
            ticker=ticker,
            drawdown_scalar=drawdown_scalar,
            raw_weight=0.0,
            reason=f"Computed non-positive stop price for {ticker}: {stop_price}",
        )

    raw_weight = config.max_loss_per_trade_pct / stop_distance_pct
    adjusted_weight = raw_weight * drawdown_scalar

    if adjusted_weight < position_limits.min_position_pct:
        return _rejected_decision(
            ticker=ticker,
            drawdown_scalar=drawdown_scalar,
            raw_weight=raw_weight,
            reason=(
                f"Below min size: {adjusted_weight:.1%} < "
                f"{position_limits.min_position_pct:.1%} after DD adjustment"
            ),
        )

    final_weight = min(max(adjusted_weight, position_limits.min_position_pct), position_limits.max_position_pct)
    dollar_amount = current_equity * final_weight
    shares = dollar_amount / entry_price

    if not fractional_shares:
        shares = float(int(shares))
        if shares <= 0:
            return _rejected_decision(
                ticker=ticker,
                drawdown_scalar=drawdown_scalar,
                raw_weight=raw_weight,
                reason="Insufficient capital for whole shares",
            )
        dollar_amount = shares * entry_price

    return PositionDecision(
        approved=True,
        ticker=ticker,
        position_pct=final_weight,
        shares=shares,
        dollar_amount=dollar_amount,
        stop_price=stop_price,
        stop_distance_pct=stop_distance_pct,
        atr_pct=atr_pct,
        drawdown_scalar=drawdown_scalar,
        raw_weight=raw_weight,
        rejection_reason=None,
    )
