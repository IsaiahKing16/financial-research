"""
portfolio_manager.py — Layer 3: Signal Ranking & Portfolio Allocation

Ranks competing BUY signals and filters them against portfolio-level
constraints (sector diversification, position limits, cooldowns) before
passing approved candidates to Layer 2 (risk_engine) for sizing.

v1: Ranking uses confidence only. Tie-break: alphabetical ticker.
v2+: Composite score with sector momentum, volatility, correlation factors.

This module is STATELESS. All portfolio state is passed in via
PortfolioSnapshot (constructed by backtest_engine each day).

Design doc: FPPE_TRADING_SYSTEM_DESIGN.md v0.3, Section 4.4
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .config import PositionLimitsConfig
from .portfolio_state import AllocationDecision, PortfolioSnapshot, RankedSignal

logger = logging.getLogger(__name__)


def rank_signals(
    buy_signals: list[dict[str, Any]],
    sector_map: dict[str, str],
) -> list[RankedSignal]:
    """Rank BUY signals by confidence (v1). Deterministic tie-breaking.

    Args:
        buy_signals: List of dicts with keys: ticker, confidence, date.
                     Optional key: sector (looked up from sector_map if absent).
        sector_map: Ticker -> sector mapping.

    Returns:
        List of RankedSignal sorted by confidence descending, then ticker
        ascending (deterministic tie-break). Each signal gets a 1-based rank.

    Raises:
        TypeError: If buy_signals is not a list.
        ValueError: If any signal is missing required keys or has invalid values.
    """
    if not isinstance(buy_signals, list):
        raise TypeError(
            f"buy_signals must be a list, got {type(buy_signals).__name__}"
        )

    if not buy_signals:
        return []

    # Validate and normalize
    normalized: list[dict[str, Any]] = []
    for sig in buy_signals:
        if "ticker" not in sig or "confidence" not in sig or "date" not in sig:
            raise ValueError(
                f"Signal missing required keys (ticker, confidence, date): {sig}"
            )
        conf = sig["confidence"]
        if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
            raise ValueError(
                f"Confidence must be float in [0, 1], got {conf} for {sig['ticker']}"
            )
        ticker = sig["ticker"]
        sector = sig.get("sector") or sector_map.get(ticker, "Unknown")
        if sector == "Unknown":
            logger.warning(
                "Ticker '%s' has no sector mapping, using 'Unknown'", ticker
            )

        normalized.append({
            "ticker": ticker,
            "confidence": float(conf),
            "sector": sector,
            "date": pd.Timestamp(sig["date"]),
            "raw_metadata": {
                k: v for k, v in sig.items()
                if k not in ("ticker", "confidence", "date", "sector", "signal")
            },
        })

    # Sort: confidence descending, ticker ascending (deterministic tie-break)
    normalized.sort(key=lambda s: (-s["confidence"], s["ticker"]))

    # Build ranked signals
    return [
        RankedSignal(
            ticker=s["ticker"],
            confidence=s["confidence"],
            sector=s["sector"],
            rank=i + 1,
            rank_score=s["confidence"],  # v1: rank_score == confidence
            date=s["date"],
            raw_metadata=s["raw_metadata"],
        )
        for i, s in enumerate(normalized)
    ]


def check_allocation(
    ranked_signal: RankedSignal,
    snapshot: PortfolioSnapshot,
    position_limits: PositionLimitsConfig,
) -> AllocationDecision:
    """Check whether a ranked signal passes portfolio-level constraints.

    Checks (in order):
        1. Already holding this ticker
        2. Ticker in active cooldown with insufficient confidence margin
        3. Sector position count at maximum

    Sector exposure % and capital availability are NOT checked here
    (they require position sizing from risk_engine / cash from backtest_engine).

    Args:
        ranked_signal: The signal to evaluate.
        snapshot: Current portfolio state (immutable).
        position_limits: Config with max_positions_per_sector.

    Returns:
        AllocationDecision with approved=True/False and rejection details.
    """
    ticker = ranked_signal.ticker
    sector = ranked_signal.sector
    confidence = ranked_signal.confidence

    # Check 1: Already holding
    if ticker in snapshot.open_tickers:
        return AllocationDecision(
            ticker=ticker,
            approved=False,
            rank=ranked_signal.rank,
            confidence=confidence,
            sector=sector,
            rejection_reason="Already holding position",
            rejection_layer="portfolio",
        )

    # Check 2: Cooldown
    if ticker in snapshot.cooldowns:
        cd = snapshot.cooldowns[ticker]
        until_date = cd.get("until_date")
        last_conf = cd.get("last_confidence", 0.0)
        signal_date = ranked_signal.date

        if until_date is not None and signal_date < pd.Timestamp(until_date):
            required_conf = last_conf + snapshot.cooldown_reentry_margin
            if confidence < required_conf:
                return AllocationDecision(
                    ticker=ticker,
                    approved=False,
                    rank=ranked_signal.rank,
                    confidence=confidence,
                    sector=sector,
                    rejection_reason=(
                        f"In cooldown until {pd.Timestamp(until_date).date()}; "
                        f"need confidence >= {required_conf:.2f}"
                    ),
                    rejection_layer="portfolio",
                )
            # Cooldown active but confidence margin met — allow through

    # Check 3: Sector position count
    # "Unknown" tickers lack a sector mapping — they are not a correlated group,
    # so enforcing concentration limits on them would falsely throttle uncorrelated
    # positions. Consistent with the Phase 1/2 path intent in backtest_engine.py.
    if sector != "Unknown":
        sector_count = snapshot.sector_position_counts.get(sector, 0)
        if sector_count >= position_limits.max_positions_per_sector:
            return AllocationDecision(
                ticker=ticker,
                approved=False,
                rank=ranked_signal.rank,
                confidence=confidence,
                sector=sector,
                rejection_reason=(
                    f"Sector {sector} at max "
                    f"{position_limits.max_positions_per_sector} positions"
                ),
                rejection_layer="portfolio",
            )

    # All checks passed
    return AllocationDecision(
        ticker=ticker,
        approved=True,
        rank=ranked_signal.rank,
        confidence=confidence,
        sector=sector,
    )


def allocate_day(
    buy_signals: list[dict[str, Any]],
    snapshot: PortfolioSnapshot,
    position_limits: PositionLimitsConfig,
    sector_map: dict[str, str],
) -> list[AllocationDecision]:
    """Rank and check all BUY signals for one date.

    Convenience function that calls rank_signals() then check_allocation()
    for each signal in rank order.

    Args:
        buy_signals: List of signal dicts for one date.
        snapshot: Current portfolio state.
        position_limits: Sector/position constraints.
        sector_map: Ticker -> sector mapping.

    Returns:
        List of AllocationDecision for ALL signals (approved and rejected),
        ordered by rank (highest confidence first).
    """
    ranked = rank_signals(buy_signals, sector_map)

    if not ranked:
        return []

    decisions: list[AllocationDecision] = []
    # Track sector counts dynamically as we approve signals within this day
    running_sector_counts = dict(snapshot.sector_position_counts)
    running_open_tickers = set(snapshot.open_tickers)

    for signal in ranked:
        # Build a running snapshot that includes signals approved earlier today
        running_snapshot = PortfolioSnapshot(
            open_tickers=frozenset(running_open_tickers),
            sector_position_counts=running_sector_counts,
            cooldowns=snapshot.cooldowns,
            cooldown_reentry_margin=snapshot.cooldown_reentry_margin,
        )
        decision = check_allocation(signal, running_snapshot, position_limits)
        decisions.append(decision)

        # Update running state if approved (so subsequent signals see this one)
        if decision.approved:
            running_open_tickers.add(signal.ticker)
            running_sector_counts[signal.sector] = (
                running_sector_counts.get(signal.sector, 0) + 1
            )

    n_approved = sum(1 for d in decisions if d.approved)
    n_rejected = len(decisions) - n_approved
    logger.debug(
        "allocate_day: %d signals -> %d approved, %d rejected",
        len(decisions), n_approved, n_rejected,
    )

    return decisions
