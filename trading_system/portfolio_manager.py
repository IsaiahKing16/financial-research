"""
trading_system/portfolio_manager.py — Phase 4 Portfolio Manager (stateless).

Two free functions:
    rank_signals(buy_signals) -> list[RankedSignal]
    allocate_day(ranked, snapshot, limits, min_position_pct) -> list[AllocationResult]

Design:
    - Stateless: all inputs passed explicitly; no globals, no instance state.
    - Deterministic: ranking is confidence DESC + ticker ASC; allocation is
      rank-ordered with a running snapshot updated after each approval.
    - Pre-sizing: these functions run BEFORE the risk engine. They do not
      compute dollar amounts or share counts — that's the risk engine's job.
    - The `cooldown` PMRejectionReason is reserved for v2; v1 does not
      implement cooldowns (the walk-forward replays a trade file that
      already encodes entry/exit timing).

Check priority inside allocate_day:
    1. already_held        — snapshot.contains_ticker(ticker)
    2. sector_count_limit  — running_sector_counts[sector] >= max_positions_per_sector
    3. sector_pct_limit    — running_sector_exposure + min_position_pct > max_sector_pct
    4. insufficient_capital — running_cash_pct < min_position_pct

Plan: docs/superpowers/plans/2026-04-08-phase4-portfolio-manager-plan.md
"""
from __future__ import annotations

from pattern_engine.contracts.signals import SignalDirection
from trading_system.config import PositionLimitsConfig
from trading_system.portfolio_state import (
    AllocationResult,
    PMRejection,
    PortfolioSnapshot,
    RankedSignal,
)
from trading_system.signal_adapter import UnifiedSignal

# ── rank_signals ──────────────────────────────────────────────────────────────

def rank_signals(buy_signals: list[UnifiedSignal]) -> list[RankedSignal]:
    """Rank BUY candidates by confidence descending, ticker ascending.

    Args:
        buy_signals: list of UnifiedSignal with signal == SignalDirection.BUY.

    Returns:
        list of RankedSignal with 1-based rank assigned in sorted order.

    Raises:
        RuntimeError: if any input signal has signal != BUY. The caller
            must pre-filter (the walk-forward only feeds BUY candidates).
    """
    if not buy_signals:
        return []

    for s in buy_signals:
        if s.signal != SignalDirection.BUY:
            raise RuntimeError(
                f"rank_signals: expected BUY, got {s.signal} for {s.ticker}"
            )

    sorted_sigs = sorted(
        buy_signals,
        key=lambda s: (-s.confidence, s.ticker),
    )

    return [
        RankedSignal(
            ticker=s.ticker,
            sector=s.sector,
            signal_date=s.date,
            confidence=s.confidence,
            rank=i + 1,
        )
        for i, s in enumerate(sorted_sigs)
    ]


# ── allocate_day ──────────────────────────────────────────────────────────────

def _reject(rs: RankedSignal, reason: str, detail: str) -> AllocationResult:
    return AllocationResult(
        ticker=rs.ticker,
        sector=rs.sector,
        signal_date=rs.signal_date,
        confidence=rs.confidence,
        rank=rs.rank,
        approved=False,
        rejection=PMRejection(
            ticker=rs.ticker,
            sector=rs.sector,
            signal_date=rs.signal_date,
            confidence=rs.confidence,
            rank=rs.rank,
            reason=reason,  # type: ignore[arg-type]
            detail=detail,
        ),
    )


def _approve(rs: RankedSignal) -> AllocationResult:
    return AllocationResult(
        ticker=rs.ticker,
        sector=rs.sector,
        signal_date=rs.signal_date,
        confidence=rs.confidence,
        rank=rs.rank,
        approved=True,
        rejection=None,
    )


def allocate_day(
    ranked_signals: list[RankedSignal],
    snapshot: PortfolioSnapshot,
    limits: PositionLimitsConfig,
    min_position_pct: float = 0.02,
) -> list[AllocationResult]:
    """Apply portfolio-level constraints to each ranked signal in rank order.

    Uses a RUNNING snapshot: approvals earlier in the day are added to
    running sector counts / sector exposure / cash commitments before the
    next signal is checked. The input `snapshot` itself is not mutated.

    Args:
        ranked_signals: output of rank_signals(), already in rank order.
        snapshot: frozen PortfolioSnapshot at the start of the day.
        limits: PositionLimitsConfig defining sector/position caps.
        min_position_pct: minimum position size used for pct_limit and
            insufficient_capital checks. Default 0.02 matches the config
            default; override if the caller is using a different floor.

    Returns:
        One AllocationResult per input signal (both approved and rejected).

    Raises:
        RuntimeError: if min_position_pct < 0.
    """
    if min_position_pct < 0.0:
        raise RuntimeError(
            f"min_position_pct must be >= 0, got {min_position_pct}"
        )

    if not ranked_signals:
        return []

    # Build running state from the snapshot. Tuples in snapshot are immutable;
    # we maintain parallel mutable mirrors for the loop.
    running_tickers: set[str] = {p.ticker for p in snapshot.open_positions}
    running_sector_counts: dict[str, int] = dict(snapshot.sector_counts)
    running_sector_exposure: dict[str, float] = dict(snapshot.sector_exposure_pct)
    running_cash_pct: float = snapshot.cash_pct

    results: list[AllocationResult] = []

    for rs in ranked_signals:
        sector = rs.sector

        # 1. already_held
        if rs.ticker in running_tickers:
            results.append(_reject(
                rs, "already_held",
                f"ticker {rs.ticker} already in open positions",
            ))
            continue

        # 2. sector_count_limit
        sector_count = running_sector_counts.get(sector, 0)
        if sector_count >= limits.max_positions_per_sector:
            results.append(_reject(
                rs, "sector_count_limit",
                f"sector {sector} at {sector_count}/"
                f"{limits.max_positions_per_sector} positions",
            ))
            continue

        # 3. sector_pct_limit
        sector_exposure = running_sector_exposure.get(sector, 0.0)
        if sector_exposure + min_position_pct > limits.max_sector_pct + 1e-9:
            results.append(_reject(
                rs, "sector_pct_limit",
                f"sector {sector} exposure {sector_exposure:.3f} + "
                f"{min_position_pct:.3f} > cap {limits.max_sector_pct:.3f}",
            ))
            continue

        # 4. insufficient_capital
        if running_cash_pct + 1e-9 < min_position_pct:
            results.append(_reject(
                rs, "insufficient_capital",
                f"running cash {running_cash_pct:.3f} < "
                f"min_position {min_position_pct:.3f}",
            ))
            continue

        # Approved — update running state.
        running_tickers.add(rs.ticker)
        running_sector_counts[sector] = sector_count + 1
        running_sector_exposure[sector] = sector_exposure + min_position_pct
        running_cash_pct -= min_position_pct
        results.append(_approve(rs))

    return results
