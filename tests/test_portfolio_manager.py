"""Tests for trading_system.portfolio_manager — Phase 4 PM core."""
from __future__ import annotations

from datetime import date

import pytest

from pattern_engine.contracts.signals import SignalDirection, SignalSource
from trading_system.config import PositionLimitsConfig
from trading_system.portfolio_manager import allocate_day, rank_signals
from trading_system.portfolio_state import (
    OpenPosition,
    PortfolioSnapshot,
)
from trading_system.signal_adapter import UnifiedSignal


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mk_signal(ticker: str, sector: str, confidence: float,
               d: date = date(2024, 1, 2)) -> UnifiedSignal:
    return UnifiedSignal(
        date=d,
        ticker=ticker,
        signal=SignalDirection.BUY,
        confidence=confidence,
        signal_source=SignalSource.KNN,
        sector=sector,
    )


def _empty_snapshot(equity: float = 10_000.0,
                    cash: float = 10_000.0) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of_date=date(2024, 1, 2),
        equity=equity, cash=cash, open_positions=(),
    )


def _mk_pos(ticker: str, sector: str, pct: float = 0.05) -> OpenPosition:
    return OpenPosition(
        ticker=ticker, sector=sector,
        entry_date=date(2024, 1, 2),
        position_pct=pct, entry_price=100.0,
    )


def _snapshot_with(*positions: OpenPosition, equity: float = 10_000.0,
                   cash: float | None = None) -> PortfolioSnapshot:
    used = sum(p.position_pct for p in positions) * equity
    if cash is None:
        cash = equity - used
    return PortfolioSnapshot(
        as_of_date=date(2024, 1, 2),
        equity=equity, cash=cash, open_positions=tuple(positions),
    )


LIMITS = PositionLimitsConfig()  # defaults: max_sector_pct=0.30, max_positions_per_sector=3


# ── rank_signals ──────────────────────────────────────────────────────────────

class TestRankSignals:
    def test_empty_input_returns_empty_list(self):
        assert rank_signals([]) == []

    def test_single_signal_gets_rank_1(self):
        sigs = [_mk_signal("AAPL", "Technology", 0.72)]
        ranked = rank_signals(sigs)
        assert len(ranked) == 1
        assert ranked[0].ticker == "AAPL"
        assert ranked[0].rank == 1

    def test_sorts_by_confidence_descending(self):
        sigs = [
            _mk_signal("AAPL", "Technology", 0.60),
            _mk_signal("MSFT", "Technology", 0.80),
            _mk_signal("JPM", "Financials", 0.70),
        ]
        ranked = rank_signals(sigs)
        assert [r.ticker for r in ranked] == ["MSFT", "JPM", "AAPL"]
        assert [r.rank for r in ranked] == [1, 2, 3]

    def test_ties_broken_by_ticker_ascending(self):
        sigs = [
            _mk_signal("MSFT", "Technology", 0.70),
            _mk_signal("AAPL", "Technology", 0.70),
            _mk_signal("GOOG", "Technology", 0.70),
        ]
        ranked = rank_signals(sigs)
        assert [r.ticker for r in ranked] == ["AAPL", "GOOG", "MSFT"]

    def test_preserves_sector_and_confidence(self):
        sigs = [_mk_signal("AAPL", "Technology", 0.72)]
        ranked = rank_signals(sigs)
        assert ranked[0].sector == "Technology"
        assert ranked[0].confidence == 0.72

    def test_rejects_non_buy_signal(self):
        sell = UnifiedSignal(
            date=date(2024, 1, 2), ticker="AAPL",
            signal=SignalDirection.SELL, confidence=0.72,
            signal_source=SignalSource.KNN, sector="Technology",
        )
        with pytest.raises(RuntimeError, match="BUY"):
            rank_signals([sell])

    def test_rejects_hold_signal(self):
        hold = UnifiedSignal(
            date=date(2024, 1, 2), ticker="AAPL",
            signal=SignalDirection.HOLD, confidence=0.72,
            signal_source=SignalSource.KNN, sector="Technology",
        )
        with pytest.raises(RuntimeError, match="BUY"):
            rank_signals([hold])

    def test_ranking_is_stable_across_calls(self):
        sigs = [
            _mk_signal("AAPL", "Technology", 0.70),
            _mk_signal("MSFT", "Technology", 0.70),
        ]
        r1 = rank_signals(sigs)
        r2 = rank_signals(sigs)
        assert [r.ticker for r in r1] == [r.ticker for r in r2]


# ── allocate_day: basic paths ─────────────────────────────────────────────────

class TestAllocateDayBasic:
    def test_empty_ranked_returns_empty(self):
        result = allocate_day([], _empty_snapshot(), LIMITS)
        assert result == []

    def test_single_signal_empty_portfolio_approved(self):
        ranked = rank_signals([_mk_signal("AAPL", "Technology", 0.72)])
        result = allocate_day(ranked, _empty_snapshot(), LIMITS)
        assert len(result) == 1
        assert result[0].approved is True
        assert result[0].rejection is None

    def test_returns_one_result_per_input(self):
        ranked = rank_signals([
            _mk_signal("AAPL", "Technology", 0.80),
            _mk_signal("MSFT", "Technology", 0.70),
            _mk_signal("JPM", "Financials", 0.65),
        ])
        result = allocate_day(ranked, _empty_snapshot(), LIMITS)
        assert len(result) == 3

    def test_rank_preserved_in_result(self):
        ranked = rank_signals([
            _mk_signal("AAPL", "Technology", 0.80),
            _mk_signal("MSFT", "Technology", 0.70),
        ])
        result = allocate_day(ranked, _empty_snapshot(), LIMITS)
        assert result[0].rank == 1
        assert result[1].rank == 2


# ── allocate_day: rejection reasons ───────────────────────────────────────────

class TestAllocateDayRejections:
    def test_already_held_rejected(self):
        snap = _snapshot_with(_mk_pos("AAPL", "Technology"))
        ranked = rank_signals([_mk_signal("AAPL", "Technology", 0.72)])
        result = allocate_day(ranked, snap, LIMITS)
        assert result[0].approved is False
        assert result[0].rejection.reason == "already_held"

    def test_sector_count_limit_rejected(self):
        snap = _snapshot_with(
            _mk_pos("AAPL", "Technology"),
            _mk_pos("MSFT", "Technology"),
            _mk_pos("GOOG", "Technology"),
        )
        ranked = rank_signals([_mk_signal("NVDA", "Technology", 0.80)])
        result = allocate_day(ranked, snap, LIMITS)
        assert result[0].approved is False
        assert result[0].rejection.reason == "sector_count_limit"

    def test_sector_pct_limit_rejected(self):
        # 3 positions in Technology at 9% each = 27% exposure.
        # Adding min_position_pct=0.05 would exceed max_sector_pct=0.30.
        snap = _snapshot_with(
            _mk_pos("AAPL", "Technology", pct=0.09),
            _mk_pos("MSFT", "Technology", pct=0.09),
            _mk_pos("GOOG", "Technology", pct=0.09),
            cash=7_300.0, equity=10_000.0,
        )
        # Override max_positions=10 so the count limit doesn't fire first.
        limits = PositionLimitsConfig(
            min_position_pct=0.02, max_position_pct=0.10,
            max_sector_pct=0.30, max_positions_per_sector=10,
        )
        ranked = rank_signals([_mk_signal("NVDA", "Technology", 0.80)])
        # min_position_pct=0.05 makes 0.27 + 0.05 = 0.32 > 0.30 → fail.
        result = allocate_day(ranked, snap, limits, min_position_pct=0.05)
        assert result[0].approved is False
        assert result[0].rejection.reason == "sector_pct_limit"

    def test_insufficient_capital_rejected(self):
        # 1% cash — below min_position_pct=0.02 → fail.
        snap = PortfolioSnapshot(
            as_of_date=date(2024, 1, 2),
            equity=10_000.0, cash=100.0,
            open_positions=(),
        )
        ranked = rank_signals([_mk_signal("AAPL", "Technology", 0.80)])
        result = allocate_day(ranked, snap, LIMITS, min_position_pct=0.02)
        assert result[0].approved is False
        assert result[0].rejection.reason == "insufficient_capital"

    def test_rejection_detail_nonempty(self):
        snap = _snapshot_with(_mk_pos("AAPL", "Technology"))
        ranked = rank_signals([_mk_signal("AAPL", "Technology", 0.72)])
        result = allocate_day(ranked, snap, LIMITS)
        assert len(result[0].rejection.detail) > 0

    def test_rejection_preserves_ticker_sector_confidence(self):
        snap = _snapshot_with(_mk_pos("AAPL", "Technology"))
        ranked = rank_signals([_mk_signal("AAPL", "Technology", 0.72)])
        result = allocate_day(ranked, snap, LIMITS)
        assert result[0].ticker == "AAPL"
        assert result[0].sector == "Technology"
        assert result[0].confidence == 0.72


# ── allocate_day: running snapshot invariant ──────────────────────────────────

class TestAllocateDayRunningSnapshot:
    def test_second_same_sector_signal_sees_first_approval(self):
        # Technology starts at 2/3. First tech signal pushes to 3/3.
        # Second tech signal must be rejected.
        snap = _snapshot_with(
            _mk_pos("AAPL", "Technology"),
            _mk_pos("MSFT", "Technology"),
        )
        ranked = rank_signals([
            _mk_signal("GOOG", "Technology", 0.80),
            _mk_signal("NVDA", "Technology", 0.70),
        ])
        result = allocate_day(ranked, snap, LIMITS)
        assert result[0].approved is True
        assert result[1].approved is False
        assert result[1].rejection.reason == "sector_count_limit"

    def test_running_sector_pct_accounts_for_new_approvals(self):
        # Tech at 20% exposure, max 30%. With min_position_pct=0.02, room for 5
        # more approvals (0.20 + 5*0.02 = 0.30) before the 6th hits 0.32 > 0.30.
        limits = PositionLimitsConfig(
            min_position_pct=0.02, max_position_pct=0.10,
            max_sector_pct=0.30, max_positions_per_sector=10,
        )
        snap = _snapshot_with(
            _mk_pos("AAPL", "Technology", pct=0.10),
            _mk_pos("MSFT", "Technology", pct=0.10),
            cash=8_000.0, equity=10_000.0,
        )
        sigs = [_mk_signal(f"TIC{i:02d}", "Technology", 0.80 - i * 0.01)
                for i in range(6)]
        ranked = rank_signals(sigs)
        result = allocate_day(ranked, snap, limits)
        approvals = [r.approved for r in result]
        assert approvals.count(True) == 5
        assert approvals.count(False) == 1
        assert result[-1].rejection.reason == "sector_pct_limit"

    def test_insufficient_capital_triggers_after_enough_approvals(self):
        limits = PositionLimitsConfig(
            min_position_pct=0.20, max_position_pct=0.25,
            max_sector_pct=1.0, max_positions_per_sector=10,
        )
        snap = PortfolioSnapshot(
            as_of_date=date(2024, 1, 2),
            equity=10_000.0, cash=5_000.0, open_positions=(),
        )
        sigs = [_mk_signal(f"TIC{i:02d}", f"Sec{i}", 0.80 - i * 0.01)
                for i in range(4)]
        ranked = rank_signals(sigs)
        result = allocate_day(ranked, snap, limits, min_position_pct=0.20)
        approvals = [r.approved for r in result]
        assert approvals.count(True) == 2
        assert all(
            (not r.approved and r.rejection.reason == "insufficient_capital")
            for r in result[2:]
        )


# ── allocate_day: determinism + edge cases ────────────────────────────────────

class TestAllocateDayDeterminism:
    def test_deterministic_across_calls(self):
        snap = _snapshot_with(_mk_pos("AAPL", "Technology"))
        ranked = rank_signals([
            _mk_signal("MSFT", "Technology", 0.70),
            _mk_signal("JPM", "Financials", 0.70),
        ])
        r1 = allocate_day(ranked, snap, LIMITS)
        r2 = allocate_day(ranked, snap, LIMITS)
        assert [x.approved for x in r1] == [x.approved for x in r2]

    def test_unknown_sector_still_processed(self):
        ranked = rank_signals([_mk_signal("XYZ", "Unknown", 0.72)])
        result = allocate_day(ranked, _empty_snapshot(), LIMITS)
        assert result[0].approved is True

    def test_raises_on_negative_min_position_pct(self):
        ranked = rank_signals([_mk_signal("AAPL", "Technology", 0.72)])
        with pytest.raises(RuntimeError):
            allocate_day(ranked, _empty_snapshot(), LIMITS,
                         min_position_pct=-0.01)

    def test_cash_exactly_at_min_position_pct_approved(self):
        # Boundary: cash == min_position_pct should be admissible (>= check
        # with 1e-9 tolerance, not strict >).
        snap = PortfolioSnapshot(
            as_of_date=date(2024, 1, 2),
            equity=10_000.0, cash=200.0,  # cash_pct = 0.02 exactly
            open_positions=(),
        )
        ranked = rank_signals([_mk_signal("AAPL", "Technology", 0.80)])
        result = allocate_day(ranked, snap, LIMITS, min_position_pct=0.02)
        assert result[0].approved is True

    def test_allocate_day_does_not_mutate_input_snapshot(self):
        # Frozen Pydantic models can't be mutated, but the running mirrors
        # inside allocate_day must not leak into the input snapshot's
        # computed properties. This test guards against accidental shared
        # state (e.g. if we ever convert to dict-backed storage).
        original = _snapshot_with(_mk_pos("AAPL", "Technology"))
        original_sector_counts = dict(original.sector_counts)
        original_cash_pct = original.cash_pct
        ranked = rank_signals([
            _mk_signal("MSFT", "Technology", 0.80),
            _mk_signal("JPM", "Financials", 0.70),
        ])
        allocate_day(ranked, original, LIMITS)
        assert original.sector_counts == original_sector_counts
        assert original.cash_pct == original_cash_pct
