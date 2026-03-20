"""Tests for trading_system.portfolio_manager — ranking, allocation, rejection."""

import pandas as pd
import pytest

from trading_system.config import PositionLimitsConfig
from trading_system.portfolio_manager import (
    allocate_day,
    check_allocation,
    rank_signals,
)
from trading_system.portfolio_state import (
    AllocationDecision,
    PortfolioSnapshot,
    RankedSignal,
)


# ── Helpers ──────────────────────────────────────────────────────

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOG": "Technology", "JPM": "Financials", "GS": "Financials",
    "BAC": "Financials", "XOM": "Energy", "CVX": "Energy",
    "UNH": "Healthcare",
}

DEFAULT_LIMITS = PositionLimitsConfig()  # max_positions_per_sector=3


def _sig(ticker: str, confidence: float, date: str = "2024-06-01", **extra) -> dict:
    return {"ticker": ticker, "confidence": confidence, "date": date, **extra}


def _empty_snapshot() -> PortfolioSnapshot:
    return PortfolioSnapshot(open_tickers=frozenset())


def _snapshot(
    open_tickers: set[str] | None = None,
    sector_counts: dict[str, int] | None = None,
    cooldowns: dict | None = None,
    margin: float = 0.05,
) -> PortfolioSnapshot:
    tickers = open_tickers or set()
    return PortfolioSnapshot(
        open_tickers=frozenset(tickers),
        sector_position_counts=sector_counts or {},
        cooldowns=cooldowns or {},
        cooldown_reentry_margin=margin,
    )


# ── rank_signals Tests ───────────────────────────────────────────

class TestRankSignals:
    def test_sorts_by_confidence_desc(self):
        signals = [
            _sig("MSFT", 0.65),
            _sig("AAPL", 0.80),
            _sig("JPM",  0.72),
        ]
        ranked = rank_signals(signals, SECTOR_MAP)
        assert ranked[0].ticker == "AAPL"
        assert ranked[1].ticker == "JPM"
        assert ranked[2].ticker == "MSFT"

    def test_rank_numbers_are_one_based(self):
        signals = [_sig("AAPL", 0.80), _sig("MSFT", 0.60)]
        ranked = rank_signals(signals, SECTOR_MAP)
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2

    def test_rank_score_equals_confidence_v1(self):
        signals = [_sig("AAPL", 0.73)]
        ranked = rank_signals(signals, SECTOR_MAP)
        assert ranked[0].rank_score == 0.73

    def test_tie_break_alphabetical(self):
        signals = [
            _sig("MSFT", 0.75),
            _sig("AAPL", 0.75),
        ]
        ranked = rank_signals(signals, SECTOR_MAP)
        assert ranked[0].ticker == "AAPL"  # A < M
        assert ranked[1].ticker == "MSFT"

    def test_empty_list_returns_empty(self):
        assert rank_signals([], SECTOR_MAP) == []

    def test_single_signal(self):
        ranked = rank_signals([_sig("AAPL", 0.70)], SECTOR_MAP)
        assert len(ranked) == 1
        assert ranked[0].rank == 1

    def test_sector_from_map_when_missing(self):
        signals = [_sig("AAPL", 0.70)]  # No 'sector' key in dict
        ranked = rank_signals(signals, SECTOR_MAP)
        assert ranked[0].sector == "Technology"

    def test_sector_from_signal_overrides_map(self):
        signals = [_sig("AAPL", 0.70, sector="CustomSector")]
        ranked = rank_signals(signals, SECTOR_MAP)
        assert ranked[0].sector == "CustomSector"

    def test_unknown_sector_when_not_in_map(self):
        signals = [_sig("ZZZZ", 0.70)]
        ranked = rank_signals(signals, {"AAPL": "Tech"})
        assert ranked[0].sector == "Unknown"

    def test_raw_metadata_preserved(self):
        signals = [_sig("AAPL", 0.70, n_matches=42, raw_prob=0.68)]
        ranked = rank_signals(signals, SECTOR_MAP)
        assert ranked[0].raw_metadata["n_matches"] == 42
        assert ranked[0].raw_metadata["raw_prob"] == 0.68

    def test_date_converted_to_timestamp(self):
        ranked = rank_signals([_sig("AAPL", 0.70, date="2024-06-15")], SECTOR_MAP)
        assert ranked[0].date == pd.Timestamp("2024-06-15")

    def test_not_a_list_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a list"):
            rank_signals("not a list", SECTOR_MAP)

    def test_missing_ticker_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            rank_signals([{"confidence": 0.7, "date": "2024-01-01"}], SECTOR_MAP)

    def test_missing_confidence_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            rank_signals([{"ticker": "AAPL", "date": "2024-01-01"}], SECTOR_MAP)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Confidence must be"):
            rank_signals([_sig("AAPL", 1.5)], SECTOR_MAP)

    def test_many_signals_ranking(self):
        """Verify ranking is stable with 10 signals of varying confidence."""
        signals = [_sig(t, c) for t, c in [
            ("AAPL", 0.80), ("MSFT", 0.75), ("GOOG", 0.75),
            ("JPM", 0.72), ("GS", 0.72), ("BAC", 0.72),
            ("XOM", 0.68), ("CVX", 0.65), ("NVDA", 0.60),
            ("UNH", 0.55),
        ]]
        ranked = rank_signals(signals, SECTOR_MAP)
        assert len(ranked) == 10
        assert ranked[0].ticker == "AAPL"  # 0.80
        # GOOG < MSFT alphabetically, both at 0.75
        assert ranked[1].ticker == "GOOG"
        assert ranked[2].ticker == "MSFT"
        # BAC < GS < JPM alphabetically, all at 0.72
        assert ranked[3].ticker == "BAC"
        assert ranked[4].ticker == "GS"
        assert ranked[5].ticker == "JPM"


# ── check_allocation Tests ───────────────────────────────────────

class TestCheckAllocation:
    def _ranked(self, ticker="AAPL", confidence=0.72, sector="Technology",
                rank=1, date="2024-06-01") -> RankedSignal:
        return RankedSignal(
            ticker=ticker, confidence=confidence, sector=sector,
            rank=rank, rank_score=confidence, date=pd.Timestamp(date),
        )

    def test_approved_on_empty_portfolio(self):
        decision = check_allocation(
            self._ranked(), _empty_snapshot(), DEFAULT_LIMITS,
        )
        assert decision.approved is True
        assert decision.rejection_reason is None

    def test_rejected_already_holding(self):
        snap = _snapshot(open_tickers={"AAPL"})
        decision = check_allocation(self._ranked(), snap, DEFAULT_LIMITS)
        assert decision.approved is False
        assert "Already holding" in decision.rejection_reason

    def test_rejected_cooldown_insufficient_confidence(self):
        snap = _snapshot(
            cooldowns={
                "AAPL": {
                    "until_date": pd.Timestamp("2024-06-10"),
                    "last_confidence": 0.70,
                },
            },
            margin=0.05,
        )
        # Signal confidence 0.72 < required 0.75 (0.70 + 0.05)
        decision = check_allocation(self._ranked(confidence=0.72), snap, DEFAULT_LIMITS)
        assert decision.approved is False
        assert "cooldown" in decision.rejection_reason.lower()
        assert "0.75" in decision.rejection_reason

    def test_approved_cooldown_sufficient_confidence(self):
        snap = _snapshot(
            cooldowns={
                "AAPL": {
                    "until_date": pd.Timestamp("2024-06-10"),
                    "last_confidence": 0.65,
                },
            },
            margin=0.05,
        )
        # Signal confidence 0.72 >= required 0.70 (0.65 + 0.05)
        decision = check_allocation(self._ranked(confidence=0.72), snap, DEFAULT_LIMITS)
        assert decision.approved is True

    def test_approved_cooldown_expired(self):
        snap = _snapshot(
            cooldowns={
                "AAPL": {
                    "until_date": pd.Timestamp("2024-05-01"),  # Before signal date
                    "last_confidence": 0.95,
                },
            },
        )
        decision = check_allocation(
            self._ranked(date="2024-06-01"), snap, DEFAULT_LIMITS,
        )
        assert decision.approved is True

    def test_rejected_sector_at_max_positions(self):
        snap = _snapshot(sector_counts={"Technology": 3})  # max is 3
        decision = check_allocation(self._ranked(), snap, DEFAULT_LIMITS)
        assert decision.approved is False
        assert "Sector Technology at max 3" in decision.rejection_reason

    def test_approved_sector_below_max(self):
        snap = _snapshot(sector_counts={"Technology": 2})
        decision = check_allocation(self._ranked(), snap, DEFAULT_LIMITS)
        assert decision.approved is True

    def test_sector_count_for_different_sector(self):
        snap = _snapshot(sector_counts={"Financials": 3})  # Different sector
        decision = check_allocation(self._ranked(sector="Technology"), snap, DEFAULT_LIMITS)
        assert decision.approved is True

    def test_rejection_layer_is_portfolio(self):
        snap = _snapshot(open_tickers={"AAPL"})
        decision = check_allocation(self._ranked(), snap, DEFAULT_LIMITS)
        assert decision.rejection_layer == "portfolio"

    def test_check_order_holding_before_cooldown(self):
        """If both holding and in cooldown, rejection says 'Already holding'."""
        snap = _snapshot(
            open_tickers={"AAPL"},
            cooldowns={"AAPL": {"until_date": pd.Timestamp("2024-06-10"), "last_confidence": 0.90}},
        )
        decision = check_allocation(self._ranked(confidence=0.72), snap, DEFAULT_LIMITS)
        assert "Already holding" in decision.rejection_reason

    def test_check_order_cooldown_before_sector(self):
        """If in cooldown AND sector full, rejection should mention cooldown."""
        snap = _snapshot(
            sector_counts={"Technology": 3},
            cooldowns={"AAPL": {"until_date": pd.Timestamp("2024-06-10"), "last_confidence": 0.90}},
        )
        decision = check_allocation(self._ranked(confidence=0.72), snap, DEFAULT_LIMITS)
        assert "cooldown" in decision.rejection_reason.lower()


# ── allocate_day Tests ───────────────────────────────────────────

class TestAllocateDay:
    def test_all_approved_on_empty_portfolio(self):
        signals = [_sig("AAPL", 0.80), _sig("JPM", 0.70)]
        decisions = allocate_day(signals, _empty_snapshot(), DEFAULT_LIMITS, SECTOR_MAP)
        assert all(d.approved for d in decisions)
        assert decisions[0].ticker == "AAPL"  # Higher confidence first

    def test_empty_signals_returns_empty(self):
        assert allocate_day([], _empty_snapshot(), DEFAULT_LIMITS, SECTOR_MAP) == []

    def test_sector_limit_blocks_fourth_in_sector(self):
        """With 3 Tech positions open, 4th Tech signal is rejected."""
        snap = _snapshot(
            open_tickers={"AAPL", "MSFT", "NVDA"},
            sector_counts={"Technology": 3},
        )
        signals = [_sig("GOOG", 0.80)]  # 4th Tech ticker
        decisions = allocate_day(signals, snap, DEFAULT_LIMITS, SECTOR_MAP)
        assert decisions[0].approved is False
        assert "Sector Technology at max" in decisions[0].rejection_reason

    def test_running_sector_count_within_day(self):
        """When 3 Tech BUY signals arrive on same day, only first 3 approved."""
        signals = [
            _sig("AAPL", 0.90), _sig("MSFT", 0.85),
            _sig("NVDA", 0.80), _sig("GOOG", 0.75),
        ]
        decisions = allocate_day(signals, _empty_snapshot(), DEFAULT_LIMITS, SECTOR_MAP)
        approved = [d for d in decisions if d.approved]
        rejected = [d for d in decisions if not d.approved]
        assert len(approved) == 3
        assert len(rejected) == 1
        assert rejected[0].ticker == "GOOG"  # Lowest confidence, 4th in sector

    def test_mixed_sectors_all_approved(self):
        signals = [
            _sig("AAPL", 0.80),  # Tech
            _sig("JPM", 0.75),   # Financials
            _sig("XOM", 0.70),   # Energy
        ]
        decisions = allocate_day(signals, _empty_snapshot(), DEFAULT_LIMITS, SECTOR_MAP)
        assert all(d.approved for d in decisions)

    def test_duplicate_ticker_rejected(self):
        """Same ticker appearing twice: second is rejected (running state tracks it)."""
        signals = [
            _sig("AAPL", 0.80),
            _sig("AAPL", 0.75),  # Duplicate
        ]
        decisions = allocate_day(signals, _empty_snapshot(), DEFAULT_LIMITS, SECTOR_MAP)
        assert decisions[0].approved is True
        assert decisions[1].approved is False
        assert "Already holding" in decisions[1].rejection_reason

    def test_preserves_rank_order(self):
        signals = [_sig("MSFT", 0.60), _sig("AAPL", 0.90)]
        decisions = allocate_day(signals, _empty_snapshot(), DEFAULT_LIMITS, SECTOR_MAP)
        assert decisions[0].rank == 1  # AAPL (0.90) ranked first
        assert decisions[1].rank == 2
