"""Tests for trading_system.portfolio_state — Phase 4 PM schemas."""
from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from trading_system.portfolio_state import (
    AllocationResult,
    OpenPosition,
    PMRejection,
    PortfolioSnapshot,
    RankedSignal,
)


# ── OpenPosition ──────────────────────────────────────────────────────────────

def test_open_position_valid_construction():
    p = OpenPosition(
        ticker="AAPL",
        sector="Technology",
        entry_date=date(2024, 1, 2),
        position_pct=0.05,
        entry_price=150.0,
    )
    assert p.ticker == "AAPL"
    assert p.position_pct == 0.05


def test_open_position_is_frozen():
    p = OpenPosition(
        ticker="AAPL", sector="Technology", entry_date=date(2024, 1, 2),
        position_pct=0.05, entry_price=150.0,
    )
    with pytest.raises(ValidationError):
        p.ticker = "MSFT"  # type: ignore[misc]


def test_open_position_rejects_lowercase_ticker():
    with pytest.raises(ValidationError):
        OpenPosition(
            ticker="aapl", sector="Technology", entry_date=date(2024, 1, 2),
            position_pct=0.05, entry_price=150.0,
        )


def test_open_position_rejects_negative_position_pct():
    with pytest.raises(ValidationError):
        OpenPosition(
            ticker="AAPL", sector="Technology", entry_date=date(2024, 1, 2),
            position_pct=-0.01, entry_price=150.0,
        )


# ── RankedSignal ──────────────────────────────────────────────────────────────

def test_ranked_signal_valid_construction():
    rs = RankedSignal(
        ticker="AAPL",
        sector="Technology",
        signal_date=date(2024, 1, 2),
        confidence=0.72,
        rank=1,
    )
    assert rs.rank == 1
    assert rs.confidence == 0.72


def test_ranked_signal_rejects_rank_zero():
    with pytest.raises(ValidationError):
        RankedSignal(
            ticker="AAPL", sector="Technology", signal_date=date(2024, 1, 2),
            confidence=0.72, rank=0,
        )


def test_ranked_signal_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        RankedSignal(
            ticker="AAPL", sector="Technology", signal_date=date(2024, 1, 2),
            confidence=1.01, rank=1,
        )


# ── PMRejection ───────────────────────────────────────────────────────────────

def test_pm_rejection_literal_reason_accepts_known_values():
    reasons = [
        "already_held", "cooldown", "sector_count_limit",
        "sector_pct_limit", "insufficient_capital",
    ]
    for r in reasons:
        rej = PMRejection(
            ticker="AAPL", sector="Technology", signal_date=date(2024, 1, 2),
            confidence=0.72, rank=1, reason=r, detail="test",
        )
        assert rej.reason == r


def test_pm_rejection_rejects_unknown_reason():
    with pytest.raises(ValidationError):
        PMRejection(
            ticker="AAPL", sector="Technology", signal_date=date(2024, 1, 2),
            confidence=0.72, rank=1, reason="made_up_reason", detail="x",
        )


# ── AllocationResult ──────────────────────────────────────────────────────────

def test_allocation_result_approved_has_no_rejection():
    ar = AllocationResult(
        ticker="AAPL", sector="Technology", signal_date=date(2024, 1, 2),
        confidence=0.72, rank=1, approved=True,
    )
    assert ar.approved is True
    assert ar.rejection is None


def test_allocation_result_rejected_must_have_rejection():
    with pytest.raises(ValidationError):
        AllocationResult(
            ticker="AAPL", sector="Technology", signal_date=date(2024, 1, 2),
            confidence=0.72, rank=1, approved=False, rejection=None,
        )


def test_allocation_result_approved_cannot_have_rejection():
    rej = PMRejection(
        ticker="AAPL", sector="Technology", signal_date=date(2024, 1, 2),
        confidence=0.72, rank=1, reason="already_held", detail="x",
    )
    with pytest.raises(ValidationError):
        AllocationResult(
            ticker="AAPL", sector="Technology", signal_date=date(2024, 1, 2),
            confidence=0.72, rank=1, approved=True, rejection=rej,
        )


# ── PortfolioSnapshot ─────────────────────────────────────────────────────────

def test_portfolio_snapshot_empty_is_valid():
    snap = PortfolioSnapshot(
        as_of_date=date(2024, 1, 2),
        equity=10_000.0,
        cash=10_000.0,
        open_positions=(),
    )
    assert snap.n_open_positions == 0
    assert snap.sector_counts == {}
    assert snap.sector_exposure_pct == {}


def test_portfolio_snapshot_computes_sector_counts_and_exposure():
    snap = PortfolioSnapshot(
        as_of_date=date(2024, 1, 2),
        equity=10_000.0,
        cash=8_000.0,
        open_positions=(
            OpenPosition(ticker="AAPL", sector="Technology",
                         entry_date=date(2024, 1, 2), position_pct=0.08,
                         entry_price=150.0),
            OpenPosition(ticker="MSFT", sector="Technology",
                         entry_date=date(2024, 1, 2), position_pct=0.07,
                         entry_price=300.0),
            OpenPosition(ticker="JPM", sector="Financials",
                         entry_date=date(2024, 1, 2), position_pct=0.05,
                         entry_price=150.0),
        ),
    )
    assert snap.n_open_positions == 3
    assert snap.sector_counts == {"Technology": 2, "Financials": 1}
    assert snap.sector_exposure_pct["Technology"] == pytest.approx(0.15)
    assert snap.sector_exposure_pct["Financials"] == pytest.approx(0.05)


def test_portfolio_snapshot_is_frozen():
    snap = PortfolioSnapshot(
        as_of_date=date(2024, 1, 2),
        equity=10_000.0, cash=10_000.0, open_positions=(),
    )
    with pytest.raises(ValidationError):
        snap.equity = 20_000.0  # type: ignore[misc]


def test_portfolio_snapshot_rejects_cash_exceeding_equity():
    with pytest.raises(ValidationError):
        PortfolioSnapshot(
            as_of_date=date(2024, 1, 2),
            equity=10_000.0, cash=10_001.0, open_positions=(),
        )


def test_portfolio_snapshot_contains_ticker_helper():
    snap = PortfolioSnapshot(
        as_of_date=date(2024, 1, 2),
        equity=10_000.0, cash=9_500.0,
        open_positions=(
            OpenPosition(ticker="AAPL", sector="Technology",
                         entry_date=date(2024, 1, 2), position_pct=0.05,
                         entry_price=150.0),
        ),
    )
    assert snap.contains_ticker("AAPL") is True
    assert snap.contains_ticker("MSFT") is False
