"""Tests for trading_system.portfolio_state dataclasses."""

import dataclasses
from datetime import datetime

import pandas as pd
import pytest

from trading_system.portfolio_state import (
    AllocationDecision,
    PortfolioSnapshot,
    RankedSignal,
)


# ── Helpers ──────────────────────────────────────────────────────

def _ranked(
    ticker: str = "AAPL",
    confidence: float = 0.72,
    sector: str = "Technology",
    rank: int = 1,
    rank_score: float = 0.72,
    date: pd.Timestamp | None = None,
    raw_metadata: dict | None = None,
) -> RankedSignal:
    return RankedSignal(
        ticker=ticker,
        confidence=confidence,
        sector=sector,
        rank=rank,
        rank_score=rank_score,
        date=date or pd.Timestamp("2024-06-01"),
        raw_metadata=raw_metadata or {},
    )


def _approved(
    ticker: str = "AAPL",
    rank: int = 1,
    confidence: float = 0.72,
    sector: str = "Technology",
) -> AllocationDecision:
    return AllocationDecision(
        ticker=ticker, approved=True, rank=rank,
        confidence=confidence, sector=sector,
    )


def _rejected(
    ticker: str = "AAPL",
    rank: int = 1,
    confidence: float = 0.72,
    sector: str = "Technology",
    reason: str = "Test rejection",
) -> AllocationDecision:
    return AllocationDecision(
        ticker=ticker, approved=False, rank=rank,
        confidence=confidence, sector=sector,
        rejection_reason=reason, rejection_layer="portfolio",
    )


# ── RankedSignal Tests ───────────────────────────────────────────

class TestRankedSignal:
    def test_valid_construction(self):
        rs = _ranked()
        assert rs.ticker == "AAPL"
        assert rs.confidence == 0.72
        assert rs.rank == 1

    def test_frozen_immutability(self):
        rs = _ranked()
        with pytest.raises(dataclasses.FrozenInstanceError):
            rs.confidence = 0.99

    def test_empty_ticker_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _ranked(ticker="")

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            _ranked(confidence=-0.1)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            _ranked(confidence=1.01)

    def test_confidence_boundary_zero(self):
        rs = _ranked(confidence=0.0, rank_score=0.0)
        assert rs.confidence == 0.0

    def test_confidence_boundary_one(self):
        rs = _ranked(confidence=1.0, rank_score=1.0)
        assert rs.confidence == 1.0

    def test_rank_zero_raises(self):
        with pytest.raises(ValueError, match="rank"):
            _ranked(rank=0)

    def test_rank_negative_raises(self):
        with pytest.raises(ValueError, match="rank"):
            _ranked(rank=-1)

    def test_rank_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="rank_score"):
            _ranked(rank_score=1.5)

    def test_raw_metadata_preserved(self):
        meta = {"n_matches": 42, "raw_prob": 0.68}
        rs = _ranked(raw_metadata=meta)
        assert rs.raw_metadata["n_matches"] == 42

    def test_default_raw_metadata_is_empty_dict(self):
        rs = _ranked()
        assert rs.raw_metadata == {}


# ── AllocationDecision Tests ─────────────────────────────────────

class TestAllocationDecision:
    def test_approved_construction(self):
        d = _approved()
        assert d.approved is True
        assert d.rejection_reason is None

    def test_rejected_construction(self):
        d = _rejected(reason="Sector full")
        assert d.approved is False
        assert d.rejection_reason == "Sector full"

    def test_frozen_immutability(self):
        d = _approved()
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.approved = False

    def test_approved_with_reason_raises(self):
        with pytest.raises(ValueError, match="must not have"):
            AllocationDecision(
                ticker="AAPL", approved=True, rank=1,
                confidence=0.72, sector="Tech",
                rejection_reason="Should not be here",
            )

    def test_rejected_without_reason_raises(self):
        with pytest.raises(ValueError, match="must have"):
            AllocationDecision(
                ticker="AAPL", approved=False, rank=1,
                confidence=0.72, sector="Tech",
                rejection_reason=None,
            )

    def test_rejection_layer_default(self):
        d = _rejected()
        assert d.rejection_layer == "portfolio"


# ── PortfolioSnapshot Tests ──────────────────────────────────────

class TestPortfolioSnapshot:
    def test_empty_snapshot(self):
        snap = PortfolioSnapshot(open_tickers=frozenset())
        assert len(snap.open_tickers) == 0
        assert snap.sector_position_counts == {}
        assert snap.cooldowns == {}

    def test_frozen_immutability(self):
        snap = PortfolioSnapshot(open_tickers=frozenset({"AAPL"}))
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.open_tickers = frozenset()

    def test_open_tickers_is_frozenset(self):
        snap = PortfolioSnapshot(open_tickers=frozenset({"AAPL", "MSFT"}))
        assert isinstance(snap.open_tickers, frozenset)
        assert "AAPL" in snap.open_tickers

    def test_sector_counts_preserved(self):
        snap = PortfolioSnapshot(
            open_tickers=frozenset({"AAPL", "MSFT"}),
            sector_position_counts={"Technology": 2},
        )
        assert snap.sector_position_counts["Technology"] == 2

    def test_cooldown_data_preserved(self):
        cd = {"AAPL": {"until_date": pd.Timestamp("2024-06-05"), "last_confidence": 0.70}}
        snap = PortfolioSnapshot(open_tickers=frozenset(), cooldowns=cd)
        assert snap.cooldowns["AAPL"]["last_confidence"] == 0.70

    def test_default_reentry_margin(self):
        snap = PortfolioSnapshot(open_tickers=frozenset())
        assert snap.cooldown_reentry_margin == 0.05
