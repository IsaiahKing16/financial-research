# Phase 3: Portfolio Manager — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a portfolio manager (Layer 3) to the FPPE trading system that ranks competing BUY signals by confidence, enforces sector diversification limits, manages capital allocation, and integrates cleanly into the existing backtest loop — with zero regression on Phase 2's 487 tests.

**Architecture:** The portfolio manager is a stateless pre-filter module (`portfolio_manager.py`) backed by pure dataclasses (`portfolio_state.py`). It sits between signal collection and risk-engine sizing in the backtest loop: signals are ranked, pre-filtered for holding/cooldown/sector-count constraints, then passed one-by-one to the existing `size_position()` + post-sizing constraint checks. The backtest engine gains a `use_portfolio_manager` flag (default `False`) that activates Layer 3 without breaking Phase 1/2 modes.

**Tech Stack:** Python 3.12, pytest, frozen dataclasses, type hints throughout. No new dependencies.

---

## File Structure

### New Files (4)

| File | Responsibility | Owner |
|------|---------------|-------|
| `trading_system/portfolio_state.py` | Pure dataclasses: `RankedSignal`, `AllocationDecision`, `PortfolioSnapshot` | Codex |
| `trading_system/portfolio_manager.py` | Core Layer 3 logic: `rank_signals()`, `check_allocation()` | Opus |
| `tests/test_portfolio_state.py` | Immutability, validation, edge cases for dataclasses | Sonnet |
| `tests/test_portfolio_manager.py` | Ranking correctness, sector limits, capital filtering, rejection reasons | Sonnet |

### Modified Files (3)

| File | Change | Lines Affected | Owner |
|------|--------|---------------|-------|
| `trading_system/backtest_engine.py` | Add `use_portfolio_manager` flag; extract ranking into portfolio_manager call at line 407–411 | ~407–573 (Phase 2 path), ~575–700 (Phase 1 path) | Opus |
| `trading_system/__init__.py` | Add exports for new classes/functions | Lines 19–28 | Sonnet |
| `tests/test_phase3_integration.py` | New file — end-to-end Phase 3 backtest tests | N/A (new) | Sonnet |

### Files NOT Modified

| File | Reason |
|------|--------|
| `trading_system/config.py` | v1 uses existing `PositionLimitsConfig.max_positions_per_sector` and `max_sector_pct`. No new config needed until v2 composite scoring. |
| `trading_system/risk_engine.py` | Phase 2 sizing is unchanged. Portfolio manager is upstream. |
| `trading_system/risk_state.py` | No new state. Portfolio snapshot is a new file. |
| `trading_system/signal_adapter.py` | Signal format unchanged. |
| `pattern_engine/*` | No changes to prediction engine. |

---

## Dependency Graph

```
Task 1: portfolio_state.py ──────────┐
  (pure dataclasses, zero deps)       │
                                      ├──→ Task 3: portfolio_manager.py ──→ Task 5: backtest_engine.py integration
Task 2: test_portfolio_state.py ─────┘           │                                    │
  (can run after Task 1)                         │                                    │
                                                 ▼                                    ▼
                                    Task 4: test_portfolio_manager.py    Task 6: test_phase3_integration.py
                                                                                      │
                                                                                      ▼
                                                                         Task 7: __init__.py exports
                                                                                      │
                                                                                      ▼
                                                                         Task 8: Full regression suite
```

**Parallel opportunities:**
- Tasks 1 + 2 can be assigned to different agents (Codex writes code, Sonnet writes tests) using the interface contract below
- Task 4 test stubs can be written once Task 3's function signatures are defined
- Tasks 1 and 2 are fully independent of Tasks 5–8

**Sequential gates:**
- Task 3 BLOCKED BY Task 1 (imports `RankedSignal`, `AllocationDecision`)
- Task 5 BLOCKED BY Task 3 (imports `rank_signals`, `check_allocation`)
- Task 6 BLOCKED BY Task 5 (needs integrated backtest engine)
- Task 8 BLOCKED BY all prior tasks

---

## Interface Contracts

### Contract 1: `RankedSignal` (portfolio_state.py → portfolio_manager.py)

```python
@dataclass(frozen=True)
class RankedSignal:
    """A BUY signal annotated with its rank position and score."""
    ticker: str
    confidence: float           # [0.0, 1.0] from FPPE calibration
    sector: str                 # From SECTOR_MAP
    rank: int                   # 1-based position in ranked list
    rank_score: float           # v1: equals confidence. v2+: composite score
    date: pd.Timestamp          # Signal date
    raw_metadata: dict[str, Any]  # Preserved from UnifiedSignal for audit
```

### Contract 2: `AllocationDecision` (portfolio_manager.py → backtest_engine.py)

```python
@dataclass(frozen=True)
class AllocationDecision:
    """Result of portfolio-level allocation check for one signal."""
    ticker: str
    approved: bool
    rank: int                           # Original rank position
    confidence: float
    sector: str
    rejection_reason: str | None = None  # None if approved
    rejection_layer: str = "portfolio"   # Always "portfolio" for PM rejections
```

### Contract 3: `PortfolioSnapshot` (backtest_engine.py → portfolio_manager.py)

```python
@dataclass(frozen=True)
class PortfolioSnapshot:
    """Read-only snapshot of current portfolio state for allocation decisions."""
    open_tickers: frozenset[str]                    # Currently held tickers
    sector_position_counts: dict[str, int]          # sector → count of open positions
    cooldowns: dict[str, dict[str, Any]]            # ticker → {until_date, last_confidence}
    cooldown_reentry_margin: float                  # Config: reentry_confidence_margin
```

### Contract 4: Function Signatures (portfolio_manager.py)

```python
def rank_signals(
    buy_signals: list[dict[str, Any]],
    sector_map: dict[str, str],
) -> list[RankedSignal]:
    """
    Rank BUY signals by confidence (v1). Deterministic tie-breaking: alphabetical ticker.

    NOTE: The design doc specifies v1 tie-breaking as "lower correlation to
    existing portfolio, then alphabetical ticker." This plan implements
    alphabetical-only tie-breaking for v1 because correlation-based
    tie-breaking requires portfolio state (correlation matrix) that is not
    available in the stateless ranking function. Correlation tie-breaking
    is deferred to v2 when CompositeRanking is implemented.

    Args:
        buy_signals: List of dicts with keys: ticker, confidence, date, sector (optional).
                     Each dict represents one BUY signal for a single date.
        sector_map: Ticker → sector mapping for signals missing 'sector' key.

    Returns:
        List of RankedSignal sorted by rank_score descending, then ticker ascending.

    Raises:
        ValueError: If buy_signals contains non-BUY signals or missing required keys.
    """

def check_allocation(
    ranked_signal: RankedSignal,
    snapshot: PortfolioSnapshot,
    position_limits: PositionLimitsConfig,
) -> AllocationDecision:
    """
    Check whether a ranked signal passes portfolio-level constraints.

    Checks (in order):
        1. Already holding this ticker → reject "Already holding position"
        2. Ticker in active cooldown with insufficient confidence margin → reject with cooldown details
        3. Sector position count at maximum → reject "Sector {X} at max {N} positions"

    Sector EXPOSURE % is NOT checked here (requires position size from risk_engine).
    Capital availability is NOT checked here (requires cash balance from backtest_engine).

    Args:
        ranked_signal: The signal to evaluate.
        snapshot: Current portfolio state (immutable).
        position_limits: Config with max_positions_per_sector.

    Returns:
        AllocationDecision with approved=True/False and rejection details.
    """

def allocate_day(
    buy_signals: list[dict[str, Any]],
    snapshot: PortfolioSnapshot,
    position_limits: PositionLimitsConfig,
    sector_map: dict[str, str],
) -> list[AllocationDecision]:
    """
    Convenience: rank + check all BUY signals for one date.

    NOTE: This function does NOT check capital availability or gross exposure.
    Those checks require the actual position size (from risk_engine) and cash
    balance (from backtest_engine), which are not available at ranking time.
    Capital and exposure checks remain in backtest_engine's post-sizing loop.

    Returns:
        List of AllocationDecision for ALL signals (approved and rejected),
        ordered by rank (highest confidence first).
    """
```

### Contract 5: BacktestEngine Integration

```python
class BacktestEngine:
    def __init__(
        self,
        config: TradingConfig = None,
        use_risk_engine: bool = False,
        use_portfolio_manager: bool = False,   # NEW — Phase 3 flag
    ):
        ...

    def run(
        self,
        signal_df: pd.DataFrame,
        price_df: pd.DataFrame,
        equal_weight_pct: float = 0.05,
        use_risk_engine: bool | None = None,
        use_portfolio_manager: bool | None = None,  # NEW — override per-run
    ) -> BacktestResults:
        ...
```

**Behavioral contract:**
- `use_portfolio_manager=True` requires `use_risk_engine=True` (Phase 3 builds on Phase 2). If `use_portfolio_manager=True` and `use_risk_engine=False`, raise `ValueError`.
- When `use_portfolio_manager=True`: replace the `day_buys.sort_values("confidence")` at line 411 with `portfolio_manager.allocate_day()`, then iterate approved signals only.
- When `use_portfolio_manager=False`: existing Phase 1/2 behavior is unchanged.

---

## Error Handling & Logging Standard

All new Phase 3 functions MUST follow these rules:

### 1. Public API Validation (RuntimeError/ValueError, NOT assert)

```python
# CORRECT:
def rank_signals(buy_signals: list[dict], ...) -> list[RankedSignal]:
    if not isinstance(buy_signals, list):
        raise TypeError(f"buy_signals must be a list, got {type(buy_signals).__name__}")
    for sig in buy_signals:
        if "ticker" not in sig or "confidence" not in sig:
            raise ValueError(f"Signal missing required keys: {sig}")
        if not 0.0 <= sig["confidence"] <= 1.0:
            raise ValueError(f"Confidence must be [0, 1], got {sig['confidence']} for {sig['ticker']}")

# WRONG (assert stripped under -O):
def rank_signals(buy_signals, ...):
    assert isinstance(buy_signals, list)  # NEVER
```

### 2. Logging Convention

```python
import logging

logger = logging.getLogger(__name__)

# Use levels appropriately:
logger.debug("Ranked %d signals for %s", len(ranked), date)           # Per-day noise
logger.info("Portfolio manager: %d approved, %d rejected", n_ok, n_rej)  # Summary per run
logger.warning("Ticker '%s' has no sector mapping, using 'Unknown'", ticker)  # Data issue
```

### 3. Graceful Degradation

- If `sector_map` is empty or missing a ticker, use `"Unknown"` sector (matches existing backtest_engine behavior at line 418–428).
- If `buy_signals` is empty, return empty list (not an error).
- If `snapshot.cooldowns` has stale entries (past `until_date`), silently skip them (treat as expired).

---

## Task 1: Create `portfolio_state.py` — Pure Dataclasses

**Agent:** Codex (isolated Python module, zero dependencies beyond stdlib + pandas)

**Files:**
- Create: `trading_system/portfolio_state.py`

- [ ] **Step 1: Write the module with all three dataclasses**

```python
"""
portfolio_state.py — Layer 3 Data Structures

Pure dataclasses for the portfolio manager. No business logic.
All frozen (immutable) except where noted.

Design doc: FPPE_TRADING_SYSTEM_DESIGN.md v0.3, Section 4.4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankedSignal:
    """A BUY signal annotated with its rank position and score.

    Args:
        ticker: Stock symbol (e.g., "AAPL").
        confidence: Calibrated probability from FPPE, [0.0, 1.0].
        sector: Sector classification from SECTOR_MAP.
        rank: 1-based position in the ranked list (1 = highest priority).
        rank_score: Scoring value used for ranking. v1: equals confidence.
        date: Signal generation date.
        raw_metadata: Preserved from UnifiedSignal for audit trail.
    """

    ticker: str
    confidence: float
    sector: str
    rank: int
    rank_score: float
    date: pd.Timestamp
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("ticker must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be [0, 1], got {self.confidence}"
            )
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if not 0.0 <= self.rank_score <= 1.0:
            raise ValueError(
                f"rank_score must be [0, 1], got {self.rank_score}"
            )


@dataclass(frozen=True)
class AllocationDecision:
    """Result of portfolio-level allocation check for one signal.

    Args:
        ticker: Stock symbol.
        approved: Whether the signal passed all portfolio constraints.
        rank: Original rank position from ranking step.
        confidence: Signal confidence (preserved for rejection logging).
        sector: Signal sector.
        rejection_reason: Human-readable reason if not approved, None if approved.
        rejection_layer: Always "portfolio" for portfolio manager rejections.
    """

    ticker: str
    approved: bool
    rank: int
    confidence: float
    sector: str
    rejection_reason: Optional[str] = None
    rejection_layer: str = "portfolio"

    def __post_init__(self) -> None:
        if self.approved and self.rejection_reason is not None:
            raise ValueError(
                "Approved decision must not have a rejection_reason"
            )
        if not self.approved and self.rejection_reason is None:
            raise ValueError(
                "Rejected decision must have a rejection_reason"
            )


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Read-only snapshot of current portfolio state for allocation decisions.

    This is constructed by backtest_engine at the start of each day's
    BUY signal processing and passed to portfolio_manager functions.

    Args:
        open_tickers: Set of currently held ticker symbols.
        sector_position_counts: Sector name -> count of open positions in that sector.
        cooldowns: Ticker -> {until_date: pd.Timestamp, last_confidence: float}.
        cooldown_reentry_margin: Config value for reentry_confidence_margin.
    """

    open_tickers: frozenset[str]
    sector_position_counts: dict[str, int] = field(default_factory=dict)
    cooldowns: dict[str, dict[str, Any]] = field(default_factory=dict)
    cooldown_reentry_margin: float = 0.05
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `cd C:\Users\Isaia\.claude\financial-research && venv\Scripts\python -c "from trading_system.portfolio_state import RankedSignal, AllocationDecision, PortfolioSnapshot; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add trading_system/portfolio_state.py
git commit -m "feat(phase3): add portfolio_state.py dataclasses — RankedSignal, AllocationDecision, PortfolioSnapshot"
```

---

## Task 2: Test `portfolio_state.py` — Dataclass Validation

**Agent:** Sonnet (scoped test writing)
**Depends on:** Task 1

**Files:**
- Create: `tests/test_portfolio_state.py`

- [ ] **Step 1: Write failing tests for RankedSignal validation**

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd C:\Users\Isaia\.claude\financial-research && venv\Scripts\python -m pytest tests/test_portfolio_state.py -v`
Expected: All pass (tests validate working code from Task 1)

- [ ] **Step 3: Commit**

```bash
git add tests/test_portfolio_state.py
git commit -m "test(phase3): add portfolio_state dataclass tests — immutability, validation, edge cases"
```

---

## Task 3: Create `portfolio_manager.py` — Core Logic

**Agent:** Opus (core algorithm, multi-file awareness)
**Depends on:** Task 1

**Files:**
- Create: `trading_system/portfolio_manager.py`

- [ ] **Step 1: Write the failing test for `rank_signals` (in test_portfolio_manager.py stub)**

Create a minimal test that imports `rank_signals` and verifies it returns signals sorted by confidence descending:

```python
# tests/test_portfolio_manager.py (initial stub — expanded in Task 4)
from trading_system.portfolio_manager import rank_signals

def test_rank_signals_sorts_by_confidence_desc():
    signals = [
        {"ticker": "MSFT", "confidence": 0.65, "date": "2024-06-01"},
        {"ticker": "AAPL", "confidence": 0.80, "date": "2024-06-01"},
        {"ticker": "JPM",  "confidence": 0.72, "date": "2024-06-01"},
    ]
    ranked = rank_signals(signals, sector_map={"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"})
    assert ranked[0].ticker == "AAPL"
    assert ranked[1].ticker == "JPM"
    assert ranked[2].ticker == "MSFT"
    assert ranked[0].rank == 1
    assert ranked[2].rank == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python -m pytest tests/test_portfolio_manager.py::test_rank_signals_sorts_by_confidence_desc -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_system.portfolio_manager'`

- [ ] **Step 3: Write `portfolio_manager.py` implementation**

```python
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
```

- [ ] **Step 4: Run the stub test to verify it passes**

Run: `venv\Scripts\python -m pytest tests/test_portfolio_manager.py::test_rank_signals_sorts_by_confidence_desc -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trading_system/portfolio_manager.py tests/test_portfolio_manager.py
git commit -m "feat(phase3): add portfolio_manager.py — rank_signals, check_allocation, allocate_day"
```

---

## Task 4: Test `portfolio_manager.py` — Full Coverage

**Agent:** Sonnet (scoped test writing)
**Depends on:** Task 3

**Files:**
- Modify: `tests/test_portfolio_manager.py` (replace stub with full suite)

- [ ] **Step 1: Write comprehensive tests**

```python
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
```

- [ ] **Step 2: Run tests**

Run: `venv\Scripts\python -m pytest tests/test_portfolio_manager.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_portfolio_manager.py
git commit -m "test(phase3): comprehensive portfolio_manager tests — ranking, allocation, sector limits"
```

---

## Task 5: Integrate into `backtest_engine.py`

**Agent:** Opus (integration, high blast-radius)
**Depends on:** Task 3

**Files:**
- Modify: `trading_system/backtest_engine.py` (lines ~1–30 imports, ~290–310 `__init__`, ~407–573 Phase 2 BUY loop)

This is the highest-risk task. The integration must:
1. Add `use_portfolio_manager` flag without breaking existing Phase 1/2 behavior
2. Replace the manual confidence sort + rejection checks with `allocate_day()` when enabled
3. Keep the post-sizing checks (sector exposure %, gross exposure, capital) in backtest_engine

- [ ] **Step 1: Run existing tests to confirm baseline**

Run: `venv\Scripts\python -m pytest tests/ -v --tb=short 2>&1 | tail -5`
Expected: All 487+ tests pass

- [ ] **Step 2: Add import at top of `backtest_engine.py`**

At line 26 (after `from .risk_state import RiskState, StopLossEvent`), add:

```python
from .portfolio_manager import allocate_day
from .portfolio_state import PortfolioSnapshot
```

- [ ] **Step 3: Add `use_portfolio_manager` parameter to `__init__`**

In `BacktestEngine.__init__()`, add the parameter and store it:

```python
def __init__(
    self,
    config: TradingConfig = None,
    use_risk_engine: bool = False,
    use_portfolio_manager: bool = False,
):
    self.config = config or DEFAULT_CONFIG
    self.use_risk_engine = use_risk_engine
    self.use_portfolio_manager = use_portfolio_manager
    if self.use_portfolio_manager and not self.use_risk_engine:
        raise ValueError(
            "use_portfolio_manager=True requires use_risk_engine=True "
            "(Phase 3 builds on Phase 2)"
        )
```

- [ ] **Step 4: Add `use_portfolio_manager` parameter to `run()`**

In `BacktestEngine.run()`, add the parameter and validate:

```python
def run(
    self,
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    equal_weight_pct: float = 0.05,
    use_risk_engine: Optional[bool] = None,
    use_portfolio_manager: Optional[bool] = None,
) -> "BacktestResults":
    risk_engine_enabled = use_risk_engine if use_risk_engine is not None else self.use_risk_engine
    pm_enabled = use_portfolio_manager if use_portfolio_manager is not None else self.use_portfolio_manager

    if pm_enabled and not risk_engine_enabled:
        raise ValueError(
            "use_portfolio_manager=True requires use_risk_engine=True "
            "(Phase 3 builds on Phase 2)"
        )
```

- [ ] **Step 5: Add portfolio manager branch in BUY signal processing (Phase 2 path)**

At line 407–411, replace the existing confidence-sort-and-iterate with a conditional branch:

```python
# ── Step 3: Process new BUY signals ──────────────────────
day_buys = signal_df[
    (signal_df["date"] == current_date) &
    (signal_df["signal"] == "BUY")
]

if pm_enabled:
    # ── Phase 3: Portfolio Manager ranks & pre-filters ────
    buy_dicts = day_buys.to_dict("records")
    snapshot = PortfolioSnapshot(
        open_tickers=frozenset(open_positions.keys()),
        sector_position_counts={
            sector: sum(
                1 for p in open_positions.values() if p.sector == sector
            )
            for sector in {p.sector for p in open_positions.values()}
        },
        cooldowns=dict(cooldowns),
        cooldown_reentry_margin=cfg_trade.reentry_confidence_margin,
    )
    allocation_decisions = allocate_day(
        buy_dicts, snapshot, cfg_pos, self.config.sector_map,
    )
    # Log portfolio-level rejections
    for dec in allocation_decisions:
        if not dec.approved:
            rejected_signals.append(RejectedSignal(
                date=current_date, ticker=dec.ticker, signal="BUY",
                confidence=dec.confidence,
                rejection_reason=dec.rejection_reason,
                rejection_layer=dec.rejection_layer,
            ))
    # Process only approved signals (already in rank order)
    approved_tickers = [
        dec.ticker for dec in allocation_decisions if dec.approved
    ]
    # Safe reindex: dedup signal_df rows, then reorder by rank via merge
    day_approved = day_buys[
        day_buys["ticker"].isin(approved_tickers)
    ].drop_duplicates(subset=["ticker"], keep="first")
    # Preserve rank order from portfolio manager
    rank_order = pd.DataFrame({"ticker": approved_tickers, "_pm_rank": range(len(approved_tickers))})
    day_buys_sorted = (
        day_approved.merge(rank_order, on="ticker")
        .sort_values("_pm_rank")
        .drop(columns=["_pm_rank"])
    )
else:
    day_buys_sorted = day_buys.sort_values("confidence", ascending=False)

for _, sig_row in day_buys_sorted.iterrows():
    ticker = sig_row["ticker"]
    confidence = sig_row["confidence"]
    sector = sig_row.get("sector", SECTOR_MAP.get(ticker, None))
    if sector is None:
        import warnings
        warnings.warn(...)
        sector = "Unknown"

    if risk_engine_enabled:
        # When pm_enabled, skip the holding/cooldown/sector-count checks
        # (already handled by portfolio_manager)
        if not pm_enabled:
            # ── Existing Phase 2 rejection checks ────────────
            if ticker in open_positions:
                ...  # existing code unchanged
                continue
            if ticker in cooldowns:
                ...  # existing code unchanged
                continue
            sector_positions = sum(...)
            if sector_positions >= cfg_pos.max_positions_per_sector:
                ...  # existing code unchanged
                continue

        # ── Price data, sizing, and post-sizing checks ────
        # (this code runs for BOTH pm_enabled and not pm_enabled)
        next_prices = price_lookup.get((next_date, ticker))
        ...  # existing sizing + post-sizing constraint code unchanged
```

**Critical details:**
1. When `pm_enabled=True`, the holding/cooldown/sector-count checks at lines 432–468 are SKIPPED (portfolio_manager already handled them). But the post-sizing checks (sector exposure %, gross exposure, capital) at lines 521–551 are KEPT (they require the actual position size from risk_engine).
2. **Drawdown halt is preserved.** The `size_position()` call (line 500) checks `risk_state` drawdown mode internally and returns `approved=False` with reason `"Drawdown halt: ..."` when drawdown exceeds the halt threshold. This protection is NOT bypassed by the portfolio manager — it runs after PM approval, in the sizing step. Do NOT add a separate drawdown halt check in the PM path; it would be redundant.
3. **Cooldown cleanup.** The existing code (line 456/605) deletes expired cooldowns from the `cooldowns` dict. When building `PortfolioSnapshot`, include ALL cooldowns (even stale ones) — the PM's `check_allocation()` safely skips expired cooldowns by comparing dates. To prevent unbounded accumulation over very long backtests, add cooldown cleanup BEFORE constructing the snapshot:
```python
# Clean up expired cooldowns before snapshot
expired = [t for t, cd in cooldowns.items() if current_date >= cd["until_date"]]
for t in expired:
    del cooldowns[t]
```

- [ ] **Step 6: Run all tests to verify zero regression**

Run: `venv\Scripts\python -m pytest tests/ -v --tb=short 2>&1 | tail -10`
Expected: All 487+ tests pass (Phase 1/2 unaffected because `use_portfolio_manager` defaults to `False`)

- [ ] **Step 7: Commit**

```bash
git add trading_system/backtest_engine.py
git commit -m "feat(phase3): integrate portfolio_manager into backtest_engine — use_portfolio_manager flag"
```

---

## Task 6: Phase 3 Integration Tests

**Agent:** Sonnet (integration test writing)
**Depends on:** Task 5

**Files:**
- Create: `tests/test_phase3_integration.py`

- [ ] **Step 1: Write end-to-end integration tests**

```python
"""
Phase 3 Integration Tests — Portfolio Manager + Risk Engine + Backtest Engine

Tests verify that portfolio_manager correctly ranks, filters, and passes
signals through the full backtest pipeline when use_portfolio_manager=True.
"""

import pandas as pd
import pytest
import dataclasses

from trading_system.backtest_engine import BacktestEngine
from trading_system.config import (
    TradingConfig,
    CapitalConfig,
    CostConfig,
    PositionLimitsConfig,
    TradeManagementConfig,
    RiskConfig,
    EvaluationConfig,
    SignalConfig,
)


# ── Helpers ──────────────────────────────────────────────────────

def _flat_price_df(
    tickers: list[str],
    n_days: int = 60,
    start: str = "2024-01-02",
    base_price: float = 100.0,
    atr_half: float = 0.01,
) -> pd.DataFrame:
    """Generate flat-price OHLCV for controlled ATR."""
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "Date": d, "Ticker": t,
                "Open": base_price,
                "High": base_price * (1 + atr_half),
                "Low": base_price * (1 - atr_half),
                "Close": base_price,
            })
    return pd.DataFrame(rows)


def _sig(
    ticker: str, sector: str, dates: list[pd.Timestamp],
    signal: str = "BUY", confidence: float = 0.75,
) -> list[dict]:
    return [
        {"date": d, "ticker": ticker, "signal": signal,
         "confidence": confidence, "sector": sector}
        for d in dates
    ]


def _test_config(**risk_overrides) -> TradingConfig:
    risk_kwargs = {
        "volatility_lookback": 20,
        "stop_loss_atr_multiple": 2.0,
        "max_loss_per_trade_pct": 0.02,
        "drawdown_brake_threshold": 0.15,
        "drawdown_halt_threshold": 0.20,
    }
    risk_kwargs.update(risk_overrides)
    return TradingConfig(
        capital=CapitalConfig(initial_capital=10_000.0),
        costs=CostConfig(slippage_bps=0.0, spread_bps=0.0),
        position_limits=PositionLimitsConfig(
            max_positions_per_sector=3,
            max_sector_pct=0.30,
        ),
        trade_management=TradeManagementConfig(),
        risk=RiskConfig(**risk_kwargs),
        evaluation=EvaluationConfig(),
        signals=SignalConfig(),
        sector_map={
            "AAPL": "Technology", "MSFT": "Technology",
            "NVDA": "Technology", "GOOG": "Technology",
            "JPM": "Financials", "GS": "Financials",
            "XOM": "Energy",
        },
    )


# ── Tests ────────────────────────────────────────────────────────

class TestPhase3BasicExecution:
    def test_backtest_completes_with_portfolio_manager(self):
        """Phase 3 mode runs without error on simple inputs."""
        cfg = _test_config()
        tickers = ["AAPL", "JPM", "XOM"]
        price_df = _flat_price_df(tickers, n_days=60)
        dates = pd.bdate_range("2024-01-23", periods=3)  # After 20-day lookback
        sigs = []
        for t, c in [("AAPL", 0.80), ("JPM", 0.70), ("XOM", 0.65)]:
            sigs.extend(_sig(t, cfg.sector_map[t], list(dates), confidence=c))
        signal_df = pd.DataFrame(sigs)

        engine = BacktestEngine(cfg, use_risk_engine=True, use_portfolio_manager=True)
        results = engine.run(signal_df, price_df)
        assert results.total_trades() >= 0
        assert len(results.daily_records) > 0

    def test_phase3_requires_phase2_at_init(self):
        """use_portfolio_manager=True without use_risk_engine=True raises at init."""
        cfg = _test_config()
        with pytest.raises(ValueError, match="requires use_risk_engine"):
            BacktestEngine(cfg, use_risk_engine=False, use_portfolio_manager=True)

    def test_phase3_requires_phase2_at_run(self):
        """Per-run override also validates Phase 2 dependency."""
        cfg = _test_config()
        engine = BacktestEngine(cfg)
        with pytest.raises(ValueError, match="requires use_risk_engine"):
            engine.run(
                pd.DataFrame(columns=["date", "ticker", "signal", "confidence"]),
                _flat_price_df(["AAPL"]),
                use_risk_engine=False,
                use_portfolio_manager=True,
            )


class TestPhase3VsPhase2:
    def test_same_single_signal_produces_same_trade(self):
        """With 1 BUY signal, Phase 3 and Phase 2 produce identical results."""
        cfg = _test_config()
        price_df = _flat_price_df(["AAPL"], n_days=60)
        dates = [pd.bdate_range("2024-01-23", periods=1)[0]]
        signal_df = pd.DataFrame(_sig("AAPL", "Technology", dates, confidence=0.80))

        engine_p2 = BacktestEngine(cfg, use_risk_engine=True, use_portfolio_manager=False)
        engine_p3 = BacktestEngine(cfg, use_risk_engine=True, use_portfolio_manager=True)

        r2 = engine_p2.run(signal_df, price_df)
        r3 = engine_p3.run(signal_df, price_df)

        assert r2.total_trades() == r3.total_trades()
        if r2.total_trades() > 0:
            assert abs(r2.final_equity() - r3.final_equity()) < 0.01


class TestPhase3SectorLimits:
    def test_fourth_tech_signal_rejected_by_portfolio_manager(self):
        """With 4 Tech BUY signals on same day, only 3 should execute."""
        cfg = _test_config()
        tickers = ["AAPL", "MSFT", "NVDA", "GOOG"]
        price_df = _flat_price_df(tickers, n_days=60)
        date = pd.bdate_range("2024-01-23", periods=1)[0]
        sigs = []
        for t, c in [("AAPL", 0.90), ("MSFT", 0.85), ("NVDA", 0.80), ("GOOG", 0.75)]:
            sigs.extend(_sig(t, "Technology", [date], confidence=c))
        signal_df = pd.DataFrame(sigs)

        engine = BacktestEngine(cfg, use_risk_engine=True, use_portfolio_manager=True)
        results = engine.run(signal_df, price_df)

        # Check rejection log for portfolio-level sector rejection
        portfolio_rejections = [
            r for r in results.rejected_signals
            if r.rejection_layer == "portfolio"
        ]
        sector_rejections = [
            r for r in portfolio_rejections
            if "Sector" in (r.rejection_reason or "")
        ]
        assert len(sector_rejections) >= 1
        assert sector_rejections[0].ticker == "GOOG"  # Lowest confidence


class TestPhase3Ranking:
    def test_highest_confidence_executes_first(self):
        """Verify rank order: highest confidence signal enters first."""
        cfg = _test_config()
        tickers = ["AAPL", "JPM"]
        price_df = _flat_price_df(tickers, n_days=60)
        date = pd.bdate_range("2024-01-23", periods=1)[0]
        sigs = [
            *_sig("JPM", "Financials", [date], confidence=0.90),
            *_sig("AAPL", "Technology", [date], confidence=0.70),
        ]
        signal_df = pd.DataFrame(sigs)

        engine = BacktestEngine(cfg, use_risk_engine=True, use_portfolio_manager=True)
        results = engine.run(signal_df, price_df)

        if results.total_trades() >= 2:
            trades = results.trades_df.sort_values("entry_date")
            first_entry = trades.iloc[0]
            assert first_entry["ticker"] == "JPM"  # Higher confidence
```

- [ ] **Step 2: Run Phase 3 integration tests**

Run: `venv\Scripts\python -m pytest tests/test_phase3_integration.py -v`
Expected: All pass

- [ ] **Step 3: Run FULL test suite to verify zero regression**

Run: `venv\Scripts\python -m pytest tests/ -v --tb=short 2>&1 | tail -10`
Expected: 487+ tests pass, 0 failures

- [ ] **Step 4: Commit**

```bash
git add tests/test_phase3_integration.py
git commit -m "test(phase3): end-to-end integration tests — Phase 3 vs Phase 2, sector limits, ranking"
```

---

## Task 7: Update Exports in `__init__.py`

**Agent:** Sonnet (small change)
**Depends on:** Task 5

**Files:**
- Modify: `trading_system/__init__.py`

- [ ] **Step 1: Add Phase 3 exports**

Add after line 28:

```python
from .portfolio_state import RankedSignal, AllocationDecision, PortfolioSnapshot
from .portfolio_manager import rank_signals, check_allocation, allocate_day
```

- [ ] **Step 2: Update module docstring**

Update the docstring to mention Layer 3:

```python
"""
FPPE Trading System v1 — Long-only backtesting and evaluation framework.

Design doc: FPPE_TRADING_SYSTEM_DESIGN.md v0.3

Exports:
    TradingConfig      — Master config dataclass. Use TradingConfig.from_profile()
                         for named risk profiles (aggressive/moderate/conservative).
    DEFAULT_CONFIG     — Pre-built instance with aggressive (default) settings.
    BacktestEngine     — Layer 1: Trade simulation with realistic friction.
    BacktestResults    — Output container returned by BacktestEngine.run().
    UnifiedSignal      — Normalized FPPE signal (all models speak this format).
    SignalDirection    — Enum: BUY / SELL / HOLD
    SignalSource       — Enum: KNN / DL / ENSEMBLE
    RankedSignal       — Layer 3: Signal annotated with rank and score.
    AllocationDecision — Layer 3: Portfolio allocation approval/rejection.
    PortfolioSnapshot  — Layer 3: Read-only portfolio state for allocation.
    SECTOR_MAP         — 52-ticker sector classification dict.
    ALL_TICKERS        — Sorted list of all 52 universe tickers.
"""
```

- [ ] **Step 3: Verify imports work**

Run: `venv\Scripts\python -c "from trading_system import RankedSignal, AllocationDecision, allocate_day; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add trading_system/__init__.py
git commit -m "feat(phase3): export portfolio_state and portfolio_manager from trading_system"
```

---

## Task 8: Full Regression Suite

**Agent:** Any (verification only)
**Depends on:** All prior tasks

- [ ] **Step 1: Run complete test suite**

Run: `venv\Scripts\python -m pytest tests/ -v`
Expected: 500+ tests pass (487 existing + ~45 new Phase 3 tests), 0 failures

- [ ] **Step 2: Verify Phase 1 mode unaffected**

Run: `venv\Scripts\python -m pytest tests/test_backtest_engine.py tests/test_review_fixes.py -v`
Expected: All pass with identical behavior (use_portfolio_manager defaults to False)

- [ ] **Step 3: Verify Phase 2 mode unaffected**

Run: `venv\Scripts\python -m pytest tests/test_phase2_integration.py tests/test_risk_engine.py -v`
Expected: All pass with identical behavior

- [ ] **Step 4: Final commit tag**

```bash
git tag -a phase3-portfolio-manager-v1 -m "Phase 3: Portfolio Manager — signal ranking, sector allocation, capital queue"
```

---

## Phase 4 Hooks — Forward-Looking Extensibility

Phase 3 is designed with the following extension points for Phase 4 (Strategy Evaluator) and beyond. These hooks exist in the code NOW but are not activated until later phases.

### Hook 1: `rank_score` Field in `RankedSignal`

**Current:** `rank_score = confidence` (v1 identity mapping)

**Phase 4+ use:** Replace with composite scoring:

```python
# v2+ example (NOT implemented in Phase 3):
rank_score = (
    0.40 * confidence_norm
  + 0.20 * sector_momentum_norm
  + 0.15 * signal_recency_norm
  + 0.10 * inv_volatility_norm
  + 0.10 * inv_correlation_norm
  + 0.05 * pattern_strength_norm  # Phase 6
)
```

**How to extend:** Create a `RankingStrategy` protocol or callable in `portfolio_manager.py`:

```python
# Phase 4 hook — add when composite scoring is needed
from typing import Protocol

class RankingStrategy(Protocol):
    def score(self, signal: dict, portfolio_state: PortfolioSnapshot) -> float: ...

class ConfidenceOnlyRanking:
    """v1 default."""
    def score(self, signal: dict, portfolio_state: PortfolioSnapshot) -> float:
        return signal["confidence"]

class CompositeRanking:
    """v2+ with configurable weights."""
    def __init__(self, weights: dict[str, float]): ...
    def score(self, signal: dict, portfolio_state: PortfolioSnapshot) -> float: ...
```

**No code changes needed now.** The `rank_score` field is already separated from `confidence`, so switching to composite scoring only requires changing the sort key in `rank_signals()`.

### Hook 2: `raw_metadata` Pass-Through

**Current:** Preserved from signal dict for audit trail.

**Phase 4+ use:** Strategy evaluator reads `raw_metadata` to compute performance attribution by signal factor:

```python
# Phase 4: analyze which metadata fields predict trade success
for trade in completed_trades:
    meta = trade.raw_metadata
    n_matches = meta.get("n_matches", 0)
    # Correlate n_matches with trade P&L...
```

**No code changes needed now.** Metadata flows through the full pipeline: UnifiedSignal → signal_df → portfolio_manager → backtest_engine → CompletedTrade.

### Hook 3: `AllocationDecision.rejection_layer`

**Current:** Always `"portfolio"` for portfolio-manager rejections.

**Phase 4+ use:** Strategy evaluator aggregates rejections by layer to identify systematic bottlenecks:

```python
# Phase 4: rejection analysis
rejection_layers = results.rejected_df.groupby("rejection_layer").size()
# → {"portfolio": 42, "risk_engine": 15, "capital": 8, "sector_limit": 23}
```

**No code changes needed now.** The `rejection_layer` field already exists in `RejectedSignal` (used since Phase 1).

### Hook 4: `PortfolioSnapshot` Extensibility

**Current:** Tracks open tickers, sector counts, cooldowns.

**Phase 4+ use:** Add fields for correlation matrix, factor exposures, or regime state:

```python
# Phase 4 extension (NOT implemented in Phase 3):
@dataclass(frozen=True)
class PortfolioSnapshot:
    open_tickers: frozenset[str]
    sector_position_counts: dict[str, int]
    cooldowns: dict[str, dict[str, Any]]
    cooldown_reentry_margin: float
    # Phase 4 additions:
    correlation_matrix: pd.DataFrame | None = None  # Pairwise correlations
    factor_exposures: dict[str, float] | None = None  # Factor tilts
    regime: str = "unknown"  # Current market regime
```

**No code changes needed now.** `PortfolioSnapshot` is a frozen dataclass — new optional fields can be added with defaults without breaking existing callers.

### Hook 5: `BacktestEngine` Layer Flags

**Current:** `use_risk_engine` (Phase 2), `use_portfolio_manager` (Phase 3).

**Phase 4+ use:** Add `use_strategy_evaluator` flag:

```python
# Phase 4 addition to BacktestEngine.__init__:
def __init__(
    self,
    config: TradingConfig = None,
    use_risk_engine: bool = False,
    use_portfolio_manager: bool = False,
    use_strategy_evaluator: bool = False,  # Phase 4
):
    if use_strategy_evaluator and not use_portfolio_manager:
        raise ValueError("Phase 4 requires Phase 3")
```

**No code changes needed now.** The flag pattern is established; adding another follows the same convention.

### Hook 6: `SharedState` (Phase 3 → Phase 4 Bridge)

The design doc specifies a `SharedState` object for inter-layer communication. Phase 3 does NOT implement SharedState (YAGNI — the `PortfolioSnapshot` is sufficient for Phase 3's read-only needs). However, Phase 4's strategy evaluator will need write access to shared state.

**When to implement:** At the start of Phase 4, create `trading_system/shared_state.py`:

```python
# Phase 4 — NOT implemented now
@dataclass
class SharedState:
    """Mutable shared state across all trading system layers."""
    risk_state: RiskState
    portfolio_snapshot: PortfolioSnapshot
    signal_commands: SignalCommands  # blocked tickers, sector blocks
    evaluation_state: EvaluationState  # rolling metrics, regime
```

**No code changes needed now.** Phase 3's `PortfolioSnapshot` is the read-only predecessor to Phase 4's mutable `SharedState`.

---

## Summary: Agent Assignment Matrix

| Task | Agent | Parallel Group | Est. Steps |
|------|-------|---------------|------------|
| 1. portfolio_state.py | Codex | A | 3 |
| 2. test_portfolio_state.py | Sonnet | A (after Task 1) | 3 |
| 3. portfolio_manager.py | Opus | B (after Task 1) | 5 |
| 4. test_portfolio_manager.py | Sonnet | B (after Task 3) | 3 |
| 5. backtest_engine.py integration | Opus | C (after Task 3) | 7 |
| 6. test_phase3_integration.py | Sonnet | C (after Task 5) | 4 |
| 7. __init__.py exports | Sonnet | C (after Task 5) | 4 |
| 8. Full regression suite | Any | D (after all) | 4 |

**Total: 8 tasks, ~33 steps, 3 parallel groups**
