# Phase 4 — Portfolio Manager Activation: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a stateless Portfolio Manager (Layer 3) into the Phase 3 walk-forward so that the system ranks competing BUY signals, enforces sector-diversification limits, and forwards only an approved subset to the risk engine for sizing — without introducing any behavioral change when the flag is off.

**Architecture:** Two free functions (`rank_signals`, `allocate_day`) in `trading_system/portfolio_manager.py` consume Pydantic schemas from `trading_system/portfolio_state.py`. The Phase 4 walk-forward replays the same `results/backtest_trades.csv` that Phase 3 did, but groups by `entry_date`, builds a `PortfolioSnapshot` from currently-open trades, runs the PM filter, and drops rejected candidates before the risk engine sizes survivors. `use_portfolio_manager` defaults to `False` so all existing tests and Phase 3 behavior are preserved byte-for-byte.

**Tech Stack:** Python 3.12, Pydantic v2 (frozen models), pandas, pytest. No new runtime dependencies. Matches Phase 2/3 style: stateless functions, `RuntimeError` for guards (not `assert`), frozen models.

**Source documents:**
- Pre-plan outline: `docs/superpowers/plans/2026-04-06-phase4-portfolio-manager-outline.md`
- Session handoff: `SESSION_HANDOFF_2026-04-08_phase4-execution.md` (in Downloads; treat as sequencing guide)
- Legacy reference (invariants only): `archive/legacy_v1/trading_system/portfolio_manager.py` + `portfolio_state.py`
- Production contracts referenced: `trading_system/signal_adapter.py` (`UnifiedSignal`), `trading_system/config.py` (`PositionLimitsConfig`), `trading_system/contracts/decisions.py`
- Structural template for walk-forward: `scripts/run_phase3_walkforward.py`

---

## 0. Preconditions (already satisfied as of 2026-04-08)

- Branch: `phase4-portfolio-manager` (created off clean `main`)
- `main` commits `04c64d2`, `cd3055d`, `9e209a2` merged (Phase 1 experiment infra, CLAUDE.md Phase 4 transition, session logs)
- `fix/phase3-risk-hardening` already merged (`ca64637`)
- Baseline tests: **696 passed, 1 skipped** (expected 678+)
- `data/52t_volnorm/val_db.parquet` present (Phase 3 walkforward depends on it)
- `results/backtest_trades.csv` (278 trades), `results/cached_signals_2024.csv` (13,104 rows) present
- `results/phase3_walkforward.tsv` exists (baseline for `--no-pm` diff)
- Test command: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`

## 1. Critical constraints (from CLAUDE.md + handoff)

1. Do NOT modify `prepare.py` or `CLAUDE.md` during execution (CLAUDE.md test-count update at session end is the only exception).
2. Do NOT modify locked settings (max_distance, confidence_threshold, horizon, regime mode, stop_loss_atr_multiple, etc.).
3. All public API guards use `RuntimeError`, never `assert`.
4. `nn_jobs=1` (Windows/Py3.12 joblib safety) — does not apply here but relevant if the script touches EngineConfig.
5. Run the full test suite after every task (command above).
6. 3-strike rule: three consecutive attempts at the same fix → STOP, log in todo list, escalate to the user.
7. Numbers require provenance: every metric in the session log must trace to a file in `results/` or a terminal output.
8. TDD per task: red → green → commit.
9. Commits: small, logical. Never use `git add -A` — always stage by explicit path.

## 2. Gate criteria (Phase 4 exit requirements — 2024 fold)

The handoff and the outline disagree on Sharpe. The plan adopts the **handoff's tighter gate** because Phase 3's 2024-fold baseline is 2.659 — requiring ≥ 2.0 means "don't lose more than ~25% of Phase 3's edge", which is the intent of "don't regress".

| ID | Metric | Bound | Source | Notes |
|---|---|---|---|---|
| G1 | Sector concentration | no sector > 30% equity at any point | handoff | Matches `PositionLimitsConfig.max_sector_pct=0.30` |
| G2 | Idle cash mean | < 35% | handoff | Mean over fold's daily equity curve |
| G3 | Idle cash p90 | < 70% | handoff | 90th percentile — tail-risk for starving the strategy |
| G4 | Sharpe (2024 fold) | ≥ 2.0 | handoff | Outline said ≥ 1.0; adopt handoff's tighter bound |
| G5 | MaxDD (2024 fold) | ≤ 10% | handoff | Phase 3 baseline 4.3% |
| G6 | Rejection reason diversity | no single reason > 60% of rejections | handoff | Sanity check — a flat distribution of rejections is healthier than a dominant cause |
| G7 | Tests | all 696 pass + ≥ 30 new | handoff | T4.0b target ≥ 8 tests, T4.0 target ≥ 25 tests |
| G8 | Runtime | < 5 min per fold | handoff | Phase 3 runs in seconds; PM is cheap |
| G9 | Zero-allocation days logged | present in output | handoff | Audit trail — days where PM rejected everything |

**Fallback protocol if G4 (Sharpe ≥ 2.0) fails:**
1. Run T4.3 rejection analysis first — is a single reason dominating? If yes, investigate whether the reason is too aggressive (e.g., `sector_pct_limit` blocking the best signals).
2. Run T4.4 (Phase 3 vs Phase 4 comparison) — are we dropping the high-confidence trades that Phase 3 sized up?
3. If PM is dropping too many high-conf trades: STOP and escalate. Do not loosen the limits without user approval (they were set by `PositionLimitsConfig` defaults that passed earlier reviews).
4. If the issue is elsewhere (e.g., the walk-forward wiring lost trades to missing data): fix the wiring bug; do not tune the PM.

## 3. File structure (what each file is responsible for)

### Files to create (Phase 4 proper)

| Path | Role | LOC target |
|---|---|---|
| `trading_system/portfolio_state.py` | Pydantic schemas: `PortfolioSnapshot`, `OpenPosition`, `RankedSignal`, `PMRejection`, `AllocationResult`. No business logic. | ~120 |
| `trading_system/portfolio_manager.py` | Two stateless free functions: `rank_signals`, `allocate_day`. Imports `PositionLimitsConfig` + `UnifiedSignal`. No class. | ~200 |
| `tests/test_portfolio_state.py` | ≥ 8 tests: schema validation, frozen semantics, edge cases. | ~180 |
| `tests/test_portfolio_manager.py` | ≥ 25 tests: ranking, tie-breaking, already-held, cooldown, sector count, sector pct, insufficient capital, zero-signal days, running-snapshot invariant. | ~450 |
| `scripts/reconcile_signals.py` | T4.0c gate: load cached 2024 signals, run matcher on 2024 fold, compare. Hard stop if parity fails. | ~130 |
| `scripts/run_phase4_walkforward.py` | Daily loop: trade-file replay → PM filter → risk engine → PnL. Mirrors `run_phase3_walkforward.py`'s structure. | ~400 |
| `scripts/analyze_pm_rejections.py` | T4.3: histograms by reason, sector, confidence bucket, top 10 tickers. | ~120 |
| `scripts/compare_phase3_vs_phase4.py` | T4.4: diff Sharpe / MaxDD / trade count between Phase 3 and Phase 4 runs. | ~80 |

### Files to modify

| Path | Change |
|---|---|
| `trading_system/config.py` | Add `use_portfolio_manager: bool = False` to `TradingConfig` (NOT to a sub-config — it's an orchestration-layer flag). Default `False` preserves Phase 3 behavior. |

### Files NOT to touch

- `pattern_engine/**` — Phase 4 is a trading-layer change only.
- `trading_system/risk_engine.py`, `position_sizer.py`, `strategy_evaluator.py` — already-working downstream components.
- `trading_system/contracts/decisions.py::AllocationDecision` — that's a post-sizing type. Phase 4 PM operates pre-sizing and must not collide.

### Design decision: why not reuse `contracts/decisions.py::AllocationDecision`?

The existing `AllocationDecision` in `contracts/decisions.py` encodes a **post-sizing** output — it has `final_position_pct`, `capital_allocated`, `evaluator_status`, `adjusted_for_evaluator`. The Phase 4 PM produces a **pre-sizing** output (ranked candidates that passed portfolio-level checks). These are distinct concerns. Putting them in the same model would force every rank output to carry sizing fields that aren't known yet. Keeping them separate (`AllocationResult` in `portfolio_state.py` for the PM output, `AllocationDecision` in `contracts/decisions.py` for the post-sizing record) is cleaner and matches Decision #2 in the pre-plan outline.

### Design decision: trade-file replay vs signal regeneration

The pre-plan outline (Decision #3) says Phase 4 should "regenerate signals from the matcher". The handoff's T4.1a pseudocode echoes this. **This plan overrides both** with a narrower scope for the following reasons:

1. Phase 3 walk-forward uses `results/backtest_trades.csv` (278 trades) — not signal regeneration. For `--no-pm` to reproduce Phase 3 exactly (T4.1b's hard requirement), Phase 4 **must** consume the same input.
2. Per-fold trade files don't exist: only 2024. Full regeneration across 6 folds is hours of runtime.
3. The PM is a *filter over ranked candidates*. Whether those candidates come from a cached file or a fresh matcher call is orthogonal to whether the PM logic is correct.
4. T4.0c reconciliation is still run (it verifies the matcher can reproduce the cached signals within tolerance — i.e. no drift from the `same_sector_boost_factor` hook commit). The signal file stays trusted.

**Impact on the plan:** `scripts/run_phase4_walkforward.py` groups `backtest_trades.csv` by `entry_date`, constructs a `UnifiedSignal`-equivalent candidate per row, runs PM filter → risk engine → surviving PnL. Zero-allocation days (G9) surface naturally when all candidates for a day are rejected.

**What we lose:** the PM cannot "unlock" signals Phase 1 suppressed because Phase 1 already decided which trades to take. If the user wants the PM to surface additional signals, that's a follow-up pass (T4.5 or a Phase 4.v2).

**Acceptance:** Flag this override in the session log. The user confirmed Option 1 (write formal plan) knowing both docs will be reconciled against reality.

## 4. Task sequencing

```
T4.0b (schemas) ──┐
                  ├──► T4.0 (PM core) ──► T4.0c (reconciliation gate) ──► T4.1a (wiring)
                  │                                                           │
                  │                                                           ▼
                  │                                                       T4.1b (flag)
                  │                                                           │
                  │                                                           ▼
                  │                                                        T4.2 (run)
                  │                                                           │
                  │                                       ┌───────────────────┤
                  │                                       ▼                   ▼
                  │                                    T4.3 (reject)      T4.4 (diff)
                  │                                       │                   │
                  └───────────────────────────────────────┴───────────────────┘
                                                          ▼
                                                  Session close + gate verdict
```

---

## Task 1 — T4.0b: Portfolio state schemas

**Files:**
- Create: `trading_system/portfolio_state.py`
- Create: `tests/test_portfolio_state.py`

**Rationale:** T4.0 (core) imports these schemas. Must exist first. Keeping data and logic in separate files matches the outline's Decision #2.

### Step 1.1 — Write the failing tests (test_portfolio_state.py)

- [ ] Create `tests/test_portfolio_state.py` with the following tests. Each test must be self-contained; use `pytest.raises` for validation errors.

```python
"""Tests for trading_system.portfolio_state — Phase 4 PM schemas."""
from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from trading_system.portfolio_state import (
    PortfolioSnapshot,
    OpenPosition,
    RankedSignal,
    PMRejection,
    AllocationResult,
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
    # Exposure is a sum of position_pct per sector.
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
```

- [ ] **Step 1.2 — Run tests; expect collection error (module does not exist)**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/test_portfolio_state.py -v
```
Expected: `ImportError` / `ModuleNotFoundError: No module named 'trading_system.portfolio_state'`.

- [ ] **Step 1.3 — Create `trading_system/portfolio_state.py`**

```python
"""
trading_system/portfolio_state.py — Phase 4 Portfolio Manager data schemas.

Pydantic v2 frozen models consumed by trading_system/portfolio_manager.py.
No business logic — schemas and computed properties only. All models are
immutable (model_config = {"frozen": True}) following the Phase 2/3 pattern.

Types:
    OpenPosition       — one currently-held position
    PortfolioSnapshot  — read-only snapshot of portfolio state on a given day
    RankedSignal       — a BUY candidate with its rank in the day's queue
    PMRejection        — structured reason a ranked signal failed portfolio checks
    AllocationResult   — final per-signal verdict (approved + optional rejection)

Design (see plan §3):
    - Distinct from trading_system/contracts/decisions.py::AllocationDecision
      which is a post-sizing output. These types are PRE-sizing.
    - `PMRejection.reason` is a Literal with 5 values so downstream histograms
      (T4.3) can't be polluted by free-form strings.
    - `PortfolioSnapshot.open_positions` is a tuple (hashable, order-stable)
      so Pydantic can freeze the model.

Plan: docs/superpowers/plans/2026-04-08-phase4-portfolio-manager-plan.md
"""
from __future__ import annotations

from datetime import date as Date
from typing import Dict, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


# ── OpenPosition ──────────────────────────────────────────────────────────────

class OpenPosition(BaseModel):
    """One currently-held position for portfolio-state purposes.

    This is a lightweight mirror of the trade row — only the fields the
    Portfolio Manager needs to reason about sector concentration and
    already-held checks. It is NOT a replacement for the trade record.
    """
    model_config = {"frozen": True}

    ticker: str = Field(min_length=1, max_length=10)
    sector: str = Field(min_length=1)
    entry_date: Date
    position_pct: float = Field(ge=0.0, le=1.0)
    entry_price: float = Field(gt=0.0)

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase; got '{v}'")
        return v


# ── PortfolioSnapshot ─────────────────────────────────────────────────────────

class PortfolioSnapshot(BaseModel):
    """Read-only snapshot of portfolio state on one trading day.

    Constructed by the walk-forward script at the start of each day before
    running the PM. Must be reconstructed (not mutated) after any approved
    allocation within the day — the `allocate_day` function does this
    internally via a running snapshot.

    Computed properties (sector_counts, sector_exposure_pct, cash_pct) are
    derived from open_positions + equity/cash and cached on first access.
    """
    model_config = {"frozen": True}

    as_of_date: Date
    equity: float = Field(gt=0.0, description="Total portfolio equity (cash + positions)")
    cash: float = Field(ge=0.0, description="Available cash")
    open_positions: Tuple[OpenPosition, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def cash_must_not_exceed_equity(self) -> "PortfolioSnapshot":
        if self.cash > self.equity:
            raise ValueError(
                f"cash ({self.cash}) must not exceed equity ({self.equity})"
            )
        return self

    @property
    def n_open_positions(self) -> int:
        return len(self.open_positions)

    @property
    def sector_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in self.open_positions:
            counts[p.sector] = counts.get(p.sector, 0) + 1
        return counts

    @property
    def sector_exposure_pct(self) -> Dict[str, float]:
        exposure: Dict[str, float] = {}
        for p in self.open_positions:
            exposure[p.sector] = exposure.get(p.sector, 0.0) + p.position_pct
        return exposure

    @property
    def cash_pct(self) -> float:
        return self.cash / self.equity if self.equity > 0 else 0.0

    def contains_ticker(self, ticker: str) -> bool:
        return any(p.ticker == ticker for p in self.open_positions)


# ── RankedSignal ──────────────────────────────────────────────────────────────

class RankedSignal(BaseModel):
    """A BUY candidate with its 1-based rank in the day's priority queue.

    v1 ranking: sort by confidence descending, ticker ascending (deterministic
    tie-break). v2+ may introduce composite scores; the schema is stable.
    """
    model_config = {"frozen": True}

    ticker: str = Field(min_length=1, max_length=10)
    sector: str = Field(min_length=1)
    signal_date: Date
    confidence: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1, description="1-based rank; 1 = highest priority")

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase; got '{v}'")
        return v


# ── PMRejection ───────────────────────────────────────────────────────────────

PMRejectionReason = Literal[
    "already_held",
    "cooldown",
    "sector_count_limit",
    "sector_pct_limit",
    "insufficient_capital",
]


class PMRejection(BaseModel):
    """Structured reason the PM rejected a ranked signal.

    The `reason` field is a Literal so T4.3 histograms are bounded. The
    `detail` field is a free-form string for audit context (e.g.,
    'sector Technology at 2/3 positions', 'cash 0.02 < min 0.02').
    """
    model_config = {"frozen": True}

    ticker: str = Field(min_length=1, max_length=10)
    sector: str = Field(min_length=1)
    signal_date: Date
    confidence: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    reason: PMRejectionReason
    detail: str = Field(min_length=1)

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase; got '{v}'")
        return v


# ── AllocationResult ──────────────────────────────────────────────────────────

class AllocationResult(BaseModel):
    """Per-signal verdict from allocate_day().

    Exactly one of the following is true:
      - approved=True and rejection=None
      - approved=False and rejection is not None

    The walk-forward script filters the returned list to approved=True to
    get the pass-through set for the risk engine.
    """
    model_config = {"frozen": True}

    ticker: str = Field(min_length=1, max_length=10)
    sector: str = Field(min_length=1)
    signal_date: Date
    confidence: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    approved: bool
    rejection: Optional[PMRejection] = None

    @model_validator(mode="after")
    def approval_state_consistency(self) -> "AllocationResult":
        if self.approved and self.rejection is not None:
            raise ValueError(
                "approved=True must not have a rejection attached"
            )
        if not self.approved and self.rejection is None:
            raise ValueError(
                "approved=False must have a rejection attached"
            )
        return self

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase; got '{v}'")
        return v
```

- [ ] **Step 1.4 — Run tests; expect all green**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/test_portfolio_state.py -v
```
Expected: all 18 tests pass (17 written above + any additions to reach ≥ 8 minimum; plan overshoots to 18 for coverage).

- [ ] **Step 1.5 — Run full test suite**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"
```
Expected: `714 passed, 1 skipped` (696 + 18 new).

- [ ] **Step 1.6 — Commit**

```bash
git add trading_system/portfolio_state.py tests/test_portfolio_state.py
git commit -m "feat(phase4): add portfolio_state schemas (T4.0b)

Pydantic v2 frozen models for the Phase 4 Portfolio Manager:
- OpenPosition, PortfolioSnapshot (with computed sector_counts,
  sector_exposure_pct, cash_pct)
- RankedSignal, PMRejection (Literal reasons), AllocationResult

No business logic — schemas only. 18 tests."
```

---

## Task 2 — T4.0: Portfolio Manager core

**Files:**
- Create: `trading_system/portfolio_manager.py`
- Create: `tests/test_portfolio_manager.py`

**Depends on:** T4.0b

**Rationale:** Stateless free functions implementing the ranking and allocation logic. Imports `PositionLimitsConfig` from `trading_system/config.py` and the schemas from T4.0b. No class. Matches the Phase 3 `risk_engine.py` style.

### API contract (locked)

```python
def rank_signals(
    buy_signals: list[UnifiedSignal],
) -> list[RankedSignal]:
    """Rank BUY candidates by confidence descending, ticker ascending.

    Returns an empty list if input is empty. Raises RuntimeError if any
    signal has signal != SignalDirection.BUY (caller must filter first).
    """


def allocate_day(
    ranked_signals: list[RankedSignal],
    snapshot: PortfolioSnapshot,
    limits: PositionLimitsConfig,
    min_position_pct: float = 0.02,
) -> list[AllocationResult]:
    """Apply portfolio-level constraints to each ranked signal in rank order.

    Check priority (first failure wins):
      1. already_held       — ticker in snapshot.open_positions
      2. cooldown           — v1: NOT CHECKED (no cooldown state in snapshot;
                              caller must pre-filter if cooldown matters)
      3. sector_count_limit — running_sector_counts[sector] >= limits.max_positions_per_sector
      4. sector_pct_limit   — running_sector_exposure_pct[sector] + min_position_pct > limits.max_sector_pct
      5. insufficient_capital — running_cash_pct < min_position_pct

    Uses a RUNNING snapshot: approvals earlier in the day are added to
    running state before checking the next signal. Deterministic given
    deterministic ranking.

    Returns one AllocationResult per input signal — approved and rejected
    alike. Caller filters to approved=True for the pass-through set.
    """
```

**Cooldown note:** v1 does NOT implement cooldowns. The walk-forward runs on `backtest_trades.csv` which already encodes entry/exit timing — cooldowns live in the trade-generation layer, not the PM filter layer. The `PMRejectionReason` Literal still lists `"cooldown"` so a future version can use it.

### Step 2.1 — Write the failing tests (test_portfolio_manager.py)

- [ ] Create `tests/test_portfolio_manager.py` with at least **26 tests** organized into 5 sections. Full body below.

```python
"""Tests for trading_system.portfolio_manager — Phase 4 PM core."""
from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from pattern_engine.contracts.signals import SignalDirection, SignalSource
from trading_system.config import PositionLimitsConfig
from trading_system.portfolio_manager import allocate_day, rank_signals
from trading_system.portfolio_state import (
    AllocationResult,
    OpenPosition,
    PortfolioSnapshot,
    RankedSignal,
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


def _empty_snapshot(equity: float = 10_000.0, cash: float = 10_000.0) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of_date=date(2024, 1, 2),
        equity=equity, cash=cash, open_positions=(),
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


def _mk_pos(ticker: str, sector: str, pct: float = 0.05) -> OpenPosition:
    return OpenPosition(
        ticker=ticker, sector=sector,
        entry_date=date(2024, 1, 2),
        position_pct=pct, entry_price=100.0,
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
        # Sector already at 3 positions (limits.max_positions_per_sector=3)
        ranked = rank_signals([_mk_signal("NVDA", "Technology", 0.80)])
        result = allocate_day(ranked, snap, LIMITS)
        assert result[0].approved is False
        assert result[0].rejection.reason == "sector_count_limit"

    def test_sector_pct_limit_rejected(self):
        # 3 positions in Technology at 9% each = 27% exposure.
        # Adding min_position_pct=0.02 would exceed max_sector_pct=0.30.
        snap = _snapshot_with(
            _mk_pos("AAPL", "Technology", pct=0.09),
            _mk_pos("MSFT", "Technology", pct=0.09),
            _mk_pos("GOOG", "Technology", pct=0.09),
            cash=7_300.0, equity=10_000.0,
        )
        # max_positions_per_sector=3 — so we'd fail count_limit first.
        # Use custom limits to isolate pct_limit. max_positions=10, max_pct=0.30
        limits = PositionLimitsConfig(
            min_position_pct=0.02, max_position_pct=0.10,
            max_sector_pct=0.30, max_positions_per_sector=10,
        )
        ranked = rank_signals([_mk_signal("NVDA", "Technology", 0.80)])
        result = allocate_day(ranked, snap, limits)
        assert result[0].approved is False
        assert result[0].rejection.reason == "sector_pct_limit"

    def test_insufficient_capital_rejected(self):
        # 98% cash committed — only 2% free, which equals min_position_pct.
        # Signal needs MORE than min_position_pct to be admissible.
        snap = PortfolioSnapshot(
            as_of_date=date(2024, 1, 2),
            equity=10_000.0, cash=100.0,  # 1% cash
            open_positions=(),  # no positions implied — edge case for testing
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
        # Technology sector starts at 2/3 positions. First approved tech
        # signal pushes it to 3/3. Second tech signal must be rejected.
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

    def test_running_sector_pct_accounts_for_min_alloc_of_today_approvals(self):
        # Tech at 20% exposure, max 30%. Each new approval burns at least
        # min_position_pct=0.02 of cap. Use max_positions_per_sector=10 to
        # isolate the pct gate.
        limits = PositionLimitsConfig(
            min_position_pct=0.02, max_position_pct=0.10,
            max_sector_pct=0.30, max_positions_per_sector=10,
        )
        snap = _snapshot_with(
            _mk_pos("AAPL", "Technology", pct=0.10),
            _mk_pos("MSFT", "Technology", pct=0.10),
            cash=8_000.0, equity=10_000.0,
        )
        # 20% exposure; room for 5 more signals at 0.02 each before hitting 0.30.
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
        # cash = 50%, min_position=20% — only 2 approvals fit.
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
            not r.approved and r.rejection.reason == "insufficient_capital"
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
        # Unknown sector behaves like any other sector for count/pct checks.
        assert result[0].approved is True

    def test_raises_on_negative_min_position_pct(self):
        ranked = rank_signals([_mk_signal("AAPL", "Technology", 0.72)])
        with pytest.raises(RuntimeError):
            allocate_day(ranked, _empty_snapshot(), LIMITS, min_position_pct=-0.01)
```

- [ ] **Step 2.2 — Run tests; expect collection error**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/test_portfolio_manager.py -v
```
Expected: `ModuleNotFoundError: No module named 'trading_system.portfolio_manager'`.

- [ ] **Step 2.3 — Create `trading_system/portfolio_manager.py`**

```python
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
    1. already_held       — snapshot.contains_ticker(ticker)
    2. sector_count_limit — running_sector_counts[sector] >= max_positions_per_sector
    3. sector_pct_limit   — running_sector_exposure + min_position_pct > max_sector_pct
    4. insufficient_capital — running_cash_pct < min_position_pct

Plan: docs/superpowers/plans/2026-04-08-phase4-portfolio-manager-plan.md
"""
from __future__ import annotations

from datetime import date as Date
from typing import Dict, List

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

def rank_signals(buy_signals: List[UnifiedSignal]) -> List[RankedSignal]:
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
    ranked_signals: List[RankedSignal],
    snapshot: PortfolioSnapshot,
    limits: PositionLimitsConfig,
    min_position_pct: float = 0.02,
) -> List[AllocationResult]:
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
    """
    if min_position_pct < 0.0:
        raise RuntimeError(
            f"min_position_pct must be >= 0, got {min_position_pct}"
        )

    if not ranked_signals:
        return []

    # Build running state from the snapshot. Tuples in snapshot are
    # immutable; we maintain parallel mutable mirrors for the loop.
    running_tickers: set[str] = {p.ticker for p in snapshot.open_positions}
    running_sector_counts: Dict[str, int] = dict(snapshot.sector_counts)
    running_sector_exposure: Dict[str, float] = dict(snapshot.sector_exposure_pct)
    running_cash_pct: float = snapshot.cash_pct

    results: List[AllocationResult] = []

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
```

- [ ] **Step 2.4 — Run PM tests**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/test_portfolio_manager.py -v
```
Expected: all 26 tests pass.

- [ ] **Step 2.5 — Run full test suite**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"
```
Expected: `740 passed, 1 skipped` (714 + 26 new).

- [ ] **Step 2.6 — Commit**

```bash
git add trading_system/portfolio_manager.py tests/test_portfolio_manager.py
git commit -m "feat(phase4): add portfolio_manager core (T4.0)

Stateless PM with two free functions:
- rank_signals: confidence DESC + ticker ASC tie-break
- allocate_day: running-snapshot filter with 4 rejection reasons
  (already_held, sector_count_limit, sector_pct_limit,
  insufficient_capital). Cooldown reserved for v2.

26 tests covering ranking, rejection paths, running-snapshot
invariant, determinism, and edge cases."
```

---

## Task 3 — T4.0c: Signal reconciliation gate

**Files:**
- Create: `scripts/reconcile_signals.py`
- No tests (it's a diagnostic script, not a library)

**Depends on:** T4.0 (for import lineage sanity), T4.0b

**Rationale:** T4.0c is a hard gate. We committed a `same_sector_boost_factor` hook to `pattern_engine/matcher.py` earlier in the session — we need to prove it didn't drift the signals vs. the cached file. If this gate fails, STOP.

### Acceptance thresholds (from handoff)

- BUY ticker overlap ≥ 95% (same tickers flagged BUY on same dates)
- Confidence RMSE < 0.01 across all matched rows

### Step 3.1 — Create `scripts/reconcile_signals.py`

- [ ] Create the script exactly as below.

```python
"""
scripts/reconcile_signals.py — T4.0c: matcher parity against cached signals.

Hard gate. Confirms that the current matcher (with the committed
`same_sector_boost_factor` hook defaulting to 1.0) reproduces
`results/cached_signals_2024.csv` within tolerance.

Thresholds (from session handoff):
    BUY ticker overlap   ≥ 95%
    Confidence RMSE       < 0.01

If either threshold fails: prints the diagnostic and exits with
code 2. Do NOT continue with Phase 4 execution.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/reconcile_signals.py

Plan: docs/superpowers/plans/2026-04-08-phase4-portfolio-manager-plan.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from trading_system.signal_adapter import simulate_signals_from_val_db

CACHED_PATH   = project_root / "results" / "cached_signals_2024.csv"
VAL_PARQUET   = project_root / "data" / "52t_volnorm" / "val_db.parquet"
TRAIN_PARQUET = project_root / "data" / "52t_volnorm" / "train_db.parquet"

OVERLAP_MIN   = 0.95
RMSE_MAX      = 0.01


def main() -> int:
    print("=" * 70)
    print("T4.0c — Signal Reconciliation Gate")
    print("=" * 70)

    if not CACHED_PATH.exists():
        print(f"  FATAL: cached signals not found at {CACHED_PATH}")
        return 2
    if not VAL_PARQUET.exists() or not TRAIN_PARQUET.exists():
        print(f"  FATAL: 52T parquet files missing at {VAL_PARQUET.parent}")
        return 2

    print("\n[1/4] Loading cached signals...")
    cached = pd.read_csv(CACHED_PATH, parse_dates=["date"])
    print(f"  {len(cached):,} rows, {cached['ticker'].nunique()} tickers")
    cached_buys = cached[cached["signal"] == "BUY"]
    print(f"  {len(cached_buys):,} BUY rows")

    print("\n[2/4] Loading train/val dbs...")
    train_db = pd.read_parquet(TRAIN_PARQUET)
    val_db = pd.read_parquet(VAL_PARQUET)
    print(f"  train={len(train_db):,}  val={len(val_db):,}")

    # Use the same sector_map build the trading system uses. For parity we
    # read it off val_db itself (simulate_signals_from_val_db only needs
    # the map to populate the `sector` column).
    sector_map = dict(
        val_db.drop_duplicates("Ticker")[["Ticker", "Sector"]].values
    ) if "Sector" in val_db.columns else {}

    print("\n[3/4] Regenerating signals via PatternMatcher...")
    regen = simulate_signals_from_val_db(
        val_db=val_db,
        train_db=train_db,
        sector_map=sector_map,
        confidence_threshold=0.65,
        min_matches=10,
    )
    regen["date"] = pd.to_datetime(regen["date"])
    regen_buys = regen[regen["signal"] == "BUY"]
    print(f"  regenerated {len(regen):,} rows, {len(regen_buys):,} BUY")

    print("\n[4/4] Comparing...")

    # Join on (date, ticker) for confidence RMSE.
    merged = cached.merge(
        regen, on=["date", "ticker"],
        suffixes=("_cached", "_regen"), how="inner",
    )
    if merged.empty:
        print("  FATAL: no overlapping (date, ticker) rows between cached and regen")
        return 2

    rmse = float(np.sqrt(np.mean(
        (merged["confidence_cached"] - merged["confidence_regen"]) ** 2
    )))

    # BUY overlap: set intersection over set union of (date, ticker) BUY pairs.
    cached_buy_keys = set(map(tuple, cached_buys[["date", "ticker"]].values))
    regen_buy_keys = set(map(tuple, regen_buys[["date", "ticker"]].values))
    if not cached_buy_keys:
        print("  FATAL: cached file has no BUY signals")
        return 2
    overlap = len(cached_buy_keys & regen_buy_keys) / len(cached_buy_keys)

    print(f"  Confidence RMSE       : {rmse:.6f}  (threshold < {RMSE_MAX})")
    print(f"  BUY ticker overlap    : {overlap:.4f}  (threshold ≥ {OVERLAP_MIN})")
    print(f"  Cached BUYs           : {len(cached_buy_keys)}")
    print(f"  Regenerated BUYs      : {len(regen_buy_keys)}")
    print(f"  Matched BUYs          : {len(cached_buy_keys & regen_buy_keys)}")

    passed = (rmse < RMSE_MAX) and (overlap >= OVERLAP_MIN)
    print("\n" + ("RECONCILIATION PASSED" if passed else "RECONCILIATION FAILED"))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3.2 — Run the reconciliation gate**

```bash
PYTHONUTF8=1 py -3.12 scripts/reconcile_signals.py
```
Expected: `RECONCILIATION PASSED` with RMSE well under 0.01 and overlap ≥ 0.95. The `same_sector_boost_factor` is 1.0 by default, so the matcher output should be byte-identical to the cached file modulo floating-point noise.

**If it FAILS: STOP.** Capture the output in the session log and escalate. Candidate diagnoses to investigate, in order:
1. Matcher's default config changed between cache generation and today (unlikely — `pattern_engine/config.py` just got committed; check its defaults vs the cache header).
2. Floating-point noise from library updates (check `pip list` diff).
3. The `same_sector_boost_factor` hook accidentally runs even with boost=1.0 (grep matcher.py for the guard).

- [ ] **Step 3.3 — Full test suite regression check**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"
```
Expected: still `740 passed, 1 skipped` — the reconciliation script has no tests, so count is unchanged.

- [ ] **Step 3.4 — Commit**

```bash
git add scripts/reconcile_signals.py
git commit -m "feat(phase4): add signal reconciliation gate (T4.0c)

Hard gate confirming PatternMatcher output parity against
results/cached_signals_2024.csv. Thresholds: BUY overlap >= 95%,
confidence RMSE < 0.01. Exits code 2 on failure.

Run: PYTHONUTF8=1 py -3.12 scripts/reconcile_signals.py"
```

---

## Task 4 — T4.1a: Walk-forward wiring

**Files:**
- Create: `scripts/run_phase4_walkforward.py`
- Modify: none (reads `results/backtest_trades.csv`, writes to `results/phase4_*`)

**Depends on:** T4.0, T4.0b, T4.0c passed

**Rationale:** The walk-forward script reuses the Phase 3 trade-replay pattern (per plan §3 design decision). It groups trades by `entry_date`, builds a `PortfolioSnapshot` from currently-open trades, runs the PM filter, and rescales surviving trade PnLs through the Phase 3 risk engine. Zero-allocation days (G9) surface when `allocate_day` rejects everything.

### Output files (5 required per handoff G9)

| Path | Contents |
|---|---|
| `results/phase4_walkforward.tsv` | Single-row fold summary: Sharpe, MaxDD, final_equity, n_placed, n_rejected, n_blocked |
| `results/phase4_equity_curve.csv` | Daily equity curve |
| `results/phase4_allocations.csv` | One row per AllocationResult (approved + rejected) for audit |
| `results/phase4_rejections.csv` | Filtered view: only rejections, with reason + detail |
| `results/phase4_gate_check.txt` | G1–G9 verdict |

### Step 4.1 — Create the walk-forward script

- [ ] Create `scripts/run_phase4_walkforward.py`. Because this is ~400 lines, the plan provides the structure + key functions and defers full body to implementation. The subagent / executor must follow the structure exactly and use Phase 3 as a template for the bits marked `[mirror phase 3]`.

**Script skeleton (structural guide — implementor fills in bodies):**

```python
"""
scripts/run_phase4_walkforward.py — Phase 4 walk-forward with Portfolio Manager.

Extends the Phase 3 trade-replay pattern with a per-day Portfolio Manager
filter:

    Phase 3:  trades.csv -> risk engine (per trade) -> PnL
    Phase 4:  trades.csv -> group by entry_date
                         -> build PortfolioSnapshot from still-open trades
                         -> rank_signals + allocate_day (per day)
                         -> risk engine (per approved) -> PnL

Gates G1-G9 (see docs/superpowers/plans/2026-04-08-phase4-portfolio-manager-plan.md §2).

Usage:
    PYTHONUTF8=1 py -3.12 scripts/run_phase4_walkforward.py
    PYTHONUTF8=1 py -3.12 scripts/run_phase4_walkforward.py --no-pm
    PYTHONUTF8=1 py -3.12 scripts/run_phase4_walkforward.py --fold 2024
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from pattern_engine.contracts.signals import SignalDirection, SignalSource
from trading_system.config import PositionLimitsConfig
from trading_system.portfolio_manager import allocate_day, rank_signals
from trading_system.portfolio_state import (
    AllocationResult,
    OpenPosition,
    PortfolioSnapshot,
)
from trading_system.signal_adapter import UnifiedSignal
from trading_system.position_sizer import SizingConfig, size_position
from trading_system.risk_engine import apply_risk_adjustments, compute_atr_pct
from trading_system.risk_overlays.liquidity_congestion import LiquidityCongestionGate


TRADES_PATH   = project_root / "results" / "backtest_trades.csv"
_52T_DIR      = project_root / "data" / "52t_volnorm"
RESULTS_DIR   = project_root / "results"

OUT_WF_TSV    = RESULTS_DIR / "phase4_walkforward.tsv"
OUT_EQUITY    = RESULTS_DIR / "phase4_equity_curve.csv"
OUT_ALLOC     = RESULTS_DIR / "phase4_allocations.csv"
OUT_REJECT    = RESULTS_DIR / "phase4_rejections.csv"
OUT_GATE      = RESULTS_DIR / "phase4_gate_check.txt"
OUT_ZERO_DAYS = RESULTS_DIR / "phase4_zero_allocation_days.csv"

INITIAL_EQUITY   = 10_000.0
TRADE_DAYS_YEAR  = 252
RISK_FREE_ANNUAL = 0.045
MIN_POSITION_PCT = 0.02


# ── Metrics (identical to Phase 3 — factor out later if both scripts stay)

def _sharpe(daily_ret: np.ndarray) -> float:
    ...  # mirror run_phase3_walkforward._sharpe


def _max_dd(equity: np.ndarray) -> float:
    ...  # mirror run_phase3_walkforward._max_dd


def _compute_b_hist(trades: pd.DataFrame) -> float:
    ...  # mirror run_phase3_walkforward._compute_b_hist


def _build_atr_lookup(val_data: pd.DataFrame) -> dict:
    ...  # mirror run_phase3_walkforward._build_atr_lookup


# ── Phase 4: PM filter over trade-replay ──────────────────────────────────────

@dataclass
class _DayContext:
    day: Date
    candidates: pd.DataFrame        # trade rows with entry_date == day
    snapshot: PortfolioSnapshot     # open positions at start of day


def _build_snapshot(
    day: Date,
    open_positions: List[OpenPosition],
    equity: float,
) -> PortfolioSnapshot:
    committed = sum(p.position_pct for p in open_positions) * equity
    cash = max(0.0, equity - committed)
    return PortfolioSnapshot(
        as_of_date=day,
        equity=equity,
        cash=cash,
        open_positions=tuple(open_positions),
    )


def _trade_row_to_signal(row: pd.Series) -> UnifiedSignal:
    return UnifiedSignal(
        date=pd.Timestamp(row["entry_date"]).date(),
        ticker=row["ticker"],
        signal=SignalDirection.BUY,
        confidence=float(row["confidence_at_entry"]),
        signal_source=SignalSource.KNN,
        sector=row.get("sector") or "Unknown",
    )


def _phase4_replay(
    trades: pd.DataFrame,
    b_hist: float,
    atr_lookup: dict,
    limits: PositionLimitsConfig,
    use_pm: bool,
) -> tuple[pd.DataFrame, List[AllocationResult], List[Date]]:
    """Group trades by entry_date, apply PM filter, rescale through risk engine.

    Returns:
        scaled_trades (DataFrame with phase4_position_pct, phase4_net_pnl, phase4_blocked)
        all_allocations (for audit)
        zero_allocation_days (list of dates where PM rejected everything)
    """
    # Structure:
    #   1. Sort trades by entry_date.
    #   2. Loop days in ascending order:
    #      a. Realize exits <= today (update running_equity / peak_equity).
    #      b. Build open_positions from trades with entry_date <= today <
    #         exit_date and already-approved-and-placed.
    #      c. Build PortfolioSnapshot.
    #      d. Gather today's candidates (trade rows with entry_date == today).
    #      e. If use_pm: rank_signals + allocate_day. Keep approved.
    #         Else: all candidates pass through (Phase 3 behavior).
    #      f. For each passing candidate: size_position + apply_risk_adjustments.
    #         Scale the original trade's net_pnl by adj.final_position_pct /
    #         original_pct. Add to pending_pnls on exit_date.
    #      g. If use_pm and no candidates approved on a day with >=1 candidate:
    #         append to zero_allocation_days.
    ...


# ── Gate check ────────────────────────────────────────────────────────────────

def _gate_check(
    summary: dict,
    allocations: List[AllocationResult],
    equity_curve: pd.DataFrame,
    fold: str,
    runtime_sec: float,
) -> tuple[bool, str]:
    # G1: sector exposure
    # G2: idle cash mean
    # G3: idle cash p90
    # G4: Sharpe >= 2.0
    # G5: MaxDD <= 10%
    # G6: no rejection reason > 60%
    # G7: external (test suite runs separately)
    # G8: runtime < 5 min
    # G9: zero-alloc days logged (presence check — file written)
    ...


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pm", action="store_true",
                        help="Disable PM — reproduces Phase 3 exactly.")
    parser.add_argument("--fold", default="2024",
                        help="Fold label (only 2024 is supported in this plan).")
    args = parser.parse_args()

    use_pm = not args.no_pm
    # ... load trades, val_data, compute b_hist, atr_lookup ...
    # ... run _phase4_replay(use_pm=use_pm) ...
    # ... build equity curve ...
    # ... write the 5 required output files ...
    # ... run _gate_check and print verdict ...
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
```

**Implementation notes for the executor:**

1. **Opening a position:** when a trade is approved via PM, add an `OpenPosition` to a running list keyed by `exit_date` so it can be removed later. Position_pct should be `adj.final_position_pct` (post-risk-engine), not the trade's original.
2. **Closing a position:** when advancing days, remove `OpenPosition` entries whose `exit_date <= today`. Realize the pending PnL into running_equity.
3. **Rejected trades still write to allocations DataFrame** — this feeds T4.3.
4. **`--no-pm` mode:** skip the `rank_signals` / `allocate_day` calls entirely. Every candidate goes through the risk engine. This must reproduce `results/phase3_walkforward.tsv` exactly (within float tolerance).
5. **Zero-allocation days:** a day is "zero-allocation" if `len(candidates) >= 1` AND `len(approved) == 0`.
6. **Equity curve:** mirror Phase 3's `_build_equity_curve` — business-day calendar, exit_date PnL aggregation.

### Step 4.2 — Validate `--no-pm` mode against Phase 3 baseline

- [ ] Run Phase 4 walk-forward with PM disabled:

```bash
PYTHONUTF8=1 py -3.12 scripts/run_phase4_walkforward.py --no-pm
```

- [ ] Compare against Phase 3 baseline. The summary row should match within 1e-6:

```bash
cat results/phase3_walkforward.tsv
cat results/phase4_walkforward.tsv
```

Key fields to compare: `sharpe`, `max_dd`, `final_equity`, `n_trades_placed`, `n_blocked`. `--no-pm` mode is the parity contract from the handoff. **If they don't match: STOP and diagnose.** Likely causes:
- Off-by-one in exit-date realization.
- A trade row silently dropped by the new day-grouping logic.
- Risk engine called with a different `SizingConfig` by accident.

### Step 4.3 — Run Phase 4 walk-forward with PM enabled

- [ ] Run the actual Phase 4 execution:

```bash
PYTHONUTF8=1 py -3.12 scripts/run_phase4_walkforward.py
```

Expected: produces all 5 output files, prints gate verdict. Runtime well under 5 minutes. Capture the terminal output.

### Step 4.4 — Full test suite

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"
```
Expected: `740 passed, 1 skipped` (unchanged — walk-forward script has no tests).

### Step 4.5 — Commit

- [ ] Commit walk-forward script + results. Results files are in `.gitignore` (`results/*.csv|tsv|txt|json`), so commit only the script.

```bash
git add scripts/run_phase4_walkforward.py
git commit -m "feat(phase4): add Phase 4 walk-forward with PM filter (T4.1a)

Per-day loop over results/backtest_trades.csv:
  build snapshot -> rank_signals -> allocate_day -> risk engine -> PnL

--no-pm flag reproduces Phase 3 exactly. Writes 5 output files:
phase4_walkforward.tsv, phase4_equity_curve.csv, phase4_allocations.csv,
phase4_rejections.csv, phase4_gate_check.txt, phase4_zero_allocation_days.csv.
Gates G1-G9 evaluated per plan §2."
```

---

## Task 5 — T4.1b: Config flag

**Files:**
- Modify: `trading_system/config.py`

**Depends on:** T4.1a

**Rationale:** The handoff wants a `use_portfolio_manager` flag in the config. Since the CLI already has `--no-pm`, the flag is for callers that drive via `TradingConfig` directly. Default `False` preserves Phase 3 behavior.

### Step 5.1 — Locate the TradingConfig

- [ ] Read `trading_system/config.py` to confirm `TradingConfig` is a frozen dataclass with sub-configs via `field(default_factory=...)`.

### Step 5.2 — Add the flag as a top-level field

- [ ] Add `use_portfolio_manager: bool = False` directly to `TradingConfig` (not to a sub-config — it's an orchestration-layer flag, not a position/risk/cost concern). Place it after `research_flags` for consistency with the frozen-dataclass pattern.

```python
# In TradingConfig:
    use_portfolio_manager: bool = False   # Phase 4: enable PM filter (default off)
```

### Step 5.3 — Add one regression test

- [ ] Add a test to an existing `tests/test_trading_config*.py` (or create `tests/test_trading_config.py` if none exists) that asserts the flag defaults to `False`.

```python
def test_use_portfolio_manager_defaults_off():
    from trading_system.config import TradingConfig
    cfg = TradingConfig()
    assert cfg.use_portfolio_manager is False
```

### Step 5.4 — Run tests

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"
```
Expected: `741 passed, 1 skipped` (740 + 1 new).

### Step 5.5 — Commit

```bash
git add trading_system/config.py tests/test_trading_config.py
git commit -m "feat(phase4): add use_portfolio_manager flag to TradingConfig (T4.1b)

Default False — preserves Phase 3 behavior. CLI flag --no-pm in
run_phase4_walkforward.py is the primary consumer; this config
entry exists for direct-driver callers."
```

---

## Task 6 — T4.2: Full walk-forward and gate evaluation

**Files:** no new files

**Depends on:** T4.1a, T4.1b

**Scope override:** the handoff says `--all-folds`, but per-fold trade files only exist for 2024. This task evaluates the **2024 fold** against G1–G9. All-folds extension is documented as follow-up (not in scope).

### Step 6.1 — Run the final Phase 4 walk-forward

- [ ] Clean run:

```bash
PYTHONUTF8=1 py -3.12 scripts/run_phase4_walkforward.py --fold 2024
```

- [ ] Capture terminal output in the session log.

### Step 6.2 — Read the gate check file

- [ ] Review `results/phase4_gate_check.txt`:

```bash
cat results/phase4_gate_check.txt
```

Expected: G1–G9 all marked `[X]`. If any gate fails:
- **G4 fails (Sharpe < 2.0):** follow the fallback protocol in plan §2. Do NOT tune limits without user approval.
- **Any other gate fails:** STOP and escalate with the output captured.

### Step 6.3 — Session log update

- [ ] Update `docs/session-logs/SESSION_2026-04-08_phase4-execution.md` with the gate verdict + metrics. Each metric must cite its source file (provenance rule).

### Step 6.4 — No commit required

Results files are gitignored. Session log commit happens at session end.

---

## Task 7 — T4.3: Rejection analysis

**Files:**
- Create: `scripts/analyze_pm_rejections.py`

**Depends on:** T4.2

**Rationale:** Histograms help diagnose whether the PM is behaving sensibly. A rejection distribution dominated by one reason (> 60%) fails G6 and is a signal the limits are miscalibrated.

### Step 7.1 — Create the analysis script

- [ ] Create `scripts/analyze_pm_rejections.py`:

```python
"""
scripts/analyze_pm_rejections.py — T4.3: histograms of PM rejection outcomes.

Reads results/phase4_rejections.csv and emits four histograms:
  1. by reason (most common → least)
  2. by sector
  3. by confidence bucket (0.65-0.70, 0.70-0.75, 0.75-0.80, >=0.80)
  4. top 10 rejected tickers

Also surfaces the G6 gate check: no single reason > 60% of rejections.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/analyze_pm_rejections.py

Outputs:
    results/phase4_rejection_analysis.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
REJECT_CSV = project_root / "results" / "phase4_rejections.csv"
OUT_TXT    = project_root / "results" / "phase4_rejection_analysis.txt"


def main() -> int:
    if not REJECT_CSV.exists():
        print(f"FATAL: {REJECT_CSV} not found. Run run_phase4_walkforward.py first.")
        return 2

    df = pd.read_csv(REJECT_CSV)
    if df.empty:
        print("No rejections — all signals approved.")
        OUT_TXT.write_text("No rejections.\n", encoding="utf-8")
        return 0

    lines: list[str] = [
        "Phase 4 Rejection Analysis",
        "=" * 50,
        f"Total rejections: {len(df)}",
        "",
    ]

    # 1. By reason
    lines.append("By reason:")
    by_reason = df["reason"].value_counts(normalize=False)
    by_reason_pct = df["reason"].value_counts(normalize=True)
    for reason, n in by_reason.items():
        lines.append(f"  {reason:28s} {n:5d}  ({by_reason_pct[reason]:6.1%})")
    lines.append("")

    # G6: no single reason > 60%
    max_reason = by_reason_pct.max()
    max_reason_name = by_reason_pct.idxmax()
    g6_pass = max_reason <= 0.60
    lines.append(
        f"G6 check: {'PASS' if g6_pass else 'FAIL'} — "
        f"max reason '{max_reason_name}' at {max_reason:.1%} "
        f"(threshold ≤ 60%)"
    )
    lines.append("")

    # 2. By sector
    lines.append("By sector:")
    for sector, n in df["sector"].value_counts().items():
        lines.append(f"  {sector:28s} {n:5d}")
    lines.append("")

    # 3. By confidence bucket
    lines.append("By confidence bucket:")
    bins = [0.00, 0.65, 0.70, 0.75, 0.80, 1.00]
    labels = ["<0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80", ">=0.80"]
    df["conf_bucket"] = pd.cut(df["confidence"], bins=bins,
                                labels=labels, include_lowest=True)
    for bucket, n in df["conf_bucket"].value_counts().sort_index().items():
        lines.append(f"  {str(bucket):28s} {n:5d}")
    lines.append("")

    # 4. Top 10 tickers
    lines.append("Top 10 rejected tickers:")
    for ticker, n in df["ticker"].value_counts().head(10).items():
        lines.append(f"  {ticker:10s} {n:5d}")
    lines.append("")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if g6_pass else 1


if __name__ == "__main__":
    sys.exit(main())
```

### Step 7.2 — Run analysis

```bash
PYTHONUTF8=1 py -3.12 scripts/analyze_pm_rejections.py
```
Expected: `G6 check: PASS` if no reason exceeds 60%.

### Step 7.3 — Commit

```bash
git add scripts/analyze_pm_rejections.py
git commit -m "feat(phase4): add PM rejection histograms (T4.3)

Histograms by reason, sector, confidence bucket, and top 10
rejected tickers. Evaluates G6 gate (no single reason > 60%).
Output: results/phase4_rejection_analysis.txt"
```

---

## Task 8 — T4.4: Phase 3 vs Phase 4 comparison

**Files:**
- Create: `scripts/compare_phase3_vs_phase4.py`

**Depends on:** T4.2

### Step 8.1 — Create the comparison script

- [ ] Create `scripts/compare_phase3_vs_phase4.py`:

```python
"""
scripts/compare_phase3_vs_phase4.py — T4.4: Phase 3 vs Phase 4 head-to-head.

Reads results/phase3_walkforward.tsv and results/phase4_walkforward.tsv and
prints a delta table. Also re-runs Phase 4 in --no-pm mode to confirm the
parity contract.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/compare_phase3_vs_phase4.py

Output:
    results/phase4_vs_phase3_comparison.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
P3 = project_root / "results" / "phase3_walkforward.tsv"
P4 = project_root / "results" / "phase4_walkforward.tsv"
OUT = project_root / "results" / "phase4_vs_phase3_comparison.txt"


def main() -> int:
    p3 = pd.read_csv(P3, sep="\t").iloc[0]
    p4 = pd.read_csv(P4, sep="\t").iloc[0]

    fields = ["sharpe", "max_dd", "final_equity", "n_trades_placed", "n_blocked"]
    lines = ["Phase 3 vs Phase 4 — 2024 fold", "=" * 50, ""]
    lines.append(f"{'Metric':24s} {'Phase 3':>12s} {'Phase 4':>12s} {'Delta':>12s}")
    lines.append("-" * 60)
    for f in fields:
        v3 = p3.get(f)
        v4 = p4.get(f)
        if v3 is None or v4 is None:
            continue
        delta = v4 - v3
        lines.append(f"{f:24s} {v3:>12.4f} {v4:>12.4f} {delta:>+12.4f}")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 8.2 — Run comparison

```bash
PYTHONUTF8=1 py -3.12 scripts/compare_phase3_vs_phase4.py
```

Capture output for the session log.

### Step 8.3 — Commit

```bash
git add scripts/compare_phase3_vs_phase4.py
git commit -m "feat(phase4): add Phase 3 vs Phase 4 diff script (T4.4)

Reads phase3_walkforward.tsv and phase4_walkforward.tsv, prints
Sharpe/MaxDD/trade count delta. Output:
results/phase4_vs_phase3_comparison.txt"
```

---

## Task 9 — Session close

**Depends on:** all previous tasks

- [ ] **Step 9.1 — Full test suite final run**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"
```
Expected: `741 passed, 1 skipped` (or higher if executor added tests beyond the minimum).

- [ ] **Step 9.2 — Update CLAUDE.md** — ONLY the test count line and, if G4 passed, mark Phase 4 COMPLETE in the Current Phase section. This is the one CLAUDE.md edit explicitly authorized by this plan.

- [ ] **Step 9.3 — Write session log** at `docs/session-logs/SESSION_2026-04-08_phase4-execution.md`. Sections to include:
  - Pre-flight state
  - Cleanup commits to main (04c64d2, cd3055d, 9e209a2)
  - Plan authoring
  - Task-by-task execution notes
  - Gate verdict with provenance for each metric
  - Outstanding follow-ups (all-folds run, cooldown implementation, etc.)

- [ ] **Step 9.4 — Commit session log + CLAUDE.md**

```bash
git add docs/session-logs/SESSION_2026-04-08_phase4-execution.md CLAUDE.md
git commit -m "docs(phase4): session log + CLAUDE.md test count update"
```

- [ ] **Step 9.5 — Do NOT push to origin** unless the user authorizes it.

---

## Follow-up / out of scope

Recorded here so a future session can pick them up without re-deriving context.

1. **All-folds walk-forward.** Requires either per-fold trade files (don't exist) or matcher regeneration per fold (minutes of runtime). Can be added as a `--fold` loop once the primary 2024 gate passes.
2. **Cooldown implementation.** The `PMRejectionReason` Literal reserves `"cooldown"`. v1 relies on the trade file already encoding entry/exit timing. A proper implementation would thread a cooldown state into `PortfolioSnapshot`.
3. **Signal regeneration mode.** The outline's Decision #3 called for regenerating signals from the matcher. This plan replaces that with trade-file replay (§3 design decision). A v2 that regenerates signals would let the PM "unlock" signals Phase 1 suppressed — distinct feature, distinct plan.
4. **2022-Bear fold fragility.** Phase 2 Kelly was -0.504 on that fold. Phase 4 inherits Phase 2's sizing rejection (emergent safety, no new code path). If the all-folds run shows 2022-Bear failing G4/G5, revisit Phase 2 Kelly parameterization first, not the PM.
5. **SLE-75 fatigue overlay redesign.** Disabled in Phase 3 per `USE_FATIGUE_OVERLAY=False` and inherited in Phase 4. Separate tracked in the Linear SLE-75 comment.

---

## Risks and open questions

- **The handoff's Sharpe ≥ 2.0 gate may be too tight.** The outline's ≥ 1.0 is the softer baseline. If 2024 Phase 4 prints Sharpe between 1.0 and 2.0, the plan treats it as a failure per G4 but the user may want to relax it — surface this choice before finalizing the gate verdict.
- **The `cached_signals_2024.csv` file was produced before the `same_sector_boost_factor` hook was committed to `pattern_engine/matcher.py`.** The default of 1.0 should make the new code path a no-op, but T4.0c exists specifically to prove that. If T4.0c fails, investigate first; do not adjust thresholds.
- **`backtest_trades.csv` has only 278 rows — a small statistical sample.** PM filter outcomes can swing visibly on small-n experiments. Interpret gate margins cautiously.

---

*Plan authored 2026-04-08 from the pre-plan outline + session handoff + live codebase survey. Executor should follow task order strictly; each task commits on its own and the full test suite runs after every task.*
