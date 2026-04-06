# Phase 3 — Risk Engine Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate real per-ticker ATR sizing, drawdown brake, and existing fatigue/congestion overlays into production via a thin stateless orchestrator.

**Architecture:** Stateless function in `trading_system/risk_engine.py` composes Phase 2 `size_position()` output with a drawdown brake scalar and overlay multipliers. `position_sizer.py` gains an optional `atr_pct` parameter that overrides `flat_atr_pct`. No new classes — flat functions only.

**Tech Stack:** Python 3.12, pytest, pandas (for walk-forward), existing `trading_system/risk_overlays/` ABCs

**Spec:** `docs/superpowers/specs/2026-04-06-phase3-risk-engine-integration-design.md`

---

## File Structure

| File | Purpose | Status |
|------|---------|--------|
| `trading_system/risk_engine.py` | New: `compute_atr_pct`, `drawdown_brake_scalar`, `AdjustedSizing`, `apply_risk_adjustments` | CREATE |
| `trading_system/position_sizer.py` | Modify: add optional `atr_pct` param to `size_position()` | MODIFY |
| `trading_system/risk_overlays/base.py` | Modify: docstring update for Phase 3 contract evolution | MODIFY (docs only) |
| `tests/test_risk_engine.py` | New: 22 unit + integration tests | CREATE |
| `tests/test_position_sizer.py` | Modify: add 3 tests for `atr_pct` override | MODIFY |
| `scripts/run_phase3_walkforward.py` | New: 6-fold walk-forward with real ATR + overlays | CREATE |
| `docs/PHASE2_RISK_ENGINE.md` | Modify: add Phase 3 progress section | MODIFY |
| `CLAUDE.md` | Modify: phase update + locked settings | MODIFY |

---

## Conventions

- All public-API guards use `raise RuntimeError(...)` (per `CLAUDE.md` rule 4 — `assert` is stripped under `-O`).
- Run tests with: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
- Commit after each task. Commit message format: `feat(phase3): <task>` or `test(phase3): <task>`.

---

## Task 1: Add `compute_atr_pct` helper (TDD)

**Files:**
- Create: `trading_system/risk_engine.py`
- Test: `tests/test_risk_engine.py`

- [ ] **Step 1: Create test file with failing tests for `compute_atr_pct`**

Create `tests/test_risk_engine.py`:

```python
"""
tests/test_risk_engine.py — Phase 3 Risk Engine Integration tests.

Covers:
  - compute_atr_pct: ATR-as-fraction-of-price helper
  - drawdown_brake_scalar: linear DD scaling 15% → 20%
  - apply_risk_adjustments: orchestrator composing DD brake + overlays
  - End-to-end pipeline: real ATR → sizing → adjustments
"""

import pytest

from trading_system.risk_engine import compute_atr_pct


class TestComputeAtrPct:
    def test_basic(self):
        """atr_14=2.0, close=100 → 0.02 (2% ATR)."""
        assert compute_atr_pct(atr_14=2.0, close=100.0) == pytest.approx(0.02)

    def test_realistic_value(self):
        """Realistic SPY-like value: atr_14=4.5, close=450 → 0.01."""
        assert compute_atr_pct(atr_14=4.5, close=450.0) == pytest.approx(0.01)

    def test_zero_atr_raises(self):
        """atr_14=0 → RuntimeError (loud failure at source)."""
        with pytest.raises(RuntimeError, match="atr_14 must be > 0"):
            compute_atr_pct(atr_14=0.0, close=100.0)

    def test_negative_atr_raises(self):
        """atr_14<0 → RuntimeError."""
        with pytest.raises(RuntimeError, match="atr_14 must be > 0"):
            compute_atr_pct(atr_14=-1.0, close=100.0)

    def test_zero_close_raises(self):
        """close=0 → RuntimeError."""
        with pytest.raises(RuntimeError, match="close must be > 0"):
            compute_atr_pct(atr_14=2.0, close=0.0)

    def test_negative_close_raises(self):
        """close<0 → RuntimeError."""
        with pytest.raises(RuntimeError, match="close must be > 0"):
            compute_atr_pct(atr_14=2.0, close=-1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/test_risk_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_system.risk_engine'`

- [ ] **Step 3: Create `risk_engine.py` with `compute_atr_pct`**

Create `trading_system/risk_engine.py`:

```python
"""
risk_engine.py — Phase 3: Risk Engine Integration (thin orchestrator).

Stateless functions composing Phase 2 sizing with drawdown brake and risk
overlays.  No state held — the caller (walk-forward script or live trader)
wires the pieces together each trading day.

Components:
  - compute_atr_pct        : atr_14 / close → ATR as fraction of price
  - drawdown_brake_scalar  : linear scalar 1.0 → 0.0 over [warn, halt] DD
  - AdjustedSizing         : result dataclass
  - apply_risk_adjustments : composes sizing × dd_scalar × overlay_multiplier

Phase 3 contract evolution:
  Risk overlays (BaseRiskOverlay subclasses) multiply POSITION SIZE in this
  orchestrator, not signal confidence as the original ABC docstring stated.
  Reason: Half-Kelly already incorporates confidence into position size, so
  re-throttling confidence would double-count the same input.

Spec: docs/superpowers/specs/2026-04-06-phase3-risk-engine-integration-design.md
Linear: Phase 3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from trading_system.position_sizer import SizingResult
from trading_system.risk_overlays.base import BaseRiskOverlay


def compute_atr_pct(atr_14: float, close: float) -> float:
    """ATR as a fraction of price.

    Computes the standard ATR/price ratio used for stop distance and
    volatility-aware position sizing.  Both inputs come from the 52T
    volnorm dataset (`atr_14` and `Close` columns).

    Args:
        atr_14: 14-period Average True Range in dollar terms.  Must be > 0.
        close:  Closing price.  Must be > 0.

    Returns:
        atr_14 / close — a positive float, typically in [0.005, 0.05] for
        liquid US equities.

    Raises:
        RuntimeError: if `atr_14 <= 0` (no volatility data) or `close <= 0`.

    Caller contract:
        Wrap calls in try/except RuntimeError and skip the trade with a
        logged reason.  Loud failure at source is preferred over silent
        rejection cascading through downstream layers.
    """
    if close <= 0:
        raise RuntimeError(f"close must be > 0, got {close}")
    if atr_14 <= 0:
        raise RuntimeError(
            f"atr_14 must be > 0, got {atr_14}; ticker has no volatility data"
        )
    return atr_14 / close
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/test_risk_engine.py::TestComputeAtrPct -v`
Expected: PASS (6/6)

- [ ] **Step 5: Commit**

```bash
git add trading_system/risk_engine.py tests/test_risk_engine.py
git commit -m "feat(phase3): add compute_atr_pct helper with TDD coverage"
```

---

## Task 2: Add `drawdown_brake_scalar` (TDD)

**Files:**
- Modify: `trading_system/risk_engine.py`
- Modify: `tests/test_risk_engine.py`

- [ ] **Step 1: Append failing tests to `tests/test_risk_engine.py`**

Append after `TestComputeAtrPct`:

```python
from trading_system.risk_engine import drawdown_brake_scalar


class TestDrawdownBrakeScalar:
    def test_no_drawdown(self):
        """dd=0 → 1.0 (full sizing)."""
        assert drawdown_brake_scalar(0.0) == pytest.approx(1.0)

    def test_below_warn(self):
        """dd=10% → 1.0 (still below 15% warn)."""
        assert drawdown_brake_scalar(0.10) == pytest.approx(1.0)

    def test_at_warn(self):
        """dd=15% → 1.0 (boundary; brake just starts)."""
        assert drawdown_brake_scalar(0.15) == pytest.approx(1.0)

    def test_midpoint(self):
        """dd=17.5% → 0.5 (linear midpoint between 15% and 20%)."""
        assert drawdown_brake_scalar(0.175) == pytest.approx(0.5)

    def test_at_halt(self):
        """dd=20% → 0.0 (full halt)."""
        assert drawdown_brake_scalar(0.20) == pytest.approx(0.0)

    def test_above_halt(self):
        """dd=25% → 0.0 (clamp at zero)."""
        assert drawdown_brake_scalar(0.25) == pytest.approx(0.0)

    def test_negative_drawdown_treated_as_zero(self):
        """dd<0 (impossible but defensive) → 1.0."""
        assert drawdown_brake_scalar(-0.05) == pytest.approx(1.0)

    def test_invalid_thresholds(self):
        """warn >= halt → RuntimeError."""
        with pytest.raises(RuntimeError, match="warn .* < halt"):
            drawdown_brake_scalar(0.10, warn=0.20, halt=0.15)

    def test_custom_thresholds(self):
        """Custom thresholds: warn=10%, halt=15%, dd=12.5% → 0.5."""
        assert drawdown_brake_scalar(0.125, warn=0.10, halt=0.15) == pytest.approx(0.5)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/test_risk_engine.py::TestDrawdownBrakeScalar -v`
Expected: FAIL with `ImportError: cannot import name 'drawdown_brake_scalar'`

- [ ] **Step 3: Add `drawdown_brake_scalar` to `risk_engine.py`**

Append to `trading_system/risk_engine.py`:

```python
def drawdown_brake_scalar(
    drawdown: float,
    warn: float = 0.15,
    halt: float = 0.20,
) -> float:
    """Linear drawdown brake scalar.

    Returns a scalar in [0.0, 1.0] used to throttle position sizing as
    portfolio drawdown approaches the halt threshold.

    Schedule:
        drawdown < warn          → 1.0  (full sizing, no brake)
        warn ≤ drawdown < halt   → linear interpolation 1.0 → 0.0
        drawdown ≥ halt          → 0.0  (halt; no new positions)

    Defaults match the roadmap spec: warn=15%, halt=20%.

    Args:
        drawdown: Current drawdown from peak equity, in [0, 1].  Negative
            values (impossible in practice) are treated as zero.
        warn: Drawdown threshold where the brake starts engaging.
        halt: Drawdown threshold where the brake reaches zero.

    Returns:
        Scalar in [0.0, 1.0] to multiply against `position_pct`.

    Raises:
        RuntimeError: if `warn >= halt`.
    """
    if warn >= halt:
        raise RuntimeError(
            f"Need warn < halt, got warn={warn}, halt={halt}"
        )
    if drawdown <= warn:
        return 1.0
    if drawdown >= halt:
        return 0.0
    # Linear interpolation: at warn → 1.0, at halt → 0.0
    span = halt - warn
    return max(0.0, 1.0 - (drawdown - warn) / span)
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/test_risk_engine.py::TestDrawdownBrakeScalar -v`
Expected: PASS (9/9)

- [ ] **Step 5: Commit**

```bash
git add trading_system/risk_engine.py tests/test_risk_engine.py
git commit -m "feat(phase3): add drawdown_brake_scalar with TDD coverage"
```

---

## Task 3: Add `atr_pct` parameter to `size_position()` (TDD)

**Files:**
- Modify: `trading_system/position_sizer.py`
- Modify: `tests/test_position_sizer.py`

- [ ] **Step 1: Append failing tests to `tests/test_position_sizer.py`**

Append at the end of the file:

```python
class TestAtrPctOverride:
    """Phase 3: real ATR override of flat_atr_pct."""

    def test_atr_pct_overrides_flat(self):
        """When atr_pct provided, it replaces config.flat_atr_pct in stop_distance."""
        # With flat_atr_pct=0.02, stop_distance = 3.0 * 0.02 = 0.06
        # With atr_pct=0.04, stop_distance = 3.0 * 0.04 = 0.12 (smaller atr_weight, smaller pos)
        cfg = SizingConfig()
        baseline = size_position(confidence=0.65, b_ratio=1.18, config=cfg)
        with_atr  = size_position(confidence=0.65, b_ratio=1.18, config=cfg, atr_pct=0.04)
        assert baseline.approved
        assert with_atr.approved
        # Larger ATR → smaller raw position (atr_weight = 0.02 / 0.12 = 0.167 vs 0.333)
        # Both might clamp to min=0.02; check the underlying atr_weight changed
        assert with_atr.atr_weight == pytest.approx(0.02 / (3.0 * 0.04))
        assert baseline.atr_weight == pytest.approx(0.02 / (3.0 * 0.02))

    def test_atr_pct_zero_rejected(self):
        """atr_pct=0 → rejected (consistent with flat_atr_pct validation)."""
        result = size_position(confidence=0.65, b_ratio=1.18, atr_pct=0.0)
        assert not result.approved
        assert "atr_pct" in (result.rejection_reason or "")

    def test_atr_pct_above_one_rejected(self):
        """atr_pct >= 1 → rejected."""
        result = size_position(confidence=0.65, b_ratio=1.18, atr_pct=1.5)
        assert not result.approved
        assert "atr_pct" in (result.rejection_reason or "")

    def test_atr_pct_none_uses_flat(self):
        """atr_pct=None → uses config.flat_atr_pct (Phase 2 backward compat)."""
        cfg = SizingConfig(flat_atr_pct=0.025)
        result = size_position(confidence=0.65, b_ratio=1.18, config=cfg, atr_pct=None)
        assert result.approved
        assert result.atr_weight == pytest.approx(0.02 / (3.0 * 0.025))
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/test_position_sizer.py::TestAtrPctOverride -v`
Expected: FAIL — `size_position()` does not accept `atr_pct` keyword argument.

- [ ] **Step 3: Modify `size_position()` to accept `atr_pct`**

In `trading_system/position_sizer.py`, change the `size_position` signature and add validation. Replace lines 130-217 (the entire function body):

```python
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
```

- [ ] **Step 4: Run new tests + Phase 2 regression**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/test_position_sizer.py -v`
Expected: PASS (28 existing + 4 new = 32)

- [ ] **Step 5: Commit**

```bash
git add trading_system/position_sizer.py tests/test_position_sizer.py
git commit -m "feat(phase3): add atr_pct override to size_position (T3.2)"
```

---

## Task 4: Add `AdjustedSizing` dataclass and `apply_risk_adjustments` (TDD)

**Files:**
- Modify: `trading_system/risk_engine.py`
- Modify: `tests/test_risk_engine.py`

- [ ] **Step 1: Append failing tests to `tests/test_risk_engine.py`**

Append after `TestDrawdownBrakeScalar`:

```python
from datetime import date

from trading_system.position_sizer import SizingResult, size_position
from trading_system.risk_engine import AdjustedSizing, apply_risk_adjustments
from trading_system.risk_overlays.fatigue_accumulation import FatigueAccumulationOverlay
from trading_system.risk_overlays.liquidity_congestion import LiquidityCongestionGate


def _make_approved_sizing(position_pct: float = 0.05) -> SizingResult:
    """Helper: create an approved SizingResult for orchestrator tests."""
    return SizingResult(
        approved=True,
        position_pct=position_pct,
        kelly_fraction=0.30,
        scaled_kelly=0.15,
        atr_weight=0.333,
        rejection_reason=None,
    )


class TestApplyRiskAdjustments:
    def test_no_overlays_no_drawdown(self):
        """No overlays, no DD → final == original."""
        sizing = _make_approved_sizing(0.05)
        adj = apply_risk_adjustments(sizing, drawdown=0.0)
        assert adj.final_position_pct == pytest.approx(0.05)
        assert adj.dd_scalar == pytest.approx(1.0)
        assert adj.overlay_multiplier == pytest.approx(1.0)
        assert not adj.blocked

    def test_dd_brake_partial_throttle(self):
        """dd=17.5% → dd_scalar=0.5 → final = 0.05 * 0.5 = 0.025."""
        sizing = _make_approved_sizing(0.05)
        adj = apply_risk_adjustments(sizing, drawdown=0.175)
        assert adj.dd_scalar == pytest.approx(0.5)
        assert adj.final_position_pct == pytest.approx(0.025)
        assert not adj.blocked

    def test_dd_halt_blocks(self):
        """dd=22% → dd_scalar=0 → blocked, reason starts with 'dd_brake:'."""
        sizing = _make_approved_sizing(0.05)
        adj = apply_risk_adjustments(sizing, drawdown=0.22)
        assert adj.final_position_pct == pytest.approx(0.0)
        assert adj.blocked
        assert adj.block_reason.startswith("dd_brake:")

    def test_rejected_sizing_passes_through(self):
        """Phase 2 rejection → blocked AdjustedSizing with original reason."""
        rejected = SizingResult(
            approved=False,
            position_pct=0.0,
            kelly_fraction=-0.05,
            scaled_kelly=0.0,
            atr_weight=0.333,
            rejection_reason="Kelly fraction non-positive",
        )
        adj = apply_risk_adjustments(rejected, drawdown=0.0)
        assert adj.blocked
        assert adj.block_reason.startswith("sizing_rejected:")
        assert "Kelly fraction non-positive" in adj.block_reason

    def test_with_fatigue_overlay(self):
        """Fatigue at score~0.4 → multiplier ~0.6 → final = 0.05 * 0.6 = 0.03."""
        sizing = _make_approved_sizing(0.05)
        fatigue = FatigueAccumulationOverlay(decay_rate=0.15)
        # Build up some fatigue: 4 BULL days
        d = date(2024, 1, 1)
        for i in range(4):
            fatigue.update(date(2024, 1, i + 1), regime_label="BULL")
        mult = fatigue.get_signal_multiplier()
        assert 0.4 < mult < 0.8  # sanity
        adj = apply_risk_adjustments(sizing, drawdown=0.0, overlays=[fatigue])
        assert adj.overlay_multiplier == pytest.approx(mult)
        assert adj.final_position_pct == pytest.approx(0.05 * mult)

    def test_with_congestion_full_block(self):
        """Congestion gate at level >= block_threshold → multiplier=0 → blocked."""
        sizing = _make_approved_sizing(0.05)
        gate = LiquidityCongestionGate(
            window=2,
            congestion_threshold=0.025,
            block_threshold=0.05,
        )
        # Inject high ATR/price ratios → exceeds block_threshold
        gate.update(date(2024, 1, 1), atr_price_ratio=0.06)
        gate.update(date(2024, 1, 2), atr_price_ratio=0.06)
        adj = apply_risk_adjustments(sizing, drawdown=0.0, overlays=[gate])
        assert adj.overlay_multiplier == pytest.approx(0.0)
        assert adj.blocked
        assert adj.block_reason.startswith("overlay:LiquidityCongestionGate")

    def test_both_overlays_multiplicative(self):
        """Two overlays compose multiplicatively."""
        sizing = _make_approved_sizing(0.05)
        fatigue = FatigueAccumulationOverlay(decay_rate=0.15)
        for i in range(3):
            fatigue.update(date(2024, 1, i + 1), regime_label="BULL")
        gate = LiquidityCongestionGate(window=2)
        # Quiet market — gate stays at 1.0
        gate.update(date(2024, 1, 1), atr_price_ratio=0.005)
        gate.update(date(2024, 1, 2), atr_price_ratio=0.005)
        adj = apply_risk_adjustments(sizing, drawdown=0.0, overlays=[fatigue, gate])
        expected = fatigue.get_signal_multiplier() * gate.get_signal_multiplier()
        assert adj.overlay_multiplier == pytest.approx(expected)

    def test_overlay_order_invariant(self):
        """Order of overlays does not change result (multiplication commutes)."""
        sizing = _make_approved_sizing(0.05)
        fatigue = FatigueAccumulationOverlay(decay_rate=0.15)
        for i in range(3):
            fatigue.update(date(2024, 1, i + 1), regime_label="BULL")
        gate = LiquidityCongestionGate(window=2)
        gate.update(date(2024, 1, 1), atr_price_ratio=0.005)
        gate.update(date(2024, 1, 2), atr_price_ratio=0.005)
        a = apply_risk_adjustments(sizing, drawdown=0.0, overlays=[fatigue, gate])
        b = apply_risk_adjustments(sizing, drawdown=0.0, overlays=[gate, fatigue])
        assert a.final_position_pct == pytest.approx(b.final_position_pct)

    def test_invalid_drawdown_raises(self):
        """drawdown > 1.0 (data bug) → RuntimeError."""
        sizing = _make_approved_sizing(0.05)
        with pytest.raises(RuntimeError, match="drawdown"):
            apply_risk_adjustments(sizing, drawdown=1.5)

    def test_block_reason_format(self):
        """block_reason follows structured format for parseable diagnostics."""
        sizing = _make_approved_sizing(0.05)
        adj = apply_risk_adjustments(sizing, drawdown=0.22)
        # Format: "dd_brake:<dd_value>"
        prefix, value = adj.block_reason.split(":", 1)
        assert prefix == "dd_brake"
        assert float(value) == pytest.approx(0.22)


class TestEndToEndIntegration:
    def test_real_atr_pipeline(self):
        """compute_atr_pct → size_position(atr_pct=) → apply_risk_adjustments."""
        atr_pct = compute_atr_pct(atr_14=4.5, close=450.0)  # 0.01
        sizing = size_position(confidence=0.68, b_ratio=1.18, atr_pct=atr_pct)
        adj = apply_risk_adjustments(sizing, drawdown=0.05)
        assert adj.original.approved
        assert adj.final_position_pct > 0
        assert not adj.blocked

    def test_zero_atr_pipeline_raises(self):
        """compute_atr_pct raises BEFORE reaching size_position (loud failure)."""
        with pytest.raises(RuntimeError, match="atr_14"):
            compute_atr_pct(atr_14=0.0, close=100.0)

    def test_synthetic_20pct_dd_scenario(self):
        """T3.4: synthetic DD progression triggers brake correctly."""
        sizing = _make_approved_sizing(0.05)
        # Simulate increasing DD: 10% → 17.5% → 22%
        steps = [
            (0.10, 1.0,  False),   # below warn
            (0.15, 1.0,  False),   # at warn boundary
            (0.175, 0.5, False),   # midpoint
            (0.20, 0.0,  True),    # at halt
            (0.22, 0.0,  True),    # past halt
        ]
        for dd, expected_scalar, expected_blocked in steps:
            adj = apply_risk_adjustments(sizing, drawdown=dd)
            assert adj.dd_scalar == pytest.approx(expected_scalar), f"dd={dd}"
            assert adj.blocked == expected_blocked, f"dd={dd}"
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/test_risk_engine.py::TestApplyRiskAdjustments tests/test_risk_engine.py::TestEndToEndIntegration -v`
Expected: FAIL with `ImportError: cannot import name 'AdjustedSizing'` (and `apply_risk_adjustments`)

- [ ] **Step 3: Add `AdjustedSizing` and `apply_risk_adjustments` to `risk_engine.py`**

Append to `trading_system/risk_engine.py` (the imports `dataclass`, `List`, `Optional`, `SizingResult`, `BaseRiskOverlay` are already at the top of the file from Task 1):

```python
# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AdjustedSizing:
    """Output of apply_risk_adjustments.

    Attrs:
        original: The pre-adjustment SizingResult from position_sizer.
        final_position_pct: Position size after DD brake and overlays.
        dd_scalar: Drawdown brake scalar in [0, 1].
        overlay_multiplier: Product of all overlay multipliers in [0, 1].
        blocked: True if final_position_pct == 0 (cannot trade).
        block_reason: Structured reason string when blocked, else None.
            Formats:
              - "sizing_rejected:<original_reason>"
              - "dd_brake:<drawdown>"
              - "overlay:<OverlayClassName>"
    """
    original: SizingResult
    final_position_pct: float
    dd_scalar: float
    overlay_multiplier: float
    blocked: bool
    block_reason: Optional[str]


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def apply_risk_adjustments(
    sizing: SizingResult,
    drawdown: float,
    overlays: Optional[List[BaseRiskOverlay]] = None,
    dd_warn: float = 0.15,
    dd_halt: float = 0.20,
) -> AdjustedSizing:
    """Compose Phase 2 sizing with DD brake and risk overlays.

    Pipeline:
      1. If sizing was rejected → propagate as blocked.
      2. Compute DD brake scalar from current drawdown.
      3. Compute overlay multiplier as product of all overlay
         signal multipliers (1.0 if no overlays).
      4. final = sizing.position_pct × dd_scalar × overlay_multiplier.
      5. If final == 0 → blocked, with structured reason indicating
         which component caused the block.

    Block reason priority (first match wins):
      sizing_rejected → dd_brake → overlay:<name>

    Phase 3 contract: overlays multiply POSITION SIZE in this orchestrator,
    not signal confidence as the original BaseRiskOverlay docstring stated.
    Half-Kelly already incorporates confidence; re-throttling it would
    double-count.

    Args:
        sizing: SizingResult from position_sizer.size_position().
        drawdown: Current portfolio drawdown from peak [0, 1].
        overlays: Optional list of BaseRiskOverlay instances.  Each must
            have already been updated for the current trading day.
        dd_warn: Drawdown brake warn threshold (default 0.15).
        dd_halt: Drawdown brake halt threshold (default 0.20).

    Returns:
        AdjustedSizing with composed final_position_pct and diagnostic fields.

    Raises:
        RuntimeError: if drawdown > 1.0 (catches upstream data bugs).
    """
    if drawdown > 1.0:
        raise RuntimeError(f"drawdown must be <= 1.0, got {drawdown}")

    # Case 1: Phase 2 rejected the sizing — propagate
    if not sizing.approved:
        return AdjustedSizing(
            original=sizing,
            final_position_pct=0.0,
            dd_scalar=0.0,
            overlay_multiplier=0.0,
            blocked=True,
            block_reason=f"sizing_rejected:{sizing.rejection_reason}",
        )

    # Case 2: compute scalars
    dd_scalar = drawdown_brake_scalar(drawdown, warn=dd_warn, halt=dd_halt)

    overlay_multiplier = 1.0
    blocking_overlay: Optional[BaseRiskOverlay] = None
    if overlays:
        for overlay in overlays:
            m = overlay.get_signal_multiplier()
            overlay_multiplier *= m
            if m == 0.0 and blocking_overlay is None:
                blocking_overlay = overlay

    final_position_pct = sizing.position_pct * dd_scalar * overlay_multiplier

    # Determine block status + structured reason (priority order)
    if final_position_pct == 0.0:
        if dd_scalar == 0.0:
            reason = f"dd_brake:{drawdown}"
        elif blocking_overlay is not None:
            reason = f"overlay:{type(blocking_overlay).__name__}"
        else:
            reason = "unknown"  # should not happen given the above cases
        return AdjustedSizing(
            original=sizing,
            final_position_pct=0.0,
            dd_scalar=dd_scalar,
            overlay_multiplier=overlay_multiplier,
            blocked=True,
            block_reason=reason,
        )

    return AdjustedSizing(
        original=sizing,
        final_position_pct=final_position_pct,
        dd_scalar=dd_scalar,
        overlay_multiplier=overlay_multiplier,
        blocked=False,
        block_reason=None,
    )
```

- [ ] **Step 4: Run all `risk_engine.py` tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/test_risk_engine.py -v`
Expected: PASS (all tests across `TestComputeAtrPct`, `TestDrawdownBrakeScalar`, `TestApplyRiskAdjustments`, `TestEndToEndIntegration`).

- [ ] **Step 5: Run full test suite to confirm no regressions**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: All previously-passing tests still pass; new tests added.

- [ ] **Step 6: Commit**

```bash
git add trading_system/risk_engine.py tests/test_risk_engine.py
git commit -m "feat(phase3): add apply_risk_adjustments orchestrator (T3.1, T3.3, T3.4)"
```

---

## Task 5: Update `BaseRiskOverlay` docstring (Phase 3 contract evolution)

**Files:**
- Modify: `trading_system/risk_overlays/base.py`

This is a documentation-only change to record the Phase 3 contract evolution.

- [ ] **Step 1: Update the module docstring in `base.py`**

In `trading_system/risk_overlays/base.py`, replace lines 10-14 (the "Signal integration:" block) with:

```python
Signal integration (Phase 3 contract):
    final_position_pct = sizing.position_pct × overlay.get_signal_multiplier()
    Overlays multiply POSITION SIZE in the risk_engine orchestrator, not
    signal confidence.  Reason: Half-Kelly position_sizer already incorporates
    confidence into position size, so re-throttling confidence would
    double-count the same input.

    Phase 1/2 historical contract (deprecated):
        effective_confidence = raw_confidence × overlay.get_signal_multiplier()
```

- [ ] **Step 2: Run risk overlay tests to verify nothing broke**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_risk_overlays.py -v`
Expected: PASS (no behavior change — docs only).

- [ ] **Step 3: Commit**

```bash
git add trading_system/risk_overlays/base.py
git commit -m "docs(phase3): update BaseRiskOverlay contract for position-size semantic"
```

---

## Task 6: Build Phase 3 walk-forward script (T3.5)

**Files:**
- Create: `scripts/run_phase3_walkforward.py`

This script mirrors `scripts/run_phase2_walkforward.py` but uses real ATR + overlays + drawdown brake. We'll start by reading the Phase 2 script and adapting it.

- [ ] **Step 1: Read the Phase 2 walk-forward script for reference**

Run: `wc -l scripts/run_phase2_walkforward.py` (in your shell) — note the structure.

Specifically note:
- How it loads `results/backtest_trades.csv` for the 2024 fold equity simulation
- How it computes Kelly fractions per fold from `data/52t_volnorm/`
- How it writes `results/phase2_walkforward.tsv` and `results/phase2_gate_check.txt`

- [ ] **Step 2: Create `scripts/run_phase3_walkforward.py`**

This script mirrors Phase 2's exact pattern: `_rescale_trades` re-sizes each trade, `_build_equity_curve` aggregates by `exit_date` onto a business-day calendar, then `_sharpe`/`_max_dd` compute on **daily** returns (not per-trade). The key difference from Phase 2 is that `_rescale_trades` now also looks up real ATR from the 52T validation data and applies `apply_risk_adjustments`.

**Verified columns in `results/backtest_trades.csv`:**
`trade_id, ticker, sector, direction, entry_date, entry_price, exit_date, exit_price, position_pct, shares, gross_pnl, entry_friction_cost, exit_friction_cost, slippage_cost, spread_cost, total_costs, net_pnl, holding_days, exit_reason, confidence_at_entry`

**Important:** `b_ratio` is not a column. Compute `b_hist = wins.mean() / losses.abs().mean()` once from the trade file before the loop, exactly as Phase 2 does.

Create the script:

```python
"""
run_phase3_walkforward.py — Phase 3: Risk Engine Integration walk-forward.

Extends Phase 2 walk-forward with:
  1. Real ATR sizing (compute_atr_pct from 52T atr_14/Close columns)
  2. Drawdown brake (15% warn, 20% halt)
  3. FatigueAccumulationOverlay (decay_rate=0.15)
  4. LiquidityCongestionGate (defaults)

Mirrors Phase 2's _rescale_trades / _build_equity_curve pattern exactly.
The only differences are:
  - _rescale_trades now uses real per-ticker ATR (compute_atr_pct)
  - _rescale_trades applies apply_risk_adjustments (DD brake + overlays)
  - PnL is scaled from the original trade's position_pct to the new
    risk-adjusted position_pct (linear scaling, same as Phase 2)

Phase 3 gate (from fppe-roadmap-v2A.md §6):
  [ ] Drawdown brake fires correctly on synthetic 20% DD scenario
  [ ] Max DD ≤ 10% on walk-forward
  [ ] Sharpe ≥ 1.0 maintained from Phase 2
  [ ] Stop-loss fires ≤ 35% of exits

Usage:
    PYTHONUTF8=1 py -3.12 scripts/run_phase3_walkforward.py

Outputs:
    results/phase3_walkforward.tsv — summary metrics
    results/phase3_gate_check.txt  — gate verdict
    results/phase3_equity_curve.csv — Phase 3 daily equity curve
    results/phase3_blocked_trades.csv — blocked trade log with reasons

Spec: docs/superpowers/specs/2026-04-06-phase3-risk-engine-integration-design.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from trading_system.position_sizer import SizingConfig, size_position
from trading_system.risk_engine import (
    apply_risk_adjustments,
    compute_atr_pct,
    drawdown_brake_scalar,
)
from trading_system.risk_overlays.fatigue_accumulation import FatigueAccumulationOverlay
from trading_system.risk_overlays.liquidity_congestion import LiquidityCongestionGate

TRADES_PATH      = project_root / "results" / "backtest_trades.csv"
_52T_DIR         = project_root / "data" / "52t_volnorm"
RESULTS_DIR      = project_root / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_WF_TSV       = RESULTS_DIR / "phase3_walkforward.tsv"
OUT_GATE         = RESULTS_DIR / "phase3_gate_check.txt"
OUT_EQUITY       = RESULTS_DIR / "phase3_equity_curve.csv"
OUT_BLOCKED      = RESULTS_DIR / "phase3_blocked_trades.csv"

INITIAL_EQUITY   = 10_000.0
TRADE_DAYS_YEAR  = 252
RISK_FREE_ANNUAL = 0.045
SPY_THRESHOLD    = 0.05  # H7 locked

SIZING_CFG = SizingConfig()


# ─── Metrics (mirrors Phase 2) ────────────────────────────────────────────────

def _sharpe(daily_ret: np.ndarray) -> float:
    if len(daily_ret) < 2 or np.std(daily_ret, ddof=1) < 1e-10:
        return float("nan")
    rf_daily = RISK_FREE_ANNUAL / TRADE_DAYS_YEAR
    excess = daily_ret - rf_daily
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(TRADE_DAYS_YEAR))


def _max_dd(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    return float(dd.max())


def _compute_b_hist(trades: pd.DataFrame) -> float:
    """Compute historical win/loss ratio from the trade file (b_ratio for Kelly)."""
    wins = trades[trades["net_pnl"] > 0]["net_pnl"]
    losses = trades[trades["net_pnl"] < 0]["net_pnl"]
    if len(wins) == 0 or len(losses) == 0:
        return 1.0
    return float(wins.mean() / losses.abs().mean())


def _build_atr_lookup(val_data: pd.DataFrame) -> dict:
    """Index 52T validation data by (date, ticker) → atr_14, Close, ret_90d."""
    df = val_data.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df.set_index(["Date", "Ticker"])[["atr_14", "Close", "ret_90d"]].to_dict("index")


# ─── Trade rescaling with Phase 3 risk engine ─────────────────────────────────

def _rescale_trades_phase3(
    trades: pd.DataFrame,
    b_hist: float,
    atr_lookup: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    """Re-size each trade with real ATR + DD brake + overlays.

    Returns:
        (scaled_trades, blocked_log)
        scaled_trades has new columns: phase3_position_pct, phase3_net_pnl, phase3_blocked
        blocked_log: list of dicts with date, ticker, reason
    """
    out = trades.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    out["exit_date"]  = pd.to_datetime(out["exit_date"])
    out = out.sort_values("entry_date").reset_index(drop=True)

    fatigue = FatigueAccumulationOverlay(decay_rate=0.15)
    congestion = LiquidityCongestionGate()

    # Running equity for DD computation. Sample only on exit dates (when PnL realizes).
    running_equity = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    pending_pnls: dict = {}  # exit_date → cumulative PnL closing that day

    phase3_pcts = []
    phase3_pnls = []
    phase3_blocked = []
    blocked_log: list[dict] = []

    last_processed_exit = None

    for _, row in out.iterrows():
        entry_dt = row["entry_date"].date()
        exit_dt = row["exit_date"].date()
        ticker = row["ticker"]

        # Realize any pending PnLs whose exit_date is on or before this entry_dt.
        # This keeps running_equity (and DD) up to date for the brake check.
        for d in sorted(pending_pnls):
            if d <= entry_dt:
                running_equity += pending_pnls[d]
                if running_equity > peak_equity:
                    peak_equity = running_equity
                del pending_pnls[d]
            else:
                break

        # Update overlays at entry_dt with SPY market data
        spy_row = atr_lookup.get((entry_dt, "SPY"))
        if spy_row is not None:
            ret_90d = spy_row.get("ret_90d", 0.0) or 0.0
            regime = "BULL" if ret_90d > SPY_THRESHOLD else "BEAR"
            fatigue.update(entry_dt, regime_label=regime)
            congestion.update(entry_dt, atr=spy_row["atr_14"], close=spy_row["Close"])

        # Lookup ATR for this ticker on entry_dt
        ticker_row = atr_lookup.get((entry_dt, ticker))
        if ticker_row is None:
            blocked_log.append({"date": entry_dt, "ticker": ticker, "reason": "missing_data"})
            phase3_pcts.append(0.0)
            phase3_pnls.append(0.0)
            phase3_blocked.append(True)
            continue
        try:
            atr_pct = compute_atr_pct(ticker_row["atr_14"], ticker_row["Close"])
        except RuntimeError:
            blocked_log.append({"date": entry_dt, "ticker": ticker, "reason": "missing_atr"})
            phase3_pcts.append(0.0)
            phase3_pnls.append(0.0)
            phase3_blocked.append(True)
            continue

        # Phase 2 sizing with real ATR
        confidence = float(row["confidence_at_entry"])
        sizing = size_position(confidence=confidence, b_ratio=b_hist, config=SIZING_CFG, atr_pct=atr_pct)

        # Current drawdown from peak
        dd = max(0.0, 1.0 - running_equity / peak_equity)

        # Phase 3 adjustments
        adj = apply_risk_adjustments(
            sizing,
            drawdown=dd,
            overlays=[fatigue, congestion],
        )
        if adj.blocked:
            blocked_log.append({"date": entry_dt, "ticker": ticker, "reason": adj.block_reason})
            phase3_pcts.append(0.0)
            phase3_pnls.append(0.0)
            phase3_blocked.append(True)
            continue

        # Scale the trade's net_pnl from original position_pct to adj.final_position_pct
        original_pct = float(row["position_pct"])
        if original_pct <= 0:
            phase3_pcts.append(0.0)
            phase3_pnls.append(0.0)
            phase3_blocked.append(True)
            continue
        scale = adj.final_position_pct / original_pct
        scaled_pnl = float(row["net_pnl"]) * scale

        phase3_pcts.append(adj.final_position_pct)
        phase3_pnls.append(scaled_pnl)
        phase3_blocked.append(False)

        # Schedule the PnL to realize on exit_dt
        pending_pnls[exit_dt] = pending_pnls.get(exit_dt, 0.0) + scaled_pnl

    out["phase3_position_pct"] = phase3_pcts
    out["phase3_net_pnl"] = phase3_pnls
    out["phase3_blocked"] = phase3_blocked
    return out, blocked_log


def _build_equity_curve(trades: pd.DataFrame, net_pnl_col: str) -> pd.DataFrame:
    """Aggregate trade PnL onto a business-day calendar (mirrors Phase 2)."""
    trades = trades.copy()
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    min_date = pd.to_datetime(trades["entry_date"].min())
    max_date = pd.to_datetime(trades["exit_date"].max())
    dates = pd.bdate_range(min_date, max_date)
    pnl_by_date = trades.groupby("exit_date")[net_pnl_col].sum()

    equity = INITIAL_EQUITY
    rows = []
    for dt in dates:
        pnl_today = float(pnl_by_date.get(dt, 0.0))
        equity += pnl_today
        rows.append({"date": dt, "equity": equity, "daily_pnl": pnl_today})
    df = pd.DataFrame(rows)
    df["daily_return"] = df["equity"].pct_change().fillna(0.0)
    return df


# ─── Synthetic 20% DD scenario test (T3.4) ────────────────────────────────────

def synthetic_dd_test() -> bool:
    """Verify the DD brake transitions through the boundary points."""
    test_points = [(0.10, 1.0), (0.15, 1.0), (0.175, 0.5), (0.20, 0.0), (0.22, 0.0)]
    for dd, expected in test_points:
        actual = drawdown_brake_scalar(dd)
        if abs(actual - expected) > 1e-9:
            print(f"  FAIL: dd={dd}, expected={expected}, got={actual}")
            return False
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("Phase 3 Risk Engine Integration — Walk-Forward")
    print("=" * 70)

    # T3.4: Synthetic DD scenario
    print("\n[1/3] Synthetic 20% DD scenario test...")
    dd_test_pass = synthetic_dd_test()
    print("  ✓ DD brake transitions correctly" if dd_test_pass else "  ✗ DD brake test failed")

    # T3.5: Walk-forward with real ATR + overlays
    print("\n[2/3] 2024 fold simulation with real ATR + overlays...")
    trades = pd.read_csv(TRADES_PATH)
    print(f"  Loaded {len(trades)} Phase 1 trades")

    b_hist = _compute_b_hist(trades)
    print(f"  b_hist = {b_hist:.4f} (computed from net_pnl wins/losses)")

    val_data = pd.read_parquet(_52T_DIR / "val_db.parquet")
    atr_lookup = _build_atr_lookup(val_data)
    print(f"  ATR lookup built: {len(atr_lookup)} (date, ticker) entries")

    scaled, blocked_log = _rescale_trades_phase3(trades, b_hist, atr_lookup)
    n_blocked = scaled["phase3_blocked"].sum()
    n_placed = len(scaled) - n_blocked
    n_stopped = (
        (~scaled["phase3_blocked"])
        & scaled["exit_reason"].astype(str).str.lower().str.startswith("stop")
    ).sum()

    eq_df = _build_equity_curve(scaled, "phase3_net_pnl")
    eq_df.to_csv(OUT_EQUITY, index=False)
    pd.DataFrame(blocked_log).to_csv(OUT_BLOCKED, index=False)

    eq_arr = eq_df["equity"].values
    daily_ret = eq_df["daily_return"].values
    sharpe = _sharpe(daily_ret)
    max_dd = _max_dd(eq_arr)
    final_eq = float(eq_arr[-1])

    print(f"  Trades placed:  {n_placed}")
    print(f"  Trades blocked: {n_blocked}")
    print(f"  Stops fired:    {n_stopped}")
    print(f"  Final equity:   ${final_eq:,.2f}")
    print(f"  Sharpe (daily): {sharpe:.3f}")
    print(f"  Max DD:         {max_dd:.1%}")

    summary = pd.DataFrame([{
        "fold": "2024",
        "n_trades_placed": int(n_placed),
        "n_blocked": int(n_blocked),
        "n_stopped": int(n_stopped),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 4),
        "final_equity": round(final_eq, 2),
        "b_hist": round(b_hist, 4),
    }])
    summary.to_csv(OUT_WF_TSV, sep="\t", index=False)

    # T3.5 gate check
    print("\n[3/3] Gate check...")
    stop_pct = n_stopped / max(1, n_placed)
    gates = {
        "DD brake fires correctly":   dd_test_pass,
        "Max DD <= 10%":              max_dd <= 0.10,
        "Sharpe >= 1.0":              sharpe >= 1.0,
        "Stops <= 35% of trades":     stop_pct <= 0.35,
    }
    all_pass = all(gates.values())
    lines = [f"Phase 3 Gate Check — {'PASS' if all_pass else 'FAIL'}", "=" * 50, ""]
    for name, passed in gates.items():
        mark = "[X]" if passed else "[ ]"
        lines.append(f"  {mark} {name}")
    lines.append("")
    lines.append(
        f"Sharpe={sharpe:.3f}, MaxDD={max_dd:.1%}, "
        f"stops={n_stopped}/{n_placed} ({stop_pct:.1%}), "
        f"blocked={n_blocked}/{len(scaled)}"
    )
    OUT_GATE.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run the script**

Run: `PYTHONUTF8=1 py -3.12 scripts/run_phase3_walkforward.py`
Expected: produces `results/phase3_walkforward.tsv`, `results/phase3_gate_check.txt`, `results/phase3_equity_curve.csv`, `results/phase3_blocked_trades.csv`. The script should run end-to-end without errors. The gate may PASS or FAIL — both are acceptable execution outcomes (FAIL means the gate criteria weren't met, which is data, not a bug).

If the script crashes with a KeyError, it likely means a column assumption is wrong. Re-read `results/backtest_trades.csv` header (expected columns documented in Step 2 above) and verify the script matches.

- [ ] **Step 4: Inspect outputs**

Read: `results/phase3_gate_check.txt`
- If all 4 gates pass → proceed to step 5
- If any gate fails → analyze the failure, determine if it's a code bug or expected behavior change, and document in the campaign doc

- [ ] **Step 5: Commit**

```bash
git add scripts/run_phase3_walkforward.py results/phase3_walkforward.tsv results/phase3_gate_check.txt results/phase3_equity_curve.csv results/phase3_blocked_trades.csv
git commit -m "feat(phase3): add walk-forward comparison script (T3.5)"
```

---

## Task 7: Update campaign doc and CLAUDE.md

**Files:**
- Modify: `docs/PHASE2_RISK_ENGINE.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `docs/PHASE2_RISK_ENGINE.md` with Phase 3 progress**

Add a new section after Phase B status:

```markdown
### Phase 3: Risk Engine Integration (COMPLETE / IN PROGRESS — fill in)

- [x] T3.1: trading_system/risk_engine.py — compute_atr_pct, drawdown_brake_scalar, apply_risk_adjustments
- [x] T3.2: position_sizer.py atr_pct override
- [x] T3.3: tests/test_risk_engine.py — N tests passing
- [x] T3.4: synthetic 20% DD scenario test
- [x] T3.5: scripts/run_phase3_walkforward.py — gate result: PASS/FAIL

Gate check (results/phase3_gate_check.txt):
  [X/ ] DD brake fires correctly
  [X/ ] Max DD <= 10%
  [X/ ] Sharpe >= 1.0
  [X/ ] Stops <= 35% of trades

Provenance: results/phase3_walkforward.tsv, results/phase3_gate_check.txt
```

Replace the bracketed values with actuals from the gate check file.

- [ ] **Step 2: Update `CLAUDE.md`**

In the **Current Phase** section, replace the existing Phase 2 COMPLETE block with Phase 3 status. Specifically:

Replace this line:
```
**Phase 2 Half-Kelly — COMPLETE. Phase 3 (Risk Engine Integration) is next.**
```

With:
```
**Phase 3 Risk Engine Integration — COMPLETE. Phase 4 (Portfolio Manager) is next.**
```

And update the Phase 3 status block to reflect actual results.

- [ ] **Step 3: Run full test suite one more time**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add docs/PHASE2_RISK_ENGINE.md CLAUDE.md
git commit -m "docs(phase3): update campaign doc and CLAUDE.md with Phase 3 results"
```

---

## Final Verification Checklist

- [ ] All new tests pass
- [ ] All existing tests still pass (no regressions)
- [ ] `results/phase3_walkforward.tsv` exists and shows reasonable Sharpe / DD numbers
- [ ] `results/phase3_gate_check.txt` documents gate verdict
- [ ] All commits follow conventional format `feat(phase3):` / `docs(phase3):` / `test(phase3):`
- [ ] `CLAUDE.md` Current Phase section reflects Phase 3 complete

## Rollback Plan

If Phase 3 walk-forward gate fails:
1. Do **not** revert the code — the unit tests are sound and the orchestrator is correct
2. The fallback (per roadmap §6) is to disable fatigue + congestion overlays and keep ATR sizing + DD brake
3. To do this, modify `run_phase3_walkforward.py` to pass `overlays=None` (or `overlays=[]`)
4. Re-run, document new gate result in campaign doc, and decide whether to proceed to Phase 4 with reduced Phase 3 scope
