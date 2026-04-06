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
