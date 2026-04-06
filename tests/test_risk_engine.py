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
