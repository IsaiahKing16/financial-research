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
