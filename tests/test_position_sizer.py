"""
tests/test_position_sizer.py — Unit tests for Phase 2 Half-Kelly position sizer.

Covers:
  - Kelly fraction computation (formula correctness, edge cases)
  - Position sizing (clamp behavior, scaling, rejections)
  - SizingConfig validation
  - SizingResult immutability

Provenance values:
  p=0.6079, b=1.1811 from results/backtest_trades.csv (278 trades, 2024 fold).
  Expected Kelly ≈ 0.2760, half-Kelly ≈ 0.1380.
"""

import math
import pytest

from trading_system.position_sizer import (
    SizingConfig,
    SizingResult,
    compute_kelly_fraction,
    size_position,
)


# ─── compute_kelly_fraction ────────────────────────────────────────────────────

class TestComputeKellyFraction:
    def test_known_values_from_backtest(self):
        """Kelly(p=0.608, b=1.181) ≈ 0.276 from 2024 backtest trades."""
        result = compute_kelly_fraction(p=0.6079, b=1.1811)
        assert abs(result - 0.2760) < 0.001

    def test_zero_edge_returns_zero(self):
        """p*b == q → Kelly == 0 (breakeven)."""
        # For b=1.0 (symmetric wins/losses), breakeven at p=0.5
        result = compute_kelly_fraction(p=0.5, b=1.0)
        assert abs(result) < 1e-10

    def test_negative_edge_returns_negative(self):
        """p < q for any b=1 → Kelly < 0."""
        result = compute_kelly_fraction(p=0.4, b=1.0)
        assert result < 0

    def test_high_edge_capped_by_formula(self):
        """Very high p with b=1 → Kelly approaches 2p-1."""
        result = compute_kelly_fraction(p=0.9, b=1.0)
        expected = 2 * 0.9 - 1  # = 0.8
        assert abs(result - expected) < 1e-9

    def test_large_b_lowers_breakeven_p(self):
        """With b=5 (5:1 win/loss), breakeven p = 1/6 ≈ 0.167."""
        # At exactly breakeven: p*b - q = 0 → p*(b+1) = 1 → p = 1/(b+1)
        b = 5.0
        breakeven_p = 1 / (b + 1)
        result = compute_kelly_fraction(p=breakeven_p, b=b)
        assert abs(result) < 1e-9

    def test_invalid_p_zero_raises(self):
        with pytest.raises(RuntimeError, match="Win probability"):
            compute_kelly_fraction(p=0.0, b=1.0)

    def test_invalid_p_one_raises(self):
        with pytest.raises(RuntimeError, match="Win probability"):
            compute_kelly_fraction(p=1.0, b=1.0)

    def test_invalid_b_zero_raises(self):
        with pytest.raises(RuntimeError, match="Win/loss ratio"):
            compute_kelly_fraction(p=0.6, b=0.0)

    def test_invalid_b_negative_raises(self):
        with pytest.raises(RuntimeError, match="Win/loss ratio"):
            compute_kelly_fraction(p=0.6, b=-1.0)


# ─── SizingConfig ─────────────────────────────────────────────────────────────

class TestSizingConfig:
    def test_defaults_are_reasonable(self):
        cfg = SizingConfig()
        assert cfg.max_loss_per_trade_pct == 0.02
        assert cfg.stop_loss_atr_multiple == 3.0
        assert cfg.flat_atr_pct == 0.02
        assert cfg.kelly_multiplier == 0.5
        assert cfg.min_position_pct == 0.02
        assert cfg.max_position_pct == 0.10

    def test_frozen_prevents_mutation(self):
        cfg = SizingConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.kelly_multiplier = 0.25  # type: ignore[misc]

    def test_atr_weight_derivation(self):
        """Default atr_weight = 0.02 / (3.0 × 0.02) = 0.333..."""
        cfg = SizingConfig()
        stop_distance = cfg.stop_loss_atr_multiple * cfg.flat_atr_pct
        atr_weight = cfg.max_loss_per_trade_pct / stop_distance
        assert abs(atr_weight - 1 / 3) < 1e-9

    def test_invalid_kelly_multiplier_raises(self):
        with pytest.raises(RuntimeError, match="kelly_multiplier"):
            SizingConfig(kelly_multiplier=0.0)

    def test_invalid_min_max_raises(self):
        with pytest.raises(RuntimeError, match="min_position_pct"):
            SizingConfig(min_position_pct=0.15, max_position_pct=0.10)


# ─── size_position ────────────────────────────────────────────────────────────

class TestSizePosition:
    """Core sizing behavior tests."""

    # Default config for convenience
    CFG = SizingConfig()

    def test_approved_with_positive_kelly(self):
        result = size_position(confidence=0.65, b_ratio=1.2)
        assert result.approved is True
        assert result.position_pct > 0
        assert result.rejection_reason is None

    def test_rejected_when_kelly_negative(self):
        """Low confidence + poor odds → negative Kelly → reject."""
        # p=0.4, b=1.0 → Kelly = -0.2
        result = size_position(confidence=0.4, b_ratio=1.0)
        assert result.approved is False
        assert result.position_pct == 0.0
        assert "non-positive" in result.rejection_reason

    def test_rejected_when_kelly_zero(self):
        """p=0.5, b=1.0 → Kelly=0 → reject (no edge)."""
        result = size_position(confidence=0.5, b_ratio=1.0)
        assert result.approved is False

    def test_clamped_to_max_when_very_high_confidence(self):
        """Very high p with large b → raw_pct > max; clamp to max_position_pct."""
        result = size_position(confidence=0.999, b_ratio=100.0, config=self.CFG)
        assert result.approved is True
        assert result.position_pct == self.CFG.max_position_pct

    def test_clamped_to_min_when_small_kelly(self):
        """Just above zero Kelly → raw_pct < min; clamp to min_position_pct."""
        cfg = SizingConfig(min_position_pct=0.02, max_position_pct=0.10)
        # Tune to produce tiny raw_pct: very large ATR multiple, tiny max_loss
        cfg_narrow = SizingConfig(
            max_loss_per_trade_pct=0.001,
            stop_loss_atr_multiple=3.0,
            flat_atr_pct=0.02,
            kelly_multiplier=0.5,
            min_position_pct=0.02,
            max_position_pct=0.10,
        )
        result = size_position(confidence=0.51, b_ratio=1.01, config=cfg_narrow)
        if result.approved:
            assert result.position_pct >= cfg_narrow.min_position_pct

    def test_scales_with_confidence(self):
        """Higher confidence → larger Kelly → larger position (before clamping)."""
        low = size_position(confidence=0.60, b_ratio=1.2, config=self.CFG)
        high = size_position(confidence=0.75, b_ratio=1.2, config=self.CFG)
        assert low.approved and high.approved
        assert high.kelly_fraction > low.kelly_fraction

    def test_half_kelly_is_half_of_full(self):
        """scaled_kelly == 0.5 × kelly_fraction with default config."""
        result = size_position(confidence=0.65, b_ratio=1.2, config=self.CFG)
        assert result.approved
        assert abs(result.scaled_kelly - 0.5 * result.kelly_fraction) < 1e-12

    def test_result_is_frozen(self):
        result = size_position(confidence=0.65, b_ratio=1.2)
        with pytest.raises((AttributeError, TypeError)):
            result.position_pct = 0.99  # type: ignore[misc]

    def test_invalid_confidence_zero(self):
        result = size_position(confidence=0.0, b_ratio=1.2)
        assert result.approved is False
        assert "confidence" in result.rejection_reason

    def test_invalid_confidence_one(self):
        result = size_position(confidence=1.0, b_ratio=1.2)
        assert result.approved is False

    def test_invalid_b_ratio_zero(self):
        result = size_position(confidence=0.65, b_ratio=0.0)
        assert result.approved is False
        assert "b_ratio" in result.rejection_reason

    def test_default_config_used_when_none(self):
        """size_position(confidence, b) uses SizingConfig() when config=None."""
        r1 = size_position(confidence=0.65, b_ratio=1.2, config=None)
        r2 = size_position(confidence=0.65, b_ratio=1.2, config=SizingConfig())
        assert r1.position_pct == r2.position_pct

    def test_provenance_known_values(self):
        """position_pct with 2024 backtest params falls in [2%, 10%]."""
        # p=0.608, b=1.181 → Kelly=0.276, half-Kelly=0.138
        # atr_weight=0.333, raw_pct = 0.333 × 0.138 ≈ 0.046
        result = size_position(confidence=0.6079, b_ratio=1.1811)
        assert result.approved is True
        assert 0.02 <= result.position_pct <= 0.10
        # raw_pct ≈ 4.6% — should not be clamped
        assert abs(result.position_pct - 0.046) < 0.005

    def test_quarter_kelly_config(self):
        """quarter_kelly_multiplier=0.25 gives half the size of half-Kelly."""
        half_cfg = SizingConfig(kelly_multiplier=0.5)
        qtr_cfg = SizingConfig(kelly_multiplier=0.25)
        half_r = size_position(confidence=0.65, b_ratio=1.5, config=half_cfg)
        qtr_r = size_position(confidence=0.65, b_ratio=1.5, config=qtr_cfg)
        # Assuming neither is clamped
        if not (half_r.position_pct == half_cfg.min_position_pct or
                half_r.position_pct == half_cfg.max_position_pct):
            assert abs(qtr_r.scaled_kelly - 0.5 * half_r.scaled_kelly) < 1e-9


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
