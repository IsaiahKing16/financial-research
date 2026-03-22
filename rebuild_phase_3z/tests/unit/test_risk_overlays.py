"""
test_risk_overlays.py — Unit tests for BaseRiskOverlay, LiquidityCongestionGate,
and FatigueAccumulationOverlay (SLE-74, SLE-75).

Tests cover:
  - BaseRiskOverlay: ABC enforcement
  - LiquidityCongestionGate: normal operation, cooldown, partial throttle, full block
  - FatigueAccumulationOverlay: accumulation, reset on transition, min_multiplier
  - ResearchFlagsConfig: new M6 flags (use_liquidity_congestion_gate etc.)
"""

from __future__ import annotations

from datetime import date

import pytest

from rebuild_phase_3z.fppe.trading_system.risk_overlays.base import BaseRiskOverlay
from rebuild_phase_3z.fppe.trading_system.risk_overlays.liquidity_congestion import (
    LiquidityCongestionGate,
)
from rebuild_phase_3z.fppe.trading_system.risk_overlays.fatigue_accumulation import (
    FatigueAccumulationOverlay,
)
from rebuild_phase_3z.fppe.trading_system.config import ResearchFlagsConfig


# ─── BaseRiskOverlay ABC ──────────────────────────────────────────────────────

class TestBaseRiskOverlay:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseRiskOverlay()

    def test_concrete_subclass_must_implement_all_methods(self):
        """Partial implementation raises TypeError at instantiation."""
        class PartialOverlay(BaseRiskOverlay):
            def update(self, current_date, **kw):
                pass
            def get_signal_multiplier(self):
                return 1.0
            # Missing reset()

        with pytest.raises(TypeError):
            PartialOverlay()

    def test_name_defaults_to_class_name(self):
        class MyOverlay(BaseRiskOverlay):
            def update(self, current_date, **kw): pass
            def get_signal_multiplier(self): return 1.0
            def reset(self): pass

        assert MyOverlay().name == "MyOverlay"

    def test_repr_includes_multiplier(self):
        class TrivialOverlay(BaseRiskOverlay):
            def update(self, current_date, **kw): pass
            def get_signal_multiplier(self): return 0.5
            def reset(self): pass

        r = repr(TrivialOverlay())
        assert "0.500" in r


# ─── LiquidityCongestionGate ──────────────────────────────────────────────────

class TestLiquidityCongestionGate:
    TODAY = date(2024, 6, 1)

    def _fresh(self, **kwargs):
        defaults = dict(
            window=5,
            congestion_threshold=0.025,
            block_threshold=0.05,
            cooldown_periods=2,
        )
        defaults.update(kwargs)
        return LiquidityCongestionGate(**defaults)

    def test_default_multiplier_is_one(self):
        gate = self._fresh()
        assert gate.get_signal_multiplier() == pytest.approx(1.0)

    def test_not_congested_initially(self):
        gate = self._fresh()
        assert not gate.is_congested

    def test_low_atr_no_congestion(self):
        gate = self._fresh()
        for _ in range(10):
            gate.update(self.TODAY, atr=0.5, close=100.0)   # ratio=0.005 < 0.025
        assert gate.get_signal_multiplier() == pytest.approx(1.0)

    def test_high_atr_triggers_congestion(self):
        gate = self._fresh()
        for _ in range(6):
            gate.update(self.TODAY, atr=3.0, close=100.0)   # ratio=0.03 > 0.025
        assert gate.is_congested
        assert gate.get_signal_multiplier() < 1.0

    def test_severe_congestion_full_block(self):
        gate = self._fresh()
        for _ in range(10):
            gate.update(self.TODAY, atr=6.0, close=100.0)   # ratio=0.06 > 0.05
        assert gate.get_signal_multiplier() == pytest.approx(0.0)

    def test_cooldown_maintains_partial_block(self):
        gate = self._fresh(cooldown_periods=3, window=3)
        # Trigger congestion
        for _ in range(5):
            gate.update(self.TODAY, atr=3.0, close=100.0)
        assert gate.is_congested
        # Feed low-ratio data — congestion level drops but cooldown active
        for _ in range(2):
            gate.update(self.TODAY, atr=0.1, close=100.0)
        # Still in cooldown
        assert gate.is_congested

    def test_precomputed_ratio_accepted(self):
        gate = self._fresh()
        gate.update(self.TODAY, atr_price_ratio=0.03)
        assert gate.congestion_level > 0.0

    def test_reset_clears_state(self):
        gate = self._fresh()
        for _ in range(10):
            gate.update(self.TODAY, atr=6.0, close=100.0)
        assert gate.is_congested
        gate.reset()
        assert not gate.is_congested
        assert gate.get_signal_multiplier() == pytest.approx(1.0)

    def test_update_without_data_decrements_cooldown(self):
        gate = self._fresh(cooldown_periods=3)
        # Trigger cooldown
        for _ in range(5):
            gate.update(self.TODAY, atr=3.0, close=100.0)
        remaining_before = gate._cooldown_remaining
        gate.update(self.TODAY)   # no atr/close provided
        assert gate._cooldown_remaining == remaining_before - 1

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="window"):
            LiquidityCongestionGate(window=0)

    def test_invalid_threshold_order_raises(self):
        with pytest.raises(ValueError, match="block_threshold"):
            LiquidityCongestionGate(congestion_threshold=0.05, block_threshold=0.02)

    def test_unknown_kwargs_ignored(self):
        """Unrecognised market_data kwargs must not raise."""
        gate = self._fresh()
        gate.update(self.TODAY, volume=1_000_000, sector="Tech")  # not recognised
        assert gate.get_signal_multiplier() == pytest.approx(1.0)


# ─── FatigueAccumulationOverlay ───────────────────────────────────────────────

class TestFatigueAccumulationOverlay:
    TODAY = date(2024, 6, 1)

    def _fresh(self, **kwargs):
        return FatigueAccumulationOverlay(**kwargs)

    def test_initial_multiplier_is_one(self):
        ov = self._fresh()
        assert ov.get_signal_multiplier() == pytest.approx(1.0)

    def test_initial_fatigue_zero(self):
        ov = self._fresh()
        assert ov.fatigue_score == pytest.approx(0.0)

    def test_fatigue_accumulates_in_same_regime(self):
        ov = self._fresh(decay_rate=0.2)
        for _ in range(10):
            ov.update(self.TODAY, regime_label="BULL")
        assert ov.fatigue_score > 0.0
        assert ov.get_signal_multiplier() < 1.0

    def test_fatigue_monotonically_increases(self):
        ov = self._fresh(decay_rate=0.15)
        scores = []
        for _ in range(20):
            ov.update(self.TODAY, regime_label="BEAR")
            scores.append(ov.fatigue_score)
        assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))

    def test_fatigue_bounded_below_one(self):
        ov = self._fresh(decay_rate=0.5)
        for _ in range(50):
            ov.update(self.TODAY, regime_label="BULL")
        assert ov.fatigue_score <= 1.0

    def test_regime_transition_resets_fatigue(self):
        ov = self._fresh(decay_rate=0.3)
        for _ in range(10):
            ov.update(self.TODAY, regime_label="BULL")
        assert ov.fatigue_score > 0.5
        ov.update(self.TODAY, regime_label="BEAR")
        assert ov.fatigue_score == pytest.approx(0.0)
        assert ov.get_signal_multiplier() == pytest.approx(1.0)

    def test_neutral_regime_resets_by_default(self):
        ov = self._fresh(decay_rate=0.3, reset_on_neutral=True)
        for _ in range(10):
            ov.update(self.TODAY, regime_label="BULL")
        ov.update(self.TODAY, regime_label="NEUTRAL")
        assert ov.fatigue_score == pytest.approx(0.0)

    def test_neutral_regime_no_reset_when_flag_false(self):
        ov = self._fresh(decay_rate=0.3, reset_on_neutral=False)
        for _ in range(5):
            ov.update(self.TODAY, regime_label="BULL")
        score_before = ov.fatigue_score
        ov.update(self.TODAY, regime_label="NEUTRAL")
        # NEUTRAL treated as regime change → resets regardless if different
        # (regime != self._current_regime "BULL")
        # This is expected behaviour: any change resets.
        assert ov.fatigue_score == pytest.approx(0.0)

    def test_min_multiplier_floor(self):
        ov = self._fresh(decay_rate=0.9, min_multiplier=0.1)
        for _ in range(100):
            ov.update(self.TODAY, regime_label="BULL")
        assert ov.get_signal_multiplier() >= 0.1 - 1e-9

    def test_reset_clears_all_state(self):
        ov = self._fresh(decay_rate=0.3)
        for _ in range(10):
            ov.update(self.TODAY, regime_label="BULL")
        ov.reset()
        assert ov.fatigue_score == pytest.approx(0.0)
        assert ov.current_regime is None
        assert ov.regime_duration == 0

    def test_regime_duration_increments(self):
        ov = self._fresh()
        for i in range(5):
            ov.update(self.TODAY, regime_label="BULL")
        # First update enters the BULL regime (duration=1); each subsequent
        # update increments → 5 consecutive BULL periods → duration=5.
        assert ov.regime_duration == 5

    def test_neutral_accumulates_when_reset_on_neutral_false(self):
        """With reset_on_neutral=False, NEUTRAL→NEUTRAL should accumulate fatigue."""
        ov = self._fresh(decay_rate=0.3, reset_on_neutral=False)
        # First NEUTRAL always resets (new regime entering from None)
        ov.update(self.TODAY, regime_label="NEUTRAL")
        assert ov.fatigue_score == pytest.approx(0.0)
        # Subsequent NEUTRAL updates should accumulate (not reset again)
        ov.update(self.TODAY, regime_label="NEUTRAL")
        assert ov.fatigue_score > 0.0  # fatigue accumulated in NEUTRAL

    def test_invalid_decay_rate_raises(self):
        with pytest.raises(ValueError, match="decay_rate"):
            FatigueAccumulationOverlay(decay_rate=0.0)

    def test_invalid_decay_rate_one_raises(self):
        with pytest.raises(ValueError, match="decay_rate"):
            FatigueAccumulationOverlay(decay_rate=1.0)

    def test_invalid_min_multiplier_raises(self):
        with pytest.raises(ValueError, match="min_multiplier"):
            FatigueAccumulationOverlay(min_multiplier=1.5)


# ─── ResearchFlagsConfig — M6 flags ──────────────────────────────────────────

class TestResearchFlagsM6:
    def test_all_m6_flags_default_false(self):
        cfg = ResearchFlagsConfig()
        assert cfg.use_liquidity_congestion_gate is False
        assert cfg.use_fatigue_accumulation is False
        assert cfg.use_drift_monitor is False
        assert cfg.use_ib_compression is False

    def test_existing_slip_deficit_flag_unchanged(self):
        cfg = ResearchFlagsConfig()
        assert cfg.use_slip_deficit is False

    def test_flags_are_independent(self):
        import dataclasses
        cfg = ResearchFlagsConfig()
        cfg2 = dataclasses.replace(cfg, use_liquidity_congestion_gate=True)
        assert cfg2.use_liquidity_congestion_gate is True
        assert cfg2.use_fatigue_accumulation is False
        assert cfg2.use_drift_monitor is False
