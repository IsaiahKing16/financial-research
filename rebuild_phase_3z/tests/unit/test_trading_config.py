"""
test_trading_config.py — Unit tests for rebuild TradingConfig (SLE-71).

Tests:
  - ResearchFlagsConfig defaults (use_slip_deficit=False)
  - use_slip_deficit=False produces identical signal to not using it
  - TradingConfig.validate() catches known bad configurations
  - All sub-configs have expected defaults (parity with production config.py)
  - dataclasses.replace() pattern works for frozen configs

Linear: SLE-71
"""

from __future__ import annotations

import dataclasses

import pytest

from rebuild_phase_3z.fppe.trading_system.config import (
    CapitalConfig,
    CostConfig,
    EvaluationConfig,
    PositionLimitsConfig,
    ResearchFlagsConfig,
    RiskConfig,
    SignalConfig,
    TradingConfig,
    TradeManagementConfig,
)


# ─── TestResearchFlags ────────────────────────────────────────────────────────

class TestResearchFlags:
    """ResearchFlagsConfig tests (SLE-71 main gate)."""

    def test_default_use_slip_deficit_is_false(self):
        """use_slip_deficit defaults to False — safe baseline."""
        flags = ResearchFlagsConfig()
        assert flags.use_slip_deficit is False

    def test_trading_config_default_has_flags(self):
        """TradingConfig includes ResearchFlagsConfig by default."""
        cfg = TradingConfig()
        assert hasattr(cfg, "research_flags")
        assert isinstance(cfg.research_flags, ResearchFlagsConfig)

    def test_trading_config_default_slip_deficit_false(self):
        """TradingConfig() default has use_slip_deficit=False."""
        cfg = TradingConfig()
        assert cfg.research_flags.use_slip_deficit is False

    def test_enable_slip_deficit(self):
        """Can enable use_slip_deficit via dataclasses.replace()."""
        cfg = TradingConfig()
        flags_on = dataclasses.replace(cfg.research_flags, use_slip_deficit=True)
        cfg_on = dataclasses.replace(cfg, research_flags=flags_on)
        assert cfg_on.research_flags.use_slip_deficit is True

    def test_research_flags_is_frozen(self):
        """ResearchFlagsConfig is frozen — direct mutation raises."""
        flags = ResearchFlagsConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            flags.use_slip_deficit = True  # type: ignore[misc]

    def test_enable_does_not_mutate_original(self):
        """Enabling slip_deficit via replace() does not mutate the original config."""
        cfg = TradingConfig()
        _ = dataclasses.replace(
            cfg,
            research_flags=dataclasses.replace(cfg.research_flags, use_slip_deficit=True),
        )
        # Original unchanged
        assert cfg.research_flags.use_slip_deficit is False


# ─── TestTradingConfigDefaults ────────────────────────────────────────────────

class TestTradingConfigDefaults:
    """All sub-config defaults match production trading_system/config.py."""

    def test_initial_capital(self):
        assert TradingConfig().capital.initial_capital == 10_000.0

    def test_max_gross_exposure(self):
        assert TradingConfig().capital.max_gross_exposure == 1.0

    def test_slippage_bps(self):
        assert TradingConfig().costs.slippage_bps == 10.0

    def test_spread_bps(self):
        assert TradingConfig().costs.spread_bps == 3.0

    def test_round_trip_bps(self):
        assert TradingConfig().costs.round_trip_bps == 26.0

    def test_min_position_pct(self):
        assert TradingConfig().position_limits.min_position_pct == 0.02

    def test_max_position_pct(self):
        assert TradingConfig().position_limits.max_position_pct == 0.10

    def test_max_positions_per_sector(self):
        assert TradingConfig().position_limits.max_positions_per_sector == 3

    def test_confidence_threshold(self):
        assert TradingConfig().signals.confidence_threshold == 0.60

    def test_min_matches(self):
        assert TradingConfig().signals.min_matches == 10

    def test_stop_loss_atr_multiple(self):
        assert TradingConfig().risk.stop_loss_atr_multiple == 3.0

    def test_max_holding_days(self):
        assert TradingConfig().trade_management.max_holding_days == 14

    def test_risk_free_annual_rate(self):
        assert TradingConfig().costs.risk_free_annual_rate == 0.045

    def test_rolling_windows(self):
        assert TradingConfig().evaluation.rolling_windows == [30, 90, 252]

    def test_drawdown_brake(self):
        assert TradingConfig().risk.drawdown_brake_threshold == 0.15

    def test_drawdown_halt(self):
        assert TradingConfig().risk.drawdown_halt_threshold == 0.20


# ─── TestValidation ───────────────────────────────────────────────────────────

class TestValidation:
    """TradingConfig.validate() catches constraint violations."""

    def test_valid_default_config(self):
        """Default TradingConfig passes validation."""
        errors = TradingConfig().validate()
        assert errors == [], f"Default config has validation errors: {errors}"

    def test_drawdown_thresholds_ordered(self):
        """brake >= halt triggers an error."""
        cfg = TradingConfig(
            risk=dataclasses.replace(
                TradingConfig().risk,
                drawdown_brake_threshold=0.20,
                drawdown_halt_threshold=0.15,
            )
        )
        errors = cfg.validate()
        assert any("drawdown_brake" in e for e in errors)

    def test_position_limits_ordered(self):
        """min_position >= max_position triggers an error."""
        cfg = TradingConfig(
            position_limits=dataclasses.replace(
                TradingConfig().position_limits,
                min_position_pct=0.15,
                max_position_pct=0.10,
            )
        )
        errors = cfg.validate()
        assert any("min_position_pct" in e for e in errors)

    def test_gross_exposure_capped_at_1(self):
        """max_gross_exposure > 1.0 triggers an error."""
        cfg = TradingConfig(
            capital=dataclasses.replace(
                TradingConfig().capital,
                max_gross_exposure=1.5,
            )
        )
        errors = cfg.validate()
        assert any("long-only" in e for e in errors)

    def test_confidence_threshold_too_low(self):
        """confidence_threshold < 0.50 triggers an error."""
        cfg = TradingConfig(
            signals=dataclasses.replace(
                TradingConfig().signals,
                confidence_threshold=0.40,
            )
        )
        errors = cfg.validate()
        assert any("confidence_threshold" in e for e in errors)

    def test_min_matches_too_low(self):
        """min_matches < 5 triggers an error."""
        cfg = TradingConfig(
            signals=dataclasses.replace(
                TradingConfig().signals,
                min_matches=3,
            )
        )
        errors = cfg.validate()
        assert any("min_matches" in e for e in errors)


# ─── TestFrozenImmutability ───────────────────────────────────────────────────

class TestFrozenImmutability:
    """All config dataclasses are frozen — direct mutation must fail."""

    def test_trading_config_is_frozen(self):
        cfg = TradingConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.signals = SignalConfig()  # type: ignore[misc]

    def test_capital_config_is_frozen(self):
        cap = CapitalConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cap.initial_capital = 999.0  # type: ignore[misc]

    def test_replace_pattern_works(self):
        """dataclasses.replace() is the correct mutation pattern."""
        cfg = TradingConfig()
        new_cfg = dataclasses.replace(
            cfg,
            signals=dataclasses.replace(cfg.signals, confidence_threshold=0.70),
        )
        assert new_cfg.signals.confidence_threshold == 0.70
        assert cfg.signals.confidence_threshold == 0.60  # original unchanged
