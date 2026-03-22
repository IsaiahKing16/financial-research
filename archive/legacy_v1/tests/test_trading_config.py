"""
tests/test_trading_config.py — Regression tests for trading_system/config.py

Covers:
  - All dataclasses are frozen (immutability enforced)
  - Default TradingConfig validates cleanly
  - from_profile() produces correct values for all three profiles
  - validate() catches each class of invalid configuration
  - CostConfig properties (total_entry_bps, total_exit_bps, round_trip_bps)
  - DEFAULT_CONFIG is a valid pre-built instance
  - summary() executes without raising
"""

import dataclasses
import pytest

from trading_system.config import (
    TradingConfig,
    CapitalConfig,
    CostConfig,
    PositionLimitsConfig,
    SignalConfig,
    TradeManagementConfig,
    RiskConfig,
    EvaluationConfig,
    DEFAULT_CONFIG,
    SECTOR_MAP,
    ALL_TICKERS,
)


# ============================================================
# Helpers
# ============================================================

def _make_config(**overrides) -> TradingConfig:
    """Build a TradingConfig with specific sub-config overrides."""
    default = TradingConfig()
    return dataclasses.replace(default, **overrides)


# ============================================================
# Frozen-dataclass immutability
# ============================================================

class TestFrozen:
    """All config dataclasses must be frozen — mutation must raise."""

    def test_capital_config_frozen(self):
        cfg = CapitalConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.initial_capital = 999.0

    def test_cost_config_frozen(self):
        cfg = CostConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.slippage_bps = 99.0

    def test_position_limits_config_frozen(self):
        cfg = PositionLimitsConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.max_position_pct = 0.99

    def test_signal_config_frozen(self):
        cfg = SignalConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.confidence_threshold = 0.99

    def test_trade_management_config_frozen(self):
        cfg = TradeManagementConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.max_holding_days = 999

    def test_risk_config_frozen(self):
        cfg = RiskConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.stop_loss_atr_multiple = 99.0

    def test_evaluation_config_frozen(self):
        cfg = EvaluationConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.min_trades_for_metrics = 999

    def test_trading_config_frozen(self):
        cfg = TradingConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.signals = SignalConfig()


# ============================================================
# Default config
# ============================================================

class TestDefaultConfig:
    """TradingConfig() and DEFAULT_CONFIG must be valid out of the box."""

    def test_default_config_validates(self):
        assert TradingConfig().validate() == []

    def test_default_config_instance_is_valid(self):
        assert DEFAULT_CONFIG.validate() == []

    def test_default_capital(self):
        assert TradingConfig().capital.initial_capital == 10_000.0

    def test_default_confidence_threshold(self):
        """0.60 is empirically optimal per 2024 sweep (see config docstring)."""
        assert TradingConfig().signals.confidence_threshold == 0.60

    def test_default_max_holding_days(self):
        """14 days is empirically optimal (peak Sharpe 1.82)."""
        assert TradingConfig().trade_management.max_holding_days == 14

    def test_sector_map_populated(self):
        assert len(TradingConfig().sector_map) == 52

    def test_all_tickers_list(self):
        assert len(ALL_TICKERS) == 52
        assert "AAPL" in ALL_TICKERS
        assert "SPY" in ALL_TICKERS


# ============================================================
# CostConfig properties
# ============================================================

class TestCostConfigProperties:
    """Property calculations must be arithmetically correct."""

    def test_round_trip_bps_default(self):
        """Default: 10 slippage + 3 spread, each side → (10+3)×2 = 26 bps."""
        cfg = CostConfig()
        assert cfg.total_entry_bps == 13.0
        assert cfg.total_exit_bps == 13.0
        assert cfg.round_trip_bps == 26.0

    def test_round_trip_bps_custom(self):
        cfg = CostConfig(slippage_bps=5.0, spread_bps=2.0)
        assert cfg.round_trip_bps == 14.0

    def test_entry_exit_symmetry(self):
        """Long-only v1 has identical entry and exit friction."""
        cfg = CostConfig()
        assert cfg.total_entry_bps == cfg.total_exit_bps


# ============================================================
# from_profile()
# ============================================================

class TestFromProfile:
    """Named profiles must produce correct values and pass validation."""

    def test_aggressive_equals_default(self):
        cfg = TradingConfig.from_profile("aggressive")
        assert cfg.signals.confidence_threshold == 0.60
        assert cfg.trade_management.max_holding_days == 14
        assert cfg.validate() == []

    def test_moderate_profile(self):
        cfg = TradingConfig.from_profile("moderate")
        assert cfg.signals.confidence_threshold == 0.63
        assert cfg.trade_management.max_holding_days == 12
        assert cfg.position_limits.max_position_pct == 0.08
        assert cfg.risk.drawdown_halt_threshold == 0.18
        assert cfg.risk.drawdown_brake_threshold == 0.12
        assert cfg.validate() == []

    def test_conservative_profile(self):
        cfg = TradingConfig.from_profile("conservative")
        assert cfg.signals.confidence_threshold == 0.68
        assert cfg.trade_management.max_holding_days == 10
        assert cfg.position_limits.max_position_pct == 0.07
        assert cfg.risk.drawdown_halt_threshold == 0.15
        assert cfg.risk.drawdown_brake_threshold == 0.10
        assert cfg.validate() == []

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            TradingConfig.from_profile("ultra_aggressive")

    def test_profiles_are_frozen(self):
        """Profiles return frozen instances — mutation must still raise."""
        cfg = TradingConfig.from_profile("moderate")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.signals = SignalConfig()

    def test_moderate_non_overridden_fields_keep_defaults(self):
        """Fields not overridden in 'moderate' must equal default values."""
        default = TradingConfig()
        moderate = TradingConfig.from_profile("moderate")
        # These fields are NOT overridden in 'moderate'
        assert moderate.capital == default.capital
        assert moderate.costs == default.costs
        assert moderate.signals.min_matches == default.signals.min_matches
        assert moderate.signals.min_agreement == default.signals.min_agreement

    def test_conservative_non_overridden_fields_keep_defaults(self):
        default = TradingConfig()
        conservative = TradingConfig.from_profile("conservative")
        assert conservative.capital == default.capital
        assert conservative.costs == default.costs


# ============================================================
# validate() — error detection
# ============================================================

class TestValidate:
    """validate() must catch each class of invalid configuration."""

    def test_valid_config_returns_empty(self):
        assert TradingConfig().validate() == []

    def test_drawdown_brake_gte_halt_flagged(self):
        cfg = _make_config(
            risk=dataclasses.replace(
                RiskConfig(),
                drawdown_brake_threshold=0.20,
                drawdown_halt_threshold=0.15,
            )
        )
        errors = cfg.validate()
        assert any("drawdown_brake" in e for e in errors)

    def test_min_position_gte_max_position_flagged(self):
        cfg = _make_config(
            position_limits=dataclasses.replace(
                PositionLimitsConfig(),
                min_position_pct=0.10,
                max_position_pct=0.08,
            )
        )
        errors = cfg.validate()
        assert any("min_position_pct" in e for e in errors)

    def test_sector_limit_too_tight_flagged(self):
        """4 positions × 5% min = 20%, but max_sector_pct=0.15 → error."""
        cfg = _make_config(
            position_limits=dataclasses.replace(
                PositionLimitsConfig(),
                max_positions_per_sector=4,
                min_position_pct=0.05,
                max_sector_pct=0.15,
            )
        )
        errors = cfg.validate()
        assert any("max_positions_per_sector" in e for e in errors)

    def test_leverage_flagged(self):
        cfg = _make_config(
            capital=dataclasses.replace(CapitalConfig(), max_gross_exposure=1.5)
        )
        errors = cfg.validate()
        assert any("long-only" in e for e in errors)

    def test_excessive_round_trip_bps_flagged(self):
        cfg = _make_config(
            costs=dataclasses.replace(CostConfig(), slippage_bps=60.0)
        )
        errors = cfg.validate()
        assert any("round_trip_bps" in e for e in errors)

    def test_confidence_below_0_5_flagged(self):
        cfg = _make_config(
            signals=dataclasses.replace(SignalConfig(), confidence_threshold=0.49)
        )
        errors = cfg.validate()
        assert any("confidence_threshold" in e for e in errors)

    def test_min_matches_too_low_flagged(self):
        cfg = _make_config(
            signals=dataclasses.replace(SignalConfig(), min_matches=3)
        )
        errors = cfg.validate()
        assert any("min_matches" in e for e in errors)

    def test_max_holding_days_zero_flagged(self):
        cfg = _make_config(
            trade_management=dataclasses.replace(TradeManagementConfig(), max_holding_days=0)
        )
        errors = cfg.validate()
        assert any("max_holding_days" in e for e in errors)

    def test_multiple_errors_all_reported(self):
        """validate() must collect ALL errors, not stop at first."""
        cfg = _make_config(
            risk=dataclasses.replace(
                RiskConfig(),
                drawdown_brake_threshold=0.25,
                drawdown_halt_threshold=0.20,
            ),
            capital=dataclasses.replace(CapitalConfig(), max_gross_exposure=2.0),
        )
        errors = cfg.validate()
        assert len(errors) >= 2


# ============================================================
# summary()
# ============================================================

class TestSummary:
    """summary() must execute without raising and include key fields."""

    def test_summary_runs(self):
        summary = TradingConfig().summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_contains_status(self):
        assert "VALID" in TradingConfig().summary()

    def test_summary_invalid_config_shows_invalid(self):
        cfg = _make_config(
            risk=dataclasses.replace(
                RiskConfig(),
                drawdown_brake_threshold=0.25,
                drawdown_halt_threshold=0.20,
            )
        )
        assert "INVALID" in cfg.summary()
