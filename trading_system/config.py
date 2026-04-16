"""
config.py — Trading system configuration for the Phase 3Z rebuild workspace.

Mirrors the production trading_system/config.py structure but:
  - Does NOT import from the production pattern_engine package (isolated workspace)
  - Adds ResearchFlagsConfig with feature flags for all research modules (SLE-71)
  - All other defaults are identical to production (parity preserved)

Key addition (SLE-71):
  ResearchFlagsConfig.use_slip_deficit = False
    When True: SlipDeficit research module is imported and applied during backtesting.
    When False: SlipDeficit is NOT imported — identical behavior to the baseline.
    The flag guards the conditional import at call sites in backtest_engine.py.

Linear: SLE-68, SLE-71
"""

from dataclasses import dataclass, field

# ─── Sub-Configs (same as production) ──────────────────────────────────────────

@dataclass(frozen=True)
class CapitalConfig:
    """Capital and account parameters."""
    initial_capital: float = 10_000.0
    fractional_shares: bool = True
    max_gross_exposure: float = 1.0


@dataclass(frozen=True)
class CostConfig:
    """Transaction cost model.

    Round-trip friction = (slippage + spread) × 2 sides = 26 bps.
    """
    slippage_bps: float = 10.0
    spread_bps: float = 3.0
    commission_per_share: float = 0.0
    risk_free_annual_rate: float = 0.045

    @property
    def total_entry_bps(self) -> float:
        return self.slippage_bps + self.spread_bps

    @property
    def total_exit_bps(self) -> float:
        return self.slippage_bps + self.spread_bps

    @property
    def round_trip_bps(self) -> float:
        return self.total_entry_bps + self.total_exit_bps


@dataclass(frozen=True)
class PositionLimitsConfig:
    """Position sizing constraints."""
    min_position_pct: float = 0.02
    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30
    max_positions_per_sector: int = 3


@dataclass(frozen=True)
class SignalConfig:
    """Signal filtering parameters."""
    confidence_threshold: float = 0.60
    min_matches: int = 10
    min_agreement: float = 0.10


@dataclass(frozen=True)
class TradeManagementConfig:
    """Rules governing trade lifecycle."""
    max_holding_days: int = 14
    cooldown_after_stop_days: int = 3
    cooldown_after_maxhold_days: int = 3
    reentry_confidence_margin: float = 0.05
    allow_same_day_churn: bool = False


@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters."""
    volatility_lookback: int = 20
    correlation_lookback: int = 60
    stop_loss_atr_multiple: float = 3.0
    max_loss_per_trade_pct: float = 0.02
    drawdown_brake_threshold: float = 0.15
    drawdown_halt_threshold: float = 0.20


@dataclass(frozen=True)
class EvaluationConfig:
    """Performance evaluation windows and thresholds."""
    rolling_windows: list[int] = field(
        default_factory=lambda: [30, 90, 252]
    )
    min_trades_for_metrics: int = 30
    baseline_random_iterations: int = 100


# ─── ResearchFlagsConfig (new in SLE-71) ───────────────────────────────────────

@dataclass(frozen=True)
class ResearchFlagsConfig:
    """Feature flags for all research modules.

    Every research module must be behind an explicit bool flag here.
    Default=False means the baseline behavior is always the safe path.

    Flags (SLE-71):
        use_slip_deficit: When True, import and apply SlipDeficit during
            backtesting. SlipDeficit adjusts simulated fill prices to account
            for intraday momentum adverse to the trade direction.
            Hard-wired True in production backtest_engine.py (SLE-71 target).

    Flags (SLE-74):
        use_liquidity_congestion_gate: When True, apply LiquidityCongestionGate
            overlay — throttles signals when ATR/price ratio exceeds threshold.

    Flags (SLE-75):
        use_fatigue_accumulation: When True, apply FatigueAccumulationOverlay —
            reduces signal confidence during extended regime runs.

    Flags (SLE-76):
        use_drift_monitor: When True, enable DriftMonitor feature-distribution
            and BSS EWMA alerting; feeds into StrategyEvaluator YELLOW status.

    Flags (SLE-72):
        use_sax_filter: When True, apply SAX symbolic filter as a second-stage
            pass over HNSW candidates, pruning those whose symbolic shape
            (PAA + digitised word) diverges from the query.

    Flags (SLE-73):
        use_wfa_rerank: When True, rerank the post-filtered top-K candidates
            by constrained DTW distance so Stage 5 sees the most temporally-
            aligned analogues first.

    Flags (SLE-78):
        use_ib_compression: When True, apply Information Bottleneck compression
            to reduce 8-dim return fingerprints before nearest-neighbour search.

    Usage:
        # Default — identical to pre-SLE-71 baseline:
        cfg = TradingConfig()
        cfg.research_flags.use_slip_deficit  # False

        # Opt-in — enables SlipDeficit:
        cfg = TradingConfig(research_flags=ResearchFlagsConfig(use_slip_deficit=True))
    """
    use_slip_deficit: bool = False
    use_sax_filter: bool = False
    use_wfa_rerank: bool = False
    use_liquidity_congestion_gate: bool = False
    use_fatigue_accumulation: bool = False
    use_drift_monitor: bool = False
    use_ib_compression: bool = False


# ─── TradingConfig (master container) ─────────────────────────────────────────

@dataclass(frozen=True)
class TradingConfig:
    """Master configuration container for the Phase 3Z rebuild.

    All sub-configs are frozen dataclasses. Mutations must use
    dataclasses.replace() on the sub-config, then construct a new
    TradingConfig. No in-place mutation permitted.

    Usage:
        cfg = TradingConfig()
        # Enable SlipDeficit research module:
        cfg2 = dataclasses.replace(
            cfg,
            research_flags=dataclasses.replace(cfg.research_flags, use_slip_deficit=True),
        )
    """
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    position_limits: PositionLimitsConfig = field(default_factory=PositionLimitsConfig)
    trade_management: TradeManagementConfig = field(default_factory=TradeManagementConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    research_flags: ResearchFlagsConfig = field(default_factory=ResearchFlagsConfig)
    use_portfolio_manager: bool = False  # Phase 4: enable PM filter (default off)

    def validate(self) -> list[str]:
        """Check internal consistency. Returns list of error messages (empty = valid)."""
        errors = []

        if self.risk.drawdown_brake_threshold >= self.risk.drawdown_halt_threshold:
            errors.append(
                f"drawdown_brake ({self.risk.drawdown_brake_threshold}) "
                f"must be < drawdown_halt ({self.risk.drawdown_halt_threshold})"
            )

        if self.position_limits.min_position_pct >= self.position_limits.max_position_pct:
            errors.append(
                f"min_position_pct ({self.position_limits.min_position_pct}) "
                f"must be < max_position_pct ({self.position_limits.max_position_pct})"
            )

        min_sector_need = (
            self.position_limits.max_positions_per_sector
            * self.position_limits.min_position_pct
        )
        if min_sector_need > self.position_limits.max_sector_pct:
            errors.append(
                f"max_positions_per_sector × min_position_pct = {min_sector_need} "
                f"exceeds max_sector_pct ({self.position_limits.max_sector_pct})"
            )

        if self.capital.max_gross_exposure > 1.0:
            errors.append(
                "v1 is long-only: max_gross_exposure must be ≤ 1.0"
            )

        if not 0.50 <= self.signals.confidence_threshold <= 1.0:
            errors.append(
                f"confidence_threshold ({self.signals.confidence_threshold}) must be in [0.50, 1.0]"
            )

        if self.signals.min_matches < 5:
            errors.append(
                f"min_matches ({self.signals.min_matches}) is dangerously low (< 5)"
            )

        return errors
