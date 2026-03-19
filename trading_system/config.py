"""
config.py — Central Configuration for FPPE Trading System v1

Single source of truth for all tunable parameters. No magic numbers
anywhere else in the codebase. Every parameter has a comment explaining
why that value was chosen.

Design doc reference: FPPE_TRADING_SYSTEM_DESIGN.md v0.3, Section 4.1
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================
# SECTOR MAP — 52 tickers across 7 sectors
# Mirrors strategy.py SECTOR_MAP with Index split out
# ============================================================

SECTOR_MAP: Dict[str, str] = {
    # Index (2)
    "SPY": "Index", "QQQ": "Index",
    # Tech (19)
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AMZN": "Tech",
    "GOOGL": "Tech", "META": "Tech", "TSLA": "Tech", "AVGO": "Tech",
    "ORCL": "Tech", "ADBE": "Tech", "CRM": "Tech", "AMD": "Tech",
    "NFLX": "Tech", "INTC": "Tech", "CSCO": "Tech", "QCOM": "Tech",
    "TXN": "Tech", "MU": "Tech", "PYPL": "Tech",
    # Finance (9)
    "JPM": "Finance", "BAC": "Finance", "WFC": "Finance", "GS": "Finance",
    "MS": "Finance", "V": "Finance", "MA": "Finance", "AXP": "Finance",
    "BRK-B": "Finance",
    # Healthcare (10)
    "LLY": "Health", "UNH": "Health", "JNJ": "Health", "ABBV": "Health",
    "MRK": "Health", "PFE": "Health", "TMO": "Health", "ISRG": "Health",
    "AMGN": "Health", "GILD": "Health",
    # Consumer (7) — includes DIS (Entertainment/Media, not Industrial)
    # NOTE: DIS was incorrectly classified as Industrial in v0.1-0.3.
    # Disney is Consumer Discretionary (Media & Entertainment). Sector
    # concentration limits apply correctly after this fix.
    "WMT": "Consumer", "COST": "Consumer", "PG": "Consumer",
    "KO": "Consumer", "PEP": "Consumer", "HD": "Consumer",
    "DIS": "Consumer",
    # Industrial (3) — Manufacturing, Aerospace, Infrastructure
    "CAT": "Industrial", "BA": "Industrial", "GE": "Industrial",
    # Energy (2)
    "XOM": "Energy", "CVX": "Energy",
}

ALL_TICKERS: List[str] = list(SECTOR_MAP.keys())


@dataclass(frozen=True)
class CapitalConfig:
    """Capital and account parameters."""
    initial_capital: float = 10_000.0      # Paper trading baseline
    fractional_shares: bool = True         # Required at $10k level
    max_gross_exposure: float = 1.0        # v1: long-only, no leverage


@dataclass(frozen=True)
class CostConfig:
    """Transaction cost model.

    Round-trip friction = (slippage + spread) × 2 sides = 26 bps.
    Revised upward from 16 bps in v0.1 after reviewer feedback:
    open execution is the noisiest print of the day.
    """
    slippage_bps: float = 10.0             # Per side; revised up from 5 bps for open execution
    spread_bps: float = 3.0                # Per side; embedded in execution price
    commission_per_share: float = 0.0      # Zero-commission broker (IBKR Lite)
    risk_free_annual_rate: float = 0.045   # For idle cash yield and Sharpe denominator

    @property
    def total_entry_bps(self) -> float:
        """Total cost on entry side."""
        return self.slippage_bps + self.spread_bps

    @property
    def total_exit_bps(self) -> float:
        """Total cost on exit side."""
        return self.slippage_bps + self.spread_bps

    @property
    def round_trip_bps(self) -> float:
        """Total round-trip friction in basis points."""
        return self.total_entry_bps + self.total_exit_bps


@dataclass(frozen=True)
class PositionLimitsConfig:
    """Position sizing constraints.

    Min 2% prevents trades where friction dominates.
    Max 10% caps single-name concentration.
    Sector limits prevent correlated blowups.
    """
    min_position_pct: float = 0.02         # Below this, friction destroys the trade
    max_position_pct: float = 0.10         # Single-name concentration cap
    max_sector_pct: float = 0.30           # Max 30% in any one sector
    max_positions_per_sector: int = 3      # Even within the 30% limit


@dataclass(frozen=True)
class SignalConfig:
    """Signal filtering parameters applied before the backtest engine.

    confidence_threshold is re-applied to cached signals when re-labeling
    BUY/SELL/HOLD at backtest time. Empirically determined via threshold
    sweep on 2024 validation data:
      - 0.65 (original): 159 BUY, 8.9% annual, Sharpe 0.97
      - 0.63: 530 BUY, 14.4% annual, Sharpe 1.39
      - 0.60 (optimal): 1876 BUY, 19.5% annual, Sharpe 1.60  ← current default
      - 0.58: 3348 BUY, 16.2% annual, Sharpe 1.17 (dilution begins)
      - 0.55: 5929 BUY, 15.9% annual, Sharpe 1.15 (continued dilution)
    Below 0.60, per-trade expectancy collapses despite higher signal volume.
    """
    confidence_threshold: float = 0.60    # Empirically optimal from 2024 sweep
    min_matches: int = 10                  # Minimum K-NN analogues required
    min_agreement: float = 0.10           # Minimum agreement spread (|prob - 0.5| × 2)


@dataclass(frozen=True)
class TradeManagementConfig:
    """Rules governing trade lifecycle.

    max_holding_days updated from 10 → 14 after empirical sweep on 2024 data:
      - 1d: -$1.22/trade, -3.9% annual (friction destroys alpha at this scale)
      - 3d: -$0.23/trade, 6.5% annual
      - 5d:  $0.67/trade, 10.5% annual
      - 7d:  $1.76/trade, 13.9% annual (FPPE projection horizon)
      - 10d: $4.13/trade, 19.5% annual
      - 14d: $6.65/trade, 22.3% annual, Sharpe 1.82  ← current default
      - 20d: $7.09/trade, 17.5% annual (fewer trades, lower annual return)
    Win rate climbs monotonically (41.6% → 60.4%) — the predicted move needs
    time to materialize. NOTE: 2024 was a bull year; re-validate in bear regimes.
    Cooldown prevents whipsaw after forced exits.
    Re-entry margin prevents noise-driven churn.
    """
    max_holding_days: int = 14            # Empirically optimal (peak Sharpe 1.82)
    cooldown_after_stop_days: int = 3      # Prevent whipsaw after stop-loss
    cooldown_after_maxhold_days: int = 3   # Prevent whipsaw after max-hold exit
    reentry_confidence_margin: float = 0.05  # Must beat prior entry by 5% to re-enter
    allow_same_day_churn: bool = False     # Cannot exit and re-enter same ticker same day


@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters.

    Position sizing is purely volatility-based in v1.
    Confidence does NOT affect position size — only ranking (Layer 3).
    This prevents the double-counting problem identified in v0.1 review.
    """
    volatility_lookback: int = 20          # Trading days for ATR calculation
    correlation_lookback: int = 60         # Trading days for pairwise correlations
    stop_loss_atr_multiple: float = 2.0    # Stop = entry ± 2×ATR
    max_loss_per_trade_pct: float = 0.02   # 2% of equity max loss per trade
    drawdown_brake_threshold: float = 0.15 # Reduce position sizes at 15% drawdown
    drawdown_halt_threshold: float = 0.20  # Halt all new trades at 20% drawdown


@dataclass(frozen=True)
class EvaluationConfig:
    """Performance evaluation windows and thresholds."""
    rolling_windows: List[int] = field(
        default_factory=lambda: [30, 90, 252]  # Short, medium, long-term
    )
    min_trades_for_metrics: int = 30       # Don't compute ratios below this
    baseline_random_iterations: int = 100  # Monte Carlo iterations for random baseline


@dataclass(frozen=True)
class TradingConfig:
    """Master configuration container. All sub-configs in one place.

    frozen=True enforces immutability: all field modifications must go through
    dataclasses.replace(), preventing accidental in-place mutations after
    construction (e.g. in test fixtures or multi-threaded signal processing).

    NOTE: sector_map is a Dict field, so hash(TradingConfig) raises TypeError
    (same limitation as EngineConfig in pattern_engine — see CLAUDE.md gotchas).
    Use repr() for identity comparisons.

    Usage:
        config = TradingConfig()
        print(config.costs.round_trip_bps)  # 26.0
        print(config.capital.initial_capital)  # 10000.0
        print(config.signals.confidence_threshold)  # 0.60
    """
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    position_limits: PositionLimitsConfig = field(default_factory=PositionLimitsConfig)
    trade_management: TradeManagementConfig = field(default_factory=TradeManagementConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    sector_map: Dict[str, str] = field(default_factory=lambda: SECTOR_MAP.copy())

    def validate(self) -> List[str]:
        """Check internal consistency. Returns list of error messages (empty = valid).

        Catches contradictions like drawdown brake > halt, or sector limits
        that don't allow enough room for max_positions_per_sector.
        """
        errors = []

        # Risk thresholds must be ordered
        if self.risk.drawdown_brake_threshold >= self.risk.drawdown_halt_threshold:
            errors.append(
                f"drawdown_brake ({self.risk.drawdown_brake_threshold}) "
                f"must be < drawdown_halt ({self.risk.drawdown_halt_threshold})"
            )

        # Position limits must be ordered
        if self.position_limits.min_position_pct >= self.position_limits.max_position_pct:
            errors.append(
                f"min_position_pct ({self.position_limits.min_position_pct}) "
                f"must be < max_position_pct ({self.position_limits.max_position_pct})"
            )

        # Sector limit must allow at least max_positions_per_sector × min_position
        min_sector_need = (
            self.position_limits.max_positions_per_sector
            * self.position_limits.min_position_pct
        )
        if min_sector_need > self.position_limits.max_sector_pct:
            errors.append(
                f"max_positions_per_sector ({self.position_limits.max_positions_per_sector}) "
                f"× min_position_pct ({self.position_limits.min_position_pct}) = {min_sector_need} "
                f"exceeds max_sector_pct ({self.position_limits.max_sector_pct})"
            )

        # Max gross exposure must be ≤ 1.0 for v1 (long-only, no leverage)
        if self.capital.max_gross_exposure > 1.0:
            errors.append(
                f"v1 is long-only: max_gross_exposure ({self.capital.max_gross_exposure}) "
                f"must be ≤ 1.0"
            )

        # Slippage should be reasonable
        if self.costs.round_trip_bps > 100:
            errors.append(
                f"round_trip_bps ({self.costs.round_trip_bps}) seems unreasonably high (>1%)"
            )

        # Risk-free rate sanity check
        if self.costs.risk_free_annual_rate < 0 or self.costs.risk_free_annual_rate > 0.15:
            errors.append(
                f"risk_free_annual_rate ({self.costs.risk_free_annual_rate}) "
                f"outside reasonable range [0, 0.15]"
            )

        # Signal config sanity checks
        if not 0.50 <= self.signals.confidence_threshold <= 1.0:
            errors.append(
                f"confidence_threshold ({self.signals.confidence_threshold}) "
                f"must be in [0.50, 1.0]. Below 0.50 means random noise, not signal."
            )
        if self.signals.min_matches < 5:
            errors.append(
                f"min_matches ({self.signals.min_matches}) is dangerously low. "
                f"K-NN probabilities are unreliable below 5 analogues; 10 is the practical minimum."
            )
        if not 0.0 <= self.signals.min_agreement <= 1.0:
            errors.append(
                f"min_agreement ({self.signals.min_agreement}) must be in [0.0, 1.0]"
            )

        # max_holding_days sanity
        if self.trade_management.max_holding_days < 1:
            errors.append("max_holding_days must be ≥ 1")
        if self.trade_management.max_holding_days > 252:
            errors.append(
                f"max_holding_days ({self.trade_management.max_holding_days}) "
                f"exceeds one trading year — likely a misconfiguration."
            )

        return errors

    @classmethod
    def from_profile(cls, profile: str) -> "TradingConfig":
        """Create a TradingConfig from a named risk profile.

        Profiles:
            "conservative" — Higher threshold (0.68), fewer/higher-quality trades,
                             lower drawdown tolerance. For users who want stable,
                             verifiable edge with minimal volatility.

            "moderate"     — Balanced threshold (0.63), intermediate parameters.
                             For users who accept some drawdown for better returns.

            "aggressive"   — Lower threshold (0.60), more trades, higher deployment.
                             Current default. Empirically optimal for 2024 data.
                             CAUTION: Bull-year bias — re-validate in bear markets.

        Usage:
            config = TradingConfig.from_profile("conservative")
            config = TradingConfig.from_profile("moderate")
            config = TradingConfig.from_profile("aggressive")  # same as TradingConfig()
        """
        if profile == "aggressive":
            # Current default — all standard values
            return cls()

        elif profile == "moderate":
            # Use dataclasses.replace() on each sub-config so all other fields
            # keep their defaults.  Direct field assignment is not possible on
            # frozen dataclasses; this is the correct mutation pattern.
            default = cls()
            return cls(
                signals=dataclasses.replace(
                    default.signals,
                    confidence_threshold=0.63,   # 530 BUY signals, 14.4% annual
                ),
                trade_management=dataclasses.replace(
                    default.trade_management,
                    max_holding_days=12,
                ),
                position_limits=dataclasses.replace(
                    default.position_limits,
                    max_position_pct=0.08,
                ),
                risk=dataclasses.replace(
                    default.risk,
                    drawdown_halt_threshold=0.18,
                    drawdown_brake_threshold=0.12,
                ),
            )

        elif profile == "conservative":
            default = cls()
            return cls(
                signals=dataclasses.replace(
                    default.signals,
                    confidence_threshold=0.68,   # ~90 est. BUY signals, higher quality
                ),
                trade_management=dataclasses.replace(
                    default.trade_management,
                    max_holding_days=10,         # Tighter, closer to FPPE horizon
                ),
                position_limits=dataclasses.replace(
                    default.position_limits,
                    max_position_pct=0.07,
                ),
                risk=dataclasses.replace(
                    default.risk,
                    drawdown_halt_threshold=0.15,
                    drawdown_brake_threshold=0.10,
                ),
            )

        else:
            raise ValueError(
                f"Unknown profile '{profile}'. Choose from: 'aggressive', 'moderate', 'conservative'"
            )

    def summary(self) -> str:
        """Print a human-readable summary of the configuration."""
        errors = self.validate()
        status = "VALID" if not errors else f"INVALID ({len(errors)} errors)"

        lines = [
            "=" * 60,
            "  FPPE Trading System v1 — Configuration Summary",
            "=" * 60,
            f"  Status: {status}",
            "",
            f"  Capital:         ${self.capital.initial_capital:,.0f}",
            f"  Fractional:      {self.capital.fractional_shares}",
            f"  Max exposure:    {self.capital.max_gross_exposure:.0%}",
            "",
            f"  Signal threshold:{self.signals.confidence_threshold:.2f}",
            f"  Min analogues:   {self.signals.min_matches}",
            f"  Min agreement:   {self.signals.min_agreement:.2f}",
            "",
            f"  Slippage:        {self.costs.slippage_bps} bps/side",
            f"  Spread:          {self.costs.spread_bps} bps/side",
            f"  Round-trip:      {self.costs.round_trip_bps} bps",
            f"  Risk-free rate:  {self.costs.risk_free_annual_rate:.1%}",
            "",
            f"  Position min:    {self.position_limits.min_position_pct:.0%}",
            f"  Position max:    {self.position_limits.max_position_pct:.0%}",
            f"  Sector max:      {self.position_limits.max_sector_pct:.0%}",
            f"  Per-sector max:  {self.position_limits.max_positions_per_sector} positions",
            "",
            f"  Max hold:        {self.trade_management.max_holding_days} days",
            f"  Stop-loss:       {self.risk.stop_loss_atr_multiple}× ATR",
            f"  Max loss/trade:  {self.risk.max_loss_per_trade_pct:.0%}",
            f"  DD brake:        {self.risk.drawdown_brake_threshold:.0%}",
            f"  DD halt:         {self.risk.drawdown_halt_threshold:.0%}",
            "",
            f"  Universe:        {len(self.sector_map)} tickers",
            f"  Sectors:         {len(set(self.sector_map.values()))}",
        ]

        if errors:
            lines.append("")
            lines.append("  VALIDATION ERRORS:")
            for e in errors:
                lines.append(f"    - {e}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# DEFAULT INSTANCE — import this for standard usage
# ============================================================

DEFAULT_CONFIG = TradingConfig()


if __name__ == "__main__":
    config = TradingConfig()
    print(config.summary())
    errors = config.validate()
    if errors:
        print("\nValidation failed!")
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print("\nConfiguration is valid.")
