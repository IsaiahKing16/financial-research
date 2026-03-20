"""
FPPE Trading System v1 — Long-only backtesting and evaluation framework.

Design doc: FPPE_TRADING_SYSTEM_DESIGN.md v0.4

Exports:
    TradingConfig      — Master config dataclass. Use TradingConfig.from_profile()
                         for named risk profiles (aggressive/moderate/conservative).
    DEFAULT_CONFIG     — Pre-built instance with aggressive (default) settings.
    BacktestEngine     — Layer 1: Trade simulation with realistic friction.
    BacktestResults    — Output container returned by BacktestEngine.run().
    UnifiedSignal      — Normalized FPPE signal (all models speak this format).
    SignalDirection    — Enum: BUY / SELL / HOLD
    SignalSource       — Enum: KNN / DL / ENSEMBLE
    SECTOR_MAP         — 52-ticker sector classification dict.
    ALL_TICKERS        — Sorted list of all 52 universe tickers.
    RiskState          — Portfolio drawdown state snapshot.
    PositionDecision   — Position sizing decision payload.
    StopLossEvent      — Stop-loss trigger event payload.
    size_position      — Position sizing + risk validation helper.
    compute_atr_pct    — 20-day ATR% computation helper.
    check_stop_loss    — Intraday low vs stop trigger helper.
    compute_drawdown_scalar — Linear brake scalar (15%→20% DD).
"""

from .config import TradingConfig, DEFAULT_CONFIG, SECTOR_MAP, ALL_TICKERS
from .backtest_engine import BacktestEngine, BacktestResults
from .signal_adapter import UnifiedSignal, SignalDirection, SignalSource
from .risk_state import RiskState, PositionDecision, StopLossEvent
from .risk_engine import (
    size_position,
    compute_atr_pct,
    check_stop_loss,
    compute_drawdown_scalar,
)

__all__ = [
    "TradingConfig",
    "DEFAULT_CONFIG",
    "SECTOR_MAP",
    "ALL_TICKERS",
    "BacktestEngine",
    "BacktestResults",
    "UnifiedSignal",
    "SignalDirection",
    "SignalSource",
    "RiskState",
    "PositionDecision",
    "StopLossEvent",
    "size_position",
    "compute_atr_pct",
    "check_stop_loss",
    "compute_drawdown_scalar",
]
