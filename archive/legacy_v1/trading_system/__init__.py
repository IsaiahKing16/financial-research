"""
FPPE Trading System v1 — Long-only backtesting and evaluation framework.

Design doc: FPPE_TRADING_SYSTEM_DESIGN.md v0.3

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

    Layer 3: Portfolio Manager
    RankedSignal       — Signal annotated with rank position and score.
    AllocationDecision — Portfolio allocation approval/rejection for one signal.
    PortfolioSnapshot  — Read-only portfolio state passed to portfolio_manager.
    rank_signals       — Rank BUY signals by confidence (v1).
    check_allocation   — Check one ranked signal against portfolio constraints.
    allocate_day       — Rank and allocate all BUY signals for one date.
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
from .portfolio_state import RankedSignal, AllocationDecision, PortfolioSnapshot
from .portfolio_manager import rank_signals, check_allocation, allocate_day
