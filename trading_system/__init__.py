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
    RiskState          — Layer 2: peak equity and drawdown mode for a backtest run.
    PositionDecision   — Layer 2: sizing/stop output from risk_engine.size_position().
    StopLossEvent      — Layer 2: recorded stop trigger (ticker, date, stop price).
    size_position      — Layer 2: volatility-based size + ATR stop (+ drawdown scalar).
    compute_atr_pct    — Layer 2: ATR as fraction of price (ta library).
    check_stop_loss    — Layer 2: intraday low vs stop.
    compute_drawdown_scalar — Layer 2: brake/halt scalar and mode string.
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
