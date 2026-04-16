"""
trading_system.contracts — Pydantic models for trade lifecycle objects.

These replace the plain @dataclass definitions in backtest_engine.py with
validated, frozen Pydantic models that enforce constraints at construction time.

Linear: SLE-57
"""

from trading_system.contracts.trades import (
    DailySnapshot,
    PositionRecord,
    TradeRecord,
)

__all__ = ["DailySnapshot", "PositionRecord", "TradeRecord"]
