"""
trading_system/exceptions.py — FPPE error hierarchy.

All trading system errors inherit from TradingSystemError.  This allows
callers to catch the base class while still distinguishing domain-specific
failures.  Never use bare `except:` or `except Exception: pass`.

Reference: PRD §7A.9 Error Hierarchy
"""

from __future__ import annotations


class TradingSystemError(Exception):
    """Base class for all FPPE trading system errors.

    Prefer subclasses for precise error handling.  Only catch
    TradingSystemError when the caller genuinely handles all subtypes.
    """


# ── Data errors ───────────────────────────────────────────────────────────────

class DataError(TradingSystemError):
    """Errors in market data acquisition, validation, or freshness."""


class MarketDataError(DataError):
    """Failure to fetch or parse market data (Polygon.io / yfinance)."""


class StaleDataError(DataError):
    """Data freshness violation — OHLCV data older than expected threshold."""


# ── Execution errors ──────────────────────────────────────────────────────────

class ExecutionError(TradingSystemError):
    """Errors in order submission or execution layer."""


class OrderRejectedError(ExecutionError):
    """Broker rejected the order (insufficient margin, invalid quantity, etc.)."""


class InsufficientFundsError(ExecutionError):
    """Insufficient capital to size the position at the computed allocation."""


# ── Model errors ──────────────────────────────────────────────────────────────

class ModelError(TradingSystemError):
    """Errors in the ML pipeline (calibration, feature extraction, KNN)."""


class CalibrationError(ModelError):
    """Calibrator failed to converge or produced NaN/Inf output probabilities."""


# ── Risk errors ───────────────────────────────────────────────────────────────

class RiskLimitError(TradingSystemError):
    """A risk guard was breached (drawdown brake, position limit, ATR limit)."""
