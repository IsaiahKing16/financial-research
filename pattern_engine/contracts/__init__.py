"""
pattern_engine.contracts — Pydantic models and Pandera schemas for cross-module boundaries.

These contracts are the single source of truth for data shapes flowing between
pattern_engine and trading_system. Any field change here must be reflected in
both producer and consumer code.

Design doc: docs/rebuild/PHASE_3Z_EXECUTION_PLAN.md §2, §4
Linear: SLE-57 (Pydantic contracts), SLE-58 (Pandera schemas)
"""

from pattern_engine.contracts.signals import SignalRecord
from pattern_engine.contracts.matcher import BaseMatcher
from pattern_engine.contracts.finite_types import FiniteFloat

__all__ = ["SignalRecord", "BaseMatcher", "FiniteFloat"]
