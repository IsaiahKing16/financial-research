"""
signal_pipeline.py — Ordered filter chain for FPPE post-query signal processing.

SignalPipeline runs a list of SignalFilterBase-compatible filters in sequence,
accumulating a combined change mask. It is the single integration point between
PatternMatcher.query() and the signal intelligence layer in run_walkforward.py.

Usage:
    pipeline = SignalPipeline(filters=[conviction_layer, mom_filter])
    signals, combined_mask = pipeline.run(np.asarray(probs), signals, val_db)

Linear: M9 (Signal Intelligence Layer)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pattern_engine.signal_filter_base import SignalFilterBase


class SignalPipeline:
    """Runs a sequence of SignalFilterBase filters in order.

    Each filter receives the signals as modified by all prior filters.
    The combined mask is the union of all per-filter change masks.

    Args:
        filters: Ordered list of SignalFilterBase instances.
    """

    def __init__(self, filters: list[SignalFilterBase]):
        self.filters = filters

    def run(
        self,
        probs: np.ndarray,
        signals: list[str],
        val_db: pd.DataFrame,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """Run all filters in sequence.

        Args:
            probs:   (N,) calibrated probabilities.
            signals: Initial signal list.
            val_db:  Validation DataFrame for this batch.
            **kwargs: Forwarded to each filter's apply() call.

        Returns:
            (filtered_signals, combined_mask):
              combined_mask is True wherever ANY filter changed a signal.
        """
        combined_mask = np.zeros(len(signals), dtype=bool)
        for filt in self.filters:
            signals, mask = filt.apply(probs, signals, val_db, **kwargs)
            combined_mask |= mask
        return signals, combined_mask
