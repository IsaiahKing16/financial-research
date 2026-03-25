"""
signal_filter_base.py — Abstract base class for all FPPE post-query signal filters.

All signal filters (SectorConvictionLayer, MomentumSignalFilter, SentimentVetoFilter)
inherit from SignalFilterBase and implement a unified apply() interface. This enables
SignalPipeline to dispatch polymorphically without knowing the concrete filter type.

Linear: M9 (Signal Intelligence Layer)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SignalFilterBase(ABC):
    """Abstract base class for post-query signal filters.

    Each filter receives the calibrated probabilities, current signals,
    and the validation DataFrame, then returns a (possibly modified)
    signal list alongside a boolean mask indicating changed positions.
    """

    @abstractmethod
    def apply(
        self,
        probs: np.ndarray,
        signals: list[str],
        val_db: pd.DataFrame,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """Apply the filter to a set of signals.

        Args:
            probs:   (N,) calibrated probabilities from PatternMatcher.query().
            signals: List of N signal strings ("BUY"/"SELL"/"HOLD").
            val_db:  Validation DataFrame for this batch (must have Ticker).
            **kwargs: Filter-specific extras (e.g. ``sentiment={}``).

        Returns:
            (filtered_signals, filter_mask):
              filtered_signals: list[str] — signals after filtering.
              filter_mask:      (N,) bool — True where this filter changed the signal.
        """
        ...
