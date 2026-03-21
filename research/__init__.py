"""
research/__init__.py — Abstract base classes for FPPE research modules.

Each research module subclasses the relevant ABC. This enforces the interface
contract so that a validated module can be promoted to production with zero
changes to its callers.

Promotion notes:
  - BaseDistanceMetric → replaces ball_tree in pattern_engine/matching.py
  - BaseCalibrator     → replaces PlattCalibrator in pattern_engine/calibration.py
                         (requires migration of signal_adapter.py callers — Phase C)
  - BaseRiskOverlay    → augments risk_engine.py (wiring is Phase C work)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


class BaseDistanceMetric(ABC):
    """Interface for distance metrics that shadow matching.py's ball_tree computation."""

    @abstractmethod
    def fit(self, X_train: np.ndarray) -> "BaseDistanceMetric":
        """Store training data for optional normalization. May be a no-op."""
        ...

    @abstractmethod
    def compute(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Compute distance from query (shape (D,)) to each candidate (shape (N, D)).

        Returns: np.ndarray of shape (N,) — one distance per candidate.
        """
        ...


class BaseCalibrator(ABC):
    """Interface matching PlattCalibrator in pattern_engine/calibration.py.

    BMACalibrator.transform() is a drop-in for PlattCalibrator.transform().
    Wiring into signal_adapter.py callers is Phase C promotion work.

    Note: generate_pdf() is NOT declared here — it is BMA-specific and will be
    called through a concrete BMACalibrator reference in Phase C integration.
    If a future calibrator also exposes a PDF method, promote it to this ABC.

    Shape note: raw_probs shape differs by subclass:
      - PlattCalibrator expects (N,) and returns (N,) — vectorized
      - BMACalibrator expects (N, K) for fit and (K,) for transform — ensemble
    Phase C signal_adapter.py migration must account for this asymmetry.
    """

    @abstractmethod
    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "BaseCalibrator":
        """Fit calibrator on training data.

        Args:
            raw_probs: Raw probabilities. Shape depends on subclass:
                       PlattCalibrator expects (N,); BMACalibrator expects (N, K).
            y_true:    Binary labels, shape (N,).
        """
        ...

    @abstractmethod
    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities in [0, 1]."""
        ...

    @property
    @abstractmethod
    def fitted(self) -> bool:
        """True after fit() has been called successfully."""
        ...


class BaseRiskOverlay(ABC):
    """Interface for additive risk overlays that augment risk_engine.py.

    No fit() method: overlays are stateless and compute entirely from the price
    series passed at call time. If a future overlay requires historical
    calibration, add fit() to this ABC at that point.
    """

    @abstractmethod
    def compute(
        self,
        prices_df: pd.DataFrame,
        positions: Optional[list] = None,
    ) -> "RiskOverlayResult":
        """Compute risk overlay signals from price data.

        Args:
            prices_df: DataFrame with a 'close' column, DatetimeIndex.
            positions: Optional list of open positions (accepted, may be ignored).

        Returns:
            RiskOverlayResult with slip_deficit, ttf_probability, tighten_stops.
        """
        ...


@dataclass
class RiskOverlayResult:
    """Output of BaseRiskOverlay.compute()."""

    slip_deficit: float      # Signed: (price - SMA_N) / SMA_N
    ttf_probability: float   # [0, 1] sigmoid of volatility Z-score
    tighten_stops: bool      # True when vol_zscore > ttf_threshold
