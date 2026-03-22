"""
calibration.py — Probability calibration for analogue matching.

Maps raw P(up) from analogue matching to calibrated probabilities
using Platt scaling (logistic regression) or isotonic regression.

The calibrator is fitted on training data only (no validation leakage).
The calibration pass during fit() uses train-as-query with identical
filtering settings as inference to ensure distribution consistency.
"""

import numpy as np


class PlattCalibrator:
    """Platt scaling via logistic regression."""

    def __init__(self):
        self._model = None

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self._model.fit(raw_probs.reshape(-1, 1), y_true)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    @property
    def fitted(self) -> bool:
        return self._model is not None


class IsotonicCalibrator:
    """Non-parametric calibration via isotonic regression."""

    def __init__(self):
        self._model = None

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(raw_probs, y_true)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return self._model.predict(raw_probs)

    @property
    def fitted(self) -> bool:
        return self._model is not None


class NoCalibrator:
    """Pass-through calibrator (identity transform)."""

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "NoCalibrator":
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return raw_probs

    @property
    def fitted(self) -> bool:
        return True


def make_calibrator(method: str):
    """Factory for creating calibrators by name.

    Args:
        method: "platt", "isotonic", or "none"

    Returns:
        Calibrator instance (PlattCalibrator, IsotonicCalibrator, or NoCalibrator)
    """
    if method == "platt":
        return PlattCalibrator()
    elif method == "isotonic":
        return IsotonicCalibrator()
    elif method == "none":
        return NoCalibrator()
    else:
        raise ValueError(f"Unknown calibration method: {method!r}. "
                         f"Choose from: 'platt', 'isotonic', 'none'")
