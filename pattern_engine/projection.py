"""
projection.py — Forward projection and signal generation.

Given a set of historical analogues (matches), computes the forward
projection (probability, mean return, agreement) and generates a
BUY/SELL/HOLD trading signal via the three-filter gate.
"""

import numpy as np
import pandas as pd


def project_forward(matches: pd.DataFrame, horizon: str = "fwd_7d_up",
                    weighting: str = "uniform") -> dict:
    """
    Given a set of historical analogues, compute the forward projection.

    Args:
        matches: DataFrame of matched historical rows with 'distance' column
        horizon: forward target column (e.g. "fwd_7d_up")
        weighting: "uniform" or "inverse" distance weighting

    Returns:
        dict with probability_up, mean_return, median_return, agreement,
        n_matches, ensemble_returns
    """
    if len(matches) == 0:
        return {
            "probability_up": 0.5,
            "mean_return": 0.0,
            "median_return": 0.0,
            "agreement": 0.0,
            "n_matches": 0,
            "ensemble_returns": np.array([]),
        }

    return_col = horizon.replace("_up", "")

    binary_outcomes = matches[horizon].values
    return_outcomes = matches[return_col].values
    distances = matches["distance"].values

    if weighting == "inverse":
        weights = 1.0 / (distances + 0.01)
        weights = weights / weights.sum()
        probability_up = np.average(binary_outcomes, weights=weights)
        mean_return = np.average(return_outcomes, weights=weights)
    else:
        probability_up = binary_outcomes.mean()
        mean_return = return_outcomes.mean()

    median_return = np.median(return_outcomes)
    agreement = abs(probability_up - 0.5) * 2

    return {
        "probability_up": float(probability_up),
        "mean_return": float(mean_return),
        "median_return": float(median_return),
        "agreement": float(agreement),
        "n_matches": len(matches),
        "ensemble_returns": return_outcomes,
    }


def generate_signal(projection: dict, threshold: float = 0.65,
                    min_agreement: float = 0.10,
                    min_matches: int = 10) -> tuple[str, str]:
    """
    Convert a forward projection into a BUY/SELL/HOLD signal.

    Three-filter gate (ALL must pass for a trade):
      1. At least min_matches valid analogues
      2. Agreement exceeds min_agreement
      3. Probability exceeds threshold (BUY) or falls below 1-threshold (SELL)

    Args:
        projection: dict from project_forward()
        threshold: probability threshold for BUY signal (1-threshold for SELL)
        min_agreement: minimum agreement spread
        min_matches: minimum number of valid matches

    Returns:
        (signal, reason) tuple where signal is "BUY", "SELL", or "HOLD"
    """
    prob = projection["probability_up"]
    agree = projection["agreement"]
    n = projection["n_matches"]

    if n < min_matches:
        return "HOLD", "insufficient_matches"
    if agree < min_agreement:
        return "HOLD", "low_agreement"
    if prob >= threshold:
        return "BUY", f"prob={prob:.3f}_agree={agree:.3f}"
    elif prob <= (1 - threshold):
        return "SELL", f"prob={prob:.3f}_agree={agree:.3f}"
    else:
        return "HOLD", f"prob={prob:.3f}_below_threshold"
