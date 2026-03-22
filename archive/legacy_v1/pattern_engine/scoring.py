"""
scoring.py — Proper scoring rules for probabilistic evaluation.

Implements Brier Score, Brier Skill Score, CRPS (optional), and
calibration bucket computation. These are the primary metrics for
evaluating the analogue matching engine's probability forecasts.
"""

import numpy as np

# Graceful import for scoringrules (optional dependency)
try:
    import scoringrules
    SCORINGRULES_AVAILABLE = True
except ImportError:
    SCORINGRULES_AVAILABLE = False


def brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Brier score: mean squared error between predicted P and binary outcome.

    Range: 0 (perfect) to 1 (worst). A naive 0.5 forecast scores 0.25.
    Lower is better.
    """
    return float(np.mean((y_pred_proba - y_true) ** 2))


def brier_skill_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Brier Skill Score: improvement over always predicting the base rate.

    BSS = 1 - (BS_model / BS_climatology)
    Range: -inf to 1.0. Positive = better than climatology. 0 = no skill.
    """
    bs_model = brier_score(y_true, y_pred_proba)
    base_rate = y_true.mean()
    bs_clim = brier_score(y_true, np.full_like(y_pred_proba, base_rate))
    if bs_clim == 0:
        return 0.0
    return float(1.0 - (bs_model / bs_clim))


def compute_crps(y_true_returns: np.ndarray,
                 ensemble_returns_list: list) -> float | None:
    """CRPS using the analogue ensemble (requires scoringrules library).

    For each query, measures the distance between the predicted CDF
    (from the analogue ensemble) and the actual outcome.
    Lower is better. Returns None if scoringrules not installed.

    Args:
        y_true_returns: array of actual forward returns
        ensemble_returns_list: list of arrays, each = analogue returns for a query
    """
    if not SCORINGRULES_AVAILABLE:
        return None

    crps_scores = []
    for obs, ensemble in zip(y_true_returns, ensemble_returns_list):
        if len(ensemble) < 2:
            continue
        try:
            score = scoringrules.crps_ensemble(float(obs), ensemble.astype(float))
            crps_scores.append(score)
        except Exception:
            continue

    if not crps_scores:
        return None
    return float(np.mean(crps_scores))


def compute_calibration(y_true: np.ndarray, y_pred_proba: np.ndarray,
                        n_buckets: int = 5) -> list[dict]:
    """Reliability calibration check.

    Splits predictions into probability buckets and checks whether
    actual UP rate matches predicted probability in each bucket.
    Perfect calibration: if model says 60%, exactly 60% went up.

    Args:
        y_true: binary ground truth
        y_pred_proba: predicted probabilities
        n_buckets: number of calibration bins

    Returns:
        list of dicts with pred_range, n, pred_prob, actual_rate, gap
    """
    buckets = []
    edges = np.linspace(0, 1, n_buckets + 1)

    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        # Last bucket uses <= hi to capture probability=1.0 (edge case fix)
        if i == n_buckets - 1:
            mask = (y_pred_proba >= lo) & (y_pred_proba <= hi)
        else:
            mask = (y_pred_proba >= lo) & (y_pred_proba < hi)

        if mask.sum() == 0:
            continue

        actual_rate = y_true[mask].mean()
        pred_center = (lo + hi) / 2
        buckets.append({
            "pred_range": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "pred_prob": round(float(pred_center), 2),
            "actual_rate": round(float(actual_rate), 4),
            "gap": round(float(actual_rate - pred_center), 4),
        })

    return buckets
