"""
pattern_engine/owa_weights.py — OWA feature weighting for Phase 7 E2.

Computes global OWA weights from mutual information rankings.
Applied in PatternMatcher.fit() when use_owa=True.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def compute_mi_ranking(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Return feature indices sorted by MI with y_train, descending (best first).

    Args:
        X_train: (N, D) feature matrix (should be scaled — same space as KNN index).
        y_train: (N,) binary labels {0, 1}.

    Returns:
        (D,) integer array — feature indices, highest MI first.
    """
    mi = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=3)
    return np.argsort(mi)[::-1]


def owa_weights(n_features: int, alpha: float) -> np.ndarray:
    """Compute OWA weight vector.

    w[i] = ((n_features - i) / n_features) ** alpha  for i in [0, n_features-1]
    Normalized so weights sum to n_features (mean = 1.0; alpha=0 → uniform).

    Index 0 = weight for the highest-MI feature.

    Args:
        n_features: Number of features (23 for returns_candle).
        alpha: Concentration exponent.
               0 = uniform, 1 = linear decay, 4 = aggressive concentration.

    Returns:
        (n_features,) float64 array.
    """
    ranks = np.arange(n_features, dtype=float)
    raw = ((n_features - ranks) / n_features) ** alpha
    total = raw.sum()
    if total < 1e-12:
        return np.ones(n_features, dtype=float)
    return raw * n_features / total


def mi_to_weight_dict(
    feature_cols: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Compute OWA weight dict keyed by column name.

    IMPORTANT: X_train must be the SCALED feature matrix (post-StandardScaler),
    not the raw feature matrix. MI ranking must match the KNN distance space.

    Args:
        feature_cols: Ordered list of column names matching X_train columns.
        X_train: (N, D) scaled feature matrix.
        y_train: (N,) binary labels.
        alpha: OWA concentration exponent.

    Returns:
        Dict mapping column name -> weight (compatible with EngineConfig.feature_weights).
    """
    ranking = compute_mi_ranking(X_train, y_train)   # indices sorted best-first
    weights_by_rank = owa_weights(len(feature_cols), alpha)
    weight_dict = {}
    for rank_pos, feat_idx in enumerate(ranking):
        col = feature_cols[feat_idx]
        weight_dict[col] = float(weights_by_rank[rank_pos])
    return weight_dict


def evaluate_owa_gate(
    baseline_bss: np.ndarray,
    enhanced_bss: np.ndarray,
    gate_delta: float = 0.001,
    gate_min_folds: int = 3,
    worst_fold_max_degradation: float = 0.0005,
) -> tuple[str, str]:
    """Evaluate the OWA gate. Returns ('PASS'/'FAIL', reason_string)."""
    deltas = enhanced_bss - baseline_bss
    n_improved = int((deltas >= gate_delta).sum())
    worst_delta = float(deltas.min())

    if n_improved < gate_min_folds:
        return "FAIL", (
            f"only {n_improved}/{len(deltas)} folds improved by >= +{gate_delta}; "
            f"worst delta={worst_delta:.5f}"
        )
    if worst_delta < -worst_fold_max_degradation:
        return "FAIL", (
            f"worst fold degraded by {worst_delta:.5f} "
            f"(limit: -{worst_fold_max_degradation}); "
            f"{n_improved}/{len(deltas)} folds otherwise improved"
        )
    return "PASS", (
        f"{n_improved}/{len(deltas)} folds improved, worst_delta={worst_delta:.5f}"
    )
