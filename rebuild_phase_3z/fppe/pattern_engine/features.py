"""
features.py — Feature column definitions and weighting utilities.

Mirrors production pattern_engine/features.py but lives in the rebuild
workspace. Imports from the production FeatureRegistry where available;
falls back to hardcoded returns_only when running in isolation.

Linear: SLE-60
"""

from __future__ import annotations

import numpy as np


# ─── Returns-only feature set (locked setting) ────────────────────────────────

RETURNS_ONLY_COLS: list[str] = [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]
"""Canonical 8-feature return fingerprint. Locked setting — do not change."""


def get_feature_cols(feature_set: str) -> list[str]:
    """Resolve feature column names for a given feature set.

    Delegates to production FeatureRegistry where available; falls back
    to hardcoded sets for isolated testing.

    Args:
        feature_set: Feature set name (e.g. "returns_only").

    Returns:
        Ordered list of column names for that feature set.

    Raises:
        ValueError: If the feature set is not registered and not hardcoded.
    """
    try:
        from pattern_engine.features import FeatureRegistry
        return FeatureRegistry.get(feature_set).columns
    except (ImportError, KeyError):
        pass

    # Hardcoded fallback for isolated rebuild workspace tests
    if feature_set == "returns_only":
        return RETURNS_ONLY_COLS

    raise ValueError(
        f"Unknown feature set '{feature_set}'. "
        "Ensure FeatureRegistry is accessible or add a hardcoded fallback."
    )


# ─── Feature weighting ────────────────────────────────────────────────────────

def apply_feature_weights(
    X: np.ndarray,
    feature_cols: list[str],
    weights: dict[str, float],
) -> np.ndarray:
    """Apply per-feature scalar weights to a feature matrix.

    Weights are applied BEFORE NN index build AND query transform.
    Higher weight = that feature influences matching distance more.
    A weight of 1.0 means no change; 0.0 would zero out the feature.

    The production convention (EngineConfig.feature_weights) uses
    "uniform" weighting for the locked returns_only set (all 1.0),
    so this function is a no-op in the baseline configuration.

    Args:
        X: (N, D) feature matrix — already scaled by StandardScaler.
        feature_cols: Column names aligned to X columns (length D).
        weights: Mapping of column name → weight scalar.
                 Columns not in weights default to weight=1.0.

    Returns:
        X_weighted: (N, D) float64 array with weights applied.
                    Always a copy — never modifies the input array.
    """
    X_weighted = X.copy()
    for i, col in enumerate(feature_cols):
        w = weights.get(col, 1.0)
        if w != 1.0:
            X_weighted[:, i] *= w
    return X_weighted
