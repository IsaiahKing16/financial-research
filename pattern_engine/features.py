"""
features.py — Feature column definitions, FeatureRegistry, and weighting utilities.

Linear: SLE-60 (original), Phase 6 (FeatureRegistry + returns_candle)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ─── Returns-only feature set (locked setting) ────────────────────────────────

RETURNS_ONLY_COLS: list[str] = [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]
"""Canonical 8-feature return fingerprint. Locked setting — do not change."""

VOL_NORM_COLS: list[str] = [f"ret_{w}d_norm" for w in [1, 3, 7, 14, 30, 45, 60, 90]]
"""Volatility-normalized return fingerprint (M9).

Each feature = ret_Xd / rolling_std(daily_returns, window=X).
Dimensionless Sharpe-like ratio — comparable across tickers with different
volatility regimes (large-cap vs mid-cap). Requires prepare.py re-run.
Candidate to replace RETURNS_ONLY_COLS once walk-forward confirms improvement.
"""


# ─── FeatureSet + FeatureRegistry ─────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureSet:
    """Descriptor for a named feature set used by EngineConfig.

    Args:
        name:        Unique identifier (matches the key in FeatureRegistry).
        columns:     Ordered list of column names. Order is HNSW-index-stable —
                     never change column order after an index has been built.
        description: Human-readable summary of what this set contains.
    """
    name: str
    columns: list[str] = field(default_factory=list)
    description: str = ""


def _build_registry() -> dict[str, FeatureSet]:
    """Build the feature registry. Deferred to avoid import-time circular deps."""
    from pattern_engine.candlestick import CANDLE_COLS  # Phase 6
    return {
        "returns_only": FeatureSet(
            name="returns_only",
            columns=list(VOL_NORM_COLS),
            description="8 vol-normalized returns — locked default (Phase 1-5)",
        ),
        "returns_candle": FeatureSet(
            name="returns_candle",
            columns=list(VOL_NORM_COLS) + list(CANDLE_COLS),
            description="8 vol-normalized returns + 15 candlestick features (Phase 6)",
        ),
    }


class _FeatureRegistry:
    """Lazy-initialized registry mapping feature set names → FeatureSet.

    Initialization is deferred so that circular imports during package load
    (features.py → candlestick.py → features.py) are avoided.
    """

    def __init__(self) -> None:
        self._registry: dict[str, FeatureSet] | None = None

    def _ensure_loaded(self) -> None:
        if self._registry is None:
            self._registry = _build_registry()

    def get(self, name: str) -> FeatureSet:
        """Return the FeatureSet for a given name.

        Args:
            name: Feature set identifier (e.g. "returns_only").

        Returns:
            FeatureSet with .columns and .description.

        Raises:
            KeyError: If name is not registered.
        """
        self._ensure_loaded()
        if name not in self._registry:  # type: ignore[operator]
            raise KeyError(
                f"Unknown feature set '{name}'. "
                f"Available: {sorted(self._registry.keys())}"  # type: ignore[union-attr]
            )
        return self._registry[name]  # type: ignore[index]

    def __contains__(self, name: str) -> bool:
        self._ensure_loaded()
        return name in self._registry  # type: ignore[operator]

    def keys(self):
        self._ensure_loaded()
        return self._registry.keys()  # type: ignore[union-attr]


FeatureRegistry = _FeatureRegistry()
"""Singleton feature set registry. Use FeatureRegistry.get("returns_only")."""


def get_feature_cols(feature_set: str) -> list[str]:
    """Resolve feature column names for a given feature set.

    Args:
        feature_set: Feature set name (e.g. "returns_only").

    Returns:
        Ordered list of column names for that feature set.

    Raises:
        ValueError: If the feature set is not registered.
    """
    try:
        return FeatureRegistry.get(feature_set).columns
    except KeyError:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. "
            f"Available: {list(FeatureRegistry.keys())}"
        ) from None


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
