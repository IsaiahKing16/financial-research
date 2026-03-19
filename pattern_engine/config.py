"""
config.py — EngineConfig frozen dataclass with proven research defaults.

All default values trace to specific sweep experiments documented in
PROJECT_GUIDE.md. nn_algorithm is derived from distance_metric
(ball_tree doesn't support cosine). feature_weights affect distance
computation and must be applied before NN index build AND query transform.
"""

import dataclasses
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for the PatternEngine.

    All defaults are proven via sweep experiments. See PROJECT_GUIDE.md
    Section 7 for evidence behind each locked setting.
    """

    # --- Matching Algorithm ---
    top_k: int = 50
    max_distance: float = 1.1019  # Quantile-calibrated, AvgK ~42
    distance_weighting: str = "uniform"  # Beats inverse (sweep 1)
    distance_metric: str = "euclidean"  # Cosine collapsed at 93.3%
    nn_jobs: int = 1  # Must be 1 on Windows/Python 3.12 (joblib deadlock)
    batch_size: int = 256

    # --- Features ---
    feature_set: str = "returns_only"  # Full 16 was 3.5x worse (sweeps 2/3)
    feature_weights: dict = field(default_factory=lambda: {
        "ret_1d": 1.0,
        "ret_3d": 1.0,
        "ret_7d": 1.5,       # Medium-term trend matters more
        "ret_14d": 1.5,
        "ret_30d": 1.0,
        "ret_45d": 0.8,
        "ret_60d": 0.8,
        "ret_90d": 0.5,      # Very long-term trend matters less
        "vol_10d": 1.2,      # Volatility regime is important context
        "vol_30d": 1.0,
        "vol_ratio": 1.0,
        "vol_abnormal": 0.8,
        "rsi_14": 1.0,
        "atr_14": 0.8,
        "price_vs_sma20": 1.2,  # Mean reversion signal
        "price_vs_sma50": 1.0,
        # Overnight/session features (neutral weights, pending sweep)
        "ret_overnight": 1.0,
        "ret_intraday": 1.0,
        "gap_magnitude": 1.0,
        "gap_direction_streak": 0.8,
        "weekend_gap": 1.0,
        "weekend_gap_magnitude": 0.8,
    })

    # --- Forward Projection ---
    projection_horizon: str = "fwd_7d_up"  # Best BSS across horizons

    # --- Signal Generation (three-filter gate) ---
    confidence_threshold: float = 0.65  # Best binary accuracy (sweep 1)
    agreement_spread: float = 0.10  # Inert at AvgK ~42 but kept for quality
    min_matches: int = 10

    # --- Cohort Filtering ---
    same_sector_only: bool = False  # Sector-only had worst BSS (sweep 1)
    exclude_same_ticker: bool = True

    # --- Regime ---
    regime_filter: bool = True
    regime_mode: str = "binary"  # "binary" (2), "multi" (4), "octet" (8)
    regime_fallback: bool = True  # Auto-widen if too few matches
    adx_threshold: float = 25.0  # ADX > threshold = Trending (octet only)

    # --- Calibration ---
    calibration_method: str = "platt"  # "platt", "isotonic", "none"

    @property
    def nn_algorithm(self) -> str:
        """Derived from distance_metric — not independently configurable.

        ball_tree for Euclidean (avoids joblib threading path).
        brute for cosine (tree indexes don't support cosine).
        """
        return "ball_tree" if self.distance_metric == "euclidean" else "brute"

    def replace(self, **kwargs) -> "EngineConfig":
        """Create a new EngineConfig with specific fields overridden."""
        return dataclasses.replace(self, **kwargs)


# Walk-forward fold definitions (expanding windows)
WALKFORWARD_FOLDS = [
    {
        "train_end": "2018-12-31",
        "val_start": "2019-01-01",
        "val_end": "2019-12-31",
        "label": "2019",
    },
    {
        "train_end": "2019-12-31",
        "val_start": "2020-01-01",
        "val_end": "2020-12-31",
        "label": "2020 (COVID)",
    },
    {
        "train_end": "2020-12-31",
        "val_start": "2021-01-01",
        "val_end": "2021-12-31",
        "label": "2021",
    },
    {
        "train_end": "2021-12-31",
        "val_start": "2022-01-01",
        "val_end": "2022-12-31",
        "label": "2022 (Bear)",
    },
    {
        "train_end": "2022-12-31",
        "val_start": "2023-01-01",
        "val_end": "2023-12-31",
        "label": "2023",
    },
    {
        "train_end": "2023-12-31",
        "val_start": "2024-01-01",
        "val_end": "2024-12-31",
        "label": "2024 (Standard Val)",
    },
]
