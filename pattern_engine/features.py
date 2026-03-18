"""
features.py — Feature sets and registry for pluggable feature selection.

FeatureRegistry allows swapping feature sets via a config string
(e.g. feature_set="returns_candle") without touching engine code.
Custom sets can be registered at runtime.
"""

from dataclasses import dataclass, field


# ============================================================
# Column definitions
# ============================================================

RETURN_WINDOWS = [1, 3, 7, 14, 30, 45, 60, 90]
RETURN_COLS = [f"ret_{w}d" for w in RETURN_WINDOWS]

SUPPLEMENT_COLS = [
    "vol_10d", "vol_30d", "vol_ratio", "vol_abnormal",
    "rsi_14", "atr_14", "price_vs_sma20", "price_vs_sma50",
]

VOL_COLS = ["vol_10d", "vol_30d", "vol_ratio", "vol_abnormal"]

# Multi-timeframe candlestick features (5 per timeframe × 3 timeframes = 15)
CANDLE_1D_COLS = [
    "candle_1d_body_to_range", "candle_1d_upper_wick",
    "candle_1d_lower_wick", "candle_1d_body_pos", "candle_1d_direction",
]
CANDLE_3D_COLS = [
    "candle_3d_body_to_range", "candle_3d_upper_wick",
    "candle_3d_lower_wick", "candle_3d_body_pos", "candle_3d_direction",
]
CANDLE_5D_COLS = [
    "candle_5d_body_to_range", "candle_5d_upper_wick",
    "candle_5d_lower_wick", "candle_5d_body_pos", "candle_5d_direction",
]
CANDLE_COLS = CANDLE_1D_COLS + CANDLE_3D_COLS + CANDLE_5D_COLS

SECTOR_COLS = [
    "sector_relative_return_7d", "spy_correlation_30d", "sector_rank_30d",
]

# Forward targets
FORWARD_WINDOWS = [1, 3, 7, 14, 30]
FORWARD_RETURN_COLS = [f"fwd_{w}d" for w in FORWARD_WINDOWS]
FORWARD_BINARY_COLS = [f"fwd_{w}d_up" for w in FORWARD_WINDOWS]


# ============================================================
# FeatureSet and Registry
# ============================================================

@dataclass
class FeatureSet:
    """A named set of feature columns for the matching engine."""
    name: str
    columns: list
    description: str = ""


class FeatureRegistry:
    """Registry of available feature sets.

    Built-in sets are registered at import time. Custom sets can be
    added via register(). Feature swapping is a config change:
    EngineConfig(feature_set="returns_candle")
    """

    _sets: dict[str, FeatureSet] = {}

    @classmethod
    def register(cls, name: str, columns: list, description: str = "") -> None:
        """Register a new feature set."""
        cls._sets[name] = FeatureSet(name=name, columns=list(columns),
                                     description=description)

    @classmethod
    def get(cls, name: str) -> FeatureSet:
        """Get a registered feature set by name."""
        if name not in cls._sets:
            available = ", ".join(sorted(cls._sets.keys()))
            raise KeyError(f"Unknown feature set: {name!r}. "
                           f"Available: {available}")
        return cls._sets[name]

    @classmethod
    def list_sets(cls) -> list[str]:
        """List all registered feature set names."""
        return sorted(cls._sets.keys())


# Register built-in feature sets
FeatureRegistry.register(
    "returns_only", RETURN_COLS,
    "Proven 8-feature return baseline (locked from sweeps 2/3)"
)
FeatureRegistry.register(
    "returns_candle", RETURN_COLS + CANDLE_COLS,
    "Returns + continuous multi-timeframe candlestick encoding (1d/3d/5d)"
)
FeatureRegistry.register(
    "returns_vol", RETURN_COLS + VOL_COLS,
    "Returns + volatility/volume features"
)
FeatureRegistry.register(
    "returns_sector", RETURN_COLS + SECTOR_COLS,
    "Returns + cross-asset sector signals"
)
FeatureRegistry.register(
    "full", RETURN_COLS + SUPPLEMENT_COLS + CANDLE_COLS + SECTOR_COLS,
    "All features combined (experimental)"
)

# CONV_LSTM hybrid feature set placeholder (requires trained network)
LSTM_LATENT_COLS = [f"lstm_latent_{i}" for i in range(16)]
FeatureRegistry.register(
    "returns_hybrid", RETURN_COLS + LSTM_LATENT_COLS,
    "Returns + CONV_LSTM latent embeddings (requires trained network)"
)
