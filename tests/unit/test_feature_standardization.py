"""P8-PRE-4: Feature standardization verification tests.

Covers:
- standardize_features flag default (True)
- standardize_features=False skips scaler
- No scaler leakage across folds
- Per-feature variance post-scaling (real data, slow)
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from pattern_engine.config import EngineConfig
from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import get_feature_cols


def test_standardize_features_default_true():
    """EngineConfig.standardize_features defaults to True."""
    cfg = EngineConfig()
    assert cfg.standardize_features is True


def test_standardize_features_false_skips_scaler():
    """When standardize_features=False, raw features are used for distance."""
    feature_cols = get_feature_cols("returns_only")  # 8 features, simpler
    n_train = 200
    # Make features with very different variances (like raw returns vs candle)
    rng = np.random.default_rng(42)
    X_high_var = rng.normal(0, 1.0, (n_train, 4))    # variance ≈ 1.0
    X_low_var = rng.normal(0, 0.07, (n_train, 4))    # variance ≈ 0.005
    X_raw = np.hstack([X_high_var, X_low_var])

    train_df = pd.DataFrame(X_raw, columns=feature_cols)
    train_df["Ticker"] = "TEST"
    train_df["Date"] = pd.date_range("2018-01-01", periods=n_train)
    train_df["fwd_7d_up"] = rng.integers(0, 2, n_train).astype(float)

    # With scaling (default): feature variances after transform should all be ~1.0
    cfg_scaled = EngineConfig(
        feature_set="returns_only", standardize_features=True, use_hnsw=False
    )
    m_scaled = PatternMatcher(cfg_scaled)
    m_scaled.fit(train_df, feature_cols)
    X_scaled_train = m_scaled._scaler.transform(X_raw)
    per_feature_var_scaled = X_scaled_train.var(axis=0)
    assert np.all(np.abs(per_feature_var_scaled - 1.0) < 0.05), (
        f"Expected unit variance after scaling, got: {per_feature_var_scaled}"
    )

    # Without scaling: low-variance features dominate less in raw space
    cfg_raw = EngineConfig(
        feature_set="returns_only", standardize_features=False, use_hnsw=False
    )
    m_raw = PatternMatcher(cfg_raw)
    m_raw.fit(train_df, feature_cols)
    # Scaler should NOT be fitted when standardize_features=False
    assert not hasattr(m_raw._scaler, "mean_"), (
        "Scaler should not be fitted when standardize_features=False"
    )
