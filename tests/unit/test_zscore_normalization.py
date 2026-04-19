"""T7.5-1 gate tests: L2 contribution balance and group_balanced_weights.

Does NOT duplicate test_feature_standardization.py tests.
Covers:
  - L2 distance contribution balance (returns vs candle share of L2²)
  - group_balanced_weights() factory correctness
  - group_balanced_weights() round-trip via apply_feature_weights
"""
import numpy as np

from pattern_engine.features import (
    VOL_NORM_COLS,
    apply_feature_weights,
    group_balanced_weights,  # will not exist yet — tests will fail
)

RETURNS_COLS = list(VOL_NORM_COLS)  # 8 features


def test_group_balanced_weights_returns_dict():
    """group_balanced_weights() returns a dict with exactly 23 keys."""
    from pattern_engine.candlestick import CANDLE_COLS
    candle_cols = list(CANDLE_COLS)
    weights = group_balanced_weights(RETURNS_COLS, candle_cols)
    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(RETURNS_COLS) | set(candle_cols)


def test_group_balanced_weights_equalizes_group_l2():
    """After applying group_balanced_weights, returns and candle groups
    contribute equal total variance to L2 distance.

    Tolerance is 5% (consistent with test_feature_standardization.py) because
    sample variance with n=1000 has std ≈ 0.045 per feature; the 5% window
    provides a stable bound without being so tight it's seed-dependent.
    """
    from pattern_engine.candlestick import CANDLE_COLS
    candle_cols = list(CANDLE_COLS)
    weights = group_balanced_weights(RETURNS_COLS, candle_cols)

    n_r = len(RETURNS_COLS)
    n_c = len(candle_cols)
    all_cols = RETURNS_COLS + candle_cols

    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, n_r + n_c))  # unit variance per feature

    X_w = apply_feature_weights(X, all_cols, weights)

    returns_group_var = X_w[:, :n_r].var(axis=0).sum()
    candle_group_var  = X_w[:, n_r:].var(axis=0).sum()

    ratio = returns_group_var / candle_group_var
    assert abs(ratio - 1.0) < 0.05, (
        f"Group L2 variance ratio expected 1.0, got {ratio:.4f}. "
        f"returns_group_var={returns_group_var:.4f}, candle_group_var={candle_group_var:.4f}"
    )


def test_group_balanced_weights_uniform_is_not_balanced():
    """Confirm uniform weights do NOT equalize group L2 — 8 returns vs 15 candles."""
    from pattern_engine.candlestick import CANDLE_COLS
    candle_cols = list(CANDLE_COLS)
    n_r = len(RETURNS_COLS)
    n_c = len(candle_cols)
    all_cols = RETURNS_COLS + candle_cols

    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, n_r + n_c))

    uniform = {col: 1.0 for col in all_cols}
    X_w = apply_feature_weights(X, all_cols, uniform)

    returns_group_var = X_w[:, :n_r].var(axis=0).sum()
    candle_group_var  = X_w[:, n_r:].var(axis=0).sum()

    expected_ratio = n_r / n_c  # 8/15 ≈ 0.533
    actual_ratio = returns_group_var / candle_group_var
    assert abs(actual_ratio - expected_ratio) < 0.05, (
        f"Expected uniform ratio ≈ {expected_ratio:.3f}, got {actual_ratio:.3f}"
    )
