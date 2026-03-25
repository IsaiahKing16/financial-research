"""Tests for prepare.compute_vol_normalized_features().

Covers: normal computation, epsilon guard against zero-vol, clip guard against
extreme ratios, and NaN propagation for rows with insufficient history.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import prepare once at module level — it has a module-level side effect
# (from pattern_engine.sector import TICKERS) that is slow to re-execute.
import prepare  # noqa: E402


def _make_df(n=30, daily_return=0.01, seed=42):
    """Minimal DataFrame with Close and ret_Xd columns."""
    rng = np.random.RandomState(seed)
    close = 100.0 * np.cumprod(1 + rng.randn(n) * daily_return)
    df = pd.DataFrame({"Close": close})
    for w in prepare.RETURN_WINDOWS:
        df[f"ret_{w}d"] = df["Close"].pct_change(w)
    return df


def test_vol_norm_columns_created():
    """compute_vol_normalized_features() produces all 8 ret_Xd_norm columns."""
    df = _make_df(n=60)
    out = prepare.compute_vol_normalized_features(df)
    for w in prepare.RETURN_WINDOWS:
        assert f"ret_{w}d_norm" in out.columns, f"Missing ret_{w}d_norm"


def test_vol_norm_values_finite_for_normal_ticker():
    """All non-NaN vol-norm values are finite after clip guard."""
    df = _make_df(n=60, daily_return=0.01)
    out = prepare.compute_vol_normalized_features(df)
    for w in prepare.RETURN_WINDOWS:
        col = out[f"ret_{w}d_norm"].dropna()
        assert np.all(np.isfinite(col.values)), f"ret_{w}d_norm has non-finite values"


def test_vol_norm_clipped_at_plus_minus_10():
    """Vol-norm values are clipped to [-10, 10] — extreme ratios cannot reach HNSW."""
    df = _make_df(n=60, daily_return=0.01)
    # Inject an abnormally large return to force extreme ratio.
    # Note: compute_vol_normalized_features() reads ret_Xd columns directly;
    # setting ret_1d = 100.0 while rolling_vol stays small forces the clip path.
    df["ret_1d"] = df["ret_1d"].fillna(0.0)
    df.loc[df.index[20], "ret_1d"] = 100.0  # +10,000% outlier
    out = prepare.compute_vol_normalized_features(df)
    col = out["ret_1d_norm"].dropna()
    assert col.max() <= 10.0, f"ret_1d_norm exceeded clip upper bound: {col.max()}"
    assert col.min() >= -10.0, f"ret_1d_norm exceeded clip lower bound: {col.min()}"


def test_epsilon_prevents_division_by_zero():
    """When rolling_std = 0.0 (constant Close prices), result is finite (not inf).

    The epsilon guard prevents division by zero. The function derives rolling_vol
    from df["Close"].pct_change() — setting Close to constant makes rolling_vol = 0.0.
    The pre-computed ret_Xd columns are not used for vol estimation; only Close matters.
    """
    df = _make_df(n=30)
    # Constant Close → daily_returns = 0.0 everywhere → rolling_std = 0.0
    df["Close"] = 100.0
    out = prepare.compute_vol_normalized_features(df)
    for w in prepare.RETURN_WINDOWS:
        col = out[f"ret_{w}d_norm"].dropna()
        assert not col.isin([np.inf, -np.inf]).any(), (
            f"ret_{w}d_norm has inf despite epsilon guard"
        )


def test_early_rows_produce_nan_before_dropna():
    """First rows (< min_periods rolling window) produce NaN vol-norm.

    These NaN rows are caught by build_analogue_database()'s dropna() call
    and never reach PatternMatcher or the HNSW guard.
    """
    df = _make_df(n=15)  # very short history — 90d window cannot fill
    out = prepare.compute_vol_normalized_features(df)
    col = out["ret_90d_norm"]
    assert col.isna().any(), (
        "Expected NaN in ret_90d_norm for short history — these rows "
        "must be caught by prepare.py's dropna() before reaching HNSW"
    )
