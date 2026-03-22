"""
test_datasets.py — Unit tests for Pandera DataFrame schemas in contracts/datasets.py.

Tests:
  - make_train_db_schema: accepts valid wide DataFrames, rejects missing cols,
    NaN values, non-binary targets
  - make_query_db_schema: accepts valid DataFrames, rejects missing cols / NaN
  - make_feature_matrix_schema: strict=True, exact columns only
  - OHLCVSchema: validates Close presence and positivity
  - SignalsOutputSchema: strict=True, validates all output columns

Linear: SLE-57
"""

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from rebuild_phase_3z.fppe.pattern_engine.contracts.datasets import (
    OHLCVSchema,
    SignalsOutputSchema,
    make_feature_matrix_schema,
    make_query_db_schema,
    make_train_db_schema,
)

# ─── Fixtures ──────────────────────────────────────────────────────────────────

FEATURE_COLS = [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]
N = 100


def _make_train_df(n: int = N, feature_cols=None) -> pd.DataFrame:
    """Generate a valid wide training DataFrame."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n),
        "Ticker": ["AAPL"] * n,
        "fwd_7d_up": rng.randint(0, 2, size=n).astype(float),
        # Extra column (should not cause strict=False schema to fail)
        "Close": rng.uniform(100, 200, size=n),
    })
    for col in feature_cols:
        df[col] = rng.randn(n)
    return df


def _make_query_df(n: int = 20, feature_cols=None) -> pd.DataFrame:
    """Generate a valid query DataFrame (no target column)."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    rng = np.random.RandomState(99)
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n),
        "Ticker": ["MSFT"] * n,
    })
    for col in feature_cols:
        df[col] = rng.randn(n)
    return df


# ─── TrainDBSchema ─────────────────────────────────────────────────────────────

class TestTrainDBSchema:

    def test_valid_dataframe_passes(self):
        schema = make_train_db_schema(FEATURE_COLS)
        df = _make_train_df()
        schema.validate(df)  # Should not raise

    def test_extra_columns_allowed(self):
        """strict=False: extra columns (Close, Volume, etc.) must not cause failure."""
        schema = make_train_db_schema(FEATURE_COLS)
        df = _make_train_df()
        df["Open"] = 150.0
        df["extra_col"] = "garbage"
        schema.validate(df)

    def test_missing_feature_col_raises(self):
        schema = make_train_db_schema(FEATURE_COLS)
        df = _make_train_df()
        df = df.drop(columns=["ret_1d"])
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_missing_date_col_raises(self):
        schema = make_train_db_schema(FEATURE_COLS)
        df = _make_train_df().drop(columns=["Date"])
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_missing_ticker_col_raises(self):
        schema = make_train_db_schema(FEATURE_COLS)
        df = _make_train_df().drop(columns=["Ticker"])
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_missing_target_col_raises(self):
        schema = make_train_db_schema(FEATURE_COLS)
        df = _make_train_df().drop(columns=["fwd_7d_up"])
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_nan_in_feature_col_raises(self):
        schema = make_train_db_schema(FEATURE_COLS)
        df = _make_train_df()
        df.loc[5, "ret_7d"] = np.nan
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_nan_in_target_col_raises(self):
        schema = make_train_db_schema(FEATURE_COLS)
        df = _make_train_df()
        df.loc[10, "fwd_7d_up"] = np.nan
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_non_binary_target_raises(self):
        schema = make_train_db_schema(FEATURE_COLS)
        df = _make_train_df()
        df.loc[0, "fwd_7d_up"] = 2.0  # Not 0 or 1
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_custom_feature_cols(self):
        """Factory function must accept any feature column list."""
        custom_cols = ["ret_1d", "ret_7d"]
        schema = make_train_db_schema(custom_cols)
        rng = np.random.RandomState(7)
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=50),
            "Ticker": ["AAPL"] * 50,
            "fwd_7d_up": rng.randint(0, 2, size=50).astype(float),
            "ret_1d": rng.randn(50),
            "ret_7d": rng.randn(50),
        })
        schema.validate(df)


# ─── QueryDBSchema ─────────────────────────────────────────────────────────────

class TestQueryDBSchema:

    def test_valid_dataframe_passes(self):
        schema = make_query_db_schema(FEATURE_COLS)
        df = _make_query_df()
        schema.validate(df)

    def test_no_target_required(self):
        """Query schema must not require fwd_7d_up."""
        schema = make_query_db_schema(FEATURE_COLS)
        df = _make_query_df()
        assert "fwd_7d_up" not in df.columns
        schema.validate(df)  # Should pass fine

    def test_extra_cols_allowed(self):
        schema = make_query_db_schema(FEATURE_COLS)
        df = _make_query_df()
        df["Close"] = 150.0
        df["fwd_7d_up"] = 1.0  # Extra but allowed
        schema.validate(df)

    def test_nan_in_feature_raises(self):
        schema = make_query_db_schema(FEATURE_COLS)
        df = _make_query_df()
        df.loc[3, "ret_14d"] = np.nan
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_missing_feature_col_raises(self):
        schema = make_query_db_schema(FEATURE_COLS)
        df = _make_query_df().drop(columns=["ret_90d"])
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)


# ─── FeatureMatrixSchema ───────────────────────────────────────────────────────

class TestFeatureMatrixSchema:

    def test_valid_matrix_passes(self):
        schema = make_feature_matrix_schema(FEATURE_COLS)
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(50, len(FEATURE_COLS)), columns=FEATURE_COLS)
        schema.validate(df)

    def test_extra_col_rejected(self):
        """strict=True: extra columns must be rejected."""
        schema = make_feature_matrix_schema(FEATURE_COLS)
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(50, len(FEATURE_COLS)), columns=FEATURE_COLS)
        df["extra"] = 0.0
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_nan_rejected(self):
        schema = make_feature_matrix_schema(FEATURE_COLS)
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(50, len(FEATURE_COLS)), columns=FEATURE_COLS)
        df.loc[0, "ret_1d"] = np.nan
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)


# ─── OHLCVSchema ──────────────────────────────────────────────────────────────

class TestOHLCVSchema:

    def _make_ohlcv(self, n: int = 20) -> pd.DataFrame:
        rng = np.random.RandomState(0)
        return pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=n),
            "Open": rng.uniform(100, 200, n),
            "High": rng.uniform(200, 250, n),
            "Low": rng.uniform(80, 100, n),
            "Close": rng.uniform(100, 200, n),
            "Volume": rng.uniform(1e6, 1e7, n),
        })

    def test_valid_passes(self):
        OHLCVSchema.validate(self._make_ohlcv())

    def test_extra_cols_allowed(self):
        """yfinance returns Adj Close etc; strict=False allows this."""
        df = self._make_ohlcv()
        df["Adj Close"] = df["Close"] * 0.99
        OHLCVSchema.validate(df)

    def test_missing_close_raises(self):
        df = self._make_ohlcv().drop(columns=["Close"])
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)

    def test_negative_close_raises(self):
        df = self._make_ohlcv()
        df.loc[0, "Close"] = -1.0
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)

    def test_nan_close_raises(self):
        df = self._make_ohlcv()
        df.loc[0, "Close"] = np.nan
        with pytest.raises(pa.errors.SchemaError):
            OHLCVSchema.validate(df)

    def test_null_volume_allowed(self):
        """Volume is nullable — thin tickers may have gaps."""
        df = self._make_ohlcv()
        df.loc[5, "Volume"] = np.nan
        OHLCVSchema.validate(df)


# ─── SignalsOutputSchema ───────────────────────────────────────────────────────

class TestSignalsOutputSchema:

    def _make_signals_df(self, n: int = 10) -> pd.DataFrame:
        rng = np.random.RandomState(5)
        return pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=n).astype(str),
            "ticker": ["AAPL"] * n,
            "signal": ["BUY"] * n,
            "confidence": rng.uniform(0.65, 0.95, n),
            "signal_source": ["KNN"] * n,
            "sector": ["Technology"] * n,
            "n_matches": [50] * n,
            "raw_prob": rng.uniform(0.5, 0.9, n),
            "mean_7d_return": rng.uniform(0.5, 3.0, n),
        })

    def test_valid_passes(self):
        SignalsOutputSchema.validate(self._make_signals_df())

    def test_extra_col_rejected(self):
        """strict=True: any extra column in output signals a pipeline change."""
        df = self._make_signals_df()
        df["extra"] = 0
        with pytest.raises(pa.errors.SchemaError):
            SignalsOutputSchema.validate(df)

    def test_invalid_signal_direction_raises(self):
        df = self._make_signals_df()
        df.loc[0, "signal"] = "STRONG_BUY"
        with pytest.raises(pa.errors.SchemaError):
            SignalsOutputSchema.validate(df)

    def test_confidence_out_of_range_raises(self):
        df = self._make_signals_df()
        df.loc[0, "confidence"] = 1.5
        with pytest.raises(pa.errors.SchemaError):
            SignalsOutputSchema.validate(df)

    def test_lowercase_ticker_raises(self):
        df = self._make_signals_df()
        df.loc[0, "ticker"] = "aapl"
        with pytest.raises(pa.errors.SchemaError):
            SignalsOutputSchema.validate(df)

    def test_negative_n_matches_raises(self):
        df = self._make_signals_df()
        df.loc[0, "n_matches"] = -1
        with pytest.raises(pa.errors.SchemaError):
            SignalsOutputSchema.validate(df)
