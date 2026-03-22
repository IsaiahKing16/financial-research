"""
test_pandera_schema.py — Tests for the Pandera-backed schema.py in rebuild workspace.

Verifies that the new schema.py (SLE-59) preserves the same error-detection semantics
as the hand-rolled original. All 13 tests from tests/test_schema.py are replicated
here using the rebuild workspace module.

Key semantic requirements:
  - SchemaError raised (not pandera.errors.SchemaError) — backward compat
  - SchemaError is a ValueError subclass
  - Missing column detection
  - NaN detection in features and target
  - Non-binary target detection
  - Minimum row count (train only)
  - Empty DataFrame detection (query only)
  - coerce=False: type coercion does NOT silently fix bad dtypes

Linear: SLE-59
"""

import numpy as np
import pandas as pd
import pytest

from pattern_engine.schema import (
    SchemaError,
    validate_query_db,
    validate_train_db,
)

# ─── Feature set used in all tests ────────────────────────────────────────────
FEATURE_SET = "returns_only"
FEATURE_COLS = [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]
TARGET = "fwd_7d_up"


# ─── Fixtures ──────────────────────────────────────────────────────────────────

def _make_train_df(n: int = 100) -> pd.DataFrame:
    """Valid wide training DataFrame."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n),
        "Ticker": ["AAPL"] * n,
        TARGET: rng.randint(0, 2, size=n).astype(float),
        "Close": rng.uniform(100, 200, n),
    })
    for col in FEATURE_COLS:
        df[col] = rng.randn(n)
    return df


def _make_query_df(n: int = 10) -> pd.DataFrame:
    """Valid query DataFrame (no target column)."""
    rng = np.random.RandomState(99)
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n),
        "Ticker": ["MSFT"] * n,
    })
    for col in FEATURE_COLS:
        df[col] = rng.randn(n)
    return df


# ─── SchemaError Type ─────────────────────────────────────────────────────────

class TestSchemaErrorType:

    def test_schema_error_is_value_error(self):
        """Backward compatibility: SchemaError must be a ValueError subclass."""
        assert issubclass(SchemaError, ValueError)

    def test_schema_error_can_be_raised(self):
        with pytest.raises(ValueError):
            raise SchemaError("test error")

    def test_schema_error_can_be_caught_as_value_error(self):
        with pytest.raises(ValueError):
            raise SchemaError("should be catchable as ValueError")


# ─── validate_train_db ────────────────────────────────────────────────────────

class TestValidateTrainDB:

    def test_valid_dataframe_passes(self):
        df = _make_train_df()
        validate_train_db(df, FEATURE_SET, TARGET)  # No exception

    def test_extra_columns_allowed(self):
        """Wide DataFrames with extra OHLCV / forward-return columns must pass."""
        df = _make_train_df()
        df["fwd_30d_return"] = 0.05
        df["Volume"] = 1e6
        validate_train_db(df, FEATURE_SET, TARGET)  # No exception

    def test_missing_date_raises(self):
        df = _make_train_df().drop(columns=["Date"])
        with pytest.raises(SchemaError, match="Date"):
            validate_train_db(df, FEATURE_SET, TARGET)

    def test_missing_ticker_raises(self):
        df = _make_train_df().drop(columns=["Ticker"])
        with pytest.raises(SchemaError, match="Ticker"):
            validate_train_db(df, FEATURE_SET, TARGET)

    def test_missing_feature_column_raises(self):
        df = _make_train_df().drop(columns=["ret_1d"])
        with pytest.raises(SchemaError):
            validate_train_db(df, FEATURE_SET, TARGET)

    def test_missing_target_raises(self):
        df = _make_train_df().drop(columns=[TARGET])
        with pytest.raises(SchemaError):
            validate_train_db(df, FEATURE_SET, TARGET)

    def test_nan_in_feature_raises(self):
        df = _make_train_df()
        df.loc[10, "ret_7d"] = np.nan
        with pytest.raises(SchemaError):
            validate_train_db(df, FEATURE_SET, TARGET)

    def test_nan_in_target_raises(self):
        df = _make_train_df()
        df.loc[5, TARGET] = np.nan
        with pytest.raises(SchemaError):
            validate_train_db(df, FEATURE_SET, TARGET)

    def test_non_binary_target_raises(self):
        df = _make_train_df()
        df.loc[0, TARGET] = 2.0  # Not 0 or 1
        with pytest.raises(SchemaError):
            validate_train_db(df, FEATURE_SET, TARGET)

    def test_too_few_rows_raises(self):
        df = _make_train_df(n=30)  # < 50 minimum
        with pytest.raises(SchemaError, match="50"):
            validate_train_db(df, FEATURE_SET, TARGET)

    def test_exactly_50_rows_passes(self):
        df = _make_train_df(n=50)
        validate_train_db(df, FEATURE_SET, TARGET)  # Should not raise

    def test_schema_error_lists_all_failures(self):
        """Error message must describe all violations (multi-error format)."""
        df = _make_train_df()
        df = df.drop(columns=["ret_1d"])
        df.loc[0, TARGET] = np.nan
        with pytest.raises(SchemaError) as exc_info:
            validate_train_db(df, FEATURE_SET, TARGET)
        # Should mention multiple issues
        error_msg = str(exc_info.value)
        assert "validation failed" in error_msg.lower() or "issues" in error_msg.lower()

    def test_unknown_feature_set_raises(self):
        df = _make_train_df()
        with pytest.raises(SchemaError, match="Unknown feature set"):
            validate_train_db(df, "nonexistent_feature_set_xyz", TARGET)


# ─── validate_query_db ────────────────────────────────────────────────────────

class TestValidateQueryDB:

    def test_valid_dataframe_passes(self):
        df = _make_query_df()
        validate_query_db(df, FEATURE_SET)  # No exception

    def test_no_target_required(self):
        """Query validation must not require the target column."""
        df = _make_query_df()
        assert TARGET not in df.columns
        validate_query_db(df, FEATURE_SET)

    def test_extra_columns_allowed(self):
        df = _make_query_df()
        df["Close"] = 150.0
        df[TARGET] = 1.0  # Present but not required
        validate_query_db(df, FEATURE_SET)

    def test_missing_date_raises(self):
        df = _make_query_df().drop(columns=["Date"])
        with pytest.raises(SchemaError, match="Date"):
            validate_query_db(df, FEATURE_SET)

    def test_missing_feature_col_raises(self):
        df = _make_query_df().drop(columns=["ret_90d"])
        with pytest.raises(SchemaError):
            validate_query_db(df, FEATURE_SET)

    def test_nan_in_feature_raises(self):
        df = _make_query_df()
        df.loc[2, "ret_30d"] = np.nan
        with pytest.raises(SchemaError):
            validate_query_db(df, FEATURE_SET)

    def test_empty_dataframe_raises(self):
        df = _make_query_df(n=0)
        with pytest.raises(SchemaError, match="empty"):
            validate_query_db(df, FEATURE_SET)

    def test_single_row_passes(self):
        df = _make_query_df(n=1)
        validate_query_db(df, FEATURE_SET)  # 1 row is valid for query
