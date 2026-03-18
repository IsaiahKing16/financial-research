"""Tests for pattern_engine.schema — DataFrame boundary validation."""

import numpy as np
import pandas as pd
import pytest
from pattern_engine.schema import (
    validate_train_db, validate_query_db, SchemaError,
)


class TestValidateTrainDb:
    """Schema validation at engine.fit() boundary."""

    @pytest.fixture
    def valid_train(self, train_db):
        """The conftest train_db should pass validation."""
        return train_db

    def test_valid_train_passes(self, valid_train):
        # Should not raise
        validate_train_db(valid_train, "returns_only", "fwd_7d_up")

    def test_missing_date_column(self, valid_train):
        df = valid_train.drop(columns=["Date"])
        with pytest.raises(SchemaError, match="Missing required column: Date"):
            validate_train_db(df, "returns_only", "fwd_7d_up")

    def test_missing_ticker_column(self, valid_train):
        df = valid_train.drop(columns=["Ticker"])
        with pytest.raises(SchemaError, match="Missing required column: Ticker"):
            validate_train_db(df, "returns_only", "fwd_7d_up")

    def test_missing_feature_columns(self, valid_train):
        df = valid_train.drop(columns=["ret_1d", "ret_3d"])
        with pytest.raises(SchemaError, match="Missing.*feature columns"):
            validate_train_db(df, "returns_only", "fwd_7d_up")

    def test_missing_projection_horizon(self, valid_train):
        df = valid_train.drop(columns=["fwd_7d_up"])
        with pytest.raises(SchemaError, match="Missing projection horizon"):
            validate_train_db(df, "returns_only", "fwd_7d_up")

    def test_nan_in_features(self, valid_train):
        df = valid_train.copy()
        df.loc[df.index[:10], "ret_1d"] = np.nan
        with pytest.raises(SchemaError, match="NaN values"):
            validate_train_db(df, "returns_only", "fwd_7d_up")

    def test_nan_in_target(self, valid_train):
        df = valid_train.copy()
        df.loc[df.index[:5], "fwd_7d_up"] = np.nan
        with pytest.raises(SchemaError, match="NaN values in target"):
            validate_train_db(df, "returns_only", "fwd_7d_up")

    def test_too_few_rows(self, valid_train):
        df = valid_train.head(10)
        with pytest.raises(SchemaError, match="only 10 rows"):
            validate_train_db(df, "returns_only", "fwd_7d_up")

    def test_non_numeric_feature(self, valid_train):
        df = valid_train.copy()
        df["ret_1d"] = "text"
        with pytest.raises(SchemaError, match="Non-numeric"):
            validate_train_db(df, "returns_only", "fwd_7d_up")

    def test_multiple_errors_reported(self, valid_train):
        """All violations reported in one SchemaError, not just the first."""
        df = valid_train.drop(columns=["Date", "Ticker"])
        with pytest.raises(SchemaError) as exc_info:
            validate_train_db(df, "returns_only", "fwd_7d_up")
        msg = str(exc_info.value)
        assert "Date" in msg
        assert "Ticker" in msg


class TestValidateQueryDb:
    """Schema validation at engine.predict() boundary."""

    @pytest.fixture
    def valid_query(self, val_db):
        return val_db

    def test_valid_query_passes(self, valid_query):
        validate_query_db(valid_query, "returns_only")

    def test_missing_feature_columns(self, valid_query):
        df = valid_query.drop(columns=["ret_1d"])
        with pytest.raises(SchemaError, match="Missing.*feature columns"):
            validate_query_db(df, "returns_only")

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["Date", "Ticker", "ret_1d"])
        with pytest.raises(SchemaError, match="empty"):
            validate_query_db(df, "returns_only")

    def test_nan_in_features(self, valid_query):
        df = valid_query.copy()
        df.loc[df.index[:5], "ret_7d"] = np.nan
        with pytest.raises(SchemaError, match="NaN values"):
            validate_query_db(df, "returns_only")

    def test_no_target_column_required(self, valid_query):
        """Query validation should NOT require fwd_7d_up."""
        df = valid_query.drop(columns=["fwd_7d_up"], errors="ignore")
        # Should not raise (query doesn't need target column)
        validate_query_db(df, "returns_only")


class TestSchemaError:
    def test_is_value_error(self):
        assert issubclass(SchemaError, ValueError)

    def test_descriptive_message(self):
        err = SchemaError("test message")
        assert str(err) == "test message"
