"""
datasets.py — Pandera DataFrame schemas for the pattern engine data pipeline.

Three schema families:
  1. TrainDBSchema / QueryDBSchema — input pipeline (strict=False, extra cols allowed)
  2. FeatureMatrixSchema — post-extraction feature array (strict=True, exact columns)
  3. OHLCVSchema — raw download validation (strict=False, extra cols OK)
  4. SignalsOutputSchema — the CSV emitted at the engine boundary (strict=True)

Design decisions:
  - strict=False for training/query DBs: these DataFrames carry many extra columns
    (OHLCV, all forward-return variants, technical indicators) alongside the ones we
    require. Rejecting extras would break the pipeline unnecessarily.
  - strict=True only for fully-specified outputs (FeatureMatrix, SignalsOutput).
  - coerce=False everywhere: silent type coercion masks upstream data bugs.
  - Factory functions accept feature_cols because the column set is dynamic
    (returns_only = 8 cols; future sets may be 40+).

Linear: SLE-57
"""

from __future__ import annotations

from typing import List

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check


# ─── Constants ────────────────────────────────────────────────────────────────

REQUIRED_INDEX_COLS = ["Date", "Ticker"]
TARGET_COL = "fwd_7d_up"  # Binary classification target (0/1)

# Minimum rows required to fit a meaningful model
MIN_TRAIN_ROWS = 50
MIN_QUERY_ROWS = 1


# ─── Shared Checks ────────────────────────────────────────────────────────────

def _no_nan_check(col_name: str) -> Check:
    """Column must contain no NaN values."""
    return Check(
        lambda s: s.notna().all(),
        error=f"Column '{col_name}' contains NaN values",
        element_wise=False,
    )


def _binary_check() -> Check:
    """Column must contain only 0 and 1 (binary classification target)."""
    return Check(
        lambda s: s.isin([0, 1]).all(),
        error=f"Target column '{TARGET_COL}' must be binary (0 or 1 only)",
        element_wise=False,
    )


# ─── TrainDBSchema ─────────────────────────────────────────────────────────────

def make_train_db_schema(feature_cols: List[str]) -> DataFrameSchema:
    """
    Return a Pandera schema for the training DataFrame.

    The training DB is wide: it contains OHLCV columns, all forward-return
    variants, technical indicators, and the feature columns we actually use.
    We validate only the columns we care about (strict=False).

    Args:
        feature_cols: List of feature column names to require and validate.
                      Example: ["ret_1d", "ret_3d", ..., "ret_90d"] for returns_only.

    Returns:
        A configured DataFrameSchema instance.

    Raises:
        pandera.errors.SchemaError: if the DataFrame fails validation.
    """
    columns: dict = {}

    # Required index columns
    # dtype=None: Date can be datetime64[us/ns], object, or str depending on
    # pandas version and construction path. Content check (no NaN) is what matters.
    columns["Date"] = Column(
        None,
        nullable=False,
        coerce=False,
        checks=_no_nan_check("Date"),
    )
    columns["Ticker"] = Column(
        None,   # str dtype varies (object vs string[pyarrow]) in pandas 3
        nullable=False,
        coerce=False,
        checks=[
            _no_nan_check("Ticker"),
            Check(lambda s: (s.astype(str).str.len() > 0).all(), error="Ticker cannot be empty string"),
        ],
    )

    # Feature columns — must be numeric with no NaN
    for col in feature_cols:
        columns[col] = Column(
            float,
            nullable=False,
            coerce=False,
            checks=_no_nan_check(col),
        )

    # Target column — binary 0/1
    columns[TARGET_COL] = Column(
        float,  # stored as float in practice (0.0 / 1.0)
        nullable=False,
        coerce=False,
        checks=[
            _no_nan_check(TARGET_COL),
            _binary_check(),
        ],
    )

    return DataFrameSchema(
        columns=columns,
        strict=False,       # Extra columns allowed (OHLCV, other forward returns, etc.)
        coerce=False,       # Never silently cast types
        name="TrainDB",
    )


# ─── QueryDBSchema ─────────────────────────────────────────────────────────────

def make_query_db_schema(feature_cols: List[str]) -> DataFrameSchema:
    """
    Return a Pandera schema for the query DataFrame (inference-time input).

    The query DB has the same structure as the training DB but does not
    require the target column (it may not exist at inference time).

    Args:
        feature_cols: List of feature column names to require and validate.

    Returns:
        A configured DataFrameSchema instance.
    """
    columns: dict = {}

    columns["Date"] = Column(
        None,   # datetime64[us/ns], object, or str — skip type enforcement
        nullable=False,
        coerce=False,
        checks=_no_nan_check("Date"),
    )
    columns["Ticker"] = Column(
        None,   # string dtype varies across pandas versions
        nullable=False,
        coerce=False,
        checks=[
            _no_nan_check("Ticker"),
            Check(lambda s: (s.astype(str).str.len() > 0).all(), error="Ticker cannot be empty string"),
        ],
    )

    for col in feature_cols:
        columns[col] = Column(
            float,
            nullable=False,
            coerce=False,
            checks=_no_nan_check(col),
        )

    return DataFrameSchema(
        columns=columns,
        strict=False,       # Query DB may carry extra columns; only validate what we need
        coerce=False,
        name="QueryDB",
    )


# ─── FeatureMatrixSchema ───────────────────────────────────────────────────────

def make_feature_matrix_schema(feature_cols: List[str]) -> DataFrameSchema:
    """
    Return a Pandera schema for the extracted feature matrix (post-StandardScaler input).

    This is the narrow matrix passed directly to the Matcher. It must contain
    exactly the specified feature columns and nothing else (strict=True).

    Args:
        feature_cols: Ordered list of feature column names. Order matters for the
                      Matcher — columns must align with the fitted scaler.

    Returns:
        A configured DataFrameSchema instance (strict=True).
    """
    columns: dict = {
        col: Column(
            float,
            nullable=False,
            coerce=False,
            checks=_no_nan_check(col),
        )
        for col in feature_cols
    }

    return DataFrameSchema(
        columns=columns,
        strict=True,        # Exact shape required — no extra columns allowed
        coerce=False,
        name="FeatureMatrix",
    )


# ─── OHLCVSchema ──────────────────────────────────────────────────────────────

# OHLCV column specifications (common to all tickers)
_OHLCV_COLUMNS = {
    "Date": Column(None, nullable=False, coerce=False, checks=_no_nan_check("Date")),
    "Open": Column(float, nullable=True, coerce=False),    # May have gaps on thin data
    "High": Column(float, nullable=True, coerce=False),
    "Low": Column(float, nullable=True, coerce=False),
    "Close": Column(float, nullable=False, coerce=False, checks=[
        _no_nan_check("Close"),
        Check(lambda s: (s > 0).all(), error="Close prices must be positive"),
    ]),
    "Volume": Column(float, nullable=True, coerce=False),
}

OHLCVSchema: DataFrameSchema = DataFrameSchema(
    columns=_OHLCV_COLUMNS,
    strict=False,           # yfinance may add Adj Close, Dividends, etc.
    coerce=False,
    name="OHLCV",
)
"""Schema for raw OHLCV data downloaded from yfinance.

strict=False because yfinance returns extra columns (Adj Close, Dividends, Stock Splits).
Close must be present and positive; Open/High/Low/Volume may have gaps.
"""


# ─── SignalsOutputSchema ───────────────────────────────────────────────────────

SignalsOutputSchema: DataFrameSchema = DataFrameSchema(
    columns={
        # dtype=None for date/string cols: str dtype varies (object vs string[pyarrow])
        "date": Column(None, nullable=False, coerce=False, checks=_no_nan_check("date")),
        "ticker": Column(None, nullable=False, coerce=False, checks=[
            _no_nan_check("ticker"),
            Check(lambda s: (s.astype(str) == s.astype(str).str.upper()).all(),
                  error="Tickers must be uppercase"),
        ]),
        "signal": Column(None, nullable=False, coerce=False, checks=[
            _no_nan_check("signal"),
            Check(lambda s: s.astype(str).isin(["BUY", "SELL", "HOLD"]).all(),
                  error="signal must be BUY, SELL, or HOLD"),
        ]),
        "confidence": Column(float, nullable=False, coerce=False, checks=[
            _no_nan_check("confidence"),
            Check(lambda s: ((s >= 0.0) & (s <= 1.0)).all(),
                  error="confidence must be in [0.0, 1.0]"),
        ]),
        "signal_source": Column(None, nullable=False, coerce=False, checks=_no_nan_check("signal_source")),
        "sector": Column(None, nullable=False, coerce=False, checks=_no_nan_check("sector")),
        "n_matches": Column(int, nullable=False, coerce=False, checks=[
            _no_nan_check("n_matches"),
            Check(lambda s: (s >= 0).all(), error="n_matches must be non-negative"),
        ]),
        "raw_prob": Column(float, nullable=False, coerce=False, checks=_no_nan_check("raw_prob")),
        "mean_7d_return": Column(float, nullable=False, coerce=False, checks=_no_nan_check("mean_7d_return")),
    },
    strict=True,    # Output schema is fully specified — no extra columns allowed
    coerce=False,
    name="SignalsOutput",
)
"""Schema for the signals CSV emitted at the pattern_engine → trading_system boundary.

Matches exactly the columns in cached_signals_2024.csv. strict=True: if extra
columns appear in output, something upstream has changed unexpectedly.
"""
