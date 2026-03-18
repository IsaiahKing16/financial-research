"""
schema.py — Lightweight DataFrame schema validation at engine boundaries.

Validates DataFrames at engine.fit() and engine.predict() entry points
to catch NaN values, missing columns, and type mismatches before they
cause cryptic failures downstream.

Uses native Python — no pandera dependency required.
"""

import numpy as np
import pandas as pd

from pattern_engine.features import (
    FeatureRegistry,
    FORWARD_RETURN_COLS,
    FORWARD_BINARY_COLS,
)


class SchemaError(ValueError):
    """Raised when a DataFrame fails schema validation."""
    pass


def validate_train_db(db: pd.DataFrame, feature_set: str,
                      projection_horizon: str) -> None:
    """Validate a training DataFrame before engine.fit().

    Checks:
      1. Required metadata columns exist (Date, Ticker)
      2. All feature columns exist and are numeric
      3. Projection horizon column exists and is binary (0/1)
      4. No NaN values in feature or target columns
      5. Minimum row count for meaningful training

    Args:
        db: training DataFrame
        feature_set: name of the feature set (e.g. "returns_only")
        projection_horizon: target column name (e.g. "fwd_7d_up")

    Raises:
        SchemaError: with a descriptive message listing all violations
    """
    errors = []
    features = FeatureRegistry.get(feature_set)
    fcols = features.columns

    # 1. Required metadata columns
    for col in ["Date", "Ticker"]:
        if col not in db.columns:
            errors.append(f"Missing required column: {col}")

    # 2. Feature columns exist and are numeric
    missing_features = [c for c in fcols if c not in db.columns]
    if missing_features:
        errors.append(
            f"Missing {len(missing_features)} feature columns: "
            f"{missing_features[:5]}{'...' if len(missing_features) > 5 else ''}"
        )
    else:
        non_numeric = [
            c for c in fcols
            if not pd.api.types.is_numeric_dtype(db[c])
        ]
        if non_numeric:
            errors.append(f"Non-numeric feature columns: {non_numeric}")

    # 3. Projection horizon column
    if projection_horizon not in db.columns:
        errors.append(f"Missing projection horizon column: {projection_horizon}")
    elif db[projection_horizon].dtype not in (np.int64, np.int32, np.float64):
        errors.append(
            f"Projection horizon '{projection_horizon}' has dtype "
            f"{db[projection_horizon].dtype}, expected numeric (0/1)"
        )

    # 4. NaN check in features and target
    if not missing_features:
        nan_counts = db[fcols].isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            worst = nan_cols.nlargest(3)
            errors.append(
                f"NaN values in {len(nan_cols)} feature columns. "
                f"Worst: {dict(worst)}"
            )

    if projection_horizon in db.columns:
        n_nan_target = db[projection_horizon].isna().sum()
        if n_nan_target > 0:
            errors.append(
                f"NaN values in target column '{projection_horizon}': "
                f"{n_nan_target} rows"
            )

    # 5. Minimum row count
    if len(db) < 50:
        errors.append(
            f"Training data has only {len(db)} rows (minimum 50 recommended)"
        )

    if errors:
        raise SchemaError(
            f"Training DataFrame validation failed ({len(errors)} issues):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def validate_query_db(db: pd.DataFrame, feature_set: str) -> None:
    """Validate a query/validation DataFrame before engine.predict().

    Lighter validation than train — no target column required
    (it may not exist for live prediction).

    Checks:
      1. Required metadata columns exist (Date, Ticker)
      2. All feature columns exist and are numeric
      3. No NaN values in feature columns
      4. At least 1 row

    Args:
        db: query/validation DataFrame
        feature_set: name of the feature set

    Raises:
        SchemaError: with a descriptive message listing all violations
    """
    errors = []
    features = FeatureRegistry.get(feature_set)
    fcols = features.columns

    # 1. Required metadata columns
    for col in ["Date", "Ticker"]:
        if col not in db.columns:
            errors.append(f"Missing required column: {col}")

    # 2. Feature columns exist and are numeric
    missing_features = [c for c in fcols if c not in db.columns]
    if missing_features:
        errors.append(
            f"Missing {len(missing_features)} feature columns: "
            f"{missing_features[:5]}{'...' if len(missing_features) > 5 else ''}"
        )
    else:
        non_numeric = [
            c for c in fcols
            if not pd.api.types.is_numeric_dtype(db[c])
        ]
        if non_numeric:
            errors.append(f"Non-numeric feature columns: {non_numeric}")

    # 3. NaN check
    if not missing_features:
        nan_counts = db[fcols].isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            worst = nan_cols.nlargest(3)
            errors.append(
                f"NaN values in {len(nan_cols)} feature columns. "
                f"Worst: {dict(worst)}"
            )

    # 4. Row count
    if len(db) == 0:
        errors.append("Query DataFrame is empty (0 rows)")

    if errors:
        raise SchemaError(
            f"Query DataFrame validation failed ({len(errors)} issues):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
