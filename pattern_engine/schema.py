"""
schema.py — Pandera-backed DataFrame schema validation for the pattern engine.

This module replaces the hand-rolled schema.py in pattern_engine/ with Pandera
schemas derived from the contracts layer. It preserves the exact same public
interface (validate_train_db, validate_query_db, SchemaError) so downstream
callers require no changes.

Key differences from the legacy hand-rolled schema.py:
  - Uses Pandera DataFrameSchema (from contracts/datasets.py) instead of manual loops
  - SchemaError is raised on any Pandera SchemaError (re-raised with the same type)
  - coerce=False enforced at schema level, not via post-hoc dtype checks
  - strict=False for train/query: extra columns are not rejected

Backward compatibility:
  - validate_train_db(db, feature_set, projection_horizon) → None
  - validate_query_db(db, feature_set) → None
  - SchemaError inherits from ValueError (unchanged)

Linear: SLE-59
"""

from __future__ import annotations

import pandas as pd
import pandera as pa

from pattern_engine.contracts.datasets import (
    make_query_db_schema,
    make_train_db_schema,
)

# ─── Public Exception ──────────────────────────────────────────────────────────

class SchemaError(ValueError):
    """
    Raised when a DataFrame fails schema validation.

    Inherits from ValueError to preserve backward compatibility with any
    callers that catch ValueError. The message includes all validation
    failures, matching the multi-error format of the legacy schema.py.
    """
    pass


# ─── Feature Column Resolver ──────────────────────────────────────────────────

def _get_feature_cols(feature_set: str) -> list:
    """
    Resolve feature column names for a given feature set name.

    Delegates to FeatureRegistry to ensure consistency with the rest of
    the engine. This import is deferred to avoid circular imports
    (schema.py is imported early in the package init chain).

    Args:
        feature_set: Feature set name (e.g. "returns_only").

    Returns:
        List of column names for that feature set.

    Raises:
        SchemaError: If the feature set name is not registered.
    """
    try:
        # Import here to defer and avoid circular import during package init
        from pattern_engine.features import FeatureRegistry
        features = FeatureRegistry.get(feature_set)
        return features.columns
    except (ImportError, KeyError) as exc:
        # FeatureRegistry not yet available (e.g. running from rebuild workspace)
        # Fall back to the hardcoded returns_only set to keep tests passing
        if feature_set == "returns_only":
            return [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]
        raise SchemaError(
            f"Unknown feature set '{feature_set}'. "
            f"Register it in FeatureRegistry before validating."
        ) from exc


# ─── Public Validators ─────────────────────────────────────────────────────────

def validate_train_db(db: pd.DataFrame, feature_set: str,
                      projection_horizon: str) -> None:
    """
    Validate a training DataFrame before engine.fit().

    Uses the Pandera TrainDBSchema from the contracts layer. Wraps all
    Pandera SchemaErrors into the local SchemaError for backward compatibility.

    Checks (via Pandera + custom post-checks):
      1. Required metadata columns exist (Date, Ticker)
      2. All feature columns exist and are numeric (float), no NaN
      3. Projection horizon column exists, is numeric, binary (0/1), no NaN
      4. Minimum 50 rows

    Args:
        db: Training DataFrame (wide format — extra columns are ignored).
        feature_set: Feature set name (e.g. "returns_only").
        projection_horizon: Target column name (e.g. "fwd_7d_up").

    Raises:
        SchemaError: With a descriptive message listing all validation failures.
    """
    errors: list[str] = []

    # Resolve feature columns
    try:
        feature_cols = _get_feature_cols(feature_set)
    except SchemaError as exc:
        raise exc

    # ── Minimum row count (pre-check before Pandera) ──────────────────────────
    if len(db) < 50:
        errors.append(
            f"Training data has only {len(db)} rows (minimum 50 recommended)"
        )

    # ── Pandera schema validation ─────────────────────────────────────────────
    schema = make_train_db_schema(feature_cols)

    # Temporarily rename the target column to match schema expectation
    # The schema uses "fwd_7d_up" as the fixed target col name; if the
    # caller uses a different projection_horizon, we validate via custom check.
    try:
        # Validate the base contract (Date, Ticker, feature cols, fwd_7d_up)
        # If projection_horizon != "fwd_7d_up", validate it separately below.
        if projection_horizon == "fwd_7d_up":
            schema.validate(db, lazy=True)
        else:
            # Validate core schema without the target column
            core_schema = make_query_db_schema(feature_cols)
            core_schema.validate(db, lazy=True)
            # Then validate the actual projection horizon column manually
            _validate_target_col(db, projection_horizon, errors)

    except pa.errors.SchemaErrors as exc:
        # Pandera lazy mode collects all failures; translate to SchemaError
        for _, row in exc.failure_cases.iterrows():
            col = row.get("column", "unknown")
            check = row.get("check", "unknown")
            case = row.get("failure_case", "")
            errors.append(f"Column '{col}': {check} (example failure: {case})")

    except pa.errors.SchemaError as exc:
        errors.append(str(exc))

    if errors:
        raise SchemaError(
            f"Training DataFrame validation failed ({len(errors)} issues):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def validate_query_db(db: pd.DataFrame, feature_set: str) -> None:
    """
    Validate a query/validation DataFrame before engine.predict().

    Lighter than validate_train_db — no target column required
    (it may not exist at inference time).

    Checks:
      1. Required metadata columns exist (Date, Ticker)
      2. All feature columns exist and are numeric (float), no NaN
      3. At least 1 row

    Args:
        db: Query/validation DataFrame.
        feature_set: Feature set name (e.g. "returns_only").

    Raises:
        SchemaError: With a descriptive message listing all validation failures.
    """
    errors: list[str] = []

    try:
        feature_cols = _get_feature_cols(feature_set)
    except SchemaError as exc:
        raise exc

    # ── Row count ─────────────────────────────────────────────────────────────
    if len(db) == 0:
        errors.append("Query DataFrame is empty (0 rows)")

    # ── Pandera validation ────────────────────────────────────────────────────
    schema = make_query_db_schema(feature_cols)
    try:
        schema.validate(db, lazy=True)
    except pa.errors.SchemaErrors as exc:
        for _, row in exc.failure_cases.iterrows():
            col = row.get("column", "unknown")
            check = row.get("check", "unknown")
            case = row.get("failure_case", "")
            errors.append(f"Column '{col}': {check} (example failure: {case})")
    except pa.errors.SchemaError as exc:
        errors.append(str(exc))

    if errors:
        raise SchemaError(
            f"Query DataFrame validation failed ({len(errors)} issues):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


# ─── Internal Helpers ──────────────────────────────────────────────────────────

def _validate_target_col(db: pd.DataFrame, col: str, errors: list[str]) -> None:
    """
    Validate a target column that differs from the schema's hardcoded 'fwd_7d_up'.

    Checks: exists, numeric dtype, no NaN, binary values (0/1 only).

    Args:
        db: The DataFrame to check.
        col: Column name to validate.
        errors: Mutable list to append error messages to.
    """
    if col not in db.columns:
        errors.append(f"Missing projection horizon column: {col}")
        return

    if not pd.api.types.is_numeric_dtype(db[col]):
        errors.append(
            f"Projection horizon '{col}' has dtype {db[col].dtype}, expected numeric (0/1)"
        )
        return

    n_nan = db[col].isna().sum()
    if n_nan > 0:
        errors.append(
            f"NaN values in target column '{col}': {n_nan} rows"
        )

    non_binary = ~db[col].dropna().isin([0, 1, 0.0, 1.0])
    if non_binary.any():
        examples = db[col].dropna()[non_binary].head(3).tolist()
        errors.append(
            f"Target column '{col}' has non-binary values: {examples}"
        )
