"""
state.py — Serializable checkpoint for a fitted pattern engine.

EngineState captures everything needed to reconstruct a fitted Matcher
from disk without re-fitting. This enables:
  - Reproducible backtests (load, run — no re-fit required)
  - M8 migration checkpointing (serialize Phase 3Z state; verify parity)
  - Overnight runner state persistence (restart without cold start)

Design decisions:
  - Scaler parameters stored as List[float] (JSON-serializable, exact float64 precision)
  - numpy arrays reconstructed on demand via @property (no dependency on numpy in
    the serialized form)
  - config_hash is SHA-256 of the EngineConfig JSON (detects config drift)
  - All fields are frozen (immutable once fitted)
  - model_config frozen=True: prevents accidental mutation after fit

Linear: SLE-57
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ─── EngineState ──────────────────────────────────────────────────────────────

class EngineState(BaseModel):
    """
    Serializable checkpoint of a fitted FPPE pattern engine.

    Captures the StandardScaler parameters and Matcher configuration
    so the engine can be reconstructed without re-fitting.

    Args:
        feature_cols: Ordered list of feature column names used during fit.
                      Order matters — must match the Matcher's training order.
        scaler_mean: StandardScaler.mean_ as a list of floats (one per feature).
        scaler_scale: StandardScaler.scale_ as a list of floats (one per feature).
        n_samples: Number of training samples the Matcher was fitted on.
        matcher_backend: Backend used ("balltree" or "hnsw").
        matcher_params: Full params dict from matcher.get_params().
        config_hash: SHA-256 of the serialized EngineConfig at fit time.
                     Detects config drift between fit and inference.
        fit_timestamp: ISO 8601 timestamp of when fit() was called.
        feature_set_name: Human-readable name of the feature set (e.g., "returns_only").
        version: Schema version for forward compatibility.

    Usage:
        state = EngineState.from_fitted(
            scaler=fitted_scaler,
            matcher=fitted_matcher,
            feature_cols=feature_cols,
            config=engine_config,
            feature_set_name="returns_only",
        )
        json_bytes = state.model_dump_json()
        restored = EngineState.model_validate_json(json_bytes)
        # Reconstruct scaler:
        mean = restored.scaler_mean_array    # np.ndarray
        scale = restored.scaler_scale_array  # np.ndarray
    """
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    # Feature column specification
    feature_cols: List[str] = Field(
        min_length=1,
        description="Ordered feature column names (must match Matcher training order)",
    )

    # StandardScaler parameters (stored as lists for JSON serializability)
    scaler_mean: List[float] = Field(
        description="StandardScaler.mean_ — one value per feature",
    )
    scaler_scale: List[float] = Field(
        description="StandardScaler.scale_ — one value per feature",
    )

    # Matcher metadata
    n_samples: int = Field(
        ge=1,
        description="Number of training samples the Matcher was fitted on",
    )
    matcher_backend: str = Field(
        description="Matcher backend name: 'balltree' or 'hnsw'",
    )
    matcher_params: Dict[str, Any] = Field(
        description="Full params dict from matcher.get_params()",
    )

    # Provenance
    config_hash: str = Field(
        description="SHA-256 hex digest of the serialized EngineConfig at fit time",
    )
    fit_timestamp: str = Field(
        description="ISO 8601 UTC timestamp of when fit() was called",
    )
    feature_set_name: str = Field(
        description="Human-readable name of the feature set (e.g., 'returns_only')",
    )
    version: str = Field(
        default="1.0",
        description="Schema version for forward compatibility",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def validate_scaler_dimensions(self) -> "EngineState":
        """scaler_mean and scaler_scale must have one entry per feature col."""
        n_features = len(self.feature_cols)
        if len(self.scaler_mean) != n_features:
            raise ValueError(
                f"scaler_mean has {len(self.scaler_mean)} entries "
                f"but feature_cols has {n_features}"
            )
        if len(self.scaler_scale) != n_features:
            raise ValueError(
                f"scaler_scale has {len(self.scaler_scale)} entries "
                f"but feature_cols has {n_features}"
            )
        return self

    @field_validator("scaler_mean", "scaler_scale")
    @classmethod
    def _require_finite_scaler(cls, v: List[float]) -> List[float]:
        """Scaler arrays must contain only finite values.

        A NaN or Inf in scaler_mean or scaler_scale would corrupt every
        distance computation silently — the KNN search would return garbage
        neighbors without any error.
        """
        bad = [i for i, x in enumerate(v) if not math.isfinite(x)]
        if bad:
            raise ValueError(
                f"Scaler values at indices {bad} are NaN or Inf. "
                "A corrupt scaler would produce invalid distances."
            )
        return v

    @field_validator("scaler_scale")
    @classmethod
    def scale_non_zero(cls, v: List[float]) -> List[float]:
        """StandardScaler.scale_ must be > 0 (zero scale → division by zero)."""
        if any(s <= 0.0 for s in v):
            raise ValueError(
                "scaler_scale contains zero or negative values; "
                "a feature column may have zero variance in the training set"
            )
        return v

    @field_validator("matcher_backend")
    @classmethod
    def valid_backend(cls, v: str) -> str:
        """Backend must be a known Matcher type."""
        known = {"balltree", "hnsw"}
        if v not in known:
            raise ValueError(f"matcher_backend '{v}' not in {known}")
        return v

    @field_validator("config_hash")
    @classmethod
    def valid_sha256(cls, v: str) -> str:
        """config_hash must look like a SHA-256 hex digest (64 hex chars)."""
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(
                f"config_hash '{v[:16]}...' does not look like a SHA-256 hex digest"
            )
        return v

    # ── Derived Properties ─────────────────────────────────────────────────────

    @property
    def scaler_mean_array(self) -> np.ndarray:
        """StandardScaler.mean_ as a numpy float64 array (shape: [n_features])."""
        return np.array(self.scaler_mean, dtype=np.float64)

    @property
    def scaler_scale_array(self) -> np.ndarray:
        """StandardScaler.scale_ as a numpy float64 array (shape: [n_features])."""
        return np.array(self.scaler_scale, dtype=np.float64)

    @property
    def n_features(self) -> int:
        """Number of features in the fitted engine."""
        return len(self.feature_cols)

    # ── Factory Constructor ────────────────────────────────────────────────────

    @classmethod
    def from_fitted(
        cls,
        scaler: Any,           # sklearn StandardScaler
        matcher: Any,          # BaseMatcher subclass (already fitted)
        feature_cols: List[str],
        config: Any,           # EngineConfig or any JSON-serializable object
        feature_set_name: str,
    ) -> "EngineState":
        """
        Create an EngineState from a fitted scaler and matcher.

        Args:
            scaler: A fitted sklearn StandardScaler instance.
            matcher: A fitted BaseMatcher instance (BallTreeMatcher or HNSWMatcher).
            feature_cols: Ordered list of feature column names.
            config: The EngineConfig used during fit (must be JSON-serializable
                    via model_dump() for Pydantic models, or vars() for dataclasses).
            feature_set_name: Human-readable name of the feature set.

        Returns:
            A frozen EngineState instance.

        Raises:
            RuntimeError: If the scaler has not been fitted.
            RuntimeError: If the matcher has not been fitted.
        """
        # Validate preconditions with RuntimeError (not assert — stripped under -O)
        if not hasattr(scaler, "mean_") or scaler.mean_ is None:
            raise RuntimeError("scaler has not been fitted; call scaler.fit() first")
        if not matcher.is_fitted:
            raise RuntimeError("matcher has not been fitted; call matcher.fit() first")

        # Serialize config to get a stable hash
        try:
            config_dict = config.model_dump() if hasattr(config, "model_dump") else vars(config)
            config_json = json.dumps(config_dict, sort_keys=True, default=str)
        except (TypeError, AttributeError) as exc:
            raise RuntimeError(
                f"Cannot serialize config for hashing: {exc}. "
                "Config must be a Pydantic model or plain dataclass."
            ) from exc

        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        fit_timestamp = datetime.now(timezone.utc).isoformat()

        return cls(
            feature_cols=list(feature_cols),
            scaler_mean=scaler.mean_.tolist(),
            scaler_scale=scaler.scale_.tolist(),
            n_samples=matcher.get_params().get("n_samples_fitted", 0),
            matcher_backend=matcher.get_params()["backend"],
            matcher_params=matcher.get_params(),
            config_hash=config_hash,
            fit_timestamp=fit_timestamp,
            feature_set_name=feature_set_name,
        )

    def verify_config_match(self, config: Any) -> bool:
        """
        Return True if the given config matches the one used at fit time.

        Args:
            config: The EngineConfig to verify against.

        Returns:
            True if SHA-256 hashes match; False if config has drifted.
        """
        try:
            config_dict = config.model_dump() if hasattr(config, "model_dump") else vars(config)
            config_json = json.dumps(config_dict, sort_keys=True, default=str)
        except (TypeError, AttributeError):
            return False

        current_hash = hashlib.sha256(config_json.encode()).hexdigest()
        return current_hash == self.config_hash
