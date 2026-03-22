"""
test_engine_state.py — Unit tests for EngineState serializable checkpoint.

Tests:
  - Construction and validation
  - scaler_mean_array / scaler_scale_array properties (numpy reconstruction)
  - Serialization round-trip (model_dump_json → model_validate_json)
  - from_fitted factory (mock scaler + matcher)
  - verify_config_match
  - Error paths (dimension mismatch, zero scale, bad backend, bad hash)

Linear: SLE-57
"""

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from rebuild_phase_3z.fppe.pattern_engine.contracts.state import EngineState


# ─── Helpers ──────────────────────────────────────────────────────────────────

FEATURE_COLS = ["ret_1d", "ret_3d", "ret_7d", "ret_14d", "ret_30d", "ret_45d", "ret_60d", "ret_90d"]
N_FEATURES = len(FEATURE_COLS)


def _make_valid_state(**kwargs) -> EngineState:
    config_dict = {"top_k": 50, "max_distance": 1.1019}
    config_json = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    defaults = dict(
        feature_cols=FEATURE_COLS,
        scaler_mean=[0.0] * N_FEATURES,
        scaler_scale=[1.0] * N_FEATURES,
        n_samples=5000,
        matcher_backend="balltree",
        matcher_params={"backend": "balltree", "n_neighbors": 50, "metric": "euclidean"},
        config_hash=config_hash,
        fit_timestamp=datetime.now(timezone.utc).isoformat(),
        feature_set_name="returns_only",
    )
    defaults.update(kwargs)
    return EngineState(**defaults)


# ─── Construction ──────────────────────────────────────────────────────────────

class TestEngineStateConstruction:

    def test_valid_construction(self):
        state = _make_valid_state()
        assert state.n_features == N_FEATURES
        assert state.matcher_backend == "balltree"
        assert state.version == "1.0"

    def test_frozen(self):
        state = _make_valid_state()
        with pytest.raises(Exception):
            state.n_samples = 9999  # type: ignore

    def test_scaler_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="scaler_mean"):
            _make_valid_state(scaler_mean=[0.0] * (N_FEATURES - 1))  # Too short

    def test_scaler_scale_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="scaler_scale"):
            _make_valid_state(scaler_scale=[1.0] * (N_FEATURES + 1))  # Too long

    def test_zero_scale_raises(self):
        with pytest.raises(ValueError, match="zero"):
            _make_valid_state(scaler_scale=[0.0] + [1.0] * (N_FEATURES - 1))

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="zero or negative"):
            _make_valid_state(scaler_scale=[-1.0] + [1.0] * (N_FEATURES - 1))

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="matcher_backend"):
            _make_valid_state(matcher_backend="kdtree")  # Not a known backend

    def test_bad_config_hash_raises(self):
        with pytest.raises(ValueError, match="SHA-256"):
            _make_valid_state(config_hash="not_a_real_hash")

    def test_empty_feature_cols_raises(self):
        with pytest.raises(ValueError):
            _make_valid_state(feature_cols=[], scaler_mean=[], scaler_scale=[])

    def test_hnsw_backend_valid(self):
        state = _make_valid_state(
            matcher_backend="hnsw",
            matcher_params={"backend": "hnsw", "n_neighbors": 50, "ef_construction": 200, "M": 16},
        )
        assert state.matcher_backend == "hnsw"


# ─── Properties ───────────────────────────────────────────────────────────────

class TestEngineStateProperties:

    def test_scaler_mean_array_shape(self):
        state = _make_valid_state()
        arr = state.scaler_mean_array
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (N_FEATURES,)
        assert arr.dtype == np.float64

    def test_scaler_scale_array_shape(self):
        state = _make_valid_state()
        arr = state.scaler_scale_array
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (N_FEATURES,)

    def test_scaler_values_preserved(self):
        mean = [0.01 * i for i in range(N_FEATURES)]
        scale = [0.5 + 0.1 * i for i in range(N_FEATURES)]
        state = _make_valid_state(scaler_mean=mean, scaler_scale=scale)
        np.testing.assert_allclose(state.scaler_mean_array, mean)
        np.testing.assert_allclose(state.scaler_scale_array, scale)

    def test_n_features(self):
        state = _make_valid_state()
        assert state.n_features == N_FEATURES


# ─── Serialization Round-Trip ──────────────────────────────────────────────────

class TestEngineStateSerialization:

    def test_json_round_trip(self):
        original = _make_valid_state()
        json_bytes = original.model_dump_json()
        restored = EngineState.model_validate_json(json_bytes)

        assert restored.feature_cols == original.feature_cols
        assert restored.matcher_backend == original.matcher_backend
        assert restored.config_hash == original.config_hash
        np.testing.assert_allclose(restored.scaler_mean_array, original.scaler_mean_array)
        np.testing.assert_allclose(restored.scaler_scale_array, original.scaler_scale_array)

    def test_json_is_valid_json(self):
        state = _make_valid_state()
        json_str = state.model_dump_json()
        parsed = json.loads(json_str)
        assert "feature_cols" in parsed
        assert "config_hash" in parsed
        assert len(parsed["config_hash"]) == 64


# ─── from_fitted Factory ───────────────────────────────────────────────────────

class MockScaler:
    """Minimal sklearn StandardScaler mock."""
    def __init__(self, n_features: int):
        self.mean_ = np.zeros(n_features)
        self.scale_ = np.ones(n_features)


class MockMatcher:
    """Minimal BaseMatcher mock."""
    def __init__(self, backend: str = "balltree"):
        self._backend = backend

    @property
    def is_fitted(self):
        return True

    def get_params(self):
        return {"backend": self._backend, "n_neighbors": 50, "metric": "euclidean", "n_samples_fitted": 5000}


class MockConfig:
    """Config mock with model_dump()."""
    def model_dump(self):
        return {"top_k": 50, "max_distance": 1.1019, "feature_set": "returns_only"}


class TestEngineStateFactory:

    def test_from_fitted_basic(self):
        scaler = MockScaler(N_FEATURES)
        matcher = MockMatcher()
        config = MockConfig()
        state = EngineState.from_fitted(
            scaler=scaler,
            matcher=matcher,
            feature_cols=FEATURE_COLS,
            config=config,
            feature_set_name="returns_only",
        )
        assert state.n_features == N_FEATURES
        assert state.matcher_backend == "balltree"
        assert state.feature_set_name == "returns_only"
        assert len(state.config_hash) == 64

    def test_from_fitted_unfitted_scaler_raises(self):
        class UnfittedScaler:
            mean_ = None  # Not fitted

        matcher = MockMatcher()
        config = MockConfig()
        with pytest.raises(RuntimeError, match="scaler has not been fitted"):
            EngineState.from_fitted(
                scaler=UnfittedScaler(),
                matcher=matcher,
                feature_cols=FEATURE_COLS,
                config=config,
                feature_set_name="returns_only",
            )

    def test_from_fitted_unfitted_matcher_raises(self):
        class UnfittedMatcher:
            @property
            def is_fitted(self):
                return False

        scaler = MockScaler(N_FEATURES)
        config = MockConfig()
        with pytest.raises(RuntimeError, match="matcher has not been fitted"):
            EngineState.from_fitted(
                scaler=scaler,
                matcher=UnfittedMatcher(),
                feature_cols=FEATURE_COLS,
                config=config,
                feature_set_name="returns_only",
            )

    def test_verify_config_match_true(self):
        scaler = MockScaler(N_FEATURES)
        matcher = MockMatcher()
        config = MockConfig()
        state = EngineState.from_fitted(
            scaler=scaler, matcher=matcher,
            feature_cols=FEATURE_COLS, config=config,
            feature_set_name="returns_only",
        )
        assert state.verify_config_match(config) is True

    def test_verify_config_match_false_on_drift(self):
        scaler = MockScaler(N_FEATURES)
        matcher = MockMatcher()
        config1 = MockConfig()
        state = EngineState.from_fitted(
            scaler=scaler, matcher=matcher,
            feature_cols=FEATURE_COLS, config=config1,
            feature_set_name="returns_only",
        )

        class DriftedConfig:
            def model_dump(self):
                return {"top_k": 100, "max_distance": 2.0}  # Different values

        assert state.verify_config_match(DriftedConfig()) is False
