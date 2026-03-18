"""Tests for pattern_engine.config."""

import dataclasses
import pytest
from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS


class TestEngineConfig:
    """Test EngineConfig frozen dataclass."""

    def test_defaults(self):
        cfg = EngineConfig()
        assert cfg.top_k == 50
        assert cfg.max_distance == 1.1019
        assert cfg.distance_weighting == "uniform"
        assert cfg.distance_metric == "euclidean"
        assert cfg.nn_jobs == 1
        assert cfg.batch_size == 256
        assert cfg.feature_set == "returns_only"
        assert cfg.projection_horizon == "fwd_7d_up"
        assert cfg.confidence_threshold == 0.65
        assert cfg.agreement_spread == 0.10
        assert cfg.min_matches == 10
        assert cfg.same_sector_only is False
        assert cfg.exclude_same_ticker is True
        assert cfg.regime_filter is True
        assert cfg.regime_mode == "binary"
        assert cfg.regime_fallback is True
        assert cfg.adx_threshold == 25.0
        assert cfg.calibration_method == "platt"
        assert cfg.cal_frac == 0.76

    def test_frozen(self):
        cfg = EngineConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.top_k = 100

    def test_nn_algorithm_euclidean(self):
        cfg = EngineConfig(distance_metric="euclidean")
        assert cfg.nn_algorithm == "ball_tree"

    def test_nn_algorithm_cosine(self):
        cfg = EngineConfig(distance_metric="cosine")
        assert cfg.nn_algorithm == "brute"

    def test_replace(self):
        cfg = EngineConfig()
        cfg2 = cfg.replace(top_k=100, max_distance=2.0)
        assert cfg2.top_k == 100
        assert cfg2.max_distance == 2.0
        assert cfg.top_k == 50  # original unchanged

    def test_feature_weights_default(self):
        cfg = EngineConfig()
        assert "ret_1d" in cfg.feature_weights
        assert "ret_7d" in cfg.feature_weights
        assert cfg.feature_weights["ret_7d"] == 1.5
        assert cfg.feature_weights["ret_90d"] == 0.5

    def test_feature_weights_independent(self):
        """Each config instance gets its own feature_weights dict."""
        cfg1 = EngineConfig()
        cfg2 = EngineConfig()
        assert cfg1.feature_weights is not cfg2.feature_weights


class TestWalkforwardFolds:
    """Test walk-forward fold definitions."""

    def test_six_folds(self):
        assert len(WALKFORWARD_FOLDS) == 6

    def test_fold_structure(self):
        for fold in WALKFORWARD_FOLDS:
            assert "train_end" in fold
            assert "val_start" in fold
            assert "val_end" in fold
            assert "label" in fold

    def test_folds_ordered(self):
        for i in range(len(WALKFORWARD_FOLDS) - 1):
            assert WALKFORWARD_FOLDS[i]["val_end"] < WALKFORWARD_FOLDS[i + 1]["val_start"]

    def test_2024_fold_label(self):
        last = WALKFORWARD_FOLDS[-1]
        assert "2024" in last["label"]
        assert last["train_end"] == "2023-12-31"
        assert last["val_start"] == "2024-01-01"
