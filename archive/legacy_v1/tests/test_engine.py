"""Tests for pattern_engine.engine."""

import os
import tempfile
import numpy as np
import pytest
from pattern_engine.config import EngineConfig
from pattern_engine.engine import PatternEngine, PredictionResult


class TestPredictionResult:
    def test_signal_counts(self):
        result = PredictionResult(
            probabilities=np.array([0.8, 0.2, 0.5]),
            calibrated_probabilities=np.array([0.8, 0.2, 0.5]),
            signals=np.array(["BUY", "SELL", "HOLD"]),
            reasons=["r1", "r2", "r3"],
            n_matches=[20, 25, 15],
            mean_returns=[0.01, -0.01, 0.0],
            ensemble_list=[np.array([]), np.array([]), np.array([])],
        )
        assert result.buy_count == 1
        assert result.sell_count == 1
        assert result.hold_count == 1
        assert result.avg_matches == 20.0


class TestPatternEngine:
    @pytest.fixture
    def config(self):
        return EngineConfig(
            regime_filter=False,
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
            batch_size=64,
        )

    @pytest.fixture
    def config_with_calibration(self):
        return EngineConfig(
            regime_filter=False,
            calibration_method="platt",
            top_k=10,
            max_distance=50.0,
            batch_size=64,
        )

    def test_fit_returns_self(self, train_db, config):
        engine = PatternEngine(config)
        result = engine.fit(train_db)
        assert result is engine
        assert engine._fitted

    def test_predict_before_fit_raises(self, val_db, config):
        engine = PatternEngine(config)
        with pytest.raises(RuntimeError, match="fit"):
            engine.predict(val_db)

    def test_predict(self, train_db, val_db, config):
        engine = PatternEngine(config)
        engine.fit(train_db)
        result = engine.predict(val_db.head(10), verbose=0)
        assert isinstance(result, PredictionResult)
        assert len(result.probabilities) == 10
        assert len(result.signals) == 10
        assert all(s in ("BUY", "SELL", "HOLD") for s in result.signals)

    def test_evaluate(self, train_db, val_db, config):
        engine = PatternEngine(config)
        engine.fit(train_db)
        metrics = engine.evaluate(val_db.head(20), verbose=0)
        assert "brier_score" in metrics
        assert "brier_skill_score" in metrics
        assert "total_samples" in metrics
        assert metrics["total_samples"] == 20

    def test_with_platt_calibration(self, train_db, val_db, config_with_calibration):
        engine = PatternEngine(config_with_calibration)
        engine.fit(train_db)
        result = engine.predict(val_db.head(10), verbose=0)
        # Calibrated probabilities should differ from raw
        assert len(result.calibrated_probabilities) == 10
        assert all(0 <= p <= 1 for p in result.calibrated_probabilities)

    def test_with_regime_filter(self, train_db, val_db):
        config = EngineConfig(
            regime_filter=True,
            regime_mode="binary",
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
        )
        engine = PatternEngine(config)
        engine.fit(train_db)
        result = engine.predict(val_db.head(10), verbose=0)
        assert len(result.probabilities) == 10

    def test_save_load_roundtrip(self, train_db, val_db, config):
        engine = PatternEngine(config)
        engine.fit(train_db)
        result1 = engine.predict(val_db.head(5), verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "engine.pkl")
            engine.save(path)

            loaded = PatternEngine.load(path)
            result2 = loaded.predict(val_db.head(5), verbose=0)

        np.testing.assert_array_almost_equal(
            result1.probabilities, result2.probabilities, decimal=10
        )
        np.testing.assert_array_equal(result1.signals, result2.signals)

    def test_method_chaining(self, train_db, val_db, config):
        metrics = PatternEngine(config).fit(train_db).evaluate(val_db.head(10), verbose=0)
        assert "brier_score" in metrics

    def test_default_config(self):
        engine = PatternEngine()
        assert engine.config.top_k == 50
        assert engine.config.feature_set == "returns_only"
