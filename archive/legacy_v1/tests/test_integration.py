"""Integration tests for the full PatternEngine pipeline.

These tests verify end-to-end workflows: fit → predict → evaluate,
walk-forward, persistence round-trip, and feature set variations.
"""

import os
import tempfile
import numpy as np
import pytest
from pattern_engine.config import EngineConfig
from pattern_engine.engine import PatternEngine


class TestFullPipeline:
    """End-to-end: fit → predict → evaluate."""

    @pytest.fixture
    def config(self):
        return EngineConfig(
            regime_filter=False,
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
            batch_size=64,
        )

    def test_fit_predict_evaluate(self, train_db, val_db, config):
        engine = PatternEngine(config)
        engine.fit(train_db)
        result = engine.predict(val_db.head(20), verbose=0)

        assert len(result.probabilities) == 20
        assert all(0 <= p <= 1 for p in result.probabilities)

        metrics = engine.evaluate(val_db.head(20), verbose=0)
        assert metrics["total_samples"] == 20
        assert 0 <= metrics["brier_score"] <= 1


class TestCalibrationPipeline:
    """Test the calibration double-pass end-to-end."""

    def test_platt_calibration(self, train_db, val_db):
        config = EngineConfig(
            regime_filter=False,
            calibration_method="platt",
            top_k=10,
            max_distance=50.0,
        )
        engine = PatternEngine(config)
        engine.fit(train_db)
        result = engine.predict(val_db.head(10), verbose=0)
        assert all(0 <= p <= 1 for p in result.calibrated_probabilities)

    def test_isotonic_calibration(self, train_db, val_db):
        config = EngineConfig(
            regime_filter=False,
            calibration_method="isotonic",
            top_k=10,
            max_distance=50.0,
        )
        engine = PatternEngine(config)
        engine.fit(train_db)
        result = engine.predict(val_db.head(10), verbose=0)
        assert all(0 <= p <= 1 for p in result.calibrated_probabilities)


class TestRegimePipeline:
    """Test regime filtering through the full pipeline."""

    def test_binary_regime(self, train_db, val_db):
        config = EngineConfig(
            regime_filter=True,
            regime_mode="binary",
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
        )
        engine = PatternEngine(config)
        engine.fit(train_db)
        metrics = engine.evaluate(val_db.head(10), verbose=0)
        assert "brier_score" in metrics

    def test_multi_regime(self, train_db, val_db):
        config = EngineConfig(
            regime_filter=True,
            regime_mode="multi",
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
        )
        engine = PatternEngine(config)
        engine.fit(train_db)
        metrics = engine.evaluate(val_db.head(10), verbose=0)
        assert "brier_score" in metrics

    def test_octet_regime(self, train_db, val_db):
        config = EngineConfig(
            regime_filter=True,
            regime_mode="octet",
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
        )
        engine = PatternEngine(config)
        engine.fit(train_db)
        metrics = engine.evaluate(val_db.head(10), verbose=0)
        assert "brier_score" in metrics


class TestPersistenceRoundTrip:
    """Test save/load produces identical predictions."""

    def test_roundtrip_no_regime(self, train_db, val_db):
        config = EngineConfig(
            regime_filter=False,
            calibration_method="platt",
            top_k=10,
            max_distance=50.0,
        )
        engine = PatternEngine(config)
        engine.fit(train_db)
        query = val_db.head(5)
        result1 = engine.predict(query, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "engine.pkl")
            engine.save(path)
            loaded = PatternEngine.load(path)
            result2 = loaded.predict(query, verbose=0)

        np.testing.assert_array_almost_equal(
            result1.calibrated_probabilities,
            result2.calibrated_probabilities,
            decimal=10,
        )

    def test_roundtrip_with_regime(self, train_db, val_db):
        config = EngineConfig(
            regime_filter=True,
            regime_mode="binary",
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
        )
        engine = PatternEngine(config)
        engine.fit(train_db)
        query = val_db.head(5)
        result1 = engine.predict(query, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "engine.pkl")
            engine.save(path)
            loaded = PatternEngine.load(path)
            result2 = loaded.predict(query, verbose=0)

        np.testing.assert_array_almost_equal(
            result1.probabilities, result2.probabilities, decimal=10
        )


class TestFeatureSetVariations:
    """Test different feature sets through the pipeline."""

    def test_returns_candle(self, train_db, val_db):
        config = EngineConfig(
            feature_set="returns_candle",
            regime_filter=False,
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
        )
        engine = PatternEngine(config)
        engine.fit(train_db)
        metrics = engine.evaluate(val_db.head(10), verbose=0)
        assert "brier_score" in metrics

    def test_returns_vol(self, train_db, val_db):
        config = EngineConfig(
            feature_set="returns_vol",
            regime_filter=False,
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
        )
        engine = PatternEngine(config)
        engine.fit(train_db)
        metrics = engine.evaluate(val_db.head(10), verbose=0)
        assert "brier_score" in metrics

    def test_returns_sector(self, train_db, val_db):
        config = EngineConfig(
            feature_set="returns_sector",
            regime_filter=False,
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
        )
        engine = PatternEngine(config)
        engine.fit(train_db)
        metrics = engine.evaluate(val_db.head(10), verbose=0)
        assert "brier_score" in metrics

    def test_full_features(self, train_db, val_db):
        config = EngineConfig(
            feature_set="full",
            regime_filter=False,
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
        )
        engine = PatternEngine(config)
        engine.fit(train_db)
        metrics = engine.evaluate(val_db.head(10), verbose=0)
        assert "brier_score" in metrics
