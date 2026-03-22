"""Tests for pattern_engine.walkforward and pattern_engine.sweep."""

import tempfile
import numpy as np
import pandas as pd
import pytest
from pattern_engine.config import EngineConfig
from pattern_engine.walkforward import WalkForwardRunner
from pattern_engine.sweep import SweepRunner
from pattern_engine.experiment_logging import ExperimentLogger


class TestWalkForwardRunner:
    @pytest.fixture
    def simple_folds(self):
        """Two short folds for fast testing."""
        return [
            {
                "train_end": "2019-06-30",
                "val_start": "2019-07-01",
                "val_end": "2019-12-31",
                "label": "2019H2",
            },
            {
                "train_end": "2019-12-31",
                "val_start": "2020-01-01",
                "val_end": "2020-06-30",
                "label": "2020H1",
            },
        ]

    @pytest.fixture
    def wf_config(self):
        return EngineConfig(
            regime_filter=False,
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
            batch_size=128,
        )

    def test_run_produces_metrics(self, synthetic_db, simple_folds, wf_config):
        runner = WalkForwardRunner(wf_config, folds=simple_folds)
        results = runner.run(synthetic_db, verbose=0)
        assert len(results) > 0
        for r in results:
            assert "fold" in r
            assert "brier_score" in r
            assert "brier_skill_score" in r

    def test_empty_fold_skipped(self, synthetic_db, wf_config):
        """A fold with dates outside the data range should be skipped."""
        folds = [
            {
                "train_end": "2010-01-01",
                "val_start": "2010-01-02",
                "val_end": "2010-12-31",
                "label": "2010 (empty)",
            }
        ]
        runner = WalkForwardRunner(wf_config, folds=folds)
        results = runner.run(synthetic_db, verbose=0)
        assert len(results) == 0

    def test_with_logger(self, synthetic_db, simple_folds, wf_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(results_dir=tmpdir)
            runner = WalkForwardRunner(wf_config, folds=simple_folds, logger=logger)
            runner.run(synthetic_db, verbose=0)
            tsv = logger.read_results()
            assert len(tsv) > 0
            assert "brier_skill_score" in tsv.columns


class TestSweepRunner:
    def test_grid_generation(self):
        configs = SweepRunner.grid(
            max_distance=[1.0, 2.0],
            regime_mode=["binary", "multi"],
        )
        assert len(configs) == 4
        assert all(isinstance(c, EngineConfig) for c in configs)

    def test_grid_with_base_config(self):
        base = EngineConfig(top_k=20)
        configs = SweepRunner.grid(base, max_distance=[1.0, 2.0])
        assert len(configs) == 2
        assert all(c.top_k == 20 for c in configs)

    def test_sweep_runs(self, synthetic_db):
        folds = [
            {
                "train_end": "2019-06-30",
                "val_start": "2019-07-01",
                "val_end": "2019-12-31",
                "label": "2019H2",
            },
        ]
        configs = [
            EngineConfig(regime_filter=False, calibration_method="none",
                         top_k=10, max_distance=50.0, batch_size=128),
            EngineConfig(regime_filter=False, calibration_method="none",
                         top_k=10, max_distance=10.0, batch_size=128),
        ]
        sweep = SweepRunner(configs, folds=folds)
        results = sweep.run(synthetic_db, verbose=0)
        assert len(results) == 2
        # Results should be sorted by BSS
        if results[0]["mean_bss"] is not None and results[1]["mean_bss"] is not None:
            assert results[0]["mean_bss"] >= results[1]["mean_bss"]


class TestExperimentLogger:
    def test_log_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(results_dir=tmpdir)
            metrics = {
                "total_samples": 100,
                "accuracy_all": 0.55,
                "brier_score": 0.24,
                "brier_skill_score": 0.01,
            }
            logger.log(metrics, EngineConfig(), experiment_name="test", fold_label="2024")
            df = logger.read_results()
            assert len(df) == 1
            assert df.iloc[0]["experiment_name"] == "test"
            assert str(df.iloc[0]["fold_label"]) == "2024"
            assert df.iloc[0]["top_k"] == 50

    def test_read_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(results_dir=tmpdir)
            df = logger.read_results()
            assert len(df) == 0

    def test_append_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(results_dir=tmpdir)
            cfg = EngineConfig()
            for i in range(3):
                logger.log({"total_samples": i}, cfg, experiment_name=f"exp_{i}")
            df = logger.read_results()
            assert len(df) == 3
