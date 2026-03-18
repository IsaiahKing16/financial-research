"""Tests for BayesianSweepRunner (Optuna-powered parameter optimization)."""

import os
import shutil
import tempfile
import numpy as np
import pytest
from pattern_engine.config import EngineConfig
from pattern_engine.sweep import BayesianSweepRunner, SweepRunner


def _make_tmpdir():
    """Create a temp dir that tolerates Windows SQLite file locks on cleanup."""
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


class TestBayesianSweepRunner:
    """Test the Bayesian sweep runner with synthetic data."""

    @pytest.fixture
    def narrow_space(self):
        """Narrow search space for fast tests."""
        return {
            "max_distance": (1.0, 2.0),
            "calibration_method": ["platt", "isotonic"],
        }

    @pytest.fixture
    def base_config(self):
        """Base config suitable for synthetic data (no regime, small k)."""
        return EngineConfig(
            regime_filter=False,
            calibration_method="none",
            top_k=10,
            max_distance=50.0,
            batch_size=64,
        )

    @pytest.fixture
    def two_folds(self):
        """Minimal fold set for fast testing."""
        return [
            {
                "train_end": "2019-06-30",
                "val_start": "2019-07-01",
                "val_end": "2019-10-01",
                "label": "fold_1",
            },
            {
                "train_end": "2019-10-01",
                "val_start": "2019-10-02",
                "val_end": "2019-12-31",
                "label": "fold_2",
            },
        ]

    def test_init_defaults(self):
        runner = BayesianSweepRunner()
        assert runner.study_name == "fppe_bayesian_sweep"
        assert runner.n_trials == 50
        assert runner.max_hours == 4.0
        assert runner.storage_path is None
        assert runner.search_space == BayesianSweepRunner.DEFAULT_SEARCH_SPACE

    def test_init_custom(self, narrow_space):
        runner = BayesianSweepRunner(
            study_name="test_study",
            n_trials=5,
            max_hours=0.1,
            search_space=narrow_space,
        )
        assert runner.study_name == "test_study"
        assert runner.n_trials == 5
        assert runner.search_space == narrow_space

    def test_build_storage_url_none(self):
        runner = BayesianSweepRunner()
        assert runner._build_storage_url() is None

    def test_build_storage_url_sqlite(self):
        runner = BayesianSweepRunner(storage_path="data/study.db")
        assert runner._build_storage_url() == "sqlite:///data/study.db"

    def test_create_study_in_memory(self):
        runner = BayesianSweepRunner(study_name="test_mem")
        study = runner._create_study()
        assert study is not None
        assert study.study_name == "test_mem"

    def test_create_study_sqlite_persistence(self):
        with _make_tmpdir() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            runner = BayesianSweepRunner(
                study_name="persist_test",
                storage_path=db_path,
            )
            study = runner._create_study()
            assert study.study_name == "persist_test"
            assert os.path.exists(db_path)

    def test_sample_config_returns_engine_config(self):
        import optuna
        runner = BayesianSweepRunner(search_space={
            "max_distance": (0.8, 2.0),
            "calibration_method": ["platt", "isotonic"],
        })
        study = optuna.create_study()
        trial = study.ask()
        config = runner._sample_config(trial)
        assert isinstance(config, EngineConfig)
        assert 0.8 <= config.max_distance <= 2.0
        assert config.calibration_method in ("platt", "isotonic")

    def test_sample_config_int_param(self):
        import optuna
        runner = BayesianSweepRunner(search_space={
            "top_k": (10, 100),
        })
        study = optuna.create_study()
        trial = study.ask()
        config = runner._sample_config(trial)
        assert isinstance(config.top_k, int)
        assert 10 <= config.top_k <= 100

    def test_run_in_memory(self, synthetic_db, narrow_space, two_folds, base_config):
        """Run 2 trials in-memory with narrow space and 2 folds."""
        runner = BayesianSweepRunner(
            study_name="test_run",
            n_trials=2,
            max_hours=0.5,
            folds=two_folds,
            search_space=narrow_space,
            base_config=base_config,
        )
        result = runner.run(synthetic_db, verbose=0)

        assert "study" in result
        assert "best_config" in result
        assert "best_bss" in result
        assert "n_trials_completed" in result
        assert result["n_trials_completed"] >= 1
        assert result["best_config"] is not None
        assert isinstance(result["best_config"], EngineConfig)

    def test_run_sqlite_resume(self, synthetic_db, narrow_space, two_folds, base_config):
        """Run 1 trial, then resume and run 1 more."""
        with _make_tmpdir() as tmpdir:
            db_path = os.path.join(tmpdir, "resume.db")

            # Run 1 trial
            runner1 = BayesianSweepRunner(
                study_name="resume_test",
                storage_path=db_path,
                n_trials=1,
                max_hours=0.5,
                folds=two_folds,
                search_space=narrow_space,
                base_config=base_config,
            )
            r1 = runner1.run(synthetic_db, verbose=0)
            assert r1["n_trials_completed"] >= 1

            # Resume with 2 more trials
            runner2 = BayesianSweepRunner(
                study_name="resume_test",
                storage_path=db_path,
                n_trials=3,  # Total budget is 3, but 1 already done
                max_hours=0.5,
                folds=two_folds,
                search_space=narrow_space,
                base_config=base_config,
            )
            r2 = runner2.run(synthetic_db, verbose=0)
            # Should have run more trials (1 existing + up to 2 new)
            assert r2["n_trials_completed"] >= 2

    def test_results_df_columns(self, synthetic_db, narrow_space, two_folds, base_config):
        runner = BayesianSweepRunner(
            study_name="df_test",
            n_trials=2,
            max_hours=0.5,
            folds=two_folds,
            search_space=narrow_space,
            base_config=base_config,
        )
        result = runner.run(synthetic_db, verbose=0)
        df = result["results_df"]
        assert df is not None
        assert "trial" in df.columns
        assert "mean_bss" in df.columns
        assert "positive_folds" in df.columns

    def test_invalid_search_space_raises(self):
        import optuna
        runner = BayesianSweepRunner(search_space={
            "max_distance": "invalid",
        })
        study = optuna.create_study()
        trial = study.ask()
        with pytest.raises(ValueError, match="Invalid search space"):
            runner._sample_config(trial)


class TestSweepRunnerGrid:
    """Ensure existing SweepRunner.grid() still works."""

    def test_grid_generation(self):
        configs = SweepRunner.grid(
            max_distance=[0.8, 1.1],
            calibration_method=["platt", "isotonic"],
        )
        assert len(configs) == 4
        assert all(isinstance(c, EngineConfig) for c in configs)
