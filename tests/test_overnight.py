"""Tests for pattern_engine.overnight (static + Bayesian modes)."""

import os
import tempfile
import pytest
from pattern_engine.config import EngineConfig
from pattern_engine.overnight import OvernightRunner
from pattern_engine.experiment_logging import ExperimentLogger


class TestOvernightRunner:
    """Test static-phase overnight runner."""

    @pytest.fixture
    def two_folds(self):
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

    @pytest.fixture
    def simple_phases(self):
        return [
            EngineConfig(
                max_distance=50.0, calibration_method="platt",
                regime_filter=False, top_k=10, batch_size=64,
            ),
            EngineConfig(
                max_distance=50.0, calibration_method="isotonic",
                regime_filter=False, top_k=10, batch_size=64,
            ),
        ]

    def test_init_defaults(self):
        runner = OvernightRunner()
        assert runner.bayesian_mode is False
        assert runner.max_hours == 6.0
        assert len(runner.phases) == 8  # 4 distances x 2 methods

    def test_init_bayesian_mode(self):
        runner = OvernightRunner(bayesian_mode=True, n_trials=10)
        assert runner.bayesian_mode is True
        assert runner.n_trials == 10

    def test_static_run(self, synthetic_db, simple_phases, two_folds):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OvernightRunner(
                phases=simple_phases,
                folds=two_folds,
                checkpoint_path=os.path.join(tmpdir, "checkpoint.json"),
                max_hours=0.5,
                integrity_check_enabled=False,
                results_dir=tmpdir,
            )
            results = runner.run(synthetic_db, verbose=0)
            assert len(results) == 2
            assert all("fold_metrics" in r or "error" in r for r in results)

    def test_static_checkpoint_resume(self, synthetic_db, simple_phases, two_folds):
        """Run 1 phase, then resume — second phase should start from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = os.path.join(tmpdir, "checkpoint.json")

            # Run with 2 phases
            runner = OvernightRunner(
                phases=simple_phases,
                folds=two_folds,
                checkpoint_path=cp,
                max_hours=0.5,
                integrity_check_enabled=False,
                results_dir=tmpdir,
            )
            results = runner.run(synthetic_db, verbose=0)
            assert len(results) == 2

            # Resume — all already completed
            runner2 = OvernightRunner(
                phases=simple_phases,
                folds=two_folds,
                checkpoint_path=cp,
                max_hours=0.5,
                integrity_check_enabled=False,
                results_dir=tmpdir,
            )
            results2 = runner2.run(synthetic_db, verbose=0)
            # No new phases should run
            assert len(results2) == 0

    def test_bayesian_run(self, synthetic_db, two_folds):
        """Bayesian mode runs Optuna study."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            runner = OvernightRunner(
                bayesian_mode=True,
                n_trials=2,
                study_name="test_overnight_bayes",
                folds=two_folds,
                max_hours=0.5,
                results_dir=tmpdir,
                checkpoint_path=os.path.join(tmpdir, "checkpoint.json"),
                search_space={
                    "max_distance": (1.0, 2.0),
                    "calibration_method": ["platt", "isotonic"],
                },
            )
            # Override phases base config for synthetic data
            results = runner.run(synthetic_db, verbose=0)
            assert len(results) == 1  # Returns [{"bayesian_result": {...}}]
            assert "bayesian_result" in results[0]

            # SQLite study file should exist
            db_path = os.path.join(tmpdir, "test_overnight_bayes.db")
            assert os.path.exists(db_path)

    def test_clear_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = os.path.join(tmpdir, "checkpoint.json")
            runner = OvernightRunner(
                checkpoint_path=cp,
                results_dir=tmpdir,
            )
            # Create a dummy checkpoint
            from pattern_engine.reliability import atomic_write_json
            atomic_write_json(cp, {"completed_phase_ids": ["p00"]})
            assert os.path.exists(cp)

            runner.clear_checkpoint()
            assert not os.path.exists(cp)
