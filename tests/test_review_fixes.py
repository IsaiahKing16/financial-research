"""Tests for code review fixes (P0-1 through P1-10).

Validates:
  - Checkpoint semantics: failed phases retryable, not marked completed
  - Config hash: full config hashing, every field changes hash
  - cal_frac removal: parameter no longer exists
  - Assert replacement: RuntimeError/ValueError instead of AssertionError
  - ProgressLog fsync: _write uses os.fsync
  - Phase health evaluation: all-failed → 'failed', partial → 'partial'
  - DataLoader empty universe guard
  - Deadline propagation in WalkForwardRunner
"""

import dataclasses
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pattern_engine.config import EngineConfig
from pattern_engine.experiment_logging import _config_hash
from pattern_engine.overnight import OvernightRunner
from pattern_engine.reliability import atomic_write_json


class TestCheckpointSemantics:
    """P0-1: Failed phases should NOT be marked as completed."""

    def test_checkpoint_format_has_phases_dict(self, synthetic_db):
        """Checkpoint should use phases dict, not completed_phase_ids set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            phases = [
                EngineConfig(
                    max_distance=50.0, calibration_method="platt",
                    regime_filter=False, top_k=10, batch_size=64,
                ),
            ]
            folds = [
                {"train_end": "2019-06-30", "val_start": "2019-07-01",
                 "val_end": "2019-10-01", "label": "f1"},
            ]
            cp_path = os.path.join(tmpdir, "cp.json")
            runner = OvernightRunner(
                phases=phases, folds=folds,
                checkpoint_path=cp_path,
                max_hours=0.5,
                integrity_check_enabled=False,
                results_dir=tmpdir,
            )
            runner.run(synthetic_db, verbose=0)

            with open(cp_path) as f:
                cp = json.load(f)
            assert "phases" in cp
            assert "p00" in cp["phases"]
            assert cp["phases"]["p00"]["status"] in ("completed", "partial")

    def test_backward_compat_old_checkpoint(self, synthetic_db):
        """Old-format checkpoints (completed_phase_ids) should still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = os.path.join(tmpdir, "cp.json")
            # Write old-format checkpoint
            atomic_write_json(cp_path, {
                "completed_phase_ids": ["p00"],
                "timestamp": "2025-01-01T00:00:00",
                "total_phases": 2,
            })
            phases = [
                EngineConfig(max_distance=50.0, regime_filter=False,
                             top_k=10, batch_size=64),
                EngineConfig(max_distance=50.0, regime_filter=False,
                             top_k=10, batch_size=64,
                             calibration_method="isotonic"),
            ]
            folds = [
                {"train_end": "2019-06-30", "val_start": "2019-07-01",
                 "val_end": "2019-10-01", "label": "f1"},
            ]
            runner = OvernightRunner(
                phases=phases, folds=folds,
                checkpoint_path=cp_path,
                max_hours=0.5,
                integrity_check_enabled=False,
                results_dir=tmpdir,
            )
            results = runner.run(synthetic_db, verbose=0)
            # p00 should be skipped (old format migrated), p01 should run
            assert len(results) == 1
            assert results[0]["phase_id"] == "p01"

    def test_phase_health_all_success(self):
        """All folds succeeded → 'completed'."""
        metrics = [
            {"fold": "2019", "brier_skill_score": 0.001},
            {"fold": "2020", "brier_skill_score": -0.002},
        ]
        assert OvernightRunner._evaluate_phase_health(metrics) == "completed"

    def test_phase_health_partial(self):
        """Some folds failed → 'partial'."""
        metrics = [
            {"fold": "2019", "brier_skill_score": 0.001},
            {"fold": "2020", "error": "something broke"},
        ]
        assert OvernightRunner._evaluate_phase_health(metrics) == "partial"

    def test_phase_health_all_failed(self):
        """All folds failed → 'failed'."""
        metrics = [
            {"fold": "2019", "error": "err1"},
            {"fold": "2020", "error": "err2"},
        ]
        assert OvernightRunner._evaluate_phase_health(metrics) == "failed"

    def test_phase_health_empty(self):
        """No folds → 'failed'."""
        assert OvernightRunner._evaluate_phase_health([]) == "failed"


class TestConfigHash:
    """P0-2: Full config hashing — every field must change the hash."""

    def test_hash_deterministic(self):
        """Same config produces same hash."""
        cfg = EngineConfig()
        assert _config_hash(cfg) == _config_hash(cfg)

    def test_hash_length(self):
        """Hash should be 16-char hex string."""
        h = _config_hash(EngineConfig())
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    @pytest.mark.parametrize("field,alt_value", [
        ("top_k", 100),
        ("max_distance", 2.0),
        ("distance_weighting", "inverse"),
        ("distance_metric", "cosine"),
        ("batch_size", 512),
        ("feature_set", "returns_full"),
        ("projection_horizon", "fwd_14d_up"),
        ("confidence_threshold", 0.80),
        ("agreement_spread", 0.20),
        ("min_matches", 20),
        ("same_sector_only", True),
        ("exclude_same_ticker", False),
        ("regime_filter", False),
        ("regime_mode", "multi"),
        ("regime_fallback", False),
        ("adx_threshold", 30.0),
        ("calibration_method", "isotonic"),
        ("nn_jobs", 2),
    ])
    def test_hash_changes_per_field(self, field, alt_value):
        """Changing any single config field must change the hash."""
        base = EngineConfig()
        modified = base.replace(**{field: alt_value})
        assert _config_hash(base) != _config_hash(modified), (
            f"Hash did not change when {field} changed from "
            f"{getattr(base, field)} to {alt_value}"
        )

    def test_hash_changes_for_feature_weights(self):
        """Changing feature_weights must change the hash."""
        base = EngineConfig()
        new_weights = dict(base.feature_weights)
        new_weights["ret_1d"] = 99.0
        modified = base.replace(feature_weights=new_weights)
        assert _config_hash(base) != _config_hash(modified)


class TestCalFracRemoval:
    """P0-3: cal_frac should no longer exist on EngineConfig."""

    def test_no_cal_frac_attribute(self):
        cfg = EngineConfig()
        assert not hasattr(cfg, "cal_frac")

    def test_cal_frac_not_in_fields(self):
        field_names = [f.name for f in dataclasses.fields(EngineConfig)]
        assert "cal_frac" not in field_names


class TestAssertReplacement:
    """P1-9: Public APIs should raise RuntimeError/ValueError, not AssertionError."""

    def test_predict_before_fit_raises_runtime_error(self):
        from pattern_engine.engine import PatternEngine
        engine = PatternEngine()
        with pytest.raises(RuntimeError, match="fit"):
            engine.predict(pd.DataFrame())

    def test_save_before_fit_raises_runtime_error(self):
        from pattern_engine.engine import PatternEngine
        engine = PatternEngine()
        with pytest.raises(RuntimeError, match="fit"):
            engine.save("dummy.pkl")

    def test_matcher_query_before_fit_raises_runtime_error(self):
        from pattern_engine.matching import Matcher
        matcher = Matcher(EngineConfig())
        with pytest.raises(RuntimeError, match="fit"):
            matcher.query(pd.DataFrame())

    def test_live_runner_no_train_raises_value_error(self):
        from pattern_engine.live import LiveSignalRunner
        runner = LiveSignalRunner()
        with pytest.raises(ValueError, match="train_db"):
            runner.run(train_db=None, query_db=pd.DataFrame())

    def test_live_runner_no_query_raises_value_error(self, train_db):
        from pattern_engine.live import LiveSignalRunner
        from pattern_engine.engine import PatternEngine
        cfg = EngineConfig(max_distance=50.0, regime_filter=False,
                           top_k=10, batch_size=64)
        engine = PatternEngine(cfg)
        engine.fit(train_db)
        runner = LiveSignalRunner(config=cfg, engine=engine)
        with pytest.raises(ValueError, match="query_db"):
            runner.run(query_db=None)


class TestProgressLogFsync:
    """P1-6: ProgressLog should fsync after write."""

    def test_write_calls_fsync(self, tmp_path):
        """Verify fsync is in the _write method source."""
        import inspect
        from pattern_engine.reliability import ProgressLog
        source = inspect.getsource(ProgressLog._write)
        assert "os.fsync" in source or "fsync" in source


class TestDeadlinePropagation:
    """P0-4: WalkForwardRunner should respect deadline_ts."""

    def test_deadline_skips_folds(self, synthetic_db):
        """An already-expired deadline should skip all folds."""
        import time
        from pattern_engine.walkforward import WalkForwardRunner
        folds = [
            {"train_end": "2019-06-30", "val_start": "2019-07-01",
             "val_end": "2019-10-01", "label": "f1"},
            {"train_end": "2019-10-01", "val_start": "2019-10-02",
             "val_end": "2019-12-31", "label": "f2"},
        ]
        runner = WalkForwardRunner(
            EngineConfig(max_distance=50.0, regime_filter=False,
                         top_k=10, batch_size=64),
            folds=folds,
        )
        # Expired deadline = 0 (epoch)
        results = runner.run(synthetic_db, verbose=0, deadline_ts=0)
        assert len(results) == 0


class TestDataLoaderGuard:
    """P1-8: DataLoader should fail on empty universe."""

    def test_empty_raw_data_raises(self):
        from pattern_engine.data import DataLoader
        loader = DataLoader()
        with pytest.raises(RuntimeError, match="No tickers survived"):
            loader.compute_features({})
