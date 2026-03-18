"""Tests for pattern_engine reliability infrastructure.

Tests crash recovery, corrupt checkpoint handling, lock files,
atomic writes, deduplication, and overnight runner resilience.
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from pattern_engine.config import EngineConfig
from pattern_engine.reliability import (
    atomic_write, atomic_write_json, safe_read_json,
    LockFile, ProgressLog,
)
from pattern_engine.experiment_logging import ExperimentLogger, _config_hash
from pattern_engine.engine import PatternEngine


class TestAtomicWrite:
    def test_writes_content(self, tmp_path):
        target = tmp_path / "test.txt"
        atomic_write(target, "hello world")
        assert target.read_text() == "hello world"

    def test_overwrites_existing(self, tmp_path):
        target = tmp_path / "test.txt"
        target.write_text("old content")
        atomic_write(target, "new content")
        assert target.read_text() == "new content"

    def test_binary_mode(self, tmp_path):
        target = tmp_path / "test.bin"
        atomic_write(target, b"\x00\x01\x02", mode="wb")
        assert target.read_bytes() == b"\x00\x01\x02"

    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "sub" / "dir" / "test.txt"
        atomic_write(target, "nested")
        assert target.read_text() == "nested"

    def test_no_temp_files_left_on_success(self, tmp_path):
        target = tmp_path / "test.txt"
        atomic_write(target, "clean")
        tmp_files = list(tmp_path.glob(".*tmp"))
        assert len(tmp_files) == 0


class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        target = tmp_path / "test.json"
        atomic_write_json(target, {"key": "value", "num": 42})
        with open(target) as f:
            data = json.load(f)
        assert data["key"] == "value"
        assert data["num"] == 42


class TestSafeReadJson:
    def test_reads_valid_json(self, tmp_path):
        target = tmp_path / "test.json"
        target.write_text('{"key": "value"}')
        data = safe_read_json(target)
        assert data["key"] == "value"

    def test_missing_file_returns_default(self, tmp_path):
        target = tmp_path / "nonexistent.json"
        data = safe_read_json(target, default={"fallback": True})
        assert data["fallback"] is True

    def test_corrupt_json_returns_default(self, tmp_path):
        target = tmp_path / "corrupt.json"
        target.write_text("{invalid json!!!")
        data = safe_read_json(target, default={"recovered": True})
        assert data["recovered"] is True

    def test_empty_file_returns_default(self, tmp_path):
        target = tmp_path / "empty.json"
        target.write_text("")
        data = safe_read_json(target, default={})
        assert data == {}


class TestLockFile:
    def test_acquires_lock(self, tmp_path):
        lock_path = tmp_path / "test.lock"
        with LockFile(lock_path) as lock:
            assert lock.acquired is True
            assert lock_path.exists()
        # Lock released
        assert not lock_path.exists()

    def test_blocks_second_acquisition(self, tmp_path):
        lock_path = tmp_path / "test.lock"
        with LockFile(lock_path) as lock1:
            assert lock1.acquired is True
            # Try to acquire while held
            with LockFile(lock_path) as lock2:
                assert lock2.acquired is False

    def test_stale_lock_cleaned_up(self, tmp_path):
        lock_path = tmp_path / "test.lock"
        # Write a lock with a PID that doesn't exist
        lock_data = {
            "pid": 99999999,  # Unlikely to be running
            "timestamp": "2020-01-01T00:00:00",
        }
        lock_path.write_text(json.dumps(lock_data))

        with LockFile(lock_path, stale_hours=0.001) as lock:
            assert lock.acquired is True

    def test_corrupt_lock_cleaned_up(self, tmp_path):
        lock_path = tmp_path / "test.lock"
        lock_path.write_text("not json!!")

        with LockFile(lock_path) as lock:
            assert lock.acquired is True


class TestProgressLog:
    def test_writes_entries(self, tmp_path):
        log = ProgressLog(tmp_path / "test.log")
        log.info("test message")
        log.error("error message")

        content = (tmp_path / "test.log").read_text()
        assert "INFO: test message" in content
        assert "ERROR: error message" in content

    def test_append_only(self, tmp_path):
        log = ProgressLog(tmp_path / "test.log")
        log.info("first")
        log.info("second")
        log.info("third")

        lines = (tmp_path / "test.log").read_text().strip().split("\n")
        assert len(lines) == 3

    def test_phase_tracking(self, tmp_path):
        log = ProgressLog(tmp_path / "test.log")
        log.phase_start("p01", "d=1.1 cal=platt")
        log.phase_end("p01", duration_sec=42.5, status="OK")
        log.fold_result("2024", bss=0.001)
        log.fold_result("2020", error="ValueError")

        content = (tmp_path / "test.log").read_text()
        assert "PHASE_START" in content
        assert "PHASE_END" in content
        assert "FOLD_OK" in content
        assert "FOLD_FAIL" in content


class TestExperimentLoggerDedup:
    @pytest.fixture
    def logger(self, tmp_path):
        return ExperimentLogger(results_dir=str(tmp_path))

    @pytest.fixture
    def sample_metrics(self):
        return {
            "total_samples": 100,
            "accuracy_all": 0.52,
            "confident_trades": 20,
            "brier_score": 0.24,
            "brier_skill_score": 0.001,
        }

    def test_first_write_creates_file(self, logger, sample_metrics, tmp_path):
        config = EngineConfig()
        result = logger.log(sample_metrics, config, "test_exp", "2024")
        assert result is True
        assert (tmp_path / "experiments.tsv").exists()

    def test_duplicate_skipped(self, logger, sample_metrics):
        config = EngineConfig()
        assert logger.log(sample_metrics, config, "test_exp", "2024") is True
        assert logger.log(sample_metrics, config, "test_exp", "2024") is False

    def test_different_fold_not_duplicate(self, logger, sample_metrics):
        config = EngineConfig()
        assert logger.log(sample_metrics, config, "test_exp", "2024") is True
        assert logger.log(sample_metrics, config, "test_exp", "2023") is True

    def test_different_config_not_duplicate(self, logger, sample_metrics):
        config1 = EngineConfig()
        config2 = EngineConfig(max_distance=1.5)
        assert logger.log(sample_metrics, config1, "test_exp", "2024") is True
        assert logger.log(sample_metrics, config2, "test_exp", "2024") is True

    def test_config_hash_column_present(self, logger, sample_metrics):
        config = EngineConfig()
        logger.log(sample_metrics, config, "test_exp", "2024")
        df = logger.read_results()
        assert "config_hash" in df.columns


class TestEngineAtomicSave:
    def test_save_load_roundtrip(self, train_db, val_db, tmp_path):
        config = EngineConfig(regime_filter=False)
        engine = PatternEngine(config)
        engine.fit(train_db)
        pred1 = engine.predict(val_db, verbose=0)

        save_path = tmp_path / "engine.pkl"
        engine.save(str(save_path))

        loaded = PatternEngine.load(str(save_path))
        pred2 = loaded.predict(val_db, verbose=0)

        assert np.allclose(
            pred1.calibrated_probabilities,
            pred2.calibrated_probabilities, atol=1e-10
        )

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PatternEngine.load(str(tmp_path / "nonexistent.pkl"))

    def test_load_corrupt_file_raises(self, tmp_path):
        corrupt = tmp_path / "corrupt.pkl"
        corrupt.write_text("not a pickle file!")
        with pytest.raises(ValueError, match="Corrupted"):
            PatternEngine.load(str(corrupt))

    def test_load_invalid_format_raises(self, tmp_path):
        import pickle
        invalid = tmp_path / "invalid.pkl"
        with open(invalid, "wb") as f:
            pickle.dump({"no_config_key": True}, f)
        with pytest.raises(ValueError, match="Invalid"):
            PatternEngine.load(str(invalid))

    def test_no_temp_files_after_save(self, train_db, tmp_path):
        config = EngineConfig(regime_filter=False)
        engine = PatternEngine(config)
        engine.fit(train_db)
        engine.save(str(tmp_path / "engine.pkl"))

        tmp_files = list(tmp_path.glob(".*tmp"))
        assert len(tmp_files) == 0


class TestConfigHash:
    def test_same_config_same_hash(self):
        c1 = EngineConfig()
        c2 = EngineConfig()
        assert _config_hash(c1) == _config_hash(c2)

    def test_different_config_different_hash(self):
        c1 = EngineConfig()
        c2 = EngineConfig(max_distance=1.5)
        assert _config_hash(c1) != _config_hash(c2)

    def test_hash_length(self):
        h = _config_hash(EngineConfig())
        assert len(h) == 12
