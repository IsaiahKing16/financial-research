"""Tests for pattern_engine.manifest — run manifests and cache fingerprinting."""

import json
import os
import tempfile

import pytest

from pattern_engine.manifest import (
    RunManifest, generate_run_id, compute_data_version,
    load_prior_context, check_data_staleness, save_data_version,
)
from pattern_engine.reliability import atomic_write_json


class TestRunManifest:
    def test_generate_run_id_format(self):
        rid = generate_run_id()
        parts = rid.split("_")
        assert len(parts) == 3  # date_time_uuid
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert len(parts[2]) == 6  # short uuid

    def test_generate_run_id_unique(self):
        ids = {generate_run_id() for _ in range(10)}
        assert len(ids) == 10

    def test_manifest_save_load(self, tmp_path):
        m = RunManifest(
            run_id="20260318_143022_a7b3ff",
            mode="static",
            started_at="2026-03-18T14:30:22",
            git_sha="abc1234",
            phases_completed=3,
            best_bss=0.00103,
        )
        m.save(str(tmp_path / "runs"))

        manifest_path = tmp_path / "runs" / m.run_id / "manifest.json"
        assert manifest_path.exists()

        loaded = RunManifest.load(manifest_path)
        assert loaded.run_id == m.run_id
        assert loaded.mode == "static"
        assert loaded.best_bss == 0.00103
        assert loaded.phases_completed == 3

    def test_manifest_json_readable(self, tmp_path):
        m = RunManifest(run_id="test_run", mode="bayesian", best_bss=-0.001)
        m.save(str(tmp_path / "runs"))
        with open(tmp_path / "runs" / "test_run" / "manifest.json") as f:
            data = json.load(f)
        assert data["run_id"] == "test_run"
        assert data["best_bss"] == -0.001


class TestDataVersion:
    def test_deterministic(self):
        v1 = compute_data_version(["SPY", "AAPL"], ["ret_1d", "ret_3d"])
        v2 = compute_data_version(["SPY", "AAPL"], ["ret_1d", "ret_3d"])
        assert v1 == v2

    def test_order_independent(self):
        v1 = compute_data_version(["AAPL", "SPY"], ["ret_3d", "ret_1d"])
        v2 = compute_data_version(["SPY", "AAPL"], ["ret_1d", "ret_3d"])
        assert v1 == v2  # sorted internally

    def test_ticker_change_changes_version(self):
        v1 = compute_data_version(["SPY", "AAPL"], ["ret_1d"])
        v2 = compute_data_version(["SPY", "MSFT"], ["ret_1d"])
        assert v1 != v2

    def test_feature_change_changes_version(self):
        v1 = compute_data_version(["SPY"], ["ret_1d", "ret_3d"])
        v2 = compute_data_version(["SPY"], ["ret_1d", "ret_7d"])
        assert v1 != v2

    def test_version_length(self):
        v = compute_data_version(["SPY"], ["ret_1d"])
        assert len(v) == 12


class TestStalenessCheck:
    def test_stale_when_no_cached(self, tmp_path):
        result = check_data_staleness("abc123", str(tmp_path))
        assert result["stale"] is True

    def test_stale_when_version_differs(self, tmp_path):
        save_data_version("old_ver", ["SPY"], ["ret_1d"], str(tmp_path))
        result = check_data_staleness("new_ver", str(tmp_path))
        assert result["stale"] is True

    def test_fresh_when_version_matches(self, tmp_path):
        save_data_version("same_ver", ["SPY"], ["ret_1d"], str(tmp_path))
        result = check_data_staleness("same_ver", str(tmp_path))
        assert result["stale"] is False


class TestPriorContext:
    def test_empty_when_no_runs(self, tmp_path):
        ctx = load_prior_context(str(tmp_path / "runs"))
        assert ctx["best_bss"] is None
        assert ctx["recent_manifests"] == []

    def test_loads_recent_manifests(self, tmp_path):
        runs_dir = str(tmp_path / "runs")
        for i in range(3):
            m = RunManifest(
                run_id=f"run_{i:02d}",
                mode="static",
                phases_completed=2,
                best_bss=0.001 * (i + 1),
                config_hash=f"hash_{i}",
            )
            m.save(runs_dir)

        ctx = load_prior_context(runs_dir)
        assert len(ctx["recent_manifests"]) == 3
        assert ctx["best_bss"] == 0.003  # max of 0.001, 0.002, 0.003

    def test_identifies_failed_runs(self, tmp_path):
        runs_dir = str(tmp_path / "runs")
        m = RunManifest(
            run_id="failed_run",
            mode="static",
            phases_completed=0,
            phases_failed=3,
        )
        m.save(runs_dir)

        ctx = load_prior_context(runs_dir)
        assert "failed_run" in ctx["failed_runs"]
