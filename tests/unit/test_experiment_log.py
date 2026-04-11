"""Tests for pattern_engine.experiment_log module."""
from __future__ import annotations
import pytest


class TestExperimentLogger:

    def test_header_writes_metadata(self, tmp_path):
        """log_header writes comment block + column header."""
        from pattern_engine.experiment_log import ExperimentLogger
        logger = ExperimentLogger(output_dir=str(tmp_path), experiment_name="test_sweep")
        logger.log_header(["trial", "max_distance", "mean_bss", "gate_pass"])
        tsv = (tmp_path / "test_sweep.tsv").read_text()
        assert "# experiment: test_sweep" in tsv
        assert "# started:" in tsv
        assert "trial\tmax_distance\tmean_bss\tgate_pass" in tsv

    def test_log_trial_appends_row(self, tmp_path):
        """log_trial appends a TSV row with correct values."""
        from pattern_engine.experiment_log import ExperimentLogger
        logger = ExperimentLogger(output_dir=str(tmp_path), experiment_name="test_sweep")
        cols = ["trial", "max_distance", "mean_bss"]
        logger.log_header(cols)
        logger.log_trial(trial_id=0, config={"max_distance": 2.5}, result={"mean_bss": 0.001})
        lines = (tmp_path / "test_sweep.tsv").read_text().strip().split("\n")
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == 2  # header + 1 data row
        assert "0\t" in data_lines[1]

    def test_incremental_append_survives_reopen(self, tmp_path):
        """Multiple log_trial calls produce valid TSV."""
        from pattern_engine.experiment_log import ExperimentLogger
        logger = ExperimentLogger(output_dir=str(tmp_path), experiment_name="append_test")
        cols = ["trial", "param_a", "score"]
        logger.log_header(cols)
        logger.log_trial(0, {"param_a": 1.0}, {"score": 0.5})
        logger.log_trial(1, {"param_a": 2.0}, {"score": 0.6})
        logger.log_trial(2, {"param_a": 3.0}, {"score": 0.7})
        lines = (tmp_path / "append_test.tsv").read_text().strip().split("\n")
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == 4  # header + 3 rows

    def test_finalize_writes_summary(self, tmp_path):
        """finalize writes a summary comment block."""
        from pattern_engine.experiment_log import ExperimentLogger
        logger = ExperimentLogger(output_dir=str(tmp_path), experiment_name="final_test")
        logger.log_header(["trial", "score"])
        logger.log_trial(0, {}, {"score": 0.5})
        class _MockResult:
            best_config = {"max_distance": 2.5}
            best_bss = 0.001
            elapsed_minutes = 5.0
        logger.finalize(best_config={"max_distance": 2.5}, sweep_result=_MockResult())
        tsv = (tmp_path / "final_test.tsv").read_text()
        assert "# best_config:" in tsv
        assert "# best_bss:" in tsv

    def test_missing_dir_created(self, tmp_path):
        """Logger creates output_dir if it doesn't exist."""
        from pattern_engine.experiment_log import ExperimentLogger
        nested = tmp_path / "sub" / "dir"
        logger = ExperimentLogger(output_dir=str(nested), experiment_name="nested")
        logger.log_header(["trial"])
        assert (nested / "nested.tsv").exists()
