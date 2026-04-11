"""
tests/unit/test_sweep.py — TDD tests for pattern_engine/sweep.py

Tests:
    TestSweepResult — dataclass field verification
    TestGridSweep — exhaustive enumeration, gate filtering
    TestOptunaSweep — basic run, persistence, resume, exception handling, TSV output
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import fields

import pandas as pd
import pytest

from pattern_engine.sweep import (
    GridSweep,
    KNN_SEARCH_SPACE,
    OptunaSweep,
    SweepResult,
)


# ── Mock objective ────────────────────────────────────────────────────────────

def _mock_objective(config: dict, full_db: pd.DataFrame) -> dict:
    md = config.get("max_distance", 2.5)
    bss = -abs(md - 2.5) + 0.001
    return {
        "mean_bss": bss,
        "trimmed_mean_bss": bss + 0.0001,
        "positive_folds": 4 if bss > 0 else 1,
        "fold_results": [
            {"fold": f"fold_{i}", "bss": bss + 0.0001 * i} for i in range(6)
        ],
        "wilcoxon_p": 0.03 if bss > 0 else 0.5,
    }


def _raising_objective(config: dict, full_db: pd.DataFrame) -> dict:
    """Raises on the second call."""
    _raising_objective._call_count = getattr(_raising_objective, "_call_count", 0) + 1
    if _raising_objective._call_count == 2:
        raise ValueError("Simulated objective failure")
    return _mock_objective(config, full_db)


# ── Small dummy DataFrame (no data loading needed) ───────────────────────────

@pytest.fixture
def dummy_db() -> pd.DataFrame:
    return pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=5), "Ticker": "T1"})


# ─────────────────────────────────────────────────────────────────────────────
# 1. TestSweepResult
# ─────────────────────────────────────────────────────────────────────────────

class TestSweepResult:
    def test_fields_exist(self):
        """SweepResult has all required fields with correct types."""
        import optuna

        study = optuna.create_study(study_name="test_fields", direction="maximize")
        results_df = pd.DataFrame(
            [{"max_distance": 2.5, "mean_bss": 0.001, "trimmed_mean_bss": 0.0011,
              "positive_folds": 4, "wilcoxon_p": 0.03}]
        )
        sr = SweepResult(
            best_config={"max_distance": 2.5},
            best_bss=0.0011,
            best_positive_folds=4,
            best_wilcoxon_p=0.03,
            results_df=results_df,
            elapsed_minutes=0.1,
            study=study,
        )

        field_names = {f.name for f in fields(sr)}
        assert "best_config" in field_names
        assert "best_bss" in field_names
        assert "best_positive_folds" in field_names
        assert "best_wilcoxon_p" in field_names
        assert "results_df" in field_names
        assert "elapsed_minutes" in field_names
        assert "study" in field_names

        assert isinstance(sr.best_config, dict)
        assert isinstance(sr.best_bss, float)
        assert isinstance(sr.best_positive_folds, int)
        assert isinstance(sr.results_df, pd.DataFrame)
        assert isinstance(sr.elapsed_minutes, float)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TestGridSweep
# ─────────────────────────────────────────────────────────────────────────────

class TestGridSweep:
    def test_exhaustive_enumeration(self, dummy_db):
        """GridSweep runs all 4 grid points; best_config is max_distance=2.5."""
        param_grid = {"max_distance": [1.5, 2.0, 2.5, 3.0]}
        sweep = GridSweep(objective_fn=_mock_objective, param_grid=param_grid)
        result = sweep.run(dummy_db, verbose=0)

        assert isinstance(result, SweepResult)
        # All 4 points must have been evaluated
        assert len(result.results_df) == 4
        # Best is 2.5 (bss = 0.001 there, all others are negative)
        assert result.best_config["max_distance"] == pytest.approx(2.5)
        assert result.best_bss > 0

    def test_gate_fn_filters(self, dummy_db):
        """Custom gate with positive_folds >= 5 penalizes configs that don't meet it."""
        # _mock_objective returns positive_folds=4 for max_distance=2.5 and 1 elsewhere
        # A gate requiring >=5 means no config passes → all penalized → best_bss is penalized
        param_grid = {"max_distance": [2.0, 2.5, 3.0]}

        def strict_gate(result_dict: dict) -> bool:
            return result_dict["positive_folds"] >= 5

        sweep = GridSweep(
            objective_fn=_mock_objective,
            param_grid=param_grid,
            gate_fn=strict_gate,
        )
        result = sweep.run(dummy_db, verbose=0)

        # All configs fail the gate, so all trimmed_mean_bss values should be penalized
        # Penalized score = max(trimmed_mean_bss - 0.05, -0.10)
        assert result.best_bss <= 0.0  # penalized
        # Verify all rows have penalized bss (none exceed a small positive threshold)
        assert all(result.results_df["trimmed_mean_bss"] <= 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TestOptunaSweep
# ─────────────────────────────────────────────────────────────────────────────

class TestOptunaSweep:
    def test_basic_run(self, dummy_db):
        """5 trials complete; returns valid SweepResult."""
        sweep = OptunaSweep(
            study_name="test_basic",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=5,
            seed=42,
        )
        result = sweep.run(dummy_db, verbose=0)

        assert isinstance(result, SweepResult)
        assert len(result.results_df) == 5
        assert result.best_bss is not None
        assert isinstance(result.best_config, dict)
        assert "max_distance" in result.best_config
        assert result.elapsed_minutes >= 0

    def test_sqlite_persistence(self, dummy_db, tmp_path):
        """storage_path creates a SQLite .db file."""
        db_file = str(tmp_path / "optuna_test.db")
        sweep = OptunaSweep(
            study_name="test_sqlite",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=3,
            storage_path=db_file,
            seed=42,
        )
        sweep.run(dummy_db, verbose=0)

        assert os.path.exists(db_file), "SQLite .db file was not created"

    def test_resume(self, dummy_db, tmp_path):
        """Run 3 trials, then resume to 5 total. results_df has 5 rows."""
        db_file = str(tmp_path / "resume_test.db")

        # Initial run: 3 trials
        sweep_initial = OptunaSweep(
            study_name="test_resume",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=3,
            storage_path=db_file,
            seed=42,
        )
        sweep_initial.run(dummy_db, verbose=0)

        # Resume: 5 total (adds 2 more)
        sweep_resume = OptunaSweep(
            study_name="test_resume",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=5,
            storage_path=db_file,
            seed=42,
        )
        result = sweep_resume.resume(dummy_db, verbose=0)

        assert isinstance(result, SweepResult)
        assert len(result.results_df) == 5

    def test_resume_requires_storage(self, dummy_db):
        """resume() raises RuntimeError if storage_path is None."""
        sweep = OptunaSweep(
            study_name="test_no_storage",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=5,
            storage_path=None,
            seed=42,
        )
        with pytest.raises(RuntimeError, match="storage_path"):
            sweep.resume(dummy_db, verbose=0)

    def test_exception_in_objective_handled(self, dummy_db):
        """If objective throws, trial is recorded with trimmed_mean_bss=-0.10; sweep continues."""
        # Reset the call counter
        _raising_objective._call_count = 0

        sweep = OptunaSweep(
            study_name="test_exception",
            objective_fn=_raising_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=3,
            seed=42,
        )
        result = sweep.run(dummy_db, verbose=0)

        # Sweep should have completed all 3 trials
        assert len(result.results_df) == 3
        # The failing trial (2nd) should have trimmed_mean_bss == -0.10
        # Use absolute tolerance since pytest.approx doesn't work with pandas Series indexing
        penalized = result.results_df[
            (result.results_df["trimmed_mean_bss"] - (-0.10)).abs() < 1e-9
        ]
        assert len(penalized) >= 1

    def test_to_tsv(self, dummy_db, tmp_path):
        """to_tsv writes a valid TSV file that can be read by pandas."""
        tsv_path = str(tmp_path / "sweep_results.tsv")
        sweep = OptunaSweep(
            study_name="test_tsv",
            objective_fn=_mock_objective,
            search_space={"max_distance": (1.0, 4.0)},
            n_trials=4,
            seed=42,
        )
        sweep.run(dummy_db, verbose=0)
        sweep.to_tsv(tsv_path)

        assert os.path.exists(tsv_path)
        df = pd.read_csv(tsv_path, sep="\t")
        assert len(df) == 4
        assert "trimmed_mean_bss" in df.columns

    def test_int_param_uses_suggest_int(self, dummy_db):
        """top_k=(20,100) with int bounds → values in results are integers."""
        sweep = OptunaSweep(
            study_name="test_int_param",
            objective_fn=_mock_objective,
            search_space={"top_k": (20, 100)},
            n_trials=5,
            seed=42,
        )
        result = sweep.run(dummy_db, verbose=0)

        assert "top_k" in result.results_df.columns
        for val in result.results_df["top_k"]:
            assert isinstance(val, int), f"Expected int, got {type(val)} for value {val}"
            assert 20 <= val <= 100


# ─────────────────────────────────────────────────────────────────────────────
# 4. KNN_SEARCH_SPACE sanity check
# ─────────────────────────────────────────────────────────────────────────────

class TestKNNSearchSpace:
    def test_knn_search_space_keys(self):
        """KNN_SEARCH_SPACE has expected keys and valid ranges."""
        assert "max_distance" in KNN_SEARCH_SPACE
        assert "top_k" in KNN_SEARCH_SPACE
        assert "cal_frac" in KNN_SEARCH_SPACE
        assert "confidence_threshold" in KNN_SEARCH_SPACE

        # top_k bounds are ints
        lo, hi = KNN_SEARCH_SPACE["top_k"]
        assert isinstance(lo, int) and isinstance(hi, int)

        # max_distance bounds are floats
        lo, hi = KNN_SEARCH_SPACE["max_distance"]
        assert isinstance(lo, float) and isinstance(hi, float)
