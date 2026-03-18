"""
overnight.py — Multi-phase overnight runner with crash isolation.

Replaces the 2,601-line strategy_overnight.py with a thin orchestrator
that cycles through config variants using WalkForwardRunner.

Supports two modes:
  1. Static phases: explicit list of EngineConfig instances (legacy)
  2. Bayesian mode: Optuna-driven parameter optimization via BayesianSweepRunner

Reliability features:
  - Lock file prevents concurrent runs
  - Atomic checkpoint writes survive mid-write crashes
  - Per-phase exception isolation (one failure doesn't kill the run)
  - Progress log to disk (not just stdout)
  - Resumable from checkpoint after restart (SQLite for Bayesian mode)
  - Time budget enforcement per-run
"""

import atexit
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS
from pattern_engine.walkforward import WalkForwardRunner
from pattern_engine.experiment_logging import ExperimentLogger
from pattern_engine.reliability import (
    atomic_write_json, safe_read_json, LockFile, ProgressLog,
)


class OvernightRunner:
    """Multi-phase overnight auto-research runner.

    Cycles through a list of config phases, running walk-forward
    validation for each. Progress is checkpointed atomically to JSON
    for safe resumption after crashes or Ctrl+C.

    Args:
        phases: list of EngineConfig instances to test
        folds: walk-forward fold definitions
        logger: ExperimentLogger for TSV output
        checkpoint_path: path for progress checkpoint JSON
        max_hours: maximum wall-clock time (default 6)
        integrity_check_enabled: run integrity check after each phase
        results_dir: directory for lock file and progress log
    """

    def __init__(self, phases: list[EngineConfig] = None,
                 folds: list[dict] = None,
                 logger: ExperimentLogger = None,
                 checkpoint_path: str = "data/results/overnight_progress.json",
                 max_hours: float = 6.0,
                 integrity_check_enabled: bool = True,
                 results_dir: str = "data/results",
                 bayesian_mode: bool = False,
                 n_trials: int = 50,
                 study_name: str = "overnight_bayesian",
                 search_space: dict = None):
        self.folds = folds or WALKFORWARD_FOLDS
        self.logger = logger or ExperimentLogger()
        self.checkpoint_path = Path(checkpoint_path)
        self.max_hours = max_hours
        self.integrity_check_enabled = integrity_check_enabled
        self._results_dir = Path(results_dir)
        self._lock_path = self._results_dir / "overnight.lock"
        self._log = ProgressLog(self._results_dir / "overnight.log")

        # Bayesian mode config
        self.bayesian_mode = bayesian_mode
        self.n_trials = n_trials
        self.study_name = study_name
        self.search_space = search_space

        # Static phases (only used when bayesian_mode=False)
        self.phases = phases or self._default_phases()

    def run(self, full_db: pd.DataFrame, verbose: int = 1) -> list[dict]:
        """Run all phases within the time budget.

        Acquires a lock file, loads checkpoint, runs remaining phases
        with per-phase exception isolation, and saves progress atomically.

        Args:
            full_db: complete analogue database
            verbose: 0=silent, 1=progress

        Returns:
            list of phase result dicts
        """
        with LockFile(self._lock_path) as lock:
            if not lock.acquired:
                msg = "Another overnight run is in progress. Exiting."
                if verbose:
                    print(f"  {msg}")
                self._log.error(msg)
                return []

            mode_label = "bayesian" if self.bayesian_mode else "static"
            phase_count = self.n_trials if self.bayesian_mode else len(self.phases)
            self._log.info(
                f"Overnight run started: {phase_count} {mode_label} phases, "
                f"{self.max_hours}h budget, PID={lock._pid}"
            )

            # Register atexit to log clean shutdown
            atexit.register(self._log.info, "Process exiting (atexit)")

            if self.bayesian_mode:
                return self._run_bayesian(full_db, verbose)
            return self._run_phases(full_db, verbose)

    def _run_phases(self, full_db: pd.DataFrame, verbose: int) -> list[dict]:
        """Core phase loop with checkpointing and error isolation."""
        start_time = time.time()
        max_seconds = self.max_hours * 3600
        results = []

        checkpoint = safe_read_json(self.checkpoint_path)
        completed = set(checkpoint.get("completed_phase_ids", []))
        # Backward compat: old checkpoint used "completed_phases" int count
        if "completed_phases" in checkpoint and not completed:
            completed = {f"p{n:02d}" for n in range(checkpoint["completed_phases"])}

        for i, config in enumerate(self.phases):
            phase_id = f"p{i:02d}"

            if phase_id in completed:
                if verbose:
                    print(f"  Phase {phase_id}: already completed (checkpoint)")
                self._log.info(f"Phase {phase_id}: skipped (checkpoint)")
                continue

            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                msg = (f"Time budget exhausted ({self.max_hours}h). "
                       f"Completed {len(completed)}/{len(self.phases)} phases.")
                if verbose:
                    print(f"\n  {msg}")
                self._log.warn(msg)
                break

            config_summary = (f"d={config.max_distance} "
                              f"cal={config.calibration_method} "
                              f"regime={config.regime_mode}")
            tag = f"overnight_{phase_id}"

            if verbose:
                remaining = (max_seconds - elapsed) / 60
                print(f"\n{'#' * 60}")
                print(f"  OVERNIGHT PHASE {i + 1}/{len(self.phases)}")
                print(f"  Config: {config_summary}")
                print(f"  Time remaining: {remaining:.0f} min")
                print(f"{'#' * 60}")

            self._log.phase_start(phase_id, config_summary)
            phase_start = time.time()

            runner = WalkForwardRunner(config, self.folds, self.logger)
            try:
                fold_metrics = runner.run(full_db, experiment_name=tag, verbose=verbose)
                phase_result = {
                    "phase": i,
                    "phase_id": phase_id,
                    "config": config,
                    "fold_metrics": fold_metrics,
                }

                # Auto integrity check after each successful phase
                if self.integrity_check_enabled:
                    phase_result["integrity"] = self._run_integrity_check(
                        config, full_db, verbose
                    )

                results.append(phase_result)
                duration = time.time() - phase_start
                self._log.phase_end(phase_id, duration, "OK")

            except Exception as e:
                duration = time.time() - phase_start
                if verbose:
                    print(f"  PHASE {i + 1} FAILED: {e}")
                self._log.phase_end(phase_id, duration, f"FAILED: {e}")
                results.append({
                    "phase": i,
                    "phase_id": phase_id,
                    "config": config,
                    "error": str(e),
                })

            # Always checkpoint, even on failure
            completed.add(phase_id)
            self._save_checkpoint(completed)

        if verbose:
            total_elapsed = (time.time() - start_time) / 60
            print(f"\n  Overnight run complete: {total_elapsed:.1f} min total, "
                  f"{len(results)} phases executed")

        self._log.info(f"Run complete: {len(results)} phases in "
                       f"{(time.time() - start_time) / 60:.1f} min")

        return results

    def _run_bayesian(self, full_db: pd.DataFrame, verbose: int) -> list[dict]:
        """Run Bayesian optimization via BayesianSweepRunner.

        Uses SQLite storage in the results directory for cross-session
        resumption. The study persists even if the process crashes.
        """
        from pattern_engine.sweep import BayesianSweepRunner

        storage_path = str(self._results_dir / f"{self.study_name}.db")
        self._log.info(f"Bayesian mode: study={self.study_name}, "
                       f"storage={storage_path}, n_trials={self.n_trials}")

        runner = BayesianSweepRunner(
            study_name=self.study_name,
            storage_path=storage_path,
            n_trials=self.n_trials,
            max_hours=self.max_hours,
            folds=self.folds,
            logger=self.logger,
            search_space=self.search_space,
        )

        result = runner.run(full_db, verbose=verbose)

        # Log summary
        self._log.info(
            f"Bayesian run complete: {result['n_trials_completed']} completed, "
            f"{result['n_trials_pruned']} pruned, "
            f"{result['n_trials_failed']} failed, "
            f"{result['elapsed_minutes']:.1f} min"
        )
        if result["best_bss"] is not None:
            self._log.info(f"Best BSS: {result['best_bss']:+.5f}")

        # Return as list[dict] for API compatibility with static mode
        return [{"bayesian_result": result}]

    def _run_integrity_check(self, config, full_db, verbose):
        """Run integrity check on first fold (quick sanity)."""
        try:
            from pattern_engine.cross_validation import CrossValidator
            fold = self.folds[0]
            full_db_dt = full_db.copy()
            full_db_dt["Date"] = pd.to_datetime(full_db_dt["Date"])
            t_db = full_db_dt[full_db_dt["Date"] <= fold["train_end"]].copy()
            v_db = full_db_dt[
                (full_db_dt["Date"] >= fold["val_start"]) &
                (full_db_dt["Date"] <= fold["val_end"])
            ].copy()
            if len(t_db) > 0 and len(v_db) > 0:
                integrity = CrossValidator.integrity_check(
                    config, t_db, v_db, verbose=0
                )
                if verbose and not integrity.get("all_passed"):
                    print(f"  WARNING: Integrity check FAILED")
                return integrity
        except Exception as e:
            self._log.warn(f"Integrity check error: {e}")
            return {"all_passed": False, "error": str(e)}
        return None

    def _default_phases(self) -> list[EngineConfig]:
        """Default overnight phase schedule (distance x calibration)."""
        distances = [1.0115, 1.1019, 1.2457, 1.5000]
        methods = ["platt", "isotonic"]
        phases = []
        for d in distances:
            for m in methods:
                phases.append(EngineConfig(max_distance=d, calibration_method=m))
        return phases

    def _save_checkpoint(self, completed: set) -> None:
        """Atomically write checkpoint to disk."""
        atomic_write_json(self.checkpoint_path, {
            "completed_phase_ids": sorted(completed),
            "timestamp": datetime.now().isoformat(),
            "total_phases": len(self.phases),
        })

    def clear_checkpoint(self) -> None:
        """Delete the checkpoint file to force a fresh run."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
