"""
overnight.py — Multi-phase overnight runner.

Replaces the 2,601-line strategy_overnight.py with a thin orchestrator
that cycles through config variants using WalkForwardRunner.

Runs on a schedule (e.g. 6-hour overnight window) with safe checkpointing.
Each phase tests a different distance/config variant.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS
from pattern_engine.walkforward import WalkForwardRunner
from pattern_engine.experiment_logging import ExperimentLogger


class OvernightRunner:
    """Multi-phase overnight auto-research runner.

    Cycles through a list of config phases, running walk-forward
    validation for each. Progress is checkpointed to JSON for
    safe resumption after Ctrl+C.

    Args:
        phases: list of EngineConfig instances to test
        folds: walk-forward fold definitions
        logger: ExperimentLogger for TSV output
        checkpoint_path: path for progress checkpoint JSON
        max_hours: maximum wall-clock time (default 6)
    """

    def __init__(self, phases: list[EngineConfig] = None,
                 folds: list[dict] = None,
                 logger: ExperimentLogger = None,
                 checkpoint_path: str = "data/results/overnight_progress.json",
                 max_hours: float = 6.0):
        self.phases = phases or self._default_phases()
        self.folds = folds or WALKFORWARD_FOLDS
        self.logger = logger or ExperimentLogger()
        self.checkpoint_path = Path(checkpoint_path)
        self.max_hours = max_hours

    def run(self, full_db: pd.DataFrame, verbose: int = 1) -> list[dict]:
        """Run all phases within the time budget.

        Args:
            full_db: complete analogue database
            verbose: 0=silent, 1=progress

        Returns:
            list of phase result dicts
        """
        start_time = time.time()
        max_seconds = self.max_hours * 3600
        results = []

        checkpoint = self._load_checkpoint()
        start_phase = checkpoint.get("completed_phases", 0)

        for i, config in enumerate(self.phases):
            if i < start_phase:
                if verbose:
                    print(f"  Phase {i}: already completed (checkpoint)")
                continue

            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                if verbose:
                    print(f"\n  Time budget exhausted ({self.max_hours}h). "
                          f"Completed {i}/{len(self.phases)} phases.")
                break

            tag = f"overnight_p{i:02d}"
            if verbose:
                remaining = (max_seconds - elapsed) / 60
                print(f"\n{'#' * 60}")
                print(f"  OVERNIGHT PHASE {i + 1}/{len(self.phases)}")
                print(f"  Config: d={config.max_distance} "
                      f"cal={config.calibration_method} "
                      f"regime={config.regime_mode}")
                print(f"  Time remaining: {remaining:.0f} min")
                print(f"{'#' * 60}")

            runner = WalkForwardRunner(config, self.folds, self.logger)
            fold_metrics = runner.run(full_db, experiment_name=tag, verbose=verbose)

            results.append({
                "phase": i,
                "config": config,
                "fold_metrics": fold_metrics,
            })

            self._save_checkpoint(i + 1)

        if verbose:
            total_elapsed = (time.time() - start_time) / 60
            print(f"\n  Overnight run complete: {total_elapsed:.1f} min total, "
                  f"{len(results)} phases executed")

        return results

    def _default_phases(self) -> list[EngineConfig]:
        """Default overnight phase schedule (distance × calibration)."""
        distances = [1.0115, 1.1019, 1.2457, 1.5000]
        methods = ["platt", "isotonic"]
        phases = []
        for d in distances:
            for m in methods:
                phases.append(EngineConfig(max_distance=d, calibration_method=m))
        return phases

    def _load_checkpoint(self) -> dict:
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                return json.load(f)
        return {}

    def _save_checkpoint(self, completed_phases: int) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump({
                "completed_phases": completed_phases,
                "timestamp": datetime.now().isoformat(),
            }, f)
