"""
pattern_engine/experiment_log.py — TSV provenance logger for sweep experiments.

Writes trial results incrementally to a TSV file with metadata comment headers.
Designed to survive interruption — each row is appended individually.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


class ExperimentLogger:
    """Incremental TSV logger for hyperparameter sweep experiments."""

    def __init__(self, output_dir: str = "results", experiment_name: str = "") -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._name = experiment_name or "experiment"
        self._path = self._dir / f"{self._name}.tsv"
        self._columns: list[str] = []

    @property
    def path(self) -> Path:
        return self._path

    def log_header(self, columns: list[str], search_space: dict | None = None,
                   locked_settings: str | None = None) -> None:
        """Write metadata comment block + column header."""
        self._columns = list(columns)
        with open(self._path, "w", encoding="utf-8") as f:
            f.write(f"# experiment: {self._name}\n")
            f.write(f"# started: {datetime.now(UTC).isoformat()}\n")
            if search_space:
                f.write(f"# search_space: {json.dumps(search_space)}\n")
            if locked_settings:
                f.write(f"# locked: {locked_settings}\n")
            f.write("\t".join(columns) + "\n")

    def log_trial(self, trial_id: int, config: dict, result: dict) -> None:
        """Append one TSV row for a completed trial."""
        row_data = {"trial": trial_id}
        row_data.update(config)
        row_data.update(result)
        values = []
        for col in self._columns:
            val = row_data.get(col, "")
            if isinstance(val, float):
                values.append(f"{val:+.6f}" if val != 0.0 else "+0.000000")
            elif isinstance(val, bool):
                values.append(str(val))
            else:
                values.append(str(val))
        with open(self._path, "a", encoding="utf-8") as f:
            f.write("\t".join(values) + "\n")

    def finalize(self, best_config: dict, sweep_result: object) -> None:
        """Append summary comment block after sweep completion."""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(f"# best_config: {json.dumps(best_config)}\n")
            bss = getattr(sweep_result, "best_bss", None)
            if bss is not None:
                f.write(f"# best_bss: {bss:+.6f}\n")
            elapsed = getattr(sweep_result, "elapsed_minutes", None)
            if elapsed is not None:
                f.write(f"# elapsed_minutes: {elapsed:.1f}\n")
