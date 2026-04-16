"""
pattern_engine/sweep.py — Optuna-backed hyperparameter sweep infrastructure.

Public API:
    SweepResult       — frozen dataclass holding sweep outcomes
    OptunaSweep       — Bayesian optimisation via Optuna (TPE sampler)
    GridSweep         — Exhaustive grid search via itertools.product
    KNN_SEARCH_SPACE  — Default search space for KNN config sweeps

Objective function contract:
    fn(config: dict, full_db: DataFrame) -> {
        "mean_bss": float,
        "trimmed_mean_bss": float,
        "fold_results": list[dict],
        "positive_folds": int,
        "wilcoxon_p": float | None,
    }

Gate contract (default: positive_folds >= 3):
    gate_fn(result_dict: dict) -> bool
    Failing trials are penalized: max(trimmed_mean_bss - 0.05, -0.10)

Provenance: Phase 3 Optuna plan, Task 5 (2026-04-11)
"""
from __future__ import annotations

import itertools
import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import optuna as _optuna_type

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

#: Default search space for KNN hyperparameter sweeps.
KNN_SEARCH_SPACE: dict[str, tuple] = {
    "max_distance": (1.0, 4.0),
    "top_k": (20, 100),
    "cal_frac": (0.5, 0.95),
    "confidence_threshold": (0.55, 0.80),
}

_PENALTY_FLOOR = -0.10
_PENALTY_OFFSET = 0.05


# ── Gate function ─────────────────────────────────────────────────────────────

def _default_gate(result_dict: dict) -> bool:
    """Default gate: at least 3 out-of-sample folds must be positive."""
    return result_dict.get("positive_folds", 0) >= 3


def _penalize(trimmed_mean_bss: float) -> float:
    """Apply penalty for gate-failing or exception trials."""
    return max(trimmed_mean_bss - _PENALTY_OFFSET, _PENALTY_FLOOR)


# ── SweepResult ───────────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    """Immutable record of a completed hyperparameter sweep."""

    best_config: dict
    best_bss: float
    best_positive_folds: int
    best_wilcoxon_p: float | None
    results_df: pd.DataFrame
    elapsed_minutes: float
    study: _optuna_type.Study | None = field(default=None, repr=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _suggest_params(
    trial: _optuna_type.Trial,
    search_space: dict[str, tuple | list],
) -> dict:
    """Map search_space spec to Optuna trial suggestions."""
    config: dict = {}
    for name, spec in search_space.items():
        if isinstance(spec, list):
            config[name] = trial.suggest_categorical(name, spec)
        elif len(spec) == 2:
            lo, hi = spec
            # Both bounds int → suggest_int; otherwise suggest_float
            if isinstance(lo, int) and isinstance(hi, int):
                config[name] = trial.suggest_int(name, lo, hi)
            else:
                config[name] = trial.suggest_float(name, float(lo), float(hi))
        else:
            raise ValueError(f"Invalid search_space spec for '{name}': {spec!r}")
    return config


def _build_results_df(rows: list[dict]) -> pd.DataFrame:
    """Build a results DataFrame from trial row records."""
    return pd.DataFrame(rows)


def _best_row(df: pd.DataFrame) -> pd.Series:
    """Return the row with the highest trimmed_mean_bss."""
    return df.loc[df["trimmed_mean_bss"].idxmax()]


# ── OptunaSweep ───────────────────────────────────────────────────────────────

class OptunaSweep:
    """
    Bayesian hyperparameter sweep using Optuna's TPE sampler.

    Parameters
    ----------
    study_name:
        Optuna study identifier (used for SQLite storage key).
    objective_fn:
        Callable matching the objective contract.
    search_space:
        ``{param_name: (lo, hi)}`` — int bounds → suggest_int, float → suggest_float,
        list → suggest_categorical.
    n_trials:
        Total number of trials to evaluate.
    max_hours:
        Wall-clock budget (passed to optuna as timeout).
    storage_path:
        If set, path to a SQLite .db file for persistent storage.
    gate_fn:
        Gate function; defaults to ``positive_folds >= 3``.
    seed:
        Random seed for TPE sampler reproducibility.
    """

    def __init__(
        self,
        study_name: str,
        objective_fn: Callable[[dict, pd.DataFrame], dict],
        search_space: dict[str, tuple | list],
        n_trials: int = 80,
        max_hours: float = 16.0,
        storage_path: str | None = None,
        gate_fn: Callable[[dict], bool] | None = None,
        seed: int = 42,
    ) -> None:
        self._study_name = study_name
        self._objective_fn = objective_fn
        self._search_space = search_space
        self._n_trials = n_trials
        self._max_hours = max_hours
        self._storage_path = storage_path
        self._gate_fn = gate_fn if gate_fn is not None else _default_gate
        self._seed = seed

        # State populated after run()
        self._study: _optuna_type.Study | None = None
        self._result: SweepResult | None = None

    def _storage_url(self) -> str | None:
        if self._storage_path is None:
            return None
        return f"sqlite:///{self._storage_path}"

    def _make_study(self, load_if_exists: bool = True) -> _optuna_type.Study:
        import optuna

        sampler = optuna.samplers.TPESampler(seed=self._seed)
        return optuna.create_study(
            study_name=self._study_name,
            direction="maximize",
            sampler=sampler,
            storage=self._storage_url(),
            load_if_exists=load_if_exists,
        )

    def _build_optuna_objective(
        self, full_db: pd.DataFrame, rows: list[dict]
    ) -> Callable:
        """Return the per-trial objective closure for Optuna."""
        gate_fn = self._gate_fn
        objective_fn = self._objective_fn

        def _objective(trial) -> float:
            config = _suggest_params(trial, self._search_space)
            try:
                result_dict = objective_fn(config, full_db)
            except Exception as exc:
                warnings.warn(
                    f"Objective raised for trial {trial.number}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                row = {**config, "trimmed_mean_bss": _PENALTY_FLOOR,
                       "mean_bss": _PENALTY_FLOOR, "positive_folds": 0,
                       "wilcoxon_p": None}
                rows.append(row)
                # Store provenance as user attributes
                trial.set_user_attr("exception", str(exc))
                trial.set_user_attr("trimmed_mean_bss", _PENALTY_FLOOR)
                trial.set_user_attr("mean_bss", _PENALTY_FLOOR)
                trial.set_user_attr("positive_folds", 0)
                trial.set_user_attr("wilcoxon_p", None)
                return _PENALTY_FLOOR

            raw_score = result_dict.get("trimmed_mean_bss", result_dict.get("mean_bss", -0.10))
            passed = gate_fn(result_dict)
            score = raw_score if passed else _penalize(raw_score)

            row = {
                **config,
                "trimmed_mean_bss": score,
                "mean_bss": result_dict.get("mean_bss", score),
                "positive_folds": result_dict.get("positive_folds", 0),
                "wilcoxon_p": result_dict.get("wilcoxon_p"),
            }
            rows.append(row)

            # Provenance attributes (all needed for resume reconstruction)
            trial.set_user_attr("trimmed_mean_bss", score)
            trial.set_user_attr("mean_bss", result_dict.get("mean_bss", score))
            trial.set_user_attr("positive_folds", result_dict.get("positive_folds", 0))
            trial.set_user_attr("wilcoxon_p", result_dict.get("wilcoxon_p"))
            trial.set_user_attr("gate_passed", passed)
            fold_results = result_dict.get("fold_results", [])
            trial.set_user_attr("fold_results", fold_results)

            return score

        return _objective

    def run(self, full_db: pd.DataFrame, verbose: int = 1) -> SweepResult:
        """
        Run a fresh Optuna study (or continue an existing one if storage_path set).

        Parameters
        ----------
        full_db:
            Full DataFrame passed verbatim to objective_fn.
        verbose:
            0 = suppress Optuna logging; 1 = default.

        Returns
        -------
        SweepResult
        """
        import optuna

        if verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        rows: list[dict] = []
        study = self._make_study(load_if_exists=True)
        self._study = study

        _objective = self._build_optuna_objective(full_db, rows)
        t0 = time.monotonic()
        study.optimize(
            _objective,
            n_trials=self._n_trials,
            timeout=self._max_hours * 3600,
        )
        elapsed = (time.monotonic() - t0) / 60.0

        results_df = _build_results_df(rows)
        self._result = self._make_sweep_result(study, results_df, elapsed)
        return self._result

    def resume(self, full_db: pd.DataFrame, verbose: int = 1) -> SweepResult:
        """
        Resume a previously stored study to reach n_trials total.

        Requires storage_path to be set. Raises RuntimeError otherwise.
        """
        if self._storage_path is None:
            raise RuntimeError(
                "resume() requires storage_path to be set. "
                "Pass storage_path= to OptunaSweep.__init__."
            )

        import optuna

        if verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.load_study(
            study_name=self._study_name,
            storage=self._storage_url(),
        )
        self._study = study

        completed = len(study.trials)
        remaining = max(0, self._n_trials - completed)

        rows: list[dict] = []
        # Rebuild rows from existing trials
        for t in study.trials:
            row = {**t.params}
            row["trimmed_mean_bss"] = t.user_attrs.get("trimmed_mean_bss", t.value)
            row["mean_bss"] = t.user_attrs.get("mean_bss", t.value)
            row["positive_folds"] = t.user_attrs.get("positive_folds", 0)
            row["wilcoxon_p"] = t.user_attrs.get("wilcoxon_p")
            rows.append(row)

        if remaining > 0:
            _objective = self._build_optuna_objective(full_db, rows)
            t0 = time.monotonic()
            study.optimize(
                _objective,
                n_trials=remaining,
                timeout=self._max_hours * 3600,
            )
            elapsed = (time.monotonic() - t0) / 60.0
        else:
            elapsed = 0.0

        results_df = _build_results_df(rows)
        self._result = self._make_sweep_result(study, results_df, elapsed)
        return self._result

    def best(self) -> dict:
        """Return best_config from the last run. Raises RuntimeError if not yet run."""
        if self._result is None:
            raise RuntimeError("Call run() or resume() before best().")
        return self._result.best_config

    def to_tsv(self, path: str) -> None:
        """Write results_df to a TSV file. Raises RuntimeError if not yet run."""
        if self._result is None:
            raise RuntimeError("Call run() or resume() before to_tsv().")
        self._result.results_df.to_csv(path, sep="\t", index=False)

    @staticmethod
    def _make_sweep_result(
        study: _optuna_type.Study,
        results_df: pd.DataFrame,
        elapsed: float,
    ) -> SweepResult:
        if results_df.empty:
            return SweepResult(
                best_config={},
                best_bss=float("nan"),
                best_positive_folds=0,
                best_wilcoxon_p=None,
                results_df=results_df,
                elapsed_minutes=elapsed,
                study=study,
            )
        best = _best_row(results_df)
        # Extract config columns (not metric columns)
        metric_cols = {"trimmed_mean_bss", "mean_bss", "positive_folds", "wilcoxon_p"}
        best_config = {k: v for k, v in best.items() if k not in metric_cols}
        return SweepResult(
            best_config=best_config,
            best_bss=float(best["trimmed_mean_bss"]),
            best_positive_folds=int(best.get("positive_folds", 0)),
            best_wilcoxon_p=best.get("wilcoxon_p"),
            results_df=results_df,
            elapsed_minutes=elapsed,
            study=study,
        )


# ── GridSweep ─────────────────────────────────────────────────────────────────

class GridSweep:
    """
    Exhaustive grid search over a discrete param_grid.

    Parameters
    ----------
    objective_fn:
        Callable matching the objective contract.
    param_grid:
        ``{param_name: [val1, val2, ...]}`` — all combinations evaluated.
    gate_fn:
        Gate function; defaults to ``positive_folds >= 3``.
    """

    def __init__(
        self,
        objective_fn: Callable[[dict, pd.DataFrame], dict],
        param_grid: dict[str, list],
        gate_fn: Callable[[dict], bool] | None = None,
    ) -> None:
        self._objective_fn = objective_fn
        self._param_grid = param_grid
        self._gate_fn = gate_fn if gate_fn is not None else _default_gate
        self._result: SweepResult | None = None

    def run(self, full_db: pd.DataFrame, verbose: int = 1) -> SweepResult:
        """
        Evaluate all grid combinations and return the best SweepResult.
        """
        keys = list(self._param_grid.keys())
        value_lists = [self._param_grid[k] for k in keys]
        combos = list(itertools.product(*value_lists))

        rows: list[dict] = []
        t0 = time.monotonic()

        for combo in combos:
            config = dict(zip(keys, combo))
            try:
                result_dict = self._objective_fn(config, full_db)
            except Exception as exc:
                warnings.warn(
                    f"GridSweep objective raised for config {config}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                row = {
                    **config,
                    "trimmed_mean_bss": _PENALTY_FLOOR,
                    "mean_bss": _PENALTY_FLOOR,
                    "positive_folds": 0,
                    "wilcoxon_p": None,
                }
                rows.append(row)
                continue

            raw_score = result_dict.get("trimmed_mean_bss", result_dict.get("mean_bss", -0.10))
            passed = self._gate_fn(result_dict)
            score = raw_score if passed else _penalize(raw_score)

            row = {
                **config,
                "trimmed_mean_bss": score,
                "mean_bss": result_dict.get("mean_bss", score),
                "positive_folds": result_dict.get("positive_folds", 0),
                "wilcoxon_p": result_dict.get("wilcoxon_p"),
            }
            rows.append(row)

            if verbose:
                logger.debug("GridSweep config=%s score=%.6f gate=%s", config, score, passed)

        elapsed = (time.monotonic() - t0) / 60.0
        results_df = _build_results_df(rows)

        if results_df.empty:
            self._result = SweepResult(
                best_config={},
                best_bss=float("nan"),
                best_positive_folds=0,
                best_wilcoxon_p=None,
                results_df=results_df,
                elapsed_minutes=elapsed,
                study=None,
            )
            return self._result

        best = _best_row(results_df)
        metric_cols = {"trimmed_mean_bss", "mean_bss", "positive_folds", "wilcoxon_p"}
        best_config = {k: v for k, v in best.items() if k not in metric_cols}

        self._result = SweepResult(
            best_config=best_config,
            best_bss=float(best["trimmed_mean_bss"]),
            best_positive_folds=int(best.get("positive_folds", 0)),
            best_wilcoxon_p=best.get("wilcoxon_p"),
            results_df=results_df,
            elapsed_minutes=elapsed,
            study=None,
        )
        return self._result

    def to_tsv(self, path: str) -> None:
        """Write results_df to a TSV file. Raises RuntimeError if not yet run."""
        if self._result is None:
            raise RuntimeError("Call run() before to_tsv().")
        self._result.results_df.to_csv(path, sep="\t", index=False)
