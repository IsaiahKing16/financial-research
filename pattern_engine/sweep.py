"""
sweep.py — Parameter sweep runners (grid + Bayesian).

Generates EngineConfig variants and runs walk-forward validation
for each, ranking results by Brier Skill Score.

BayesianSweepRunner uses Optuna's TPE sampler to steer toward
high-BSS parameter regions, replacing exhaustive grid search.
SQLite-backed storage enables cross-session resumption.
"""

import dataclasses
import time
from itertools import product

import numpy as np
import pandas as pd

from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS
from pattern_engine.walkforward import WalkForwardRunner
from pattern_engine.experiment_logging import ExperimentLogger


class SweepRunner:
    """Run walk-forward validation across a grid of config variants.

    Args:
        configs: list of EngineConfig variants to test
        folds: walk-forward fold definitions
        logger: optional ExperimentLogger for TSV output
    """

    def __init__(self, configs: list[EngineConfig],
                 folds: list[dict] = None,
                 logger: ExperimentLogger = None):
        self.configs = configs
        self.folds = folds or WALKFORWARD_FOLDS
        self.logger = logger

    def run(self, full_db: pd.DataFrame, experiment_name: str = "sweep",
            verbose: int = 1) -> list[dict]:
        """Run walk-forward for each config and return ranked results.

        Args:
            full_db: complete analogue database
            experiment_name: base name for TSV logging
            verbose: 0=silent, 1=progress

        Returns:
            list of result dicts sorted by mean BSS (descending)
        """
        all_results = []

        for i, config in enumerate(self.configs):
            tag = f"{experiment_name}_c{i:02d}"
            if verbose:
                print(f"\n{'#' * 60}")
                print(f"  SWEEP CONFIG {i + 1}/{len(self.configs)}: {tag}")
                print(f"  max_distance={config.max_distance} | "
                      f"feature_set={config.feature_set} | "
                      f"regime_mode={config.regime_mode}")
                print(f"{'#' * 60}")

            runner = WalkForwardRunner(config, self.folds, self.logger)
            fold_metrics = runner.run(full_db, experiment_name=tag, verbose=verbose)

            bss_values = [m["brier_skill_score"] for m in fold_metrics
                          if m.get("brier_skill_score") is not None]
            mean_bss = sum(bss_values) / len(bss_values) if bss_values else None

            all_results.append({
                "config_index": i,
                "config": config,
                "fold_metrics": fold_metrics,
                "mean_bss": mean_bss,
                "positive_folds": sum(1 for b in bss_values if b > 0),
                "total_folds": len(bss_values),
            })

        # Sort by mean BSS descending
        all_results.sort(key=lambda r: r["mean_bss"] or float("-inf"), reverse=True)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  SWEEP RESULTS (ranked by mean BSS)")
            print(f"{'=' * 60}")
            for r in all_results:
                bss = r["mean_bss"]
                sign = "+" if bss and bss > 0 else ""
                print(f"  Config {r['config_index']:2d}: "
                      f"Mean BSS={sign}{bss:.5f}  "
                      f"Positive folds={r['positive_folds']}/{r['total_folds']}")
            print(f"{'=' * 60}\n")

        return all_results

    @staticmethod
    def grid(base_config: EngineConfig = None, **param_lists) -> list[EngineConfig]:
        """Generate a grid of EngineConfig variants.

        Args:
            base_config: base config to vary from (defaults to EngineConfig())
            **param_lists: parameter names mapped to lists of values to try

        Returns:
            list of EngineConfig instances, one per combination

        Example:
            configs = SweepRunner.grid(
                max_distance=[0.8, 1.0, 1.1019],
                regime_mode=["binary", "multi"],
            )
        """
        base = base_config or EngineConfig()
        keys = list(param_lists.keys())
        values = list(param_lists.values())

        configs = []
        for combo in product(*values):
            overrides = dict(zip(keys, combo))
            configs.append(dataclasses.replace(base, **overrides))

        return configs


class BayesianSweepRunner:
    """Optuna-powered Bayesian parameter optimization for PatternEngine.

    Uses TPE (Tree-structured Parzen Estimator) sampling to explore the
    hyperparameter space efficiently, steering toward configs that maximize
    mean BSS across walk-forward folds.

    Key differences from SweepRunner:
      - Trials are guided by previous results (not exhaustive grid)
      - SQLite storage enables cross-session resumption
      - Per-trial exception isolation (one crash doesn't kill the study)
      - Time-budget enforcement with graceful shutdown
      - Pruning of unpromising trials after early folds

    Args:
        study_name: Optuna study identifier (for persistence/resumption)
        storage_path: SQLite path for study state (None = in-memory)
        n_trials: maximum number of trials to run
        max_hours: wall-clock time budget (default 4.0)
        folds: walk-forward fold definitions
        logger: optional ExperimentLogger for TSV output
        base_config: base EngineConfig to override (None = defaults)
        search_space: dict mapping param names to (low, high) tuples or
            lists of categorical values. If None, uses default space.
    """

    # Default Bayesian search space — covers the proven parameter ranges
    # with room to explore adjacent regions
    DEFAULT_SEARCH_SPACE = {
        "max_distance": (0.8, 2.0),           # float range
        "top_k": (20, 100),                    # int range
        "calibration_method": ["platt", "isotonic"],  # categorical
        "cal_frac": (0.5, 0.95),              # float range
        "confidence_threshold": (0.55, 0.80),  # float range
        "regime_mode": ["binary", "multi"],    # categorical
    }

    def __init__(
        self,
        study_name: str = "fppe_bayesian_sweep",
        storage_path: str = None,
        n_trials: int = 50,
        max_hours: float = 4.0,
        folds: list[dict] = None,
        logger: ExperimentLogger = None,
        base_config: EngineConfig = None,
        search_space: dict = None,
    ):
        self.study_name = study_name
        self.storage_path = storage_path
        self.n_trials = n_trials
        self.max_hours = max_hours
        self.folds = folds or WALKFORWARD_FOLDS
        self.logger = logger
        self.base_config = base_config or EngineConfig()
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE
        self._study = None

    def _build_storage_url(self) -> str | None:
        """Build SQLite storage URL from path, or None for in-memory."""
        if self.storage_path is None:
            return None
        return f"sqlite:///{self.storage_path}"

    def _create_study(self):
        """Create or load an Optuna study with TPE sampler."""
        import optuna

        storage = self._build_storage_url()
        self._study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction="maximize",  # Maximize mean BSS
            load_if_exists=True,   # Resume from SQLite if available
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,   # Don't prune first 5 trials
                n_warmup_steps=2,     # Don't prune first 2 folds
            ),
        )
        return self._study

    def _sample_config(self, trial) -> EngineConfig:
        """Sample an EngineConfig from the search space using Optuna trial.

        Handles float ranges, int ranges, and categorical parameters.
        """
        overrides = {}

        for param, space in self.search_space.items():
            if isinstance(space, list):
                # Categorical
                overrides[param] = trial.suggest_categorical(param, space)
            elif isinstance(space, tuple) and len(space) == 2:
                low, high = space
                # Determine type from EngineConfig defaults
                default_val = getattr(self.base_config, param, None)
                if isinstance(default_val, int):
                    overrides[param] = trial.suggest_int(param, low, high)
                else:
                    overrides[param] = trial.suggest_float(param, low, high)
            else:
                raise ValueError(
                    f"Invalid search space for {param}: {space}. "
                    f"Use (low, high) tuple or list of categories."
                )

        return self.base_config.replace(**overrides)

    def _objective(self, trial, full_db: pd.DataFrame, verbose: int) -> float:
        """Optuna objective function: run walk-forward, return mean BSS.

        Each trial samples a config, runs all folds, and reports the
        mean BSS. Early folds report intermediate values for pruning.
        """
        import optuna

        config = self._sample_config(trial)
        tag = f"bayesian_t{trial.number:03d}"

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  TRIAL {trial.number}: {tag}")
            print(f"  max_distance={config.max_distance:.4f} | "
                  f"top_k={config.top_k} | "
                  f"cal={config.calibration_method} | "
                  f"cal_frac={config.cal_frac:.2f}")
            print(f"  confidence={config.confidence_threshold:.2f} | "
                  f"regime={config.regime_mode}")
            print(f"{'=' * 60}")

        runner = WalkForwardRunner(config, self.folds, self.logger)
        fold_metrics = runner.run(
            full_db, experiment_name=tag, verbose=max(0, verbose - 1)
        )

        # Extract BSS values and report intermediate for pruning
        bss_values = []
        for step, m in enumerate(fold_metrics):
            bss = m.get("brier_skill_score")
            if bss is not None:
                bss_values.append(bss)
                # Report running mean BSS for pruning
                running_mean = np.mean(bss_values)
                trial.report(running_mean, step)

                if trial.should_prune():
                    if verbose:
                        print(f"  Trial {trial.number} pruned at fold {step + 1} "
                              f"(running BSS={running_mean:+.5f})")
                    raise optuna.TrialPruned()

        if not bss_values:
            return float("-inf")

        mean_bss = float(np.mean(bss_values))
        positive_folds = sum(1 for b in bss_values if b > 0)

        if verbose:
            print(f"  Trial {trial.number} complete: "
                  f"mean BSS={mean_bss:+.5f}, "
                  f"positive folds={positive_folds}/{len(bss_values)}")

        # Store fold details as trial user attributes
        trial.set_user_attr("positive_folds", positive_folds)
        trial.set_user_attr("total_folds", len(bss_values))
        trial.set_user_attr("fold_bss", bss_values)
        # EngineConfig contains dict fields (feature_weights) so hash() fails.
        # Use a deterministic string representation instead.
        trial.set_user_attr("config_repr", repr(config))

        return mean_bss

    def run(self, full_db: pd.DataFrame, verbose: int = 1) -> dict:
        """Run Bayesian optimization study.

        Args:
            full_db: complete analogue database
            verbose: 0=silent, 1=progress, 2=fold-level detail

        Returns:
            dict with keys: study, best_config, best_bss, n_trials,
                           results_df, elapsed_minutes
        """
        import optuna

        # Suppress Optuna's default INFO logging (we have our own output)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = self._create_study()
        start_time = time.time()
        max_seconds = self.max_hours * 3600

        # Count existing trials (for resumed studies)
        existing_trials = len([
            t for t in study.trials
            if t.state in (optuna.trial.TrialState.COMPLETE,
                           optuna.trial.TrialState.PRUNED)
        ])
        remaining = max(0, self.n_trials - existing_trials)

        if verbose and existing_trials > 0:
            print(f"  Resuming study '{self.study_name}': "
                  f"{existing_trials} trials already complete, "
                  f"{remaining} remaining")

        if remaining == 0:
            if verbose:
                print(f"  Study already has {existing_trials} trials. "
                      f"Increase n_trials to run more.")
        else:
            # Time-budget callback
            def _time_budget_callback(study, trial):
                elapsed = time.time() - start_time
                if elapsed > max_seconds:
                    study.stop()

            # Per-trial exception isolation via catch parameter
            study.optimize(
                lambda trial: self._objective(trial, full_db, verbose),
                n_trials=remaining,
                timeout=max_seconds,
                callbacks=[_time_budget_callback],
                catch=(Exception,),  # Isolate per-trial failures
            )

        elapsed_min = (time.time() - start_time) / 60

        # Build results
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        result = {
            "study": study,
            "best_config": None,
            "best_bss": None,
            "n_trials_completed": len(completed),
            "n_trials_pruned": len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ]),
            "n_trials_failed": len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.FAIL
            ]),
            "elapsed_minutes": round(elapsed_min, 1),
            "results_df": None,
        }

        if completed:
            best = study.best_trial
            result["best_bss"] = best.value
            result["best_config"] = self.base_config.replace(**{
                k: v for k, v in best.params.items()
                if hasattr(self.base_config, k)
            })

            # Build summary DataFrame
            rows = []
            for t in completed:
                row = {
                    "trial": t.number,
                    "mean_bss": t.value,
                    "positive_folds": t.user_attrs.get("positive_folds", 0),
                    "total_folds": t.user_attrs.get("total_folds", 0),
                    **t.params,
                }
                rows.append(row)
            result["results_df"] = pd.DataFrame(rows).sort_values(
                "mean_bss", ascending=False
            ).reset_index(drop=True)

        if verbose:
            self._print_summary(result)

        return result

    def _print_summary(self, result: dict):
        """Print optimization summary."""
        print(f"\n{'=' * 60}")
        print(f"  BAYESIAN SWEEP RESULTS")
        print(f"{'=' * 60}")
        print(f"  Study: {self.study_name}")
        print(f"  Completed: {result['n_trials_completed']} | "
              f"Pruned: {result['n_trials_pruned']} | "
              f"Failed: {result['n_trials_failed']}")
        print(f"  Time: {result['elapsed_minutes']:.1f} min")

        if result["best_config"] is not None:
            best = result["best_config"]
            bss = result["best_bss"]
            print(f"\n  Best BSS: {bss:+.5f}")
            print(f"  Best config:")
            print(f"    max_distance = {best.max_distance:.4f}")
            print(f"    top_k        = {best.top_k}")
            print(f"    cal_method   = {best.calibration_method}")
            print(f"    cal_frac     = {best.cal_frac:.2f}")
            print(f"    confidence   = {best.confidence_threshold:.2f}")
            print(f"    regime_mode  = {best.regime_mode}")

        if result["results_df"] is not None and len(result["results_df"]) > 0:
            print(f"\n  Top 5 trials:")
            top5 = result["results_df"].head(5)
            for _, row in top5.iterrows():
                print(f"    Trial {int(row['trial']):3d}: "
                      f"BSS={row['mean_bss']:+.5f}  "
                      f"d={row.get('max_distance', '?'):.4f}  "
                      f"k={int(row.get('top_k', 0))}  "
                      f"cal={row.get('calibration_method', '?')}")
        print(f"{'=' * 60}\n")
