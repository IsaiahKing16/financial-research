"""
sweep.py — Parameter sweep runner.

Generates EngineConfig variants and runs walk-forward validation
for each, ranking results by Brier Skill Score.
"""

import dataclasses
from itertools import product

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
