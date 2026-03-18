"""
experiment_logging.py — TSV experiment logger with provenance tracking.

Clean break from results/results_analogue.tsv. New column schema
includes all EngineConfig fields. Preserves the PROJECT_GUIDE
provenance requirement: every metric must trace to a TSV row.

New columns are always appended at the end — never insert in the middle.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from pattern_engine.config import EngineConfig


class ExperimentLogger:
    """Appends experiment results to TSV with full config provenance."""

    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / "experiments.tsv"

    def log(self, metrics: dict, config: EngineConfig,
            experiment_name: str = "", fold_label: str = "") -> None:
        """Append one experiment result row to the TSV file.

        Args:
            metrics: dict from evaluate_probabilistic()
            config: EngineConfig used for this experiment
            experiment_name: name/tag for this experiment
            fold_label: walk-forward fold label (e.g. "2024")
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        row = {
            # Metadata
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": experiment_name,
            "fold_label": fold_label,

            # Metrics
            "total_samples": metrics.get("total_samples", 0),
            "accuracy_all": metrics.get("accuracy_all", 0.0),
            "confident_trades": metrics.get("confident_trades", 0),
            "confident_pct": metrics.get("confident_pct", 0.0),
            "accuracy_confident": metrics.get("accuracy_confident", 0.0),
            "precision_confident": metrics.get("precision_confident", 0.0),
            "recall_confident": metrics.get("recall_confident", 0.0),
            "f1_confident": metrics.get("f1_confident", 0.0),
            "brier_score": metrics.get("brier_score"),
            "brier_skill_score": metrics.get("brier_skill_score"),
            "crps": metrics.get("crps"),
            "horizon": metrics.get("horizon", ""),
            "avg_matches": metrics.get("avg_matches", 0.0),
            "buy_signals": metrics.get("buy_signals", 0),
            "sell_signals": metrics.get("sell_signals", 0),
            "hold_signals": metrics.get("hold_signals", 0),

            # Config provenance
            "top_k": config.top_k,
            "max_distance": config.max_distance,
            "distance_weighting": config.distance_weighting,
            "distance_metric": config.distance_metric,
            "nn_jobs": config.nn_jobs,
            "batch_size": config.batch_size,
            "feature_set": config.feature_set,
            "feature_weights": json.dumps(config.feature_weights),
            "projection_horizon": config.projection_horizon,
            "confidence_threshold": config.confidence_threshold,
            "agreement_spread": config.agreement_spread,
            "min_matches": config.min_matches,
            "same_sector_only": config.same_sector_only,
            "exclude_same_ticker": config.exclude_same_ticker,
            "regime_filter": config.regime_filter,
            "regime_mode": config.regime_mode,
            "regime_fallback": config.regime_fallback,
            "adx_threshold": config.adx_threshold,
            "calibration_method": config.calibration_method,
            "cal_frac": config.cal_frac,
        }

        df_row = pd.DataFrame([row])

        if self.results_file.exists():
            df_row.to_csv(self.results_file, mode="a", header=False,
                          index=False, sep="\t")
        else:
            df_row.to_csv(self.results_file, index=False, sep="\t")

    def read_results(self) -> pd.DataFrame:
        """Read all experiment results from the TSV file."""
        if not self.results_file.exists():
            return pd.DataFrame()
        return pd.read_csv(self.results_file, sep="\t")
