"""
experiment_logging.py — TSV experiment logger with provenance tracking.

Clean break from results/results_analogue.tsv. New column schema
includes all EngineConfig fields. Preserves the PROJECT_GUIDE
provenance requirement: every metric must trace to a TSV row.

Reliability features:
  - fsync after each append (data hits disk before returning)
  - Deduplication: skip if (experiment_name, fold_label, config_hash) exists
  - New columns always appended at the end — never insert in the middle
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from pattern_engine.config import EngineConfig


def _config_hash(config: EngineConfig) -> str:
    """Deterministic hash of ALL config fields for deduplication.

    Previously hashed only 8 of ~20 fields, causing collisions between
    configs that differed on confidence_threshold, agreement_spread,
    distance_weighting, projection_horizon, feature_weights,
    exclude_same_ticker, same_sector_only, regime_fallback, adx_threshold,
    or min_matches.  Two materially different configs would share a hash and
    one would be silently skipped, wasting Bayesian sweep budget and
    producing incorrect deduplication.

    Fix: serialize the full config dict with sorted keys so every field
    contributes to the hash.  feature_weights is a nested dict —
    sort_keys=True ensures stable serialization regardless of insertion order.
    """
    import dataclasses
    full_dict = dataclasses.asdict(config)
    canonical = json.dumps(full_dict, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


class ExperimentLogger:
    """Appends experiment results to TSV with full config provenance."""

    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / "experiments.tsv"

    def log(self, metrics: dict, config: EngineConfig,
            experiment_name: str = "", fold_label: str = "",
            skip_duplicates: bool = True) -> bool:
        """Append one experiment result row to the TSV file.

        Args:
            metrics: dict from evaluate_probabilistic()
            config: EngineConfig used for this experiment
            experiment_name: name/tag for this experiment
            fold_label: walk-forward fold label (e.g. "2024")
            skip_duplicates: if True, skip row if already logged

        Returns:
            True if row was written, False if skipped (duplicate)
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        cfg_hash = _config_hash(config)

        # Dedup check
        if skip_duplicates and self._is_duplicate(experiment_name, fold_label, cfg_hash):
            return False

        row = {
            # Metadata
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": experiment_name,
            "fold_label": fold_label,
            "config_hash": cfg_hash,

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

        if not self.results_file.exists():
            # First write — include header, use atomic write
            fd, tmp = tempfile.mkstemp(
                dir=str(self.results_dir), suffix=".tsv.tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(df_row.to_csv(index=False, sep="\t"))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, str(self.results_file))
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        else:
            # Append with fsync — ensures row hits disk
            row_tsv = df_row.to_csv(index=False, sep="\t", header=False)
            with open(self.results_file, "a") as f:
                f.write(row_tsv)
                f.flush()
                os.fsync(f.fileno())

        return True

    def _is_duplicate(self, experiment_name: str, fold_label: str,
                      cfg_hash: str) -> bool:
        """Check if this experiment+fold+config already has a row."""
        if not self.results_file.exists():
            return False
        try:
            df = pd.read_csv(self.results_file, sep="\t", usecols=[
                "experiment_name", "fold_label", "config_hash"
            ], dtype=str)
            mask = (
                (df["experiment_name"] == str(experiment_name)) &
                (df["fold_label"] == str(fold_label)) &
                (df["config_hash"] == str(cfg_hash))
            )
            return bool(mask.any())
        except (KeyError, pd.errors.EmptyDataError):
            # config_hash column might not exist in old TSVs
            return False

    def read_results(self) -> pd.DataFrame:
        """Read all experiment results from the TSV file."""
        if not self.results_file.exists():
            return pd.DataFrame()
        return pd.read_csv(self.results_file, sep="\t")
