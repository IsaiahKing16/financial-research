"""
walkforward.py — Walk-forward validation runner.

Implements expanding-window walk-forward validation: for each fold,
trains on all data up to train_end, validates on val_start to val_end.
Each fold builds a fresh PatternEngine instance.
"""

import pandas as pd

from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS
from pattern_engine.engine import PatternEngine
from pattern_engine.evaluation import print_metrics
from pattern_engine.experiment_logging import ExperimentLogger


class WalkForwardRunner:
    """Walk-forward validation with expanding training windows.

    Args:
        config: EngineConfig for each fold's engine
        folds: list of fold dicts with train_end, val_start, val_end, label
        logger: optional ExperimentLogger for TSV output
    """

    def __init__(self, config: EngineConfig = None,
                 folds: list[dict] = None,
                 logger: ExperimentLogger = None):
        self.config = config or EngineConfig()
        self.folds = folds or WALKFORWARD_FOLDS
        self.logger = logger

    def run(self, full_db: pd.DataFrame, experiment_name: str = "walkforward",
            verbose: int = 1) -> list[dict]:
        """Run walk-forward validation across all folds.

        Args:
            full_db: complete analogue database (all dates)
            experiment_name: name for TSV logging
            verbose: 0=silent, 1=progress

        Returns:
            list of metrics dicts, one per fold
        """
        full_db = full_db.copy()
        full_db["Date"] = pd.to_datetime(full_db["Date"])

        all_metrics = []

        for fold in self.folds:
            label = fold["label"]
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"  FOLD: {label}")
                print(f"  Train: <= {fold['train_end']} | Val: {fold['val_start']} → {fold['val_end']}")
                print(f"{'=' * 60}")

            # Temporal split for this fold
            train_db = full_db[full_db["Date"] <= fold["train_end"]].copy()
            val_db = full_db[
                (full_db["Date"] >= fold["val_start"]) &
                (full_db["Date"] <= fold["val_end"])
            ].copy()

            if len(train_db) == 0 or len(val_db) == 0:
                if verbose:
                    print(f"  Skipping {label}: empty split "
                          f"(train={len(train_db)}, val={len(val_db)})")
                continue

            if verbose:
                print(f"  Train: {len(train_db):,} rows | Val: {len(val_db):,} rows")

            # Build and evaluate engine for this fold
            engine = PatternEngine(self.config)
            engine.fit(train_db)
            metrics = engine.evaluate(val_db, verbose=verbose)

            if verbose:
                print_metrics(metrics, label=label)

            # Log to TSV
            if self.logger:
                self.logger.log(
                    metrics=metrics,
                    config=self.config,
                    experiment_name=experiment_name,
                    fold_label=label,
                )

            all_metrics.append({"fold": label, **metrics})

        # Summary
        if verbose and all_metrics:
            print(f"\n{'=' * 60}")
            print(f"  WALK-FORWARD SUMMARY")
            print(f"{'=' * 60}")
            for m in all_metrics:
                bss = m.get("brier_skill_score", 0)
                sign = "+" if bss and bss > 0 else ""
                print(f"  {m['fold']:<20s} BSS={sign}{bss:.5f}  "
                      f"AvgK={m.get('avg_matches', 0):.1f}  "
                      f"Trades={m.get('confident_trades', 0)}")

            bss_values = [m["brier_skill_score"] for m in all_metrics
                          if m.get("brier_skill_score") is not None]
            if bss_values:
                positive_folds = sum(1 for b in bss_values if b > 0)
                print(f"\n  Positive BSS folds: {positive_folds}/{len(bss_values)}")
                print(f"  Mean BSS: {sum(bss_values) / len(bss_values):+.5f}")
            print(f"{'=' * 60}\n")

        return all_metrics
