"""
walkforward.py — Walk-forward validation runner.

Implements expanding-window walk-forward validation: for each fold,
trains on all data up to train_end, validates on val_start to val_end.
Each fold builds a fresh PatternEngine instance.
"""

import time

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
        cross_validate: if True, run 2 additional config variants per fold
            and log agreement rates for cross-model validation
    """

    def __init__(self, config: EngineConfig = None,
                 folds: list[dict] = None,
                 logger: ExperimentLogger = None,
                 cross_validate: bool = False):
        self.config = config or EngineConfig()
        self.folds = folds or WALKFORWARD_FOLDS
        self.logger = logger
        self.cross_validate = cross_validate

    def run(self, full_db: pd.DataFrame, experiment_name: str = "walkforward",
            verbose: int = 1, deadline_ts: float = None) -> list[dict]:
        """Run walk-forward validation across all folds.

        Args:
            full_db: complete analogue database (all dates)
            experiment_name: name for TSV logging
            verbose: 0=silent, 1=progress
            deadline_ts: absolute timestamp deadline (None = no deadline).
                If set, skips remaining folds when time budget is exhausted.

        Returns:
            list of metrics dicts, one per fold
        """
        full_db = full_db.copy()
        full_db["Date"] = pd.to_datetime(full_db["Date"])

        all_metrics = []

        for fold in self.folds:
            # Check deadline before starting an expensive fold
            if deadline_ts is not None and time.time() > deadline_ts:
                if verbose:
                    print(f"  Time budget exhausted — skipping remaining folds")
                break

            label = fold["label"]
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"  FOLD: {label}")
                print(f"  Train: <= {fold['train_end']} | Val: {fold['val_start']} -> {fold['val_end']}")
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

            # Build and evaluate engine for this fold (with error isolation)
            try:
                engine = PatternEngine(self.config)
                engine.fit(train_db)
                metrics = engine.evaluate(val_db, verbose=verbose)

                # Cross-model validation (optional)
                if self.cross_validate:
                    try:
                        from pattern_engine.cross_validation import CrossValidator
                        xval_configs = [
                            self.config,
                            self.config.replace(max_distance=self.config.max_distance * 0.9),
                            self.config.replace(calibration_method=(
                                "isotonic" if self.config.calibration_method == "platt" else "platt"
                            )),
                        ]
                        xval = CrossValidator(xval_configs)
                        xval_result = xval.run(train_db, val_db, verbose=0)
                        disagreements = xval.flag_disagreements()
                        metrics["xval_disagreement_count"] = len(disagreements)
                        metrics["xval_unanimous_pct"] = round(
                            sum(1 for pred_row in zip(*[p.signals for p in xval_result.predictions])
                                if len(set(pred_row)) == 1) / len(val_db) * 100, 1
                        )
                        if verbose:
                            print(f"  Cross-validation: {metrics['xval_unanimous_pct']:.1f}% "
                                  f"unanimous, {len(disagreements)} disagreements")
                    except Exception as xval_err:
                        if verbose:
                            print(f"  Cross-validation error (non-fatal): {xval_err}")

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

            except Exception as e:
                if verbose:
                    print(f"  FOLD {label} FAILED: {e}")
                all_metrics.append({
                    "fold": label,
                    "error": str(e),
                    "brier_skill_score": None,
                })

        # Summary
        if verbose and all_metrics:
            print(f"\n{'=' * 60}")
            print(f"  WALK-FORWARD SUMMARY")
            print(f"{'=' * 60}")
            for m in all_metrics:
                bss = m.get("brier_skill_score")
                if bss is None:
                    print(f"  {m['fold']:<20s} BSS=FAILED  "
                          f"Error={m.get('error', 'unknown')}")
                    continue
                sign = "+" if bss > 0 else ""
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
