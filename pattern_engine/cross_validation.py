"""
cross_validation.py — Cross-model validation framework.

Runs multiple PatternEngine configs on the same data and compares
their outputs for consistency, agreement, and ensemble consensus.

Capabilities:
  - Agreement matrix: which configs agree on direction per ticker/date
  - Disagreement flagging: surface predictions where models diverge
  - Consensus signals: only emit when N-of-M models agree
  - Integrity checks: determinism, persistence round-trip, calibration sanity
"""

import tempfile
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

from pattern_engine.config import EngineConfig
from pattern_engine.engine import PatternEngine, PredictionResult
from pattern_engine.scoring import brier_skill_score


@dataclass
class CrossValidationResult:
    """Container for cross-validation outputs."""
    configs: list
    predictions: list  # list of PredictionResult per config
    query_db: pd.DataFrame = None
    _agreement_df: pd.DataFrame = None
    _consensus_df: pd.DataFrame = None


class CrossValidator:
    """Run multiple PatternEngine configs and cross-check outputs.

    Models can validate each other — disagreement between configs flags
    low-confidence predictions, while consensus across configs produces
    ultra-high-conviction signals.

    Args:
        configs: list of EngineConfig instances to compare
        consensus_threshold: minimum configs that must agree to emit a signal
            (default: majority vote, >50% of models)
    """

    def __init__(self, configs: list[EngineConfig],
                 consensus_threshold: int = None):
        self.configs = configs
        self.consensus_threshold = consensus_threshold or (len(configs) // 2 + 1)
        self._result = None

    def run(self, train_db: pd.DataFrame, val_db: pd.DataFrame,
            verbose: int = 1) -> CrossValidationResult:
        """Fit all models on train_db, predict on val_db, compare outputs.

        Args:
            train_db: training DataFrame
            val_db: validation/query DataFrame
            verbose: 0=silent, 1=progress

        Returns:
            CrossValidationResult with all predictions and comparison data
        """
        predictions = []

        for i, config in enumerate(self.configs):
            if verbose:
                print(f"  CrossValidator: fitting config {i + 1}/{len(self.configs)} "
                      f"(d={config.max_distance}, cal={config.calibration_method})")

            engine = PatternEngine(config)
            engine.fit(train_db)
            result = engine.predict(val_db, verbose=0)
            predictions.append(result)

        self._result = CrossValidationResult(
            configs=self.configs,
            predictions=predictions,
            query_db=val_db,
        )

        if verbose:
            self._print_summary()

        return self._result

    def agreement_matrix(self) -> pd.DataFrame:
        """Build NxM matrix: rows=query indices, cols=config labels, values=signal.

        Returns:
            DataFrame where each column is a config's signal array
        """
        if self._result is None:
            raise RuntimeError("Call run() before accessing results.")

        data = {}
        for i, (config, pred) in enumerate(
            zip(self._result.configs, self._result.predictions)
        ):
            label = f"cfg{i}_d{config.max_distance}_cal{config.calibration_method}"
            data[label] = pred.signals

        df = pd.DataFrame(data)

        # Add ticker info if available
        if self._result.query_db is not None and "Ticker" in self._result.query_db.columns:
            df.insert(0, "Ticker", self._result.query_db["Ticker"].values)
        if self._result.query_db is not None and "Date" in self._result.query_db.columns:
            df.insert(0, "Date", self._result.query_db["Date"].values)

        return df

    def flag_disagreements(self, min_spread: float = 0.15) -> pd.DataFrame:
        """Return rows where probability spread across models exceeds threshold.

        High spread = models disagree = low confidence regardless of
        individual model scores.

        Args:
            min_spread: minimum probability spread to flag (default 0.15)

        Returns:
            DataFrame of flagged rows with spread and per-model probabilities
        """
        if self._result is None:
            raise RuntimeError("Call run() before accessing results.")

        n_queries = len(self._result.predictions[0].calibrated_probabilities)
        rows = []

        for idx in range(n_queries):
            probs = [
                float(pred.calibrated_probabilities[idx])
                for pred in self._result.predictions
            ]
            spread = max(probs) - min(probs)

            if spread >= min_spread:
                row = {"query_idx": idx, "spread": round(spread, 4)}

                # Add ticker if available
                if (self._result.query_db is not None and
                        "Ticker" in self._result.query_db.columns):
                    row["Ticker"] = self._result.query_db.iloc[idx]["Ticker"]

                for i, p in enumerate(probs):
                    row[f"cfg{i}_prob"] = round(p, 4)

                signals = [
                    self._result.predictions[i].signals[idx]
                    for i in range(len(self._result.predictions))
                ]
                row["signal_agreement"] = len(set(signals)) == 1
                rows.append(row)

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values("spread", ascending=False).reset_index(drop=True)
        return df

    def consensus_signals(self) -> pd.DataFrame:
        """Only emit signals where >= consensus_threshold models agree.

        Ultra-high-conviction filter: if 3 of 4 model variants all say BUY,
        emit BUY. Otherwise emit HOLD.

        Returns:
            DataFrame with Ticker, ConsensusSignal, AgreementRatio, AvgProbability
        """
        if self._result is None:
            raise RuntimeError("Call run() before accessing results.")

        n_queries = len(self._result.predictions[0].calibrated_probabilities)
        n_models = len(self._result.predictions)
        rows = []

        for idx in range(n_queries):
            signals = [pred.signals[idx] for pred in self._result.predictions]
            probs = [
                float(pred.calibrated_probabilities[idx])
                for pred in self._result.predictions
            ]

            # Count votes per signal type
            from collections import Counter
            votes = Counter(signals)
            top_signal, top_count = votes.most_common(1)[0]

            if top_count >= self.consensus_threshold:
                consensus = top_signal
            else:
                consensus = "HOLD"

            row = {
                "ConsensusSignal": consensus,
                "AgreementRatio": round(top_count / n_models, 2),
                "AvgProbability": round(np.mean(probs), 4),
                "ProbSpread": round(max(probs) - min(probs), 4),
                "VotesFor": top_count,
                "TotalModels": n_models,
            }

            if (self._result.query_db is not None and
                    "Ticker" in self._result.query_db.columns):
                row["Ticker"] = self._result.query_db.iloc[idx]["Ticker"]

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort: consensus BUY/SELL first, then by agreement
        if len(df) > 0:
            df["_sort"] = df["ConsensusSignal"].map({"BUY": 0, "SELL": 1, "HOLD": 2})
            df = df.sort_values(
                ["_sort", "AgreementRatio", "AvgProbability"],
                ascending=[True, False, False]
            ).drop(columns=["_sort"]).reset_index(drop=True)

        return df

    @staticmethod
    def integrity_check(config: EngineConfig, train_db: pd.DataFrame,
                        val_db: pd.DataFrame, verbose: int = 1) -> dict:
        """Verify pipeline consistency and correctness.

        Checks:
        1. Determinism: fit twice with same data → identical predictions
        2. Persistence: save/load round-trip → identical predictions
        3. Calibration sanity: train-set probabilities are reasonable
        4. BSS sanity: score is within expected range

        Args:
            config: EngineConfig to test
            train_db: training DataFrame
            val_db: validation DataFrame
            verbose: 0=silent, 1=progress

        Returns:
            dict with check results (all True = healthy pipeline)
        """
        results = {}

        # Check 1: Determinism
        if verbose:
            print("  Integrity: determinism check...")
        engine1 = PatternEngine(config)
        engine1.fit(train_db)
        pred1 = engine1.predict(val_db, verbose=0)

        engine2 = PatternEngine(config)
        engine2.fit(train_db)
        pred2 = engine2.predict(val_db, verbose=0)

        probs_match = np.allclose(
            pred1.calibrated_probabilities,
            pred2.calibrated_probabilities,
            atol=1e-10,
        )
        signals_match = np.array_equal(pred1.signals, pred2.signals)
        results["determinism_probs"] = bool(probs_match)
        results["determinism_signals"] = bool(signals_match)

        # Check 2: Persistence round-trip
        if verbose:
            print("  Integrity: persistence round-trip...")
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_engine.pkl"
            engine1.save(str(save_path))
            engine_loaded = PatternEngine.load(str(save_path))
            pred_loaded = engine_loaded.predict(val_db, verbose=0)

            persistence_match = np.allclose(
                pred1.calibrated_probabilities,
                pred_loaded.calibrated_probabilities,
                atol=1e-10,
            )
            results["persistence_round_trip"] = bool(persistence_match)

        # Check 3: Calibration sanity
        if verbose:
            print("  Integrity: calibration sanity...")
        cal_probs = pred1.calibrated_probabilities
        results["calibration_in_range"] = bool(
            np.all(cal_probs >= 0) and np.all(cal_probs <= 1)
        )
        results["calibration_not_degenerate"] = bool(
            np.std(cal_probs) > 0.001  # Not all the exact same value
        )

        # Check 4: BSS sanity
        if verbose:
            print("  Integrity: BSS sanity check...")
        horizon = config.projection_horizon
        if horizon in val_db.columns:
            y_true = val_db[horizon].values
            bss = brier_skill_score(y_true, cal_probs)
            results["bss_value"] = round(bss, 6)
            results["bss_not_catastrophic"] = bool(bss > -1.0)  # Worse than -1 = broken
        else:
            results["bss_value"] = None
            results["bss_not_catastrophic"] = None
            results["bss_skipped"] = True

        # Summary — None values indicate skipped checks, flag them
        checks = [v for v in results.values() if isinstance(v, bool)]
        results["all_passed"] = all(checks)

        if verbose:
            status = "PASS" if results["all_passed"] else "FAIL"
            print(f"  Integrity check: {status} "
                  f"({sum(checks)}/{len(checks)} checks passed)")
            for k, v in results.items():
                if k == "all_passed":
                    continue
                marker = "OK" if v is True else ("FAIL" if v is False else str(v))
                print(f"    {k}: {marker}")

        return results

    def _print_summary(self):
        """Print cross-validation summary."""
        n_queries = len(self._result.predictions[0].signals)
        n_models = len(self._result.predictions)

        print(f"\n  Cross-Validation Summary ({n_models} models, {n_queries} queries)")
        print(f"  {'─' * 50}")

        for i, (config, pred) in enumerate(
            zip(self._result.configs, self._result.predictions)
        ):
            print(f"  Config {i}: d={config.max_distance} "
                  f"cal={config.calibration_method} → "
                  f"BUY={pred.buy_count} SELL={pred.sell_count} "
                  f"HOLD={pred.hold_count}")

        # Agreement rate
        all_signals = np.column_stack(
            [pred.signals for pred in self._result.predictions]
        )
        unanimous = sum(
            1 for row in all_signals if len(set(row)) == 1
        )
        print(f"\n  Unanimous agreement: {unanimous}/{n_queries} "
              f"({unanimous / n_queries * 100:.1f}%)")

        # Probability spread stats
        spreads = []
        for idx in range(n_queries):
            probs = [
                float(pred.calibrated_probabilities[idx])
                for pred in self._result.predictions
            ]
            spreads.append(max(probs) - min(probs))
        spreads = np.array(spreads)
        print(f"  Prob spread: mean={spreads.mean():.4f} "
              f"max={spreads.max():.4f} "
              f"p90={np.percentile(spreads, 90):.4f}")
