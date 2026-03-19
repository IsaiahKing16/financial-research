"""
engine.py — PatternEngine: the core historical analogue matching engine.

Single entry point for the entire prediction pipeline:
  engine = PatternEngine(config)
  engine.fit(train_db)
  result = engine.predict(query_db)
  metrics = engine.evaluate(val_db)

Encapsulates the calibration double-pass inside fit():
  1. Fit scaler on training features
  2. Build NN index on scaled training data
  3. Compute regime labels for training SPY rows
  4. Run matching on train-as-query (with regime_filter matching inference)
  5. Fit Platt/isotonic calibrator on raw probabilities + ground truth
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from pattern_engine.config import EngineConfig
from pattern_engine.features import FeatureRegistry
from pattern_engine.matching import Matcher
from pattern_engine.calibration import make_calibrator
from pattern_engine.regime import RegimeLabeler
from pattern_engine.evaluation import evaluate_probabilistic
from pattern_engine.projection import generate_signal
from pattern_engine.schema import validate_train_db, validate_query_db


class PredictionResult:
    """Container for prediction outputs."""

    def __init__(self, probabilities, calibrated_probabilities, signals,
                 reasons, n_matches, mean_returns, ensemble_list):
        self.probabilities = probabilities
        self.calibrated_probabilities = calibrated_probabilities
        self.signals = signals
        self.reasons = reasons
        self.n_matches = n_matches
        self.mean_returns = mean_returns
        self.ensemble_list = ensemble_list

    @property
    def buy_count(self) -> int:
        return int((self.signals == "BUY").sum())

    @property
    def sell_count(self) -> int:
        return int((self.signals == "SELL").sum())

    @property
    def hold_count(self) -> int:
        return int((self.signals == "HOLD").sum())

    @property
    def avg_matches(self) -> float:
        return float(np.mean(self.n_matches))


class PatternEngine:
    """Historical analogue matching engine.

    Usage:
        config = EngineConfig()              # Proven research defaults
        engine = PatternEngine(config)
        engine.fit(train_db)                 # Fit scaler, NN index, calibrator
        result = engine.predict(query_db)    # Generate predictions
        metrics = engine.evaluate(val_db)    # Predict + score

        # Persistence
        engine.save("engine_state.pkl")
        engine = PatternEngine.load("engine_state.pkl")
    """

    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()
        self._features = FeatureRegistry.get(self.config.feature_set)
        self._matcher = None
        self._calibrator = None
        self._regime_labeler = None
        self._train_db = None
        self._fitted = False

    def fit(self, train_db: pd.DataFrame) -> "PatternEngine":
        """Fit all engine components on training data.

        This encapsulates the calibration double-pass:
        1. Fit scaler + NN index on training features
        2. Fit regime labeler on training SPY rows
        3. Run matching on train-as-query (calibration pass)
        4. Fit calibrator on raw probabilities from that pass

        CRITICAL: The calibration pass uses regime_filter=True (when enabled)
        to match the inference-time distribution. Mismatched distributions
        silently corrupt the calibrator.

        Args:
            train_db: Training DataFrame with feature columns

        Returns:
            self (for method chaining)
        """
        cfg = self.config

        # Schema validation at boundary
        validate_train_db(train_db, cfg.feature_set, cfg.projection_horizon)

        self._train_db = train_db.copy()
        fcols = self._features.columns

        # Step 1: Fit regime labeler
        if cfg.regime_filter:
            self._regime_labeler = RegimeLabeler(
                mode=cfg.regime_mode,
                adx_threshold=cfg.adx_threshold,
            )
            self._regime_labeler.fit(train_db)
        else:
            self._regime_labeler = None

        # Step 2: Fit matcher (scaler + NN index)
        self._matcher = Matcher(cfg)
        self._matcher.fit(train_db, fcols, regime_labeler=self._regime_labeler)

        # Step 3: Calibration double-pass
        # Run matching on train-as-query with identical filtering settings
        if cfg.calibration_method != "none":
            cal_probs, _, _, _, _, _ = self._matcher.query(
                train_db, regime_labeler=self._regime_labeler, verbose=0
            )

            # Get ground truth for training data
            horizon_binary = cfg.projection_horizon
            y_true = train_db[horizon_binary].values

            # Fit calibrator
            self._calibrator = make_calibrator(cfg.calibration_method)
            self._calibrator.fit(cal_probs, y_true)
        else:
            self._calibrator = make_calibrator("none")

        self._fitted = True
        return self

    def predict(self, query_db: pd.DataFrame, verbose: int = 1) -> PredictionResult:
        """Run analogue matching + calibration + signal generation on query data.

        Args:
            query_db: DataFrame of query rows
            verbose: 0=silent, 1=progress

        Returns:
            PredictionResult with probabilities, signals, etc.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        # Schema validation at boundary
        validate_query_db(query_db, self.config.feature_set)

        # Run matching
        raw_probs, signals, reasons, n_matches, mean_returns, ensembles = \
            self._matcher.query(query_db, regime_labeler=self._regime_labeler,
                                verbose=verbose)

        # Apply calibration
        calibrated_probs = self._calibrator.transform(raw_probs)

        # Re-generate signals using calibrated probabilities
        cfg = self.config
        cal_signals = []
        cal_reasons = []
        for i in range(len(calibrated_probs)):
            projection = {
                "probability_up": float(calibrated_probs[i]),
                "agreement": abs(float(calibrated_probs[i]) - 0.5) * 2,
                "n_matches": n_matches[i],
            }
            signal, reason = generate_signal(
                projection,
                threshold=cfg.confidence_threshold,
                min_agreement=cfg.agreement_spread,
                min_matches=cfg.min_matches,
            )
            cal_signals.append(signal)
            cal_reasons.append(reason)

        return PredictionResult(
            probabilities=raw_probs,
            calibrated_probabilities=calibrated_probs,
            signals=np.array(cal_signals),
            reasons=cal_reasons,
            n_matches=n_matches,
            mean_returns=mean_returns,
            ensemble_list=ensembles,
        )

    def evaluate(self, val_db: pd.DataFrame, verbose: int = 1) -> dict:
        """Predict on val_db and score against ground truth.

        Args:
            val_db: Validation DataFrame with feature and target columns
            verbose: 0=silent, 1=progress

        Returns:
            dict of evaluation metrics
        """
        result = self.predict(val_db, verbose=verbose)
        cfg = self.config

        horizon_binary = cfg.projection_horizon
        horizon_return = horizon_binary.replace("_up", "")

        y_true_binary = val_db[horizon_binary].values
        y_true_returns = val_db[horizon_return].values

        metrics = evaluate_probabilistic(
            y_true_binary=y_true_binary,
            y_true_returns=y_true_returns,
            probabilities=result.calibrated_probabilities,
            ensemble_returns_list=result.ensemble_list,
            signals=result.signals,
            horizon_label=horizon_return,
        )

        metrics["avg_matches"] = result.avg_matches
        metrics["buy_signals"] = result.buy_count
        metrics["sell_signals"] = result.sell_count
        metrics["hold_signals"] = result.hold_count

        return metrics

    def save(self, path: str) -> None:
        """Serialize fitted engine state to disk (atomic write).

        Uses temp+fsync+rename so a crash mid-write preserves the old file.
        Saves: config, scaler, NN index, calibrator, regime thresholds.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before save().")
        import tempfile, os

        state = {
            "version": "2.1",
            "config": self.config,
            "matcher_scaler": self._matcher.scaler,
            "matcher_nn_index": self._matcher._nn_index,
            "matcher_feature_cols": self._matcher._feature_cols,
            "calibrator": self._calibrator,
            "regime_labeler": self._regime_labeler,
            "train_db": self._train_db,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str) -> "PatternEngine":
        """Load a fitted engine from disk with validation.

        Raises ValueError on corrupted or incompatible state files.

        Returns:
            Fitted PatternEngine instance ready for predict()/evaluate()
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Engine state not found: {path}")

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ImportError) as e:
            raise ValueError(f"Corrupted engine state at {path}: {e}") from e

        if not isinstance(state, dict) or "config" not in state:
            raise ValueError(f"Invalid engine state format at {path}")

        # Version check
        version = state.get("version", "1.0")
        if version not in ("2.0", "2.1"):
            raise ValueError(
                f"Incompatible engine state version {version!r} at {path}. "
                f"Re-fit and re-save the engine."
            )

        engine = cls(config=state["config"])
        engine._train_db = state["train_db"]
        engine._calibrator = state["calibrator"]
        engine._regime_labeler = state["regime_labeler"]

        # Reconstruct matcher with saved components
        engine._matcher = Matcher(state["config"])
        engine._matcher._scaler = state["matcher_scaler"]
        engine._matcher._nn_index = state["matcher_nn_index"]
        engine._matcher._feature_cols = state["matcher_feature_cols"]
        engine._matcher._train_db = state["train_db"]
        engine._matcher._fitted = True

        # Re-compute regime labels for training data (G1: defensive None set)
        if engine._regime_labeler is not None and engine._regime_labeler.fitted:
            engine._matcher._regime_labels_train = engine._regime_labeler.label(
                state["train_db"], reference_db=state["train_db"]
            )
        else:
            engine._matcher._regime_labels_train = None

        engine._features = FeatureRegistry.get(state["config"].feature_set)
        engine._fitted = True
        return engine
