"""Tests for pattern_engine.cross_validation."""

import numpy as np
import pandas as pd
import pytest

from pattern_engine.config import EngineConfig
from pattern_engine.cross_validation import CrossValidator, CrossValidationResult


class TestCrossValidator:
    @pytest.fixture
    def two_configs(self):
        """Two distinct EngineConfig variants for cross-validation."""
        return [
            EngineConfig(regime_filter=False),
            EngineConfig(max_distance=1.25, regime_filter=False),
        ]

    def test_run_returns_result(self, train_db, val_db, two_configs):
        xval = CrossValidator(two_configs)
        result = xval.run(train_db, val_db, verbose=0)
        assert isinstance(result, CrossValidationResult)
        assert len(result.predictions) == 2

    def test_predictions_same_length(self, train_db, val_db, two_configs):
        xval = CrossValidator(two_configs)
        xval.run(train_db, val_db, verbose=0)
        n = len(val_db)
        for pred in xval._result.predictions:
            assert len(pred.calibrated_probabilities) == n
            assert len(pred.signals) == n

    def test_agreement_matrix_shape(self, train_db, val_db, two_configs):
        xval = CrossValidator(two_configs)
        xval.run(train_db, val_db, verbose=0)
        matrix = xval.agreement_matrix()
        assert len(matrix) == len(val_db)
        # 2 config columns + Ticker + Date
        assert matrix.shape[1] >= 2

    def test_flag_disagreements_returns_df(self, train_db, val_db, two_configs):
        xval = CrossValidator(two_configs)
        xval.run(train_db, val_db, verbose=0)
        disagreements = xval.flag_disagreements(min_spread=0.0)
        assert isinstance(disagreements, pd.DataFrame)

    def test_consensus_signals_shape(self, train_db, val_db, two_configs):
        xval = CrossValidator(two_configs)
        xval.run(train_db, val_db, verbose=0)
        consensus = xval.consensus_signals()
        assert len(consensus) == len(val_db)
        assert "ConsensusSignal" in consensus.columns
        assert "AgreementRatio" in consensus.columns

    def test_consensus_valid_signals(self, train_db, val_db, two_configs):
        xval = CrossValidator(two_configs)
        xval.run(train_db, val_db, verbose=0)
        consensus = xval.consensus_signals()
        valid = {"BUY", "SELL", "HOLD"}
        assert set(consensus["ConsensusSignal"].unique()).issubset(valid)

    def test_consensus_threshold(self, train_db, val_db):
        """With threshold=3 and only 2 configs, everything should be HOLD."""
        configs = [
            EngineConfig(regime_filter=False),
            EngineConfig(max_distance=1.25, regime_filter=False),
        ]
        xval = CrossValidator(configs, consensus_threshold=3)
        xval.run(train_db, val_db, verbose=0)
        consensus = xval.consensus_signals()
        assert all(consensus["ConsensusSignal"] == "HOLD")


class TestIntegrityCheck:
    def test_determinism(self, train_db, val_db):
        config = EngineConfig(regime_filter=False)
        result = CrossValidator.integrity_check(config, train_db, val_db, verbose=0)
        assert result["determinism_probs"] is True
        assert result["determinism_signals"] is True

    def test_persistence_round_trip(self, train_db, val_db):
        config = EngineConfig(regime_filter=False)
        result = CrossValidator.integrity_check(config, train_db, val_db, verbose=0)
        assert result["persistence_round_trip"] is True

    def test_calibration_sanity(self, train_db, val_db):
        config = EngineConfig(regime_filter=False)
        result = CrossValidator.integrity_check(config, train_db, val_db, verbose=0)
        assert result["calibration_in_range"] is True

    def test_all_passed(self, train_db, val_db):
        config = EngineConfig(regime_filter=False)
        result = CrossValidator.integrity_check(config, train_db, val_db, verbose=0)
        assert result["all_passed"] is True
