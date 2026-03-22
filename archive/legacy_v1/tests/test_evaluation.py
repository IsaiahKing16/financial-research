"""Tests for pattern_engine.evaluation."""

import numpy as np
import pytest
from pattern_engine.evaluation import (
    evaluate_from_signals, evaluate_probabilistic, print_metrics,
)


class TestEvaluateFromSignals:
    def test_basic_evaluation(self):
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 1])
        probs = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.6, 0.4, 0.75])
        signals = np.array(["BUY", "SELL", "BUY", "SELL", "BUY", "HOLD", "HOLD", "BUY"])
        result = evaluate_from_signals(y_true, probs, signals)

        assert "total_samples" in result
        assert result["total_samples"] == 8
        assert "accuracy_all" in result
        assert "confident_trades" in result
        assert result["confident_trades"] == 6  # 5 BUY + 1 SELL (- 2 HOLD = wait, SELL counts)
        assert 0 <= result["accuracy_all"] <= 1

    def test_no_confident_trades(self):
        y_true = np.array([1, 0, 1])
        probs = np.array([0.6, 0.4, 0.55])
        signals = np.array(["HOLD", "HOLD", "HOLD"])
        result = evaluate_from_signals(y_true, probs, signals)
        assert result["confident_trades"] == 0
        assert result["accuracy_confident"] == 0.0

    def test_all_confident(self):
        y_true = np.array([1, 0, 1])
        probs = np.array([0.9, 0.1, 0.8])
        signals = np.array(["BUY", "SELL", "BUY"])
        result = evaluate_from_signals(y_true, probs, signals)
        assert result["confident_trades"] == 3
        assert result["confident_pct"] == 1.0


class TestEvaluateProbabilistic:
    def test_full_evaluation(self):
        rng = np.random.RandomState(42)
        n = 50
        y_true_binary = rng.randint(0, 2, n)
        y_true_returns = rng.randn(n) * 0.02
        probs = rng.rand(n)
        signals = np.where(probs > 0.65, "BUY",
                           np.where(probs < 0.35, "SELL", "HOLD"))
        ensembles = [rng.randn(10) * 0.02 for _ in range(n)]

        result = evaluate_probabilistic(
            y_true_binary, y_true_returns, probs, ensembles, signals,
            horizon_label="fwd_7d"
        )

        assert "brier_score" in result
        assert "brier_skill_score" in result
        assert "calibration_buckets" in result
        assert "horizon" in result
        assert result["horizon"] == "fwd_7d"
        assert 0 <= result["brier_score"] <= 1


class TestPrintMetrics:
    def test_does_not_raise(self, capsys):
        metrics = {
            "total_samples": 100,
            "accuracy_all": 0.55,
            "confident_trades": 30,
            "confident_pct": 0.3,
            "accuracy_confident": 0.6,
            "precision_confident": 0.65,
            "f1_confident": 0.62,
            "brier_score": 0.24,
            "brier_skill_score": 0.04,
            "crps": None,
            "calibration_buckets": [
                {"pred_range": "0.6-0.8", "n": 20, "pred_prob": 0.7,
                 "actual_rate": 0.68, "gap": -0.02},
            ],
            "horizon": "fwd_7d",
        }
        print_metrics(metrics, label="test")
        captured = capsys.readouterr()
        assert "PROBABILISTIC EVALUATION" in captured.out
        assert "Brier Score" in captured.out
