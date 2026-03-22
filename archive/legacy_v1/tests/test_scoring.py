"""Tests for pattern_engine.scoring."""

import numpy as np
import pytest
from pattern_engine.scoring import (
    brier_score, brier_skill_score, compute_crps, compute_calibration,
)


class TestBrierScore:
    """Test Brier score computation."""

    def test_perfect_prediction(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(y_true, y_pred) == 0.0

    def test_worst_prediction(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier_score(y_true, y_pred) == 1.0

    def test_naive_prediction(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        assert brier_score(y_true, y_pred) == 0.25

    def test_range(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        y_pred = rng.rand(100)
        bs = brier_score(y_true, y_pred)
        assert 0 <= bs <= 1


class TestBrierSkillScore:
    """Test Brier Skill Score computation."""

    def test_no_skill(self):
        """Predicting base rate has BSS = 0."""
        y_true = np.array([1, 1, 1, 0, 0])
        base_rate = y_true.mean()
        y_pred = np.full(5, base_rate)
        assert abs(brier_skill_score(y_true, y_pred)) < 1e-10

    def test_perfect_skill(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_skill_score(y_true, y_pred) == 1.0

    def test_worse_than_naive(self):
        """Worse-than-baseline prediction has negative BSS."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier_skill_score(y_true, y_pred) < 0

    def test_positive_skill(self):
        """A somewhat good prediction should have positive BSS."""
        y_true = np.array([1, 0, 1, 0, 1, 1])
        y_pred = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.6])
        bss = brier_skill_score(y_true, y_pred)
        assert bss > 0

    def test_all_same_outcome(self):
        """All same outcome → BS_clim = 0 → BSS = 0."""
        y_true = np.ones(10)
        y_pred = np.ones(10) * 0.9
        assert brier_skill_score(y_true, y_pred) == 0.0


class TestCRPS:
    """Test CRPS computation (optional dependency)."""

    def test_returns_float_or_none(self):
        y_true = np.array([0.01, -0.02, 0.03])
        ensembles = [np.random.randn(20) * 0.02 for _ in range(3)]
        result = compute_crps(y_true, ensembles)
        # Either a float (scoringrules installed) or None
        assert result is None or isinstance(result, float)

    def test_empty_ensembles(self):
        y_true = np.array([0.01])
        ensembles = [np.array([])]
        result = compute_crps(y_true, ensembles)
        assert result is None


class TestCalibration:
    """Test calibration bucket computation."""

    def test_basic_buckets(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0.1, 0.15, 0.2, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9, 0.95])
        buckets = compute_calibration(y_true, y_pred, n_buckets=5)
        assert len(buckets) > 0
        for b in buckets:
            assert "pred_range" in b
            assert "n" in b
            assert "pred_prob" in b
            assert "actual_rate" in b
            assert "gap" in b

    def test_edge_case_probability_one(self):
        """Probability = 1.0 should be captured by the last bucket."""
        y_true = np.array([1, 1])
        y_pred = np.array([0.99, 1.0])
        buckets = compute_calibration(y_true, y_pred, n_buckets=5)
        total_n = sum(b["n"] for b in buckets)
        assert total_n == 2

    def test_empty_buckets_skipped(self):
        y_true = np.array([1, 1])
        y_pred = np.array([0.9, 0.95])
        buckets = compute_calibration(y_true, y_pred, n_buckets=5)
        # Only the high-probability bucket should be populated
        assert len(buckets) == 1
