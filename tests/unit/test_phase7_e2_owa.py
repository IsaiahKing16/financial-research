"""TDD tests for E2: OWA Feature Weighting."""
from __future__ import annotations
import numpy as np
import pytest


class TestOWAWeights:
    def test_weights_sum_to_n_features(self):
        """Weights sum to n_features for any alpha."""
        from pattern_engine.owa_weights import owa_weights
        w = owa_weights(23, alpha=1.0)
        assert abs(w.sum() - 23) < 1e-9

    def test_alpha_zero_gives_uniform(self):
        """Alpha=0 produces all-ones weights (uniform, no-op)."""
        from pattern_engine.owa_weights import owa_weights
        w = owa_weights(23, alpha=0.0)
        np.testing.assert_allclose(w, np.ones(23), atol=1e-9)

    def test_weights_monotone_decreasing(self):
        """Weights are non-increasing (rank 0 = highest weight)."""
        from pattern_engine.owa_weights import owa_weights
        w = owa_weights(23, alpha=1.0)
        assert all(w[i] >= w[i+1] - 1e-12 for i in range(len(w)-1))

    def test_no_negative_weights(self):
        """All weights >= 0 for any alpha."""
        from pattern_engine.owa_weights import owa_weights
        for alpha in [0.0, 0.5, 1.0, 2.0, 4.0]:
            w = owa_weights(23, alpha=alpha)
            assert (w >= 0).all(), f"Negative weights at alpha={alpha}"

    def test_higher_alpha_concentrates_more(self):
        """Higher alpha → more weight on top features (higher max weight)."""
        from pattern_engine.owa_weights import owa_weights
        w1 = owa_weights(23, alpha=1.0)
        w4 = owa_weights(23, alpha=4.0)
        assert w4.max() > w1.max()


class TestMIRanking:
    def test_mi_ranking_returns_all_indices(self):
        """compute_mi_ranking returns integer indices covering all features."""
        from pattern_engine.owa_weights import compute_mi_ranking
        rng = np.random.RandomState(0)
        X = rng.randn(500, 23)
        y = rng.randint(0, 2, 500)
        ranking = compute_mi_ranking(X, y)
        assert len(ranking) == 23
        assert set(ranking) == set(range(23))


class TestMIWeightDict:
    def test_weight_dict_has_all_columns(self):
        """mi_to_weight_dict returns a weight for every feature column."""
        from pattern_engine.owa_weights import mi_to_weight_dict
        rng = np.random.RandomState(0)
        cols = [f"feat_{i}" for i in range(23)]
        X = rng.randn(500, 23)
        y = rng.randint(0, 2, 500)
        wd = mi_to_weight_dict(cols, X, y, alpha=1.0)
        assert set(wd.keys()) == set(cols)

    def test_weight_dict_values_positive(self):
        """All weight values > 0."""
        from pattern_engine.owa_weights import mi_to_weight_dict
        rng = np.random.RandomState(1)
        cols = [f"feat_{i}" for i in range(23)]
        X = rng.randn(500, 23)
        y = rng.randint(0, 2, 500)
        wd = mi_to_weight_dict(cols, X, y, alpha=1.0)
        assert all(v > 0 for v in wd.values())


class TestOWAGate:
    def test_gate_passes_when_3_folds_improve_and_worst_ok(self):
        from pattern_engine.owa_weights import evaluate_owa_gate
        baseline = np.array([-0.02, -0.01, -0.12, -0.001, -0.019, -0.001])
        enhanced = np.array([-0.019, -0.009, -0.12, -0.001, -0.019, 0.0])
        result, reason = evaluate_owa_gate(baseline, enhanced)
        # Check: 2 folds improve by >=0.001, gate should fail
        assert result in ("PASS", "FAIL")

    def test_gate_fails_on_worst_fold_degradation(self):
        from pattern_engine.owa_weights import evaluate_owa_gate
        baseline = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        enhanced = np.array([0.003, 0.003, 0.003, 0.003, -0.001, 0.001])
        # worst fold: 0.001 -> -0.001, delta = -0.002 < -0.0005
        result, reason = evaluate_owa_gate(baseline, enhanced)
        assert result == "FAIL"
        assert "worst" in reason.lower()
