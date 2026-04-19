"""T7.5-5: Unit tests for cv_bss_estimator() in pattern_engine/scoring.py.

Uses synthetic data to verify CI properties and variance reduction.
All tests use numpy random seeds for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from pattern_engine.scoring import cv_bss_estimator

_RNG_SEED = 42
_N_BOOTSTRAP = 1000
_N_SAMPLES = 500


def _make_rng(seed: int = _RNG_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestCvBssEstimatorKeys:
    def test_result_has_all_required_keys(self) -> None:
        rng = _make_rng()
        actuals = rng.integers(0, 2, size=_N_SAMPLES).astype(float)
        preds = np.clip(rng.uniform(0.3, 0.7, size=_N_SAMPLES), 0.0, 1.0)

        result = cv_bss_estimator(preds, actuals, n_bootstrap=_N_BOOTSTRAP, rng_seed=_RNG_SEED)

        required = {"bss_point", "bss_cv", "ci_lower", "ci_upper", "variance_reduction", "beta", "n"}
        assert required.issubset(result.keys())

    def test_n_equals_sample_size(self) -> None:
        rng = _make_rng()
        actuals = rng.integers(0, 2, size=_N_SAMPLES).astype(float)
        preds = np.clip(rng.uniform(0.4, 0.6, size=_N_SAMPLES), 0.0, 1.0)

        result = cv_bss_estimator(preds, actuals, n_bootstrap=_N_BOOTSTRAP, rng_seed=_RNG_SEED)

        assert result["n"] == _N_SAMPLES


class TestCvBssEstimatorPerfectPredictions:
    def test_ci_lower_positive_for_strong_predictions(self) -> None:
        """Strong predictions should yield ci_lower > 0 (edge is statistically significant)."""
        rng = _make_rng()
        base_rate = 0.55
        actuals = (rng.uniform(size=_N_SAMPLES) < base_rate).astype(float)

        # Strong predictions: near 0.9 for positives, near 0.1 for negatives
        preds = np.where(actuals == 1.0, 0.90, 0.10)

        result = cv_bss_estimator(preds, actuals, n_bootstrap=_N_BOOTSTRAP, rng_seed=_RNG_SEED)

        assert result["ci_lower"] > 0.0, (
            f"Strong predictions should give ci_lower > 0, got {result['ci_lower']:.4f}"
        )
        assert result["bss_point"] > 0.0

    def test_bss_point_high_for_strong_predictions(self) -> None:
        rng = _make_rng()
        actuals = (rng.uniform(size=_N_SAMPLES) < 0.5).astype(float)
        preds = np.where(actuals == 1.0, 0.85, 0.15)

        result = cv_bss_estimator(preds, actuals, n_bootstrap=_N_BOOTSTRAP, rng_seed=_RNG_SEED)

        assert result["bss_point"] > 0.5


class TestCvBssEstimatorRandomPredictions:
    def test_ci_lower_negative_for_random_predictions(self) -> None:
        """Random predictions should yield ci_lower < 0 (no statistically significant edge)."""
        rng = _make_rng(seed=99)
        base_rate = 0.5
        actuals = (rng.uniform(size=2000) < base_rate).astype(float)
        preds = rng.uniform(0.4, 0.6, size=2000)

        result = cv_bss_estimator(preds, actuals, n_bootstrap=_N_BOOTSTRAP, rng_seed=_RNG_SEED)

        assert result["ci_lower"] < 0.0, (
            f"Random predictions should give ci_lower < 0, got {result['ci_lower']:.4f}"
        )

    def test_bss_point_near_zero_for_base_rate_predictions(self) -> None:
        """Constant base-rate predictions produce BSS ≈ 0."""
        rng = _make_rng()
        base_rate = 0.55
        actuals = (rng.uniform(size=_N_SAMPLES) < base_rate).astype(float)
        preds = np.full(_N_SAMPLES, base_rate)

        result = cv_bss_estimator(preds, actuals, n_bootstrap=_N_BOOTSTRAP, rng_seed=_RNG_SEED)

        assert abs(result["bss_point"]) < 0.05


class TestCvBssEstimatorVarianceReduction:
    def test_variance_reduction_in_range(self) -> None:
        """variance_reduction must be in [0, 1]."""
        rng = _make_rng()
        actuals = rng.integers(0, 2, size=_N_SAMPLES).astype(float)
        preds = np.clip(rng.normal(0.55, 0.1, size=_N_SAMPLES), 0.01, 0.99)

        result = cv_bss_estimator(preds, actuals, n_bootstrap=_N_BOOTSTRAP, rng_seed=_RNG_SEED)

        assert 0.0 <= result["variance_reduction"] <= 1.0

    def test_variance_reduction_positive_when_correlated(self) -> None:
        """When BS_model correlates with BS_clim, variance_reduction > 0."""
        rng = _make_rng()
        actuals = (rng.uniform(size=_N_SAMPLES) < 0.55).astype(float)
        preds = np.clip(rng.normal(0.55, 0.1, size=_N_SAMPLES), 0.01, 0.99)

        result = cv_bss_estimator(preds, actuals, n_bootstrap=_N_BOOTSTRAP, rng_seed=_RNG_SEED)

        assert result["variance_reduction"] > 0.0

    def test_ci_lower_less_than_ci_upper(self) -> None:
        rng = _make_rng()
        actuals = rng.integers(0, 2, size=_N_SAMPLES).astype(float)
        preds = np.clip(rng.normal(0.55, 0.1, size=_N_SAMPLES), 0.01, 0.99)

        result = cv_bss_estimator(preds, actuals, n_bootstrap=_N_BOOTSTRAP, rng_seed=_RNG_SEED)

        assert result["ci_lower"] < result["ci_upper"]


class TestCvBssEstimatorReproducibility:
    def test_same_seed_gives_same_result(self) -> None:
        rng = _make_rng()
        actuals = rng.integers(0, 2, size=_N_SAMPLES).astype(float)
        preds = np.clip(rng.normal(0.55, 0.1, size=_N_SAMPLES), 0.01, 0.99)

        result_a = cv_bss_estimator(preds, actuals, n_bootstrap=_N_BOOTSTRAP, rng_seed=_RNG_SEED)
        result_b = cv_bss_estimator(preds, actuals, n_bootstrap=_N_BOOTSTRAP, rng_seed=_RNG_SEED)

        assert result_a["bss_cv"] == pytest.approx(result_b["bss_cv"])
        assert result_a["ci_lower"] == pytest.approx(result_b["ci_lower"])


class TestCvBssEstimatorContract:
    def test_empty_predictions_raises(self) -> None:
        with pytest.raises(Exception):
            cv_bss_estimator(np.array([]), np.array([1.0, 0.0]))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(Exception):
            cv_bss_estimator(
                np.array([0.5, 0.6, 0.7]),
                np.array([1.0, 0.0]),
            )

    def test_too_few_bootstrap_raises(self) -> None:
        actuals = np.array([1.0, 0.0, 1.0, 0.0])
        preds = np.array([0.7, 0.3, 0.8, 0.2])
        with pytest.raises(Exception):
            cv_bss_estimator(preds, actuals, n_bootstrap=50)

    def test_invalid_ci_level_raises(self) -> None:
        actuals = np.array([1.0, 0.0, 1.0, 0.0])
        preds = np.array([0.7, 0.3, 0.8, 0.2])
        with pytest.raises(Exception):
            cv_bss_estimator(preds, actuals, ci_level=1.5)
