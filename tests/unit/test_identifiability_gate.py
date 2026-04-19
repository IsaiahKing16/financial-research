"""T7.5-3: Unit tests for identifiability_gate() in pattern_engine/diagnostics.py.

Verifies:
  - PASS when training_n / k >= min_ratio (vacuously true on production config)
  - FAIL when k > training_n / min_ratio
  - Contract guards on invalid inputs
"""

from __future__ import annotations

import pytest

from pattern_engine.diagnostics import identifiability_gate

_PRODUCTION_TOP_K = 50
_PRODUCTION_MIN_RATIO = 20.0


class TestIdentifiabilityGatePass:
    def test_production_config_passes(self) -> None:
        """Current production config: top_k=50, ~50K training rows → ratio ≈ 1000."""
        training_n = 50_000
        result = identifiability_gate(training_n=training_n, k=_PRODUCTION_TOP_K)
        assert result["verdict"] == "PASS"

    def test_small_fold_still_passes(self) -> None:
        """Even a small fold with 2000 training rows passes at k=50 (ratio=40)."""
        result = identifiability_gate(training_n=2_000, k=_PRODUCTION_TOP_K)
        assert result["verdict"] == "PASS"
        assert result["ratio"] == pytest.approx(40.0)

    def test_ratio_at_exact_threshold_passes(self) -> None:
        """Ratio == min_ratio exactly is a PASS."""
        training_n = int(_PRODUCTION_MIN_RATIO * _PRODUCTION_TOP_K)  # 1000
        result = identifiability_gate(training_n=training_n, k=_PRODUCTION_TOP_K)
        assert result["verdict"] == "PASS"
        assert result["ratio"] == pytest.approx(_PRODUCTION_MIN_RATIO)


class TestIdentifiabilityGateFail:
    def test_fails_when_k_exceeds_ratio(self) -> None:
        """k=100 with training_n=1000 → ratio=10 < 20 → FAIL."""
        result = identifiability_gate(training_n=1_000, k=100)
        assert result["verdict"] == "FAIL"

    def test_fails_just_below_threshold(self) -> None:
        """training_n / k = 19.9 → FAIL."""
        training_n = 199
        k = 10
        result = identifiability_gate(training_n=training_n, k=k)
        assert result["verdict"] == "FAIL"
        assert result["ratio"] == pytest.approx(19.9)

    def test_custom_min_ratio_respected(self) -> None:
        """Custom min_ratio of 50 fails where default 20 would pass."""
        result = identifiability_gate(training_n=1_000, k=_PRODUCTION_TOP_K, min_ratio=50.0)
        assert result["verdict"] == "FAIL"


class TestIdentifiabilityGateResultStructure:
    def test_result_has_required_keys(self) -> None:
        result = identifiability_gate(training_n=10_000, k=_PRODUCTION_TOP_K)
        required = {"verdict", "ratio", "training_n", "effective_params", "min_ratio"}
        assert required.issubset(result.keys())

    def test_effective_params_equals_k(self) -> None:
        """effective_params is k per Hastie et al. effective df for local methods."""
        k = 50
        result = identifiability_gate(training_n=10_000, k=k)
        assert result["effective_params"] == k

    def test_ratio_formula_correct(self) -> None:
        training_n = 3_000
        k = 75
        result = identifiability_gate(training_n=training_n, k=k)
        assert result["ratio"] == pytest.approx(training_n / k)

    def test_min_ratio_preserved_in_result(self) -> None:
        custom_ratio = 35.0
        result = identifiability_gate(training_n=10_000, k=_PRODUCTION_TOP_K, min_ratio=custom_ratio)
        assert result["min_ratio"] == pytest.approx(custom_ratio)


class TestIdentifiabilityGateContract:
    def test_zero_training_n_raises(self) -> None:
        with pytest.raises(Exception):
            identifiability_gate(training_n=0, k=_PRODUCTION_TOP_K)

    def test_zero_k_raises(self) -> None:
        with pytest.raises(Exception):
            identifiability_gate(training_n=10_000, k=0)

    def test_negative_training_n_raises(self) -> None:
        with pytest.raises(Exception):
            identifiability_gate(training_n=-1, k=_PRODUCTION_TOP_K)

    def test_zero_min_ratio_raises(self) -> None:
        with pytest.raises(Exception):
            identifiability_gate(training_n=10_000, k=_PRODUCTION_TOP_K, min_ratio=0.0)
