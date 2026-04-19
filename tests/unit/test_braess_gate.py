"""T7.5-2: Unit tests for braess_gate() in pattern_engine/diagnostics.py.

run_walkforward is mocked — no real 52T walk-forward compute in unit tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from pattern_engine.diagnostics import braess_gate

_TARGET = "pattern_engine.diagnostics.run_walkforward"

_COLS_WITH = [f"f{i}" for i in range(23)]
_COLS_WITHOUT = [f"f{i}" for i in range(8)]

_PASS_THRESHOLD = 4  # wins_with >= 4/6 → PASS


def _make_wf_result(bss_vals: list[float]) -> dict[str, Any]:
    return {
        "fold_results": [{"bss": b} for b in bss_vals],
        "mean_bss": sum(bss_vals) / len(bss_vals),
        "positive_folds": sum(1 for b in bss_vals if b > 0),
    }


class TestBraessGatePass:
    def test_returns_pass_when_with_wins_four_of_six(self) -> None:
        bss_with    = [0.01, 0.02, 0.03, 0.005, -0.01, 0.01]
        bss_without = [0.005, 0.01, 0.01, 0.001, -0.005, 0.02]

        with patch(_TARGET, side_effect=[
            _make_wf_result(bss_with),
            _make_wf_result(bss_without),
        ]):
            result = braess_gate(None, _COLS_WITH, _COLS_WITHOUT)  # type: ignore[arg-type]

        assert result["verdict"] == "PASS"
        assert result["wins_with"] >= _PASS_THRESHOLD

    def test_returns_six_fold_deltas(self) -> None:
        bss_with    = [0.01, 0.02, 0.03, 0.005, -0.01, 0.01]
        bss_without = [0.005, 0.01, 0.01, 0.001, -0.005, 0.02]

        with patch(_TARGET, side_effect=[
            _make_wf_result(bss_with),
            _make_wf_result(bss_without),
        ]):
            result = braess_gate(None, _COLS_WITH, _COLS_WITHOUT)  # type: ignore[arg-type]

        assert len(result["fold_deltas"]) == 6

    def test_fold_deltas_correct_sign(self) -> None:
        bss_with    = [0.02, 0.01, 0.03, 0.01, -0.01, 0.01]
        bss_without = [0.01, 0.02, 0.01, 0.005, -0.005, 0.005]

        with patch(_TARGET, side_effect=[
            _make_wf_result(bss_with),
            _make_wf_result(bss_without),
        ]):
            result = braess_gate(None, _COLS_WITH, _COLS_WITHOUT)  # type: ignore[arg-type]

        expected_deltas = [w - wo for w, wo in zip(bss_with, bss_without, strict=True)]
        for actual, expected in zip(result["fold_deltas"], expected_deltas, strict=True):
            assert abs(actual - expected) < 1e-12


class TestBraessGateFail:
    def test_returns_fail_when_with_wins_three_or_fewer(self) -> None:
        bss_with    = [0.01, -0.01, 0.01, -0.02, -0.03, 0.005]
        bss_without = [0.02,  0.01, 0.02,  0.01,  0.01, 0.001]

        with patch(_TARGET, side_effect=[
            _make_wf_result(bss_with),
            _make_wf_result(bss_without),
        ]):
            result = braess_gate(None, _COLS_WITH, _COLS_WITHOUT)  # type: ignore[arg-type]

        assert result["verdict"] == "FAIL"
        assert result["wins_with"] < _PASS_THRESHOLD

    def test_wins_with_equals_three_is_fail(self) -> None:
        bss_with    = [0.02, 0.02, 0.02, -0.02, -0.02, -0.02]
        bss_without = [0.01, 0.01, 0.01,  0.01,  0.01,  0.01]

        with patch(_TARGET, side_effect=[
            _make_wf_result(bss_with),
            _make_wf_result(bss_without),
        ]):
            result = braess_gate(None, _COLS_WITH, _COLS_WITHOUT)  # type: ignore[arg-type]

        assert result["verdict"] == "FAIL"
        assert result["wins_with"] == 3


class TestBraessGateContract:
    def test_empty_feature_cols_with_raises(self) -> None:
        with pytest.raises(Exception):
            braess_gate(None, [], _COLS_WITHOUT)  # type: ignore[arg-type]

    def test_empty_feature_cols_without_raises(self) -> None:
        with pytest.raises(Exception):
            braess_gate(None, _COLS_WITH, [])  # type: ignore[arg-type]

    def test_nan_bss_raises_runtime_error(self) -> None:
        import math
        bss_with    = [0.01, float("nan"), 0.01, 0.01, 0.01, 0.01]
        bss_without = [0.005, 0.01, 0.005, 0.005, 0.005, 0.005]

        with patch(_TARGET, side_effect=[
            _make_wf_result(bss_with),
            _make_wf_result(bss_without),
        ]):
            with pytest.raises(RuntimeError, match="NaN BSS"):
                braess_gate(None, _COLS_WITH, _COLS_WITHOUT)  # type: ignore[arg-type]

    def test_result_contains_required_keys(self) -> None:
        bss_with    = [0.01, 0.02, 0.03, 0.01, 0.01, 0.01]
        bss_without = [0.005, 0.01, 0.01, 0.005, 0.005, 0.005]

        with patch(_TARGET, side_effect=[
            _make_wf_result(bss_with),
            _make_wf_result(bss_without),
        ]):
            result = braess_gate(None, _COLS_WITH, _COLS_WITHOUT)  # type: ignore[arg-type]

        required_keys = {"verdict", "wins_with", "fold_deltas", "mean_bss_with", "mean_bss_without", "n_folds"}
        assert required_keys.issubset(result.keys())
