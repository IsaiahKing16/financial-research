"""
tests/unit/test_phase6_bss_comparison.py

Unit tests for the two pure-logic functions in scripts/phase6_bss_comparison.py:
  - _find_winner_from_df: identifies smallest max_distance where all N folds have AvgK≥gate
  - _evaluate_gate: classifies fold-win counts as PASS / DRAW / FAIL
"""
import io
import pytest
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase6_bss_comparison import _find_winner_from_df, _evaluate_gate


# ── _find_winner_from_df ──────────────────────────────────────────────────────

def _make_sweep_df(avgk_by_distance: dict[float, list[float]]) -> pd.DataFrame:
    """Build a minimal sweep DataFrame with columns [max_distance, fold, AvgK]."""
    rows = []
    for d, avgks in sorted(avgk_by_distance.items()):
        for fi, k in enumerate(avgks):
            rows.append({"max_distance": d, "fold": fi, "AvgK": k})
    return pd.DataFrame(rows)


def test_find_winner_returns_smallest_passing_distance():
    """When d=2.5 is the smallest where all 6 folds pass, returns 2.5."""
    df = _make_sweep_df({
        2.0: [19.8, 21.6, 22.7, 23.9, 23.1, 23.5],   # fold 0 fails (19.8 < 20)
        2.5: [41.2, 42.2, 43.2, 43.7, 43.5, 43.0],   # all pass
        3.0: [48.6, 48.7, 49.1, 49.2, 49.1, 48.8],   # all pass (but larger)
    })
    assert _find_winner_from_df(df) == 2.5


def test_find_winner_skips_larger_when_smaller_already_passes():
    """Returns first (smallest) winner, ignoring later values."""
    df = _make_sweep_df({
        1.5: [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],  # all pass — winner
        2.5: [45.0, 45.0, 45.0, 45.0, 45.0, 45.0],  # also pass, but larger
    })
    assert _find_winner_from_df(df) == 1.5


def test_find_winner_returns_none_when_no_value_passes():
    """Returns None when no max_distance achieves AvgK≥20 on all folds."""
    df = _make_sweep_df({
        2.0: [19.8, 21.6, 22.7, 23.9, 23.1, 23.5],  # fold 0 always fails
        2.5: [19.0, 21.0, 22.0, 23.0, 23.0, 23.0],  # fold 0 still fails
    })
    assert _find_winner_from_df(df) is None


def test_find_winner_respects_custom_gate():
    """Custom avgk_gate parameter is respected."""
    df = _make_sweep_df({
        1.0: [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],  # all ≥ 10, fails ≥ 20
        2.0: [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],  # all ≥ both gates
    })
    assert _find_winner_from_df(df, avgk_gate=10) == 1.0
    assert _find_winner_from_df(df, avgk_gate=20) == 2.0


def test_find_winner_exact_boundary():
    """AvgK exactly equal to the gate (20.0) counts as passing."""
    df = _make_sweep_df({
        2.0: [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    })
    assert _find_winner_from_df(df) == 2.0


# ── _evaluate_gate ────────────────────────────────────────────────────────────

def test_evaluate_gate_pass_at_4_of_6():
    bss_a = [-0.01, 0.002, -0.003, 0.001, -0.002, 0.003]
    bss_b = [ 0.01, 0.004,  0.001, 0.002, -0.001, 0.005]
    # candle wins folds 0, 1, 2, 3, 5 = 5/6 → PASS
    assert _evaluate_gate(bss_a, bss_b) == "PASS"


def test_evaluate_gate_pass_at_exactly_4():
    bss_a = [0.010, 0.010, -0.001, -0.001, 0.010, 0.010]
    bss_b = [0.011, 0.011,  0.000,  0.001, 0.009, 0.009]
    # candle wins folds 0,1,2,3 = 4/6 → PASS
    assert _evaluate_gate(bss_a, bss_b) == "PASS"


def test_evaluate_gate_draw_at_3_of_6():
    bss_a = [0.01, 0.01, 0.01, -0.01, -0.01, -0.01]
    bss_b = [0.02, 0.02, 0.02, -0.02, -0.02, -0.02]
    # candle wins folds 3,4,5 = 3/6 → DRAW
    assert _evaluate_gate(bss_a, bss_b) == "DRAW"


def test_evaluate_gate_fail_at_2_of_6():
    bss_a = [0.01, 0.01, 0.01, 0.01, -0.01, -0.01]
    bss_b = [0.00, 0.00, 0.00, 0.00,  0.00,  0.00]
    # candle wins folds 4,5 = 2/6 → FAIL
    assert _evaluate_gate(bss_a, bss_b) == "FAIL"


def test_evaluate_gate_pass_all_6():
    bss_a = [-0.01] * 6
    bss_b = [ 0.01] * 6
    assert _evaluate_gate(bss_a, bss_b) == "PASS"


def test_evaluate_gate_fail_zero_wins():
    bss_a = [0.01] * 6
    bss_b = [0.00] * 6
    assert _evaluate_gate(bss_a, bss_b) == "FAIL"
