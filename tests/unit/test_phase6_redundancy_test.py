"""
tests/unit/test_phase6_redundancy_test.py

Unit tests for the pure-logic function in scripts/phase6_redundancy_test.py:
  - _evaluate_redundancy_gate: classifies 22D vs 23D fold-win counts as DROP / KEEP
"""
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase6_redundancy_test import _evaluate_redundancy_gate


# Gate rule: 22D wins ≥ 3/6 → DROP body_position; otherwise KEEP.

def test_drop_at_exactly_3():
    """22D wins 3/6 folds → DROP (equality with threshold counts as pass)."""
    bss_23d = [0.010, 0.010, 0.010, -0.010, -0.010, -0.010]
    bss_22d = [0.009, 0.009, 0.009,  0.000,  0.000,  0.000]
    # 22D wins folds 3, 4, 5 = 3/6
    assert _evaluate_redundancy_gate(bss_23d, bss_22d) == "DROP"


def test_drop_at_more_than_3():
    """22D wins 5/6 folds → DROP."""
    bss_23d = [0.005, 0.005, 0.005, 0.005, 0.005, 0.010]
    bss_22d = [0.006, 0.006, 0.006, 0.006, 0.006, 0.009]
    # 22D wins folds 0–4 = 5/6
    assert _evaluate_redundancy_gate(bss_23d, bss_22d) == "DROP"


def test_keep_at_2_wins():
    """22D wins only 2/6 folds → KEEP."""
    bss_23d = [0.010, 0.010, 0.010, 0.010, -0.010, -0.010]
    bss_22d = [0.009, 0.009, 0.009, 0.009,  0.000,  0.000]
    # 22D wins folds 4, 5 = 2/6
    assert _evaluate_redundancy_gate(bss_23d, bss_22d) == "KEEP"


def test_keep_at_zero_wins():
    """22D never beats 23D → KEEP."""
    bss_23d = [0.01] * 6
    bss_22d = [0.00] * 6
    assert _evaluate_redundancy_gate(bss_23d, bss_22d) == "KEEP"


def test_drop_at_all_6_wins():
    """22D wins all 6 folds → DROP."""
    bss_23d = [0.00] * 6
    bss_22d = [0.01] * 6
    assert _evaluate_redundancy_gate(bss_23d, bss_22d) == "DROP"


def test_tie_does_not_count_as_22d_win():
    """Exact tie (22D BSS == 23D BSS) does NOT count as a 22D win."""
    bss_23d = [0.010, 0.010, 0.010, 0.010, 0.010, 0.005]
    bss_22d = [0.010, 0.010, 0.010, 0.010, 0.010, 0.006]
    # 22D strictly beats 23D only on fold 5 = 1/6 → KEEP
    assert _evaluate_redundancy_gate(bss_23d, bss_22d) == "KEEP"
