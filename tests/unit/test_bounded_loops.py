"""Verify bounded loop discipline per PRD §7A (Power of 10 Rule 2).

No `while` loops should exist in production code. All iteration bounds
are expressed as MAX_* constants in pattern_engine/config.py.
"""
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent.parent


def test_no_while_loops_in_production_code():
    """Zero `while` loops in pattern_engine/ or trading_system/ production code.

    This test is the CI enforcement for Power of 10 Rule 2 (Bounded Loops).
    If this test fails, a while loop was introduced without review.
    """
    dirs_to_check = [
        REPO_ROOT / "pattern_engine",
        REPO_ROOT / "trading_system",
    ]
    violations = []
    while_pattern = re.compile(r"^\s*while\s+", re.MULTILINE)

    for d in dirs_to_check:
        for py_file in d.rglob("*.py"):
            # Skip test files and __pycache__
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
            text = py_file.read_text(encoding="utf-8", errors="replace")
            for match in while_pattern.finditer(text):
                line_num = text[:match.start()].count("\n") + 1
                violations.append(f"{py_file.relative_to(REPO_ROOT)}:{line_num}")

    assert not violations, (
        f"Found {len(violations)} while loop(s) in production code "
        f"(Power of 10 Rule 2 violation):\n" + "\n".join(violations)
    )


def test_max_constants_exist():
    """MAX_* constants must be defined in pattern_engine/config.py."""
    from pattern_engine import config

    required_constants = [
        "MAX_CALIBRATION_ITER",
        "MAX_ORDER_RETRIES",
        "MAX_WALK_FORWARD_FOLDS",
        "MAX_HNSW_ELEMENTS",
        "MAX_BATCH_ITER",
    ]
    missing = [c for c in required_constants if not hasattr(config, c)]
    assert not missing, f"Missing MAX_* constants in config.py: {missing}"


def test_max_constants_are_positive_ints():
    """All MAX_* constants are positive integers."""
    from pattern_engine import config

    for name in dir(config):
        if name.startswith("MAX_"):
            value = getattr(config, name)
            assert isinstance(value, int) and value > 0, (
                f"config.{name} = {value!r} — expected positive int"
            )
