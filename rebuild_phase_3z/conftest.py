"""
rebuild_phase_3z/conftest.py — pytest configuration for Phase 3Z rebuild tests.

Registers custom marks to suppress PytestUnknownMarkWarning.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (excluded from standard CI; run with -m slow)",
    )
