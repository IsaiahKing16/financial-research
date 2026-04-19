"""T7.5-4: Regression guard — HMM look-ahead audit.

Audit result (2026-04-19):
  grep hmmlearn pattern_engine/ trading_system/       → 0 hits
  grep smoothed_marginal pattern_engine/ trading_system/ → 0 hits
  grep predict_proba pattern_engine/ trading_system/   → 1 hit: matcher.py:80
      (sklearn LogisticRegression.predict_proba — Platt calibrator, forward-only, correct)

Verdict: G7.5-4 PASS — no HMM look-ahead contamination present.
"""

import pathlib


_PRODUCTION_ROOTS = [
    pathlib.Path("pattern_engine"),
    pathlib.Path("trading_system"),
]


def _collect_sources() -> list[tuple[pathlib.Path, str]]:
    sources: list[tuple[pathlib.Path, str]] = []
    for root in _PRODUCTION_ROOTS:
        if root.exists():
            for py in sorted(root.rglob("*.py")):
                sources.append((py, py.read_text(encoding="utf-8")))
    return sources


def test_hmmlearn_not_in_production_source() -> None:
    """hmmlearn must never appear in production paths (Kim smoother is a look-ahead trap)."""
    for path, text in _collect_sources():
        assert "hmmlearn" not in text, (
            f"hmmlearn found in {path} — use statsmodels filtered_marginal_probabilities instead"
        )


def test_smoothed_marginal_not_in_production_source() -> None:
    """smoothed_marginal_probabilities is a look-ahead method and must not appear."""
    for path, text in _collect_sources():
        assert "smoothed_marginal" not in text, (
            f"smoothed_marginal found in {path} — forward-only filtered_marginal_probabilities required"
        )


def test_predict_proba_only_in_matcher() -> None:
    """predict_proba must appear only in matcher.py (Platt calibrator — forward-only).

    Any new predict_proba call in pattern_engine must be audited for look-ahead risk.
    """
    violations: list[str] = []
    for path, text in _collect_sources():
        if "predict_proba" in text:
            if path.name != "matcher.py":
                violations.append(str(path))
    assert not violations, (
        f"predict_proba found outside matcher.py in: {violations!r}. "
        "Audit for HMM look-ahead contamination before merging."
    )
