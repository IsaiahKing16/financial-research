"""Tests for pattern_engine/walkforward.py — walk-forward fold runner, BSS, data loading.

TDD tests written before implementation per Phase 3 Optuna plan (Tasks 1-3).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── TestBSS ──────────────────────────────────────────────────────────────────


class TestBSS:
    """Test Brier Skill Score computation."""

    def test_perfect_predictions(self):
        from pattern_engine.walkforward import _bss

        y = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        probs = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        assert _bss(probs, y) == pytest.approx(1.0)

    def test_climatological_predictions(self):
        from pattern_engine.walkforward import _bss

        y = np.array([1.0, 0.0, 1.0, 0.0])
        base_rate = y.mean()  # 0.5
        probs = np.full(len(y), base_rate)
        assert _bss(probs, y) == pytest.approx(0.0)

    def test_worse_than_climatology(self):
        from pattern_engine.walkforward import _bss

        y = np.array([1.0, 0.0, 1.0, 0.0])
        # Predict the opposite of truth
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        assert _bss(probs, y) < 0

    def test_constant_labels_returns_zero(self):
        from pattern_engine.walkforward import _bss

        y = np.ones(10)
        probs = np.full(10, 0.8)
        # base_rate=1.0, bs_ref=0.0 → should return 0.0
        assert _bss(probs, y) == pytest.approx(0.0)


# ── TestMurphyDecomposition ──────────────────────────────────────────────────


class TestMurphyDecomposition:
    """Test Murphy decomposition of Brier Score."""

    def test_uncertainty_matches_base_rate(self):
        from pattern_engine.walkforward import _murphy_decomposition

        y = np.array([1.0, 0.0] * 50)  # 50/50 split
        probs = np.random.default_rng(42).uniform(0.3, 0.7, 100)
        _, _, unc = _murphy_decomposition(probs, y)
        assert unc == pytest.approx(0.25, abs=1e-6)

    def test_perfect_predictions_have_zero_reliability(self):
        from pattern_engine.walkforward import _murphy_decomposition

        y = np.array([1.0, 0.0] * 50)
        probs = y.copy()  # perfect calibration
        rel, _, _ = _murphy_decomposition(probs, y)
        assert rel == pytest.approx(0.0, abs=1e-6)

    def test_constant_labels_return_nan(self):
        from pattern_engine.walkforward import _murphy_decomposition

        y = np.ones(50)
        probs = np.full(50, 0.9)
        rel, res, unc = _murphy_decomposition(probs, y)
        assert math.isnan(rel)
        assert math.isnan(res)


# ── TestBetaCalibrator ───────────────────────────────────────────────────────


class TestBetaCalibrator:
    """Test _BetaCalibrator wrapping betacal."""

    def test_fit_transform_returns_probabilities(self):
        from pattern_engine.walkforward import _BetaCalibrator

        rng = np.random.default_rng(42)
        raw = rng.uniform(0.1, 0.9, 200)
        y = (rng.random(200) > 0.5).astype(float)

        cal = _BetaCalibrator()
        cal.fit(raw, y)
        transformed = cal.transform(raw)

        assert transformed.shape == raw.shape
        assert np.all(transformed >= 0.0)
        assert np.all(transformed <= 1.0)


# ── TestH7HoldRegime ─────────────────────────────────────────────────────────


class TestH7HoldRegime:
    """Test H7 HOLD regime application."""

    def test_bear_rows_get_base_rate(self):
        from pattern_engine.walkforward import _apply_h7_hold_regime

        # Build minimal val_db: SPY with some bear and some bull dates
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        spy_rows = pd.DataFrame({
            "Date": dates,
            "Ticker": "SPY",
            "ret_90d": [0.02, 0.08, 0.02, 0.08, 0.02,
                        0.08, 0.02, 0.08, 0.02, 0.08],
        })
        stock_rows = pd.DataFrame({
            "Date": dates,
            "Ticker": "AAPL",
            "ret_90d": [0.1] * 10,
        })
        val_db = pd.concat([spy_rows, stock_rows], ignore_index=True)
        train_db = val_db.copy()  # not used when SPY is in val_db

        probs = np.array([0.6] * 20)  # 10 SPY + 10 AAPL
        base_rate = 0.5

        probs_hold, bear_mask = _apply_h7_hold_regime(
            val_db=val_db,
            train_db=train_db,
            base_rate=base_rate,
            probs=probs,
            threshold=0.05,
        )

        # Bear rows (ret_90d=0.02 < 0.05) should have probs replaced with base_rate
        bear_indices = np.where(bear_mask)[0]
        assert len(bear_indices) > 0, "Expected some bear rows"
        assert np.all(probs_hold[bear_mask] == base_rate)

        # Bull rows (ret_90d=0.08 >= 0.05) should preserve original probs
        bull_indices = np.where(~bear_mask)[0]
        assert len(bull_indices) > 0, "Expected some bull rows"
        assert np.all(probs_hold[~bear_mask] == 0.6)


# ── TestBuildCfg ─────────────────────────────────────────────────────────────


class TestBuildCfg:
    """Test _build_cfg configuration builder."""

    def test_locked_defaults(self):
        from pattern_engine.walkforward import _build_cfg

        cfg = _build_cfg()
        assert cfg.max_distance == 2.5
        assert cfg.top_k == 50
        assert cfg.distance_weighting == "uniform"
        assert cfg.distance_metric == "euclidean"
        assert cfg.nn_jobs == 1
        assert cfg.use_hnsw is True
        assert cfg.regime_filter is False

    def test_overrides_applied(self):
        from pattern_engine.walkforward import _build_cfg

        cfg = _build_cfg({"max_distance": 1.5, "top_k": 30})
        assert cfg.max_distance == 1.5
        assert cfg.top_k == 30

    def test_unknown_field_raises(self):
        from pattern_engine.walkforward import _build_cfg

        with pytest.raises(RuntimeError, match="unknown EngineConfig field"):
            _build_cfg({"nonexistent_field": 42})

    def test_cal_frac_stripped_and_injected(self):
        from pattern_engine.walkforward import _build_cfg

        overrides = {"cal_frac": 0.80}
        cfg = _build_cfg(overrides)
        # cal_frac is not a real EngineConfig field, so it shouldn't raise
        # It should be injected via setattr
        assert getattr(cfg, "cal_frac", None) == 0.80
        # Original dict should NOT be mutated
        assert "cal_frac" in overrides


# ── TestRunFold ──────────────────────────────────────────────────────────────


class TestRunFold:
    """Test run_fold() with mocked PatternMatcher."""

    def test_return_dict_keys(self, synthetic_full_db, monkeypatch):
        """run_fold returns a dict with all expected keys."""
        from pattern_engine import walkforward as wf_module
        from pattern_engine.config import WALKFORWARD_FOLDS

        # Augment with candlestick features
        full_db = wf_module._augment_with_candlestick(synthetic_full_db)

        # Mock PatternMatcher to avoid real computation
        class _MockMatcher:
            def __init__(self, cfg):
                pass

            def fit(self, train_db, feature_cols):
                pass

            def query(self, val_db, verbose=0):
                n = len(val_db)
                probs = np.full(n, 0.55)
                signals = ["HOLD"] * n
                return probs, signals, [None] * n, [10] * n, [None] * n, [None] * n

        monkeypatch.setattr(wf_module, "PatternMatcher", _MockMatcher)

        fold = WALKFORWARD_FOLDS[-1]  # 2024-Val
        result = wf_module.run_fold(fold=fold, full_db=full_db)

        expected_keys = {
            "fold", "bss", "n_scored", "n_total", "base_rate",
            "mean_prob", "reliability", "resolution", "uncertainty",
        }
        assert set(result.keys()) == expected_keys
        assert result["fold"] == fold["label"]
        assert isinstance(result["bss"], float)
        assert isinstance(result["n_scored"], int)

    def test_empty_val_returns_nan(self, synthetic_full_db):
        """Fold with no data in validation window returns NaN BSS."""
        from pattern_engine import walkforward as wf_module

        full_db = wf_module._augment_with_candlestick(synthetic_full_db)

        # Create a fold with dates outside the data range
        fake_fold = {
            "label": "empty-fold",
            "train_end": "2010-12-31",
            "val_start": "2011-01-01",
            "val_end": "2011-12-31",
        }
        result = wf_module.run_fold(fold=fake_fold, full_db=full_db)
        assert math.isnan(result["bss"])
        assert result["n_scored"] == 0
        assert result["n_total"] == 0


# ── TestRunWalkforward ───────────────────────────────────────────────────────


class TestRunWalkforward:
    """Test run_walkforward() orchestrator."""

    def test_return_dict_shape(self, monkeypatch):
        """run_walkforward returns expected top-level keys."""
        from pattern_engine import walkforward as wf_module

        # Mock run_fold to return fixed results
        def _mock_run_fold(fold, full_db, feature_cols=None, cfg_overrides=None):
            return {
                "fold": fold["label"],
                "bss": 0.01,
                "n_scored": 100,
                "n_total": 200,
                "base_rate": 0.5,
                "mean_prob": 0.52,
                "reliability": 0.001,
                "resolution": 0.002,
                "uncertainty": 0.25,
            }

        monkeypatch.setattr(wf_module, "run_fold", _mock_run_fold)

        fake_db = pd.DataFrame()  # Not used by mock
        result = wf_module.run_walkforward(fake_db)

        assert "mean_bss" in result
        assert "trimmed_mean_bss" in result
        assert "positive_folds" in result
        assert "fold_results" in result
        assert "wilcoxon_p" in result
        assert len(result["fold_results"]) == 6

    def test_trimmed_mean_drops_worst(self, monkeypatch):
        """Trimmed mean should drop the worst fold."""
        from pattern_engine import walkforward as wf_module
        from pattern_engine.config import WALKFORWARD_FOLDS

        bss_values = [0.10, -0.50, 0.05, 0.08, 0.03, 0.07]

        def _mock_run_fold(fold, full_db, feature_cols=None, cfg_overrides=None):
            idx = next(
                i for i, f in enumerate(WALKFORWARD_FOLDS)
                if f["label"] == fold["label"]
            )
            return {
                "fold": fold["label"],
                "bss": bss_values[idx],
                "n_scored": 100,
                "n_total": 200,
                "base_rate": 0.5,
                "mean_prob": 0.52,
                "reliability": 0.001,
                "resolution": 0.002,
                "uncertainty": 0.25,
            }

        monkeypatch.setattr(wf_module, "run_fold", _mock_run_fold)

        fake_db = pd.DataFrame()
        result = wf_module.run_walkforward(fake_db)

        # Mean of all 6
        assert result["mean_bss"] == pytest.approx(np.mean(bss_values))

        # Trimmed mean: drop worst (-0.50), average remaining 5
        remaining = sorted(bss_values)[1:]  # drop lowest
        assert result["trimmed_mean_bss"] == pytest.approx(np.mean(remaining))

        # Positive folds: 5 out of 6
        assert result["positive_folds"] == 5


# ── TestParity ───────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestParity:
    """Parity: walkforward.run_fold() matches baseline TSV on real data."""

    def test_fold_2019_parity(self):
        """BSS from walkforward.run_fold on fold 2019 matches baseline TSV."""
        from pattern_engine.walkforward import load_and_augment_db, run_fold
        from pattern_engine.config import WALKFORWARD_FOLDS

        full_db = load_and_augment_db()
        fold = WALKFORWARD_FOLDS[0]  # 2019
        result = run_fold(fold, full_db)

        import csv
        baseline_path = Path("results/phase7/baseline_23d.tsv")
        if not baseline_path.exists():
            pytest.skip("Baseline TSV not found — run phase7_baseline.py first")

        with open(baseline_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["fold"] == "2019":
                    expected_bss = float(row["bss"])
                    break
            else:
                pytest.fail("2019 fold not found in baseline TSV")

        assert result["bss"] == pytest.approx(expected_bss, abs=1e-6), \
            f"Parity failure: walkforward BSS={result['bss']}, baseline BSS={expected_bss}"
