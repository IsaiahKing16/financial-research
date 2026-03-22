"""Tests for pattern_engine.projection."""

import numpy as np
import pandas as pd
import pytest
from pattern_engine.projection import project_forward, generate_signal


class TestProjectForward:
    """Test forward projection from analogue matches."""

    def _make_matches(self, n=20, prob_up=0.7, mean_ret=0.02):
        """Create synthetic match DataFrame."""
        rng = np.random.RandomState(42)
        ups = int(n * prob_up)
        fwd = np.concatenate([rng.rand(ups) * 0.05, -rng.rand(n - ups) * 0.05])
        rng.shuffle(fwd)
        return pd.DataFrame({
            "fwd_7d": fwd,
            "fwd_7d_up": (fwd > 0).astype(int),
            "distance": rng.rand(n) * 0.5 + 0.1,
        })

    def test_empty_matches(self):
        matches = pd.DataFrame(columns=["fwd_7d", "fwd_7d_up", "distance"])
        result = project_forward(matches)
        assert result["probability_up"] == 0.5
        assert result["n_matches"] == 0
        assert result["agreement"] == 0.0
        assert len(result["ensemble_returns"]) == 0

    def test_uniform_weighting(self):
        matches = self._make_matches(n=20, prob_up=0.8)
        result = project_forward(matches, weighting="uniform")
        assert result["n_matches"] == 20
        # 80% probability should yield ~0.8
        assert 0.5 < result["probability_up"] <= 1.0
        assert result["agreement"] > 0

    def test_inverse_weighting(self):
        matches = self._make_matches()
        result = project_forward(matches, weighting="inverse")
        assert result["n_matches"] == 20
        assert 0 <= result["probability_up"] <= 1.0

    def test_all_up(self):
        df = pd.DataFrame({
            "fwd_7d": [0.01, 0.02, 0.03],
            "fwd_7d_up": [1, 1, 1],
            "distance": [0.5, 0.6, 0.7],
        })
        result = project_forward(df)
        assert result["probability_up"] == 1.0
        assert result["agreement"] == 1.0
        assert result["mean_return"] > 0

    def test_all_down(self):
        df = pd.DataFrame({
            "fwd_7d": [-0.01, -0.02, -0.03],
            "fwd_7d_up": [0, 0, 0],
            "distance": [0.5, 0.6, 0.7],
        })
        result = project_forward(df)
        assert result["probability_up"] == 0.0
        assert result["agreement"] == 1.0
        assert result["mean_return"] < 0

    def test_ensemble_returns(self):
        matches = self._make_matches(n=10)
        result = project_forward(matches)
        assert len(result["ensemble_returns"]) == 10


class TestGenerateSignal:
    """Test signal generation via three-filter gate."""

    def test_buy_signal(self):
        proj = {"probability_up": 0.75, "agreement": 0.5, "n_matches": 30}
        signal, reason = generate_signal(proj, threshold=0.65)
        assert signal == "BUY"

    def test_sell_signal(self):
        proj = {"probability_up": 0.25, "agreement": 0.5, "n_matches": 30}
        signal, reason = generate_signal(proj, threshold=0.65)
        assert signal == "SELL"

    def test_hold_below_threshold(self):
        proj = {"probability_up": 0.55, "agreement": 0.1, "n_matches": 30}
        signal, reason = generate_signal(proj, threshold=0.65)
        assert signal == "HOLD"
        assert "below_threshold" in reason

    def test_hold_insufficient_matches(self):
        proj = {"probability_up": 0.9, "agreement": 0.8, "n_matches": 5}
        signal, reason = generate_signal(proj, threshold=0.65, min_matches=10)
        assert signal == "HOLD"
        assert "insufficient_matches" in reason

    def test_hold_low_agreement(self):
        proj = {"probability_up": 0.9, "agreement": 0.05, "n_matches": 30}
        signal, reason = generate_signal(proj, threshold=0.65, min_agreement=0.10)
        assert signal == "HOLD"
        assert "low_agreement" in reason

    def test_exact_threshold_buy(self):
        proj = {"probability_up": 0.65, "agreement": 0.3, "n_matches": 20}
        signal, _ = generate_signal(proj, threshold=0.65)
        assert signal == "BUY"

    def test_exact_threshold_sell(self):
        proj = {"probability_up": 0.35, "agreement": 0.3, "n_matches": 20}
        signal, _ = generate_signal(proj, threshold=0.65)
        assert signal == "SELL"
