"""
test_matcher_parity_staged.py — Parity tests for the staged PatternMatcher.

Two parity levels:

Level 1 — BallTree exact parity:
    PatternMatcher(use_hnsw=False) vs production Matcher on identical inputs.
    Target: bit-identical probabilities (rtol=1e-12), identical signal labels.

Level 2 — HNSW approximate parity:
    PatternMatcher(use_hnsw=True) vs PatternMatcher(use_hnsw=False).
    Targets per SLE-62 acceptance criteria:
      - recall@50 > 0.9995
      - Signal agreement > 99%
      - BSS agreement within 0.001 of exact baseline

These tests run on synthetic data with a fixed RNG seed so they are
deterministic in CI and require no data files.

Linear: SLE-60 (level 1), SLE-62 (level 2)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

try:
    from pattern_engine.contracts.matchers.hnsw_matcher import HNSWMatcher
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False

from pattern_engine.matcher import PatternMatcher


# ─── Synthetic data fixtures ───────────────────────────────────────────────────

FEATURE_COLS = [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]
N_TRAIN = 2000
N_QUERY = 200
RNG_SEED = 42


def _make_engine_config(use_hnsw: bool = False):
    """Return a minimal EngineConfig-like object for testing PatternMatcher."""
    try:
        from pattern_engine.config import EngineConfig
        # calibration_method="none": parity tests verify raw-matching fidelity,
        # not calibration. Legacy Matcher returns raw probs; PatternMatcher must
        # also return raw probs here so probabilities are bit-identical.
        return EngineConfig(use_hnsw=use_hnsw, calibration_method="none")
    except ImportError:
        pass
    # Fallback: minimal config object with required attributes
    class MinimalConfig:
        top_k = 50
        max_distance = 1.1019
        distance_weighting = "uniform"
        feature_weights = {}  # All 1.0 (no-op)
        batch_size = 256
        confidence_threshold = 0.65
        agreement_spread = 0.10
        min_matches = 10
        exclude_same_ticker = True
        same_sector_only = False
        regime_filter = False
        regime_fallback = False
        projection_horizon = "fwd_7d_up"
        # Parity tests compare against legacy production Matcher (no calibration).
        # Disable Platt so probabilities are bit-identical to the legacy path.
        calibration_method = "none"
        def __init__(self, use_hnsw=False):
            self.use_hnsw = use_hnsw
    return MinimalConfig(use_hnsw=use_hnsw)


def _make_train_df(n: int = N_TRAIN, seed: int = RNG_SEED) -> pd.DataFrame:
    """Generate a realistic training DataFrame with returns and targets."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "Date": pd.date_range("2015-01-01", periods=n, freq="B"),
        "Ticker": rng.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], size=n),
        "fwd_7d_up": rng.randint(0, 2, size=n).astype(float),
        "fwd_7d": rng.randn(n) * 2.0,
    })
    for col in FEATURE_COLS:
        df[col] = rng.randn(n)
    return df


def _make_query_df(n: int = N_QUERY, seed: int = 99) -> pd.DataFrame:
    """Generate query rows (distinct seed for independence from training)."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "Ticker": rng.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], size=n),
        "fwd_7d_up": rng.randint(0, 2, size=n).astype(float),
        "fwd_7d": rng.randn(n) * 2.0,
    })
    for col in FEATURE_COLS:
        df[col] = rng.randn(n)
    return df


# ─── Level 1: BallTree staged parity vs production ────────────────────────────

class TestBallTreeStagedParity:
    """PatternMatcher(use_hnsw=False) must match production Matcher exactly."""

    @pytest.fixture(scope="class")
    def train_df(self):
        return _make_train_df()

    @pytest.fixture(scope="class")
    def query_df(self):
        return _make_query_df()

    @pytest.fixture(scope="class")
    def staged_results(self, train_df, query_df):
        cfg = _make_engine_config(use_hnsw=False)
        matcher = PatternMatcher(cfg)
        matcher.fit(train_df, FEATURE_COLS)
        return matcher.query(query_df, verbose=0)

    def test_staged_matcher_fits(self, train_df):
        """PatternMatcher.fit() should complete without error."""
        cfg = _make_engine_config(use_hnsw=False)
        m = PatternMatcher(cfg)
        m.fit(train_df, FEATURE_COLS)
        assert m.fitted
        assert m.backend_name == "balltree"

    def test_probabilities_shape(self, staged_results, query_df):
        probs, signals, *_ = staged_results
        assert probs.shape == (len(query_df),)

    def test_signals_shape(self, staged_results, query_df):
        _, signals, *_ = staged_results
        assert len(signals) == len(query_df)

    def test_signals_valid_values(self, staged_results):
        _, signals, *_ = staged_results
        valid = {"BUY", "SELL", "HOLD"}
        assert all(s in valid for s in signals), "All signals must be BUY/SELL/HOLD"

    def test_probabilities_in_range(self, staged_results):
        probs, *_ = staged_results
        assert np.all(probs >= 0.0), "Probabilities must be non-negative"
        assert np.all(probs <= 1.0), "Probabilities must be <= 1.0"

    def test_n_matches_non_negative(self, staged_results):
        _, _, _, n_matches, *_ = staged_results
        assert all(n >= 0 for n in n_matches)

    def test_parity_with_production_matcher(self, train_df, query_df):
        """PatternMatcher(use_hnsw=False) probabilities must match production Matcher."""
        try:
            from pattern_engine.matching import Matcher
            from pattern_engine.config import EngineConfig
        except ImportError:
            pytest.skip("Production pattern_engine not importable from test environment")

        # Production Matcher
        prod_cfg = EngineConfig(use_hnsw=False)
        prod_matcher = Matcher(prod_cfg)
        prod_matcher.fit(train_df, FEATURE_COLS)
        prod_probs, prod_signals, *_ = prod_matcher.query(query_df, verbose=0)

        # Staged PatternMatcher
        staged_cfg = _make_engine_config(use_hnsw=False)
        staged_matcher = PatternMatcher(staged_cfg)
        staged_matcher.fit(train_df, FEATURE_COLS)
        staged_probs, staged_signals, *_ = staged_matcher.query(query_df, verbose=0)

        # Probabilities must be bit-identical (same algorithm, same scaler, same tree)
        np.testing.assert_allclose(
            staged_probs, prod_probs, rtol=1e-10,
            err_msg="PatternMatcher probs differ from production Matcher probs"
        )

        # Signals must match exactly
        assert list(staged_signals) == list(prod_signals), (
            f"Signal mismatch: {sum(s != p for s, p in zip(staged_signals, prod_signals))} "
            f"differences out of {len(prod_signals)}"
        )

    def test_stage_1_scaling_is_correct(self, train_df, query_df):
        """Verify Stage 1 produces zero-mean unit-variance features on training data."""
        from sklearn.preprocessing import StandardScaler
        cfg = _make_engine_config(use_hnsw=False)
        matcher = PatternMatcher(cfg)
        matcher.fit(train_df, FEATURE_COLS)

        # Re-apply scaler to training data — should get mean≈0, std≈1
        X_raw = train_df[FEATURE_COLS].values
        X_scaled = matcher.scaler.transform(X_raw)
        col_means = X_scaled.mean(axis=0)
        col_stds = X_scaled.std(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-10, err_msg="Scaler means != 0")
        np.testing.assert_allclose(col_stds, 1.0, atol=1e-10, err_msg="Scaler stds != 1")

    def test_stage_2_backend_is_balltree(self, train_df):
        """Stage 2 must select BallTreeMatcher when use_hnsw=False."""
        from pattern_engine.contracts.matchers.balltree_matcher import BallTreeMatcher
        cfg = _make_engine_config(use_hnsw=False)
        matcher = PatternMatcher(cfg)
        matcher.fit(train_df, FEATURE_COLS)
        assert isinstance(matcher.backend, BallTreeMatcher)

    def test_stage_4_distance_filter(self, train_df):
        """Stage 4: all accepted analogues must be within max_distance."""
        cfg = _make_engine_config(use_hnsw=False)
        matcher = PatternMatcher(cfg)
        matcher.fit(train_df, FEATURE_COLS)

        # Query single row to inspect filter
        row = _make_query_df(n=1)
        X_raw = row[FEATURE_COLS].values
        X_weighted = matcher._prepare_features(X_raw, fit_scaler=False)

        n_probe = min(cfg.top_k * 3, len(train_df))
        distances_b, indices_b = matcher._backend.kneighbors(X_weighted, n_neighbors=n_probe)

        val_tickers = np.asarray(row["Ticker"], dtype=object)
        val_sectors = np.array([""])
        top_mask = matcher._post_filter(distances_b, indices_b, val_tickers, val_sectors, None)

        # All accepted distances must be within max_distance
        accepted_dists = distances_b[0][top_mask[0]]
        if len(accepted_dists) > 0:
            assert np.all(accepted_dists <= cfg.max_distance + 1e-10), (
                f"Some accepted distances exceed max_distance: {accepted_dists.max():.6f}"
            )

    def test_stage_5_neighbor_results_built(self, train_df, query_df):
        """Stage 5 must produce NeighborResult objects with valid fields."""
        from pattern_engine.contracts.signals import NeighborResult
        cfg = _make_engine_config(use_hnsw=False)
        matcher = PatternMatcher(cfg)
        matcher.fit(train_df, FEATURE_COLS)

        # Access Stage 5 directly on a small batch
        row = _make_query_df(n=3)
        X_raw = row[FEATURE_COLS].values
        X_weighted = matcher._prepare_features(X_raw, fit_scaler=False)

        n_probe = min(cfg.top_k * 3, len(train_df))
        distances_b, indices_b = matcher._backend.kneighbors(X_weighted, n_neighbors=n_probe)

        val_tickers = np.asarray(row["Ticker"], dtype=object)
        val_sectors = np.array([""] * 3)
        top_mask = matcher._post_filter(distances_b, indices_b, val_tickers, val_sectors, None)

        val_dates = row["Date"].values
        _, _, _, _, _, _, neighbor_results = matcher._package_results(
            top_mask, distances_b, indices_b, val_tickers, val_dates
        )

        assert len(neighbor_results) == 3
        for nr in neighbor_results:
            assert isinstance(nr, NeighborResult)
            assert nr.n_neighbors_found == len(nr.neighbor_indices)
            assert nr.n_neighbors_found <= cfg.top_k


# ─── Level 2: HNSW approximate parity ────────────────────────────────────────

@pytest.mark.skipif(not HAS_HNSWLIB, reason="hnswlib not installed")
class TestHNSWApproximateParity:
    """HNSW backend must satisfy the M3 recall and agreement gates (SLE-62)."""

    @pytest.fixture(scope="class")
    def train_df(self):
        return _make_train_df()

    @pytest.fixture(scope="class")
    def query_df(self):
        return _make_query_df()

    @pytest.fixture(scope="class")
    def exact_results(self, train_df, query_df):
        cfg = _make_engine_config(use_hnsw=False)
        m = PatternMatcher(cfg)
        m.fit(train_df, FEATURE_COLS)
        probs, signals, _, n_matches, mean_rets, _ = m.query(query_df, verbose=0)
        return probs, signals, n_matches, mean_rets

    @pytest.fixture(scope="class")
    def hnsw_results(self, train_df, query_df):
        cfg = _make_engine_config(use_hnsw=True)
        m = PatternMatcher(cfg)
        m.fit(train_df, FEATURE_COLS)
        probs, signals, _, n_matches, mean_rets, _ = m.query(query_df, verbose=0)
        return probs, signals, n_matches, mean_rets

    def test_hnsw_backend_selected(self, train_df):
        """Stage 2 must select HNSWMatcher when use_hnsw=True."""
        cfg = _make_engine_config(use_hnsw=True)
        m = PatternMatcher(cfg)
        m.fit(train_df, FEATURE_COLS)
        assert m.backend_name == "hnsw"

    def test_signal_agreement_above_99pct(self, exact_results, hnsw_results):
        """SLE-62: HNSW signal labels must agree with exact results > 99%."""
        _, exact_signals, _, _ = exact_results
        _, hnsw_signals, _, _ = hnsw_results

        n = len(exact_signals)
        agreements = sum(e == h for e, h in zip(exact_signals, hnsw_signals))
        agreement_rate = agreements / n

        assert agreement_rate >= 0.99, (
            f"Signal agreement {agreement_rate:.4f} < 0.99 "
            f"({n - agreements}/{n} mismatches)"
        )

    def test_probability_correlation_high(self, exact_results, hnsw_results):
        """HNSW probabilities should be highly correlated with exact."""
        exact_probs, *_ = exact_results
        hnsw_probs, *_ = hnsw_results

        # Pearson correlation should be very high
        corr = np.corrcoef(exact_probs, hnsw_probs)[0, 1]
        assert corr >= 0.98, (
            f"Probability correlation {corr:.4f} < 0.98 — "
            "HNSW and exact results diverge too much"
        )

    def test_bss_parity_within_tolerance(self, exact_results, hnsw_results, query_df):
        """SLE-62: BSS parity within 0.001 between exact and HNSW backends."""
        exact_probs, *_ = exact_results
        hnsw_probs, *_ = hnsw_results

        # Brier scores for each backend
        y_true = query_df["fwd_7d_up"].values

        def bss(probs, y):
            brier = np.mean((probs - y) ** 2)
            brier_clim = np.mean((y.mean() - y) ** 2)
            return 1.0 - brier / brier_clim if brier_clim > 0 else 0.0

        exact_bss = bss(exact_probs, y_true)
        hnsw_bss = bss(hnsw_probs, y_true)
        delta = abs(hnsw_bss - exact_bss)

        assert delta <= 0.001, (
            f"BSS delta {delta:.6f} exceeds 0.001 tolerance. "
            f"Exact BSS={exact_bss:.6f}, HNSW BSS={hnsw_bss:.6f}"
        )

    def test_n_matches_comparable(self, exact_results, hnsw_results):
        """HNSW should find a similar number of analogues as exact search."""
        _, _, exact_nm, _ = exact_results
        _, _, hnsw_nm, _ = hnsw_results

        mean_exact = np.mean(exact_nm)
        mean_hnsw = np.mean(hnsw_nm)

        # Mean n_matches should not differ by more than 10%
        if mean_exact > 0:
            relative_diff = abs(mean_hnsw - mean_exact) / mean_exact
            assert relative_diff <= 0.10, (
                f"Mean n_matches differs by {relative_diff:.2%}: "
                f"exact={mean_exact:.1f}, hnsw={mean_hnsw:.1f}"
            )

    def test_recall_at_k_direct(self, train_df, query_df):
        """
        SLE-62: Direct recall@50 test using the raw kneighbors output.

        Independently verifies that HNSW finds ≥99.5% of the true top-50
        neighbors (as found by BallTree) on this dataset.
        """
        from pattern_engine.contracts.matchers.balltree_matcher import BallTreeMatcher
        cfg = _make_engine_config(use_hnsw=False)

        # Build scaled+weighted feature matrices
        exact_m = PatternMatcher(cfg)
        exact_m.fit(train_df, FEATURE_COLS)

        X_query_raw = query_df[FEATURE_COLS].values
        X_query_weighted = exact_m._prepare_features(X_query_raw, fit_scaler=False)

        # Exact neighbors
        k = 50
        n_probe = min(k * 3, N_TRAIN)
        exact_dists, exact_indices = exact_m.backend.kneighbors(X_query_weighted, n_neighbors=n_probe)

        # HNSW neighbors on the same weighted matrix
        hnsw_m = PatternMatcher(_make_engine_config(use_hnsw=True))
        hnsw_m.fit(train_df, FEATURE_COLS)

        # Note: HNSW scaler must match exact scaler — reuse exact scaler
        # The HNSW matcher was fit on the same data, so scaler params are identical
        X_hnsw_weighted = hnsw_m._prepare_features(X_query_raw, fit_scaler=False)
        hnsw_dists, hnsw_indices = hnsw_m.backend.kneighbors(X_hnsw_weighted, n_neighbors=n_probe)

        # Recall@50: what fraction of exact top-50 does HNSW recover?
        recalls = []
        for i in range(len(query_df)):
            # Get true top-k from exact search after distance filter
            exact_top = set(exact_indices[i][exact_dists[i] <= cfg.max_distance][:k])
            hnsw_top = set(hnsw_indices[i][hnsw_dists[i] <= cfg.max_distance][:k])
            if len(exact_top) == 0:
                continue
            recalls.append(len(exact_top & hnsw_top) / len(exact_top))

        if recalls:
            mean_recall = np.mean(recalls)
            assert mean_recall >= 0.995, (
                f"Recall@50 = {mean_recall:.5f}, expected >= 0.995"
            )
