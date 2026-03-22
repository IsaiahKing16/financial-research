"""
test_end_to_end_parity.py — Level 3: End-to-end pipeline parity (SLE-80).

Three verification levels:

  Level 3a — Determinism:
      PatternMatcher on seeded data is bit-reproducible (same output every run).
      No data files required. Always runs in CI.

  Level 3b — Snapshot regression:
      Compare current output against the frozen parity_snapshot.json artifact.
      Catches silent regressions from algorithm changes or dependency upgrades.
      Requires artifacts/baselines/parity_snapshot.json (generate once, commit).

  Level 3c — Production oracle (slow, local only):
      Compare PatternMatcher BSS against the frozen baseline BSS=+0.00103.
      Requires the 52 ticker CSV files and the cached signals CSV.
      Skipped in CI. Run with: pytest -m slow

Frozen baseline (from rebuild_start_report.md):
    BSS (2024 fold): +0.00103
    Tolerances:      BSS rtol=1e-5, Sharpe rtol=1e-4, signal counts exact

To regenerate snapshot after a deliberate algorithm change:
    python scripts/generate_parity_snapshot.py
    git add rebuild_phase_3z/artifacts/baselines/parity_snapshot.json
    git commit -m "Update parity snapshot after [reason]"

Linear: SLE-80
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rebuild_phase_3z.fppe.pattern_engine.matcher import PatternMatcher

# ─── Reference dataset constants ──────────────────────────────────────────────
# These MUST match scripts/generate_parity_snapshot.py exactly.

FEATURE_COLS = [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]
N_TRAIN = 2000
N_QUERY = 400
RNG_SEED_TRAIN = 42
RNG_SEED_QUERY = 99

SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2]  # rebuild_phase_3z/
    / "artifacts" / "baselines" / "parity_snapshot.json"
)

BSS_PRODUCTION_BASELINE = +0.00103   # frozen from rebuild_start_report.md
BSS_PRODUCTION_RTOL = 1e-4           # ±0.01% relative tolerance on the 2024 fold BSS


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _make_train_df(n: int = N_TRAIN, seed: int = RNG_SEED_TRAIN) -> pd.DataFrame:
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


def _make_query_df(n: int = N_QUERY, seed: int = RNG_SEED_QUERY) -> pd.DataFrame:
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


def _synthetic_config():
    """
    Config matching scripts/generate_parity_snapshot.py.

    max_distance is relaxed vs. production (1.1019) because i.i.d. normal
    features have expected L2 distance ~4.0 after StandardScaler — the
    production value would filter out almost all analogues on synthetic data.
    """
    class SyntheticConfig:
        top_k = 50
        max_distance = 4.5
        distance_weighting = "uniform"
        feature_weights = {}
        batch_size = 256
        confidence_threshold = 0.55
        agreement_spread = 0.05
        min_matches = 5
        exclude_same_ticker = True
        same_sector_only = False
        regime_filter = False
        regime_fallback = False
        projection_horizon = "fwd_7d_up"
        use_hnsw = False
        # Parity tests use synthetic random labels (50/50) — calibration
        # would map everything to ~0.5 (base rate), destroying all signal
        # diversity.  The snapshot tests verify raw-matching determinism, not
        # calibration.  Production EngineConfig uses 'platt' (locked setting).
        calibration_method = "none"
    return SyntheticConfig()


def _bss(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Brier Skill Score vs. climatological baseline."""
    brier = float(np.mean((probs - y_true) ** 2))
    brier_clim = float(np.var(y_true))
    return 1.0 - brier / brier_clim if brier_clim > 0 else 0.0


# ─── Level 3a: Determinism ────────────────────────────────────────────────────

class TestDeterminism:
    """
    Level 3a: PatternMatcher output is bit-reproducible.

    Two independent calls with identical data and config must produce
    bit-identical probabilities and signal labels. This tests:
      - StandardScaler fitting is deterministic (no random init)
      - BallTree construction is deterministic
      - Post-filter and packaging are pure functions
    """

    @pytest.fixture(scope="class")
    def train_df(self):
        return _make_train_df()

    @pytest.fixture(scope="class")
    def query_df(self):
        return _make_query_df()

    def _run(self, train_df, query_df):
        m = PatternMatcher(_synthetic_config())
        m.fit(train_df, FEATURE_COLS)
        return m.query(query_df, verbose=0)

    def test_probabilities_bit_identical(self, train_df, query_df):
        """Two runs produce identical probability arrays."""
        probs1, *_ = self._run(train_df, query_df)
        probs2, *_ = self._run(train_df, query_df)
        np.testing.assert_array_equal(
            probs1, probs2,
            err_msg="Probabilities differ between two identical runs — non-determinism detected",
        )

    def test_signals_identical(self, train_df, query_df):
        """Two runs produce identical signal arrays."""
        _, sigs1, *_ = self._run(train_df, query_df)
        _, sigs2, *_ = self._run(train_df, query_df)
        assert list(sigs1) == list(sigs2), (
            f"Signal mismatch: {sum(a != b for a, b in zip(sigs1, sigs2))} differences"
        )

    def test_n_matches_identical(self, train_df, query_df):
        """Two runs produce identical n_matches arrays."""
        _, _, _, nm1, *_ = self._run(train_df, query_df)
        _, _, _, nm2, *_ = self._run(train_df, query_df)
        assert nm1 == nm2, "n_matches differ between runs"

    def test_mean_returns_identical(self, train_df, query_df):
        """Two runs produce identical mean_returns."""
        _, _, _, _, mr1, *_ = self._run(train_df, query_df)
        _, _, _, _, mr2, *_ = self._run(train_df, query_df)
        np.testing.assert_array_equal(
            np.array(mr1, dtype=float),
            np.array(mr2, dtype=float),
            err_msg="mean_returns differ between runs",
        )

    def test_output_shape_stable(self, train_df, query_df):
        """Output arrays all have length == len(query_df)."""
        probs, sigs, reasons, nm, mr, ens = self._run(train_df, query_df)
        N = len(query_df)
        assert len(probs) == N
        assert len(sigs) == N
        assert len(reasons) == N
        assert len(nm) == N
        assert len(mr) == N
        assert len(ens) == N


# ─── Level 3b: Snapshot regression ───────────────────────────────────────────

class TestSnapshotRegression:
    """
    Level 3b: Current output must match the committed parity snapshot.

    The snapshot is generated once by scripts/generate_parity_snapshot.py
    and committed to the repo. CI compares every run against it.

    If this test fails:
      1. Investigate whether the change was intentional (algorithm change,
         dependency upgrade with different BallTree tie-breaking, etc.)
      2. If intentional: run generate_parity_snapshot.py and commit the new snapshot.
      3. If unintentional: you found a regression. Debug before proceeding.
    """

    @pytest.fixture(scope="class")
    def snapshot(self):
        if not SNAPSHOT_PATH.exists():
            pytest.skip(
                f"Parity snapshot not found: {SNAPSHOT_PATH}. "
                "Run: python scripts/generate_parity_snapshot.py"
            )
        return json.loads(SNAPSHOT_PATH.read_text())

    @pytest.fixture(scope="class")
    def current_results(self):
        train_df = _make_train_df()
        query_df = _make_query_df()
        cfg = _synthetic_config()
        m = PatternMatcher(cfg)
        m.fit(train_df, FEATURE_COLS)
        probs, signals, _, n_matches, mean_returns, _ = m.query(query_df, verbose=0)
        y_true = query_df["fwd_7d_up"].values
        return {
            "probs": np.asarray(probs),
            "signals": list(signals),
            "n_matches": list(n_matches),
            "mean_returns": list(mean_returns),
            "y_true": y_true,
        }

    def test_snapshot_schema_version(self, snapshot):
        """Snapshot must have the expected schema version."""
        assert snapshot.get("_schema_version") == "SLE-80-v1", (
            f"Unexpected snapshot schema: {snapshot.get('_schema_version')}"
        )

    def test_signal_counts_exact(self, snapshot, current_results):
        """Signal counts (BUY/SELL/HOLD) must exactly match the snapshot."""
        signals = current_results["signals"]
        buy = int(np.sum(np.asarray(signals) == "BUY"))
        sell = int(np.sum(np.asarray(signals) == "SELL"))
        hold = int(np.sum(np.asarray(signals) == "HOLD"))

        expected = snapshot["metrics"]
        assert buy == expected["buy_count"], (
            f"BUY count: got {buy}, expected {expected['buy_count']}"
        )
        assert sell == expected["sell_count"], (
            f"SELL count: got {sell}, expected {expected['sell_count']}"
        )
        assert hold == expected["hold_count"], (
            f"HOLD count: got {hold}, expected {expected['hold_count']}"
        )

    def test_mean_probability_within_tolerance(self, snapshot, current_results):
        """Mean probability must match snapshot within rtol=1e-7."""
        current_mean = float(np.mean(current_results["probs"]))
        expected_mean = snapshot["metrics"]["mean_probability"]
        rtol = snapshot["tolerances"]["probability_rtol"]
        np.testing.assert_allclose(
            current_mean, expected_mean, rtol=rtol,
            err_msg=f"Mean probability drifted: {current_mean:.8f} vs {expected_mean:.8f}",
        )

    def test_bss_within_tolerance(self, snapshot, current_results):
        """BSS must match snapshot within rtol=1e-5."""
        probs = current_results["probs"]
        y_true = current_results["y_true"]
        current_bss = _bss(probs, y_true)
        expected_bss = snapshot["metrics"]["bss"]
        rtol = snapshot["tolerances"]["bss_rtol"]
        np.testing.assert_allclose(
            current_bss, expected_bss, rtol=rtol,
            err_msg=f"BSS drifted: {current_bss:.8f} vs {expected_bss:.8f}",
        )

    def test_mean_n_matches_within_tolerance(self, snapshot, current_results):
        """Mean n_matches must match snapshot within rtol=1e-7."""
        current_mn = float(np.mean(current_results["n_matches"]))
        expected_mn = snapshot["metrics"]["mean_n_matches"]
        np.testing.assert_allclose(
            current_mn, expected_mn, rtol=1e-7,
            err_msg=f"Mean n_matches drifted: {current_mn:.4f} vs {expected_mn:.4f}",
        )

    def test_dataset_config_unchanged(self, snapshot):
        """Snapshot dataset config must match this file's constants."""
        ds = snapshot["dataset"]
        assert ds["n_train"] == N_TRAIN, f"n_train mismatch: {ds['n_train']} vs {N_TRAIN}"
        assert ds["n_query"] == N_QUERY, f"n_query mismatch: {ds['n_query']} vs {N_QUERY}"
        assert ds["rng_seed_train"] == RNG_SEED_TRAIN
        assert ds["rng_seed_query"] == RNG_SEED_QUERY
        assert ds["feature_cols"] == FEATURE_COLS


# ─── Level 3c: Production oracle (slow — local only) ─────────────────────────

@pytest.mark.slow
class TestProductionOracleParity:
    """
    Level 3c: Verify BSS against the frozen production baseline.

    Requires the 52 ticker CSV data files and the production feature DB.
    Too slow for CI (data loading + full walk-forward).
    Run locally: pytest rebuild_phase_3z/tests/parity/ -m slow -v

    Gate:
        BSS (2024 fold) must be within rtol=1e-4 of +0.00103.
        Source: rebuild_phase_3z/artifacts/baselines/rebuild_start_report.md
    """

    def _try_load_production_data(self):
        """Load production train/val DBs if available."""
        try:
            from pattern_engine.config import EngineConfig
            from pattern_engine.data import DataLoader
        except ImportError:
            pytest.skip("Production pattern_engine not importable")

        import os
        data_dir = Path(__file__).resolve().parents[4] / "data"
        if not data_dir.exists():
            pytest.skip(f"Data directory not found: {data_dir}")

        cfg = EngineConfig()
        loader = DataLoader(cfg)
        try:
            train_db, val_db = loader.temporal_split()
        except Exception as e:
            pytest.skip(f"Could not load production data: {e}")

        return train_db, val_db, cfg

    def test_production_bss_within_tolerance(self):
        """BSS on real 2024 validation fold must match frozen baseline ±rtol=1e-4."""
        train_db, val_db, cfg = self._try_load_production_data()

        try:
            from pattern_engine.config import EngineConfig
            prod_cfg = EngineConfig(use_hnsw=False, nn_jobs=1)
        except ImportError:
            pytest.skip("EngineConfig not importable")

        feature_cols = [c for c in train_db.columns if c.startswith("ret_")]
        if not feature_cols:
            pytest.skip("No return columns found in production data")

        m = PatternMatcher(prod_cfg)
        m.fit(train_db, feature_cols)
        probs, *_ = m.query(val_db, verbose=0)

        y_true = val_db["fwd_7d_up"].values
        bss = _bss(np.asarray(probs), y_true)

        np.testing.assert_allclose(
            bss, BSS_PRODUCTION_BASELINE, rtol=BSS_PRODUCTION_RTOL,
            err_msg=(
                f"Production BSS {bss:.6f} is outside tolerance of "
                f"frozen baseline {BSS_PRODUCTION_BASELINE:.5f} "
                f"(rtol={BSS_PRODUCTION_RTOL})"
            ),
        )

    def test_production_signal_counts_stable(self):
        """
        Signal counts on the 2024 validation fold must be within ±5% of
        the frozen baseline (191 total trades).

        Note: exact counts are not required for Level 3c (the production
        Matcher and rebuild PatternMatcher use the same algorithm, so
        counts should be equal; the ±5% band accommodates data file drift).
        """
        train_db, val_db, cfg = self._try_load_production_data()

        try:
            from pattern_engine.config import EngineConfig
            prod_cfg = EngineConfig(use_hnsw=False, nn_jobs=1)
        except ImportError:
            pytest.skip("EngineConfig not importable")

        feature_cols = [c for c in train_db.columns if c.startswith("ret_")]
        m = PatternMatcher(prod_cfg)
        m.fit(train_db, feature_cols)
        _, signals, *_ = m.query(val_db, verbose=0)

        buy_count = int(np.sum(np.asarray(signals) == "BUY"))
        # Frozen baseline: 191 total trades. ±5% tolerance.
        baseline_trades = 191
        assert abs(buy_count - baseline_trades) / baseline_trades <= 0.05, (
            f"BUY count {buy_count} deviates >5% from frozen baseline {baseline_trades}"
        )
