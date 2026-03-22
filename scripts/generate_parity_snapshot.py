"""
generate_parity_snapshot.py — Generate the frozen parity snapshot artifact.

Run this script once when PatternMatcher output is known-good, then commit
the resulting JSON. The parity test suite (test_end_to_end_parity.py) reads
this snapshot on every CI run to detect regressions.

Usage:
    python scripts/generate_parity_snapshot.py

Output:
    rebuild_phase_3z/artifacts/baselines/parity_snapshot.json

When to re-generate:
    - After a deliberate algorithm change that alters matcher output
    - After upgrading sklearn or numpy versions (may change BallTree tie-breaking)
    - Never re-generate to hide a regression — investigate first

Linear: SLE-80
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = REPO_ROOT / "rebuild_phase_3z" / "artifacts" / "baselines" / "parity_snapshot.json"

# Add repo root so imports work from scripts/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Reference dataset (must match test_end_to_end_parity.py exactly) ──────────
FEATURE_COLS = [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]
N_TRAIN = 2000
N_QUERY = 400
RNG_SEED_TRAIN = 42
RNG_SEED_QUERY = 99


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


def _get_engine_config():
    # NOTE: max_distance=1.1019 is calibrated for real financial return data.
    # On i.i.d. standard normal synthetic features, the expected L2 distance
    # between two 8-dimensional points is ~sqrt(8*2) ≈ 4.0 after StandardScaler.
    # We use a relaxed max_distance=4.5 for synthetic data so signals are generated.
    # The PRODUCTION parity tests (with real data) use the locked max_distance=1.1019.
    class SyntheticConfig:
        top_k = 50
        max_distance = 4.5            # relaxed for synthetic 8d normal features
        distance_weighting = "uniform"
        feature_weights = {}          # all 1.0 (no-op)
        batch_size = 256
        confidence_threshold = 0.55   # slightly relaxed to get BUY/SELL signals
        agreement_spread = 0.05
        min_matches = 5               # relaxed for sparser synthetic data
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


def generate_snapshot() -> dict:
    from rebuild_phase_3z.fppe.pattern_engine.matcher import PatternMatcher

    print("Generating parity snapshot...")
    print(f"  Training rows: {N_TRAIN} (seed={RNG_SEED_TRAIN})")
    print(f"  Query rows:    {N_QUERY} (seed={RNG_SEED_QUERY})")

    train_df = _make_train_df()
    query_df = _make_query_df()

    cfg = _get_engine_config()
    matcher = PatternMatcher(cfg)
    matcher.fit(train_df, FEATURE_COLS)
    print(f"  Fitted PatternMatcher (backend={matcher.backend_name})")

    probs, signals, reasons, n_matches, mean_returns, _ = matcher.query(query_df, verbose=0)
    print(f"  Query complete: {len(signals)} rows")

    y_true = query_df["fwd_7d_up"].values
    bss = _bss(np.asarray(probs), y_true)

    buy_count = int(np.sum(np.asarray(signals) == "BUY"))
    sell_count = int(np.sum(np.asarray(signals) == "SELL"))
    hold_count = int(np.sum(np.asarray(signals) == "HOLD"))

    snapshot = {
        "_schema_version": "SLE-80-v1",
        "_generated_at": datetime.utcnow().isoformat() + "Z",
        "_description": "Frozen parity snapshot — PatternMatcher on seeded synthetic data",
        "dataset": {
            "n_train": N_TRAIN,
            "n_query": N_QUERY,
            "rng_seed_train": RNG_SEED_TRAIN,
            "rng_seed_query": RNG_SEED_QUERY,
            "feature_cols": FEATURE_COLS,
        },
        "config": {
            "backend": matcher.backend_name,
            "top_k": cfg.top_k,
            "max_distance": cfg.max_distance,
            "confidence_threshold": cfg.confidence_threshold,
            "agreement_spread": cfg.agreement_spread,
            "min_matches": cfg.min_matches,
        },
        "metrics": {
            "bss": bss,
            "mean_probability": float(np.mean(probs)),
            "std_probability": float(np.std(probs)),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
            "mean_n_matches": float(np.mean(n_matches)),
            "fraction_actionable": float((buy_count + sell_count) / len(signals)),
        },
        "tolerances": {
            "bss_rtol": 1e-5,
            "probability_rtol": 1e-7,
            "signal_counts_exact": True,
        },
    }

    print(f"\n  BSS: {bss:.6f}")
    print(f"  Signals: {buy_count} BUY / {sell_count} SELL / {hold_count} HOLD")
    print(f"  Mean prob: {snapshot['metrics']['mean_probability']:.6f}")
    print(f"  Mean n_matches: {snapshot['metrics']['mean_n_matches']:.1f}")

    return snapshot


def main():
    snapshot = generate_snapshot()
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"\nOK Snapshot written to: {SNAPSHOT_PATH.relative_to(REPO_ROOT)}")
    print("  Commit this file. CI will compare future runs against it.")


if __name__ == "__main__":
    main()
