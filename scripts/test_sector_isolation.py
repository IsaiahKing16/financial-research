"""
test_sector_isolation.py — Test same_sector_only=True on Fold 6 (2024-Val).

Hypothesis: Cross-sector analogue contamination is diluting signal.
A Tech stock's return fingerprint matching a Utility stock's historical
period is structurally meaningless — sector context is load-bearing.

This script compares fold 6 BSS with same_sector_only=False vs True,
using the same fitted HNSW index (refit once, sector filter is post-filter).

Usage:
    python scripts/test_sector_isolation.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import RETURNS_ONLY_COLS

DATA_DIR     = REPO_ROOT / "data"
FEATURE_COLS = RETURNS_ONLY_COLS

FOLD6 = {"train_end": "2023-12-31", "val_start": "2024-01-01", "val_end": "2024-12-31"}


@dataclass
class TestConfig:
    top_k: int = 50
    max_distance: float = 1.1019
    distance_weighting: str = "uniform"
    feature_weights: dict = field(default_factory=dict)
    batch_size: int = 256
    confidence_threshold: float = 0.65
    agreement_spread: float = 0.05
    min_matches: int = 5
    exclude_same_ticker: bool = True
    same_sector_only: bool = False        # toggled below
    regime_filter: bool = False
    regime_fallback: bool = False
    projection_horizon: str = "fwd_7d_up"
    calibration_method: str = "platt"
    cal_max_samples: int = 100_000
    use_hnsw: bool = True
    use_sax_filter: bool = False
    use_wfa_rerank: bool = False
    use_ib_compression: bool = False


def bss(probs: np.ndarray, y_true: np.ndarray) -> float:
    brier      = float(np.mean((probs - y_true) ** 2))
    brier_clim = float(np.var(y_true))
    return (1.0 - brier / brier_clim) if brier_clim > 0 else 0.0


def report(label: str, probs, signals, n_matches, val_db, cfg):
    probs_arr = np.asarray(probs)
    y_true    = val_db[cfg.projection_horizon].values.astype(np.float64)
    sigs      = np.array(signals)
    score     = bss(probs_arr, y_true)
    print(f"\n  [{label}]")
    print(f"    AvgK   : {np.mean(n_matches):.1f}")
    print(f"    Probs  : mean={probs_arr.mean():.4f}  std={probs_arr.std():.4f}"
          f"  >0.65: {(probs_arr > 0.65).mean():.1%}")
    print(f"    BSS    : {score:+.6f}")
    print(f"    Signals: BUY={int((sigs=='BUY').sum())}  "
          f"SELL={int((sigs=='SELL').sum())}  HOLD={int((sigs=='HOLD').sum())}")
    return score


def main():
    print("=" * 68)
    print("  Sector Isolation Test — Fold 6 (2024-Val), 585-ticker universe")
    print("=" * 68)

    full_db = pd.concat([pd.read_parquet(DATA_DIR / "train_db.parquet"),
                         pd.read_parquet(DATA_DIR / "val_db.parquet")],
                        ignore_index=True)
    full_db["Date"] = pd.to_datetime(full_db["Date"])

    train_db = full_db[full_db["Date"] <= pd.Timestamp(FOLD6["train_end"])].dropna(
        subset=["fwd_7d_up"]).copy()
    val_db   = full_db[
        (full_db["Date"] >= pd.Timestamp(FOLD6["val_start"])) &
        (full_db["Date"] <= pd.Timestamp(FOLD6["val_end"]))
    ].dropna(subset=["fwd_7d_up"]).copy()

    print(f"\n  train={len(train_db):,}  val={len(val_db):,}")

    # ── A. Baseline: same_sector_only=False ──────────────────────────────────
    print("\n  Fitting (same_sector_only=False, calibration=platt)...")
    cfg_off = TestConfig(same_sector_only=False)
    t0 = time.time()
    m_off = PatternMatcher(cfg_off)
    m_off.fit(train_db, FEATURE_COLS)
    print(f"  Fit done in {time.time()-t0:.0f}s")

    p, s, _, n, _, _ = m_off.query(val_db, verbose=0)
    bss_off = report("same_sector_only=False (current locked)", p, s, n, val_db, cfg_off)

    # ── B. same_sector_only=True ─────────────────────────────────────────────
    # No refit needed — sector filter is Stage-4 post-filter only
    print("\n  Switching to same_sector_only=True (no refit)...")
    m_off.config.same_sector_only = True
    t0 = time.time()
    p2, s2, _, n2, _, _ = m_off.query(val_db, verbose=0)
    elapsed = time.time() - t0
    bss_on = report(f"same_sector_only=True (elapsed {elapsed:.0f}s)", p2, s2, n2, val_db, cfg_off)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    delta = bss_on - bss_off
    print(f"  BSS delta: {delta:+.6f}  "
          f"({'improvement' if delta > 0 else 'regression'})")
    print(f"  Baseline (52-ticker fold 6): BSS=+0.00103")
    if bss_on > bss_off:
        print("\n  → Sector isolation HELPS. Consider same_sector_only=True.")
        print("    Next: run full 6-fold walk-forward with same_sector_only=True.")
    else:
        print("\n  → Sector isolation does NOT help. Cross-sector noise is not")
        print("    the primary driver. Next: investigate σ-normalized features.")
    print("=" * 68)


if __name__ == "__main__":
    main()
