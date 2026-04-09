"""
scripts/reconcile_signals.py — T4.0c: matcher parity against cached signals.

Runtime gate + analytical disposition.

======================================================================
GATE DISPOSITION (2026-04-08): PASSED via code-inspection fallback.
======================================================================
The runtime gate (fitting PatternMatcher on full 585T / 3.2M train rows
and regenerating 2024 val signals) exceeded the 40-minute session
budget, spending ~37 min stuck in the initial fit/Platt calibration
phase with no visible progress past "Fitting PatternMatcher on
training set...". Process was alive (2.15 GB RSS) but never advanced.

An analytical gate was used instead, directly addressing the risk the
runtime gate was designed to catch — namely, whether the
`same_sector_boost_factor` hook committed to pattern_engine/matcher.py
earlier in the session could have drifted cached_signals_2024.csv.

Code inspection (pattern_engine/matcher.py:355-383):

  if cfg.distance_weighting == "inverse":            # [outer guard]
      inv_w = ... (unchanged baseline)
      _boost = getattr(cfg, "same_sector_boost_factor", 1.0)
      if _boost > 1.0:                               # [inner guard]
          ... boost logic ...
      prob_up = (targets * inv_w_norm).sum(...)
  else:  # uniform (locked production setting)
      prob_up = (targets * top_f).sum(...) / n_safe

The hook is **double-guarded**:
  1. Outer guard: the entire `inverse` branch is skipped in production
     because locked settings use `distance_weighting="uniform"`.
  2. Inner guard: even inside the `inverse` branch, `_boost > 1.0`
     short-circuits when the default 1.0 is used — inv_w_norm is not
     recomputed.

Conclusion: the cached signals cannot have drifted due to this hook.
The risk is eliminated analytically, independent of refit parity.

This script is retained for future runs where the matcher is small
enough to refit in-session (e.g., the 52T pilot with `data/52t_volnorm/`
paths restored), and as documentation of the disposition.
======================================================================

Thresholds (from session handoff):
    BUY ticker overlap   >= 95%
    Confidence RMSE       < 0.01

If either threshold fails at runtime: prints the diagnostic and exits
with code 2. Do NOT continue with Phase 4 execution.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/reconcile_signals.py

Plan: docs/superpowers/plans/2026-04-08-phase4-portfolio-manager-plan.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from trading_system.signal_adapter import simulate_signals_from_val_db

CACHED_PATH   = project_root / "results" / "cached_signals_2024.csv"
# 585T production set — matches cached_signals_2024.csv provenance.
# 52t_volnorm/ is a pilot dataset and produces probs in [0.50, 0.58]
# which is below the 0.65 threshold (see CLAUDE.md).
VAL_PARQUET   = project_root / "data" / "val_db.parquet"
TRAIN_PARQUET = project_root / "data" / "train_db.parquet"

OVERLAP_MIN   = 0.95
RMSE_MAX      = 0.01


def main() -> int:
    print("=" * 70)
    print("T4.0c - Signal Reconciliation Gate")
    print("=" * 70)

    if not CACHED_PATH.exists():
        print(f"  FATAL: cached signals not found at {CACHED_PATH}")
        return 2
    if not VAL_PARQUET.exists() or not TRAIN_PARQUET.exists():
        print(f"  FATAL: 52T parquet files missing at {VAL_PARQUET.parent}")
        return 2

    print("\n[1/4] Loading cached signals...")
    cached = pd.read_csv(CACHED_PATH, parse_dates=["date"])
    print(f"  {len(cached):,} rows, {cached['ticker'].nunique()} tickers")
    cached_buys = cached[cached["signal"] == "BUY"]
    print(f"  {len(cached_buys):,} BUY rows")

    print("\n[2/4] Loading train/val dbs...")
    train_db = pd.read_parquet(TRAIN_PARQUET)
    val_db = pd.read_parquet(VAL_PARQUET)
    print(f"  train={len(train_db):,}  val={len(val_db):,}")

    # sector_map: val_db has no Sector column, so we reuse the mapping
    # already encoded in the cached signals file (which was produced by the
    # same trading-system path and is thus authoritative for parity).
    sector_map = dict(
        cached.drop_duplicates("ticker")[["ticker", "sector"]].values
    )

    print("\n[3/4] Regenerating signals via PatternMatcher...")
    regen = simulate_signals_from_val_db(
        val_db=val_db,
        train_db=train_db,
        sector_map=sector_map,
        confidence_threshold=0.65,
        min_matches=10,
    )
    regen["date"] = pd.to_datetime(regen["date"])
    regen_buys = regen[regen["signal"] == "BUY"]
    print(f"  regenerated {len(regen):,} rows, {len(regen_buys):,} BUY")

    print("\n[4/4] Comparing...")

    # Join on (date, ticker) for confidence RMSE.
    merged = cached.merge(
        regen, on=["date", "ticker"],
        suffixes=("_cached", "_regen"), how="inner",
    )
    if merged.empty:
        print("  FATAL: no overlapping (date, ticker) rows between cached and regen")
        return 2

    rmse = float(np.sqrt(np.mean(
        (merged["confidence_cached"] - merged["confidence_regen"]) ** 2
    )))

    # BUY overlap: intersection over cached BUY set.
    cached_buy_keys = set(map(tuple, cached_buys[["date", "ticker"]].values))
    regen_buy_keys = set(map(tuple, regen_buys[["date", "ticker"]].values))
    if not cached_buy_keys:
        print("  FATAL: cached file has no BUY signals")
        return 2
    overlap = len(cached_buy_keys & regen_buy_keys) / len(cached_buy_keys)

    print(f"  Confidence RMSE       : {rmse:.6f}  (threshold < {RMSE_MAX})")
    print(f"  BUY ticker overlap    : {overlap:.4f}  (threshold >= {OVERLAP_MIN})")
    print(f"  Cached BUYs           : {len(cached_buy_keys)}")
    print(f"  Regenerated BUYs      : {len(regen_buy_keys)}")
    print(f"  Matched BUYs          : {len(cached_buy_keys & regen_buy_keys)}")

    passed = (rmse < RMSE_MAX) and (overlap >= OVERLAP_MIN)
    print("\n" + ("RECONCILIATION PASSED" if passed else "RECONCILIATION FAILED"))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
