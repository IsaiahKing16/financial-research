"""
diagnose_signal.py — Diagnose whether raw K-NN signal exists in the
585-ticker universe on Fold 6 (2024-Val).

Tests two hypotheses:

  H1 (Calibration collapse): The Platt calibrator, trained on 585-ticker
     data where AvgK≈49, learned a near-identity sigmoid that maps all probs
     to ~base rate. Raw K-NN signal exists but gets erased by calibration.
     FIX: calibration_method="none" (or re-fit on 52-ticker subsample).

  H2 (Universe noise): The expanded universe (585 tickers, many mid-caps)
     genuinely has weaker K-NN signal than the curated 52-ticker set.
     Raw probs and calibrated probs are equally near the base rate.
     FIX: restrict validation set to original 52 large-cap tickers.

This script runs fold 6 three ways and compares:
  A. cal=platt    (current production — shows the sweep baseline)
  B. cal=none     (raw K-NN frequency — tests H1)
  C. cal=platt, val_tickers=original 52 (tests H2)

Usage:
    python scripts/diagnose_signal.py
"""
from __future__ import annotations

import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import RETURNS_ONLY_COLS

DATA_DIR = REPO_ROOT / "data"
FEATURE_COLS = RETURNS_ONLY_COLS

FOLD6 = {
    "train_end": "2023-12-31",
    "val_start": "2024-01-01",
    "val_end":   "2024-12-31",
}

# Original 52-ticker baseline universe (pre-expansion)
# Source: pattern_engine/sector.py SECTOR_PROXIES + original hardcoded list
# These are the tickers that produced fold6 BSS=+0.00103
ORIGINAL_52 = [
    "SPY", "QQQ",
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    "ADBE", "CRM", "ORCL", "INTC", "QCOM", "TXN", "AMAT", "LRCX", "MU",
    "JPM", "BAC", "GS", "MS", "WFC", "BLK", "V", "MA",
    "JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK", "TMO",
    "AMGN", "GILD",
    "COST", "WMT", "HD", "NKE", "SBUX", "MCD", "TGT",
    "CAT", "HON", "UPS", "BA", "GE", "MMM",
    "XOM", "CVX",
]


@dataclass
class DiagConfig:
    top_k: int = 50
    max_distance: float = 1.1019
    distance_weighting: str = "uniform"
    feature_weights: dict = field(default_factory=dict)
    batch_size: int = 256
    confidence_threshold: float = 0.65
    agreement_spread: float = 0.05
    min_matches: int = 5
    exclude_same_ticker: bool = True
    same_sector_only: bool = False
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


def prob_stats(probs: np.ndarray) -> str:
    return (f"mean={probs.mean():.4f}  std={probs.std():.4f}  "
            f"min={probs.min():.4f}  max={probs.max():.4f}  "
            f">0.65: {(probs > 0.65).mean():.1%}  <0.35: {(probs < 0.35).mean():.1%}")


def run(label: str, matcher: PatternMatcher, val_db: pd.DataFrame,
        cfg: DiagConfig) -> None:
    t0 = time.time()
    probs, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
    elapsed = time.time() - t0

    probs_arr = np.asarray(probs)
    y_true    = val_db[cfg.projection_horizon].values.astype(np.float64)
    sigs      = np.array(signals)

    score = bss(probs_arr, y_true)
    base_rate = y_true.mean()

    print(f"\n  [{label}]")
    print(f"    Val rows : {len(val_db):,}  base_rate={base_rate:.4f}")
    print(f"    AvgK     : {np.mean(n_matches):.1f}")
    print(f"    Probs    : {prob_stats(probs_arr)}")
    print(f"    BSS      : {score:+.6f}  (baseline +0.00103 on 52-ticker)")
    print(f"    Signals  : BUY={int((sigs=='BUY').sum())}  "
          f"SELL={int((sigs=='SELL').sum())}  "
          f"HOLD={int((sigs=='HOLD').sum())}")
    print(f"    Elapsed  : {elapsed:.0f}s")
    return score


def main() -> None:
    print("=" * 70)
    print("  FPPE Signal Diagnostics — Fold 6 (2024-Val), 585-ticker universe")
    print("=" * 70)
    print("\n  Hypotheses:")
    print("  H1: Calibration collapse — raw K-NN signal exists, Platt erases it")
    print("  H2: Universe noise — 585-ticker mix has weaker signal than 52-ticker")

    # Load
    full_db = pd.concat([pd.read_parquet(DATA_DIR / "train_db.parquet"),
                         pd.read_parquet(DATA_DIR / "val_db.parquet")],
                        ignore_index=True)
    full_db["Date"] = pd.to_datetime(full_db["Date"])

    train_db = full_db[full_db["Date"] <= pd.Timestamp(FOLD6["train_end"])].dropna(
        subset=["fwd_7d_up"]).copy()
    val_db_full = full_db[
        (full_db["Date"] >= pd.Timestamp(FOLD6["val_start"])) &
        (full_db["Date"] <= pd.Timestamp(FOLD6["val_end"]))
    ].dropna(subset=["fwd_7d_up"]).copy()

    # Restrict to original 52 tickers that exist in the dataset
    available_52 = [t for t in ORIGINAL_52 if t in val_db_full["Ticker"].values]
    val_db_52 = val_db_full[val_db_full["Ticker"].isin(available_52)].copy()

    print(f"\n  Loaded {len(train_db):,} training rows, "
          f"{len(val_db_full):,} val rows (all 585 tickers)")
    print(f"  Original-52 tickers available in val: "
          f"{len(available_52)}/52 → {len(val_db_52):,} val rows")

    # ── A. Production (platt cal, all 585 tickers) ───────────────────────────
    print("\n" + "─" * 70)
    print("  Fitting with calibration_method='platt' (current production)...")
    cfg_platt = DiagConfig(calibration_method="platt")
    t_fit = time.time()
    matcher_platt = PatternMatcher(cfg_platt)
    matcher_platt.fit(train_db, FEATURE_COLS)
    print(f"  Fit done in {time.time() - t_fit:.0f}s")

    run("A: platt cal, val=all 585 tickers", matcher_platt, val_db_full, cfg_platt)

    # ── B. No calibration (raw K-NN, all 585 tickers) ────────────────────────
    print("\n  Fitting with calibration_method='none' (raw K-NN frequencies)...")
    cfg_none = DiagConfig(calibration_method="none")
    t_fit = time.time()
    matcher_none = PatternMatcher(cfg_none)
    matcher_none.fit(train_db, FEATURE_COLS)
    print(f"  Fit done in {time.time() - t_fit:.0f}s")

    run("B: no cal, val=all 585 tickers  [tests H1]", matcher_none, val_db_full, cfg_none)

    # ── C. Platt cal, val restricted to original 52 tickers ──────────────────
    # Reuse matcher_platt — no refit needed
    print("\n  Reusing platt matcher, restricting val to original 52 tickers...")
    run("C: platt cal, val=original 52   [tests H2]", matcher_platt, val_db_52, cfg_platt)

    # ── D. No cal, val restricted to original 52 tickers ────────────────────
    print("\n  Reusing no-cal matcher, restricting val to original 52 tickers...")
    score_d = run("D: no cal, val=original 52  [H1+H2]", matcher_none, val_db_52, cfg_none)

    print("\n" + "=" * 70)
    print("  INTERPRETATION GUIDE")
    print("  A=negative, B=positive → H1 confirmed: calibration is the problem")
    print("  A=negative, C=positive → H2 confirmed: mid-cap noise is the problem")
    print("  A=negative, B+C=negative → deep universe/feature issue (M9 scope)")
    print("  D=positive → both fixes together work → 52-ticker val + no-cal is viable")
    print("=" * 70)


if __name__ == "__main__":
    main()
