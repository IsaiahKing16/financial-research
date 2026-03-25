"""
test_asymmetric_universe.py — Asymmetric universe hypothesis test.

Hypothesis: The 585-ticker expansion hurts BSS because mid-cap forward return
labels (fwd_7d_up) are near-random (high vol = coin-flip 7-day outcomes).
If true, using all 585 tickers as the ANALOGUE POOL but evaluating BSS only
on the original 52 large-cap tickers should recover positive BSS.

Design:
  - Train:  Full 585-ticker HNSW index (more analogues = better coverage)
  - Query:  Full 585-ticker val set
  - Eval:   BSS computed on 52-ticker subset of val results only

Compares three conditions on Fold 6 (2024-Val):
  A. 585-pool, 585-val  (current production — vol-norm)
  B. 585-pool, 52-val   (asymmetric — the hypothesis)
  C. 52-pool,  52-val   (original 52-ticker baseline recreation)

Usage:
    python scripts/test_asymmetric_universe.py
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
from pattern_engine.features import VOL_NORM_COLS, RETURNS_ONLY_COLS

DATA_DIR = REPO_ROOT / "data"

# Original 52 large-cap tickers (from data/processed/train_db.parquet)
LARGE_CAP_52 = [
    'AAPL', 'ABBV', 'ADBE', 'AMD',  'AMGN', 'AMZN', 'AVGO', 'AXP',
    'BA',   'BAC',  'BRK-B','CAT',  'COST', 'CRM',  'CSCO', 'CVX',
    'DIS',  'GE',   'GILD', 'GOOGL','GS',   'HD',   'INTC', 'ISRG',
    'JNJ',  'JPM',  'KO',   'LLY',  'MA',   'META', 'MRK',  'MS',
    'MSFT', 'MU',   'NFLX', 'NVDA', 'ORCL', 'PEP',  'PFE',  'PG',
    'PYPL', 'QCOM', 'QQQ',  'SPY',  'TMO',  'TSLA', 'TXN',  'UNH',
    'V',    'WFC',  'WMT',  'XOM',
]

FOLD6 = {
    "label":     "2024-Val",
    "train_end": "2023-12-31",
    "val_start": "2024-01-01",
    "val_end":   "2024-12-31",
}


@dataclass
class Cfg:
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


def directional_acc(probs: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((probs > 0.5).astype(int) == y_true.astype(int)))


def report(label: str, probs: np.ndarray, y_true: np.ndarray,
           n_matches, signals) -> None:
    sigs = np.array(signals)
    print(f"\n[{label}]")
    print(f"  Val rows  : {len(y_true):,}  base_rate={y_true.mean():.4f}")
    print(f"  AvgK      : {np.mean(n_matches):.1f}")
    print(f"  Probs     : mean={probs.mean():.4f}  std={probs.std():.4f}  "
          f"min={probs.min():.4f}  max={probs.max():.4f}  "
          f">0.65: {(probs > 0.65).mean():.1%}  <0.35: {(probs < 0.35).mean():.1%}")
    b = bss(probs, y_true)
    print(f"  BSS       : {b:+.6f}  {'★ POSITIVE' if b > 0 else '(baseline: +0.00103 on 52-ticker)'}")
    print(f"  Accuracy  : {directional_acc(probs, y_true):.1%}")
    print(f"  Signals   : BUY={int((sigs=='BUY').sum())}  "
          f"SELL={int((sigs=='SELL').sum())}  HOLD={int((sigs=='HOLD').sum())}")


def main() -> None:
    print("=" * 70)
    print("  Asymmetric Universe Test — Fold 6 (2024-Val)")
    print("=" * 70)
    print("  Hypothesis: mid-cap fwd_7d_up labels are near-random (high vol).")
    print("  Test: 585-ticker analogue pool + 52-ticker signal evaluation")
    print("  Expected: if H is true → B shows positive BSS; if false → all negative\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    full_db_585 = pd.concat([
        pd.read_parquet(DATA_DIR / "train_db.parquet"),
        pd.read_parquet(DATA_DIR / "val_db.parquet"),
    ], ignore_index=True)
    full_db_585["Date"] = pd.to_datetime(full_db_585["Date"])

    full_db_52 = full_db_585[full_db_585["Ticker"].isin(LARGE_CAP_52)].copy()

    print(f"  585-ticker DB: {len(full_db_585):,} rows")
    print(f"   52-ticker DB: {len(full_db_52):,} rows")

    def split(db, feature_cols):
        train = db[db["Date"] <= pd.Timestamp(FOLD6["train_end"])].dropna(
            subset=["fwd_7d_up"]).copy()
        val   = db[(db["Date"] >= pd.Timestamp(FOLD6["val_start"])) &
                   (db["Date"] <= pd.Timestamp(FOLD6["val_end"]))].dropna(
            subset=["fwd_7d_up"]).copy()
        return train, val

    # ── Condition A: 585-pool, 585-val (current production, vol-norm) ─────────
    print(f"\n{'─'*70}")
    print("Fitting A: 585-pool, vol-norm features...")
    train_585, val_585 = split(full_db_585, VOL_NORM_COLS)
    cfg = Cfg()
    t0 = time.time()
    m585 = PatternMatcher(cfg)
    m585.fit(train_585, VOL_NORM_COLS)
    print(f"  Fit done in {time.time()-t0:.0f}s  "
          f"(train={len(train_585):,}, val={len(val_585):,})")

    probs_a, sigs_a, _, nk_a, _, _ = m585.query(val_585, verbose=0)
    report("A: 585-pool, 585-val [current production — vol-norm]",
           np.asarray(probs_a), val_585["fwd_7d_up"].values.astype(float),
           nk_a, sigs_a)

    # ── Condition B: 585-pool, 52-val (asymmetric) ────────────────────────────
    print(f"\n{'─'*70}")
    print("Reusing 585 matcher → restricting val to 52 large-cap tickers...")
    val_52_from_585 = val_585[val_585["Ticker"].isin(LARGE_CAP_52)].copy()
    probs_b, sigs_b, _, nk_b, _, _ = m585.query(val_52_from_585, verbose=0)
    report("B: 585-pool, 52-val [ASYMMETRIC — the hypothesis]",
           np.asarray(probs_b), val_52_from_585["fwd_7d_up"].values.astype(float),
           nk_b, sigs_b)

    # ── Condition C: 52-pool, 52-val, vol-norm ────────────────────────────────
    print(f"\n{'─'*70}")
    print("Fitting C: 52-pool, vol-norm features...")
    train_52, val_52 = split(full_db_52, VOL_NORM_COLS)
    t0 = time.time()
    m52 = PatternMatcher(Cfg())
    m52.fit(train_52, VOL_NORM_COLS)
    print(f"  Fit done in {time.time()-t0:.0f}s  "
          f"(train={len(train_52):,}, val={len(val_52):,})")
    probs_c, sigs_c, _, nk_c, _, _ = m52.query(val_52, verbose=0)
    report("C: 52-pool, 52-val [vol-norm]",
           np.asarray(probs_c), val_52["fwd_7d_up"].values.astype(float),
           nk_c, sigs_c)

    # ── Condition D: 52-pool, 52-val, RAW RETURNS (true baseline recreation) ──
    # The original +0.00103 baseline used raw returns, not vol-norm.
    # If D recovers +0.00103, vol-norm is the regression source for 52T.
    print(f"\n{'─'*70}")
    print("Fitting D: 52-pool, RAW return features (true baseline recreation)...")
    # Raw return features require data/processed/train_db.parquet (52-ticker, raw)
    raw_52_path = REPO_ROOT / "data" / "processed"
    raw_train = pd.read_parquet(raw_52_path / "train_db.parquet")
    raw_val   = pd.read_parquet(raw_52_path / "val_db.parquet")
    for df in (raw_train, raw_val):
        df["Date"] = pd.to_datetime(df["Date"])
    raw_train = raw_train[raw_train["Date"] <= pd.Timestamp(FOLD6["train_end"])].dropna(
        subset=["fwd_7d_up"]).copy()
    raw_val_f6 = raw_val[(raw_val["Date"] >= pd.Timestamp(FOLD6["val_start"])) &
                         (raw_val["Date"] <= pd.Timestamp(FOLD6["val_end"]))].dropna(
        subset=["fwd_7d_up"]).copy()
    t0 = time.time()
    m52r = PatternMatcher(Cfg())
    m52r.fit(raw_train, RETURNS_ONLY_COLS)
    print(f"  Fit done in {time.time()-t0:.0f}s  "
          f"(train={len(raw_train):,}, val={len(raw_val_f6):,})")
    probs_d, sigs_d, _, nk_d, _, _ = m52r.query(raw_val_f6, verbose=0)
    report("D: 52-pool, 52-val [RAW RETURNS — true baseline recreation]",
           np.asarray(probs_d), raw_val_f6["fwd_7d_up"].values.astype(float),
           nk_d, sigs_d)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  INTERPRETATION")
    bss_a = bss(np.asarray(probs_a), val_585["fwd_7d_up"].values.astype(float))
    bss_b = bss(np.asarray(probs_b), val_52_from_585["fwd_7d_up"].values.astype(float))
    bss_c = bss(np.asarray(probs_c), val_52["fwd_7d_up"].values.astype(float))
    bss_d = bss(np.asarray(probs_d), raw_val_f6["fwd_7d_up"].values.astype(float))
    print(f"  A (585/585, vol-norm): {bss_a:+.6f}")
    print(f"  B (585/52,  vol-norm): {bss_b:+.6f}  ← asymmetric")
    print(f"  C (52/52,   vol-norm): {bss_c:+.6f}  ← vol-norm 52T")
    print(f"  D (52/52,   raw ret):  {bss_d:+.6f}  ← TRUE baseline recreation")
    print(f"  Baseline:              +0.001030  ← original (raw returns, locked)")
    if bss_d > 0.0005:
        print("\n  ★ D recovers baseline — vol-norm is the regression source for 52T.")
        print("    Vol-norm helps 585T (-0.00073→-0.00031) but hurts 52T (+0.00103→C).")
        print("    Solution: feature set should be UNIVERSE-DEPENDENT.")
    elif bss_d < 0:
        print("\n  → D is negative — HNSW may differ from BallTree baseline.")
        print("    The +0.00103 baseline may have been a BallTree-specific result.")
    print("=" * 70)


if __name__ == "__main__":
    main()
