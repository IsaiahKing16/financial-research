"""
run_walkforward.py — 6-fold expanding-window walk-forward validation.

Uses the production PatternMatcher (Phase 3Z) with locked settings from CLAUDE.md.
Reads full_db.parquet produced by prepare.py — no re-download required.

Usage:
    python scripts/run_walkforward.py

Expected baseline (52-ticker, pre-expansion):
    Fold 1 (2019) BSS: -0.01211   Acc: 61.2%
    Fold 2 (2020) BSS: -0.00142   Acc: 57.4%
    Fold 3 (2021) BSS: -0.00131   Acc: 57.7%
    Fold 4 (2022) BSS: -0.03208   Acc: 48.7%
    Fold 5 (2023) BSS: -0.00029   Acc: 56.2%
    Fold 6 (2024) BSS: +0.00103   Acc: 56.6%  ← only positive fold
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import RETURNS_ONLY_COLS, VOL_NORM_COLS

DATA_DIR = REPO_ROOT / "data"           # prepare.py writes here (root data/)
PROCESSED_DIR = DATA_DIR / "processed"  # legacy path (pre-expansion files)

# M9: switched to volatility-normalized features.
# Diagnostic evidence (diagnose_signal.py 2026-03-24): raw return magnitudes
# produce spurious cross-ticker K-NN matches in heterogeneous 585-ticker universe.
# To revert to locked baseline: FEATURE_COLS = RETURNS_ONLY_COLS
FEATURE_COLS = VOL_NORM_COLS  # 8 vol-normalized features: ret_Xd / rolling_std_Xd


def load_full_db() -> pd.DataFrame:
    """Load the full analogue database (train + val splits only; test excluded).

    Search order — first matching pair wins:
      1. data/train_db.parquet + data/val_db.parquet       ← prepare.py output
      2. data/processed/train_db.parquet + val_db.parquet  ← legacy fallback
    Single-file fallback: data/full_analogue_db.parquet
    """
    candidates = [DATA_DIR, PROCESSED_DIR]
    for base in candidates:
        t, v = base / "train_db.parquet", base / "val_db.parquet"
        if t.exists() and v.exists():
            train = pd.read_parquet(t)
            val   = pd.read_parquet(v)
            combined = pd.concat([train, val], ignore_index=True)
            print(f"  Loaded train+val: {len(combined):,} rows from {base}")
            return combined

    for fallback in (DATA_DIR / "full_analogue_db.parquet",
                     PROCESSED_DIR / "full_db.parquet"):
        if fallback.exists():
            print(f"  WARNING: using fallback {fallback}")
            return pd.read_parquet(fallback)

    raise FileNotFoundError(
        f"No analogue database found in {DATA_DIR}. Run prepare.py first."
    )


# ── Locked config (CLAUDE.md — do not change without new experiment evidence) ─

@dataclass
class WalkForwardConfig:
    """Minimal config shim with all attributes PatternMatcher reads via getattr."""
    # Locked settings
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
    # Research pilots — all off
    use_sax_filter: bool = False
    use_wfa_rerank: bool = False
    use_ib_compression: bool = False
    journal_top_n: int = 25   # 0=disabled, 5/10/25=capture top-N analogues per BUY/SELL


# ── Walk-forward fold definitions (expanding training window) ─────────────────

FOLDS = [
    {"label": "2019",       "train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2019-12-31"},
    {"label": "2020-COVID", "train_end": "2019-12-31", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"label": "2021",       "train_end": "2020-12-31", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"label": "2022-Bear",  "train_end": "2021-12-31", "val_start": "2022-01-01", "val_end": "2022-12-31"},
    {"label": "2023",       "train_end": "2022-12-31", "val_start": "2023-01-01", "val_end": "2023-12-31"},
    {"label": "2024-Val",   "train_end": "2023-12-31", "val_start": "2024-01-01", "val_end": "2024-12-31"},
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def bss(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Brier Skill Score vs. climatological baseline."""
    brier = float(np.mean((probs - y_true) ** 2))
    brier_clim = float(np.var(y_true))
    return (1.0 - brier / brier_clim) if brier_clim > 0 else 0.0


def accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Directional accuracy: fraction of rows where prob>0.5 matches y_true.

    Uses the full prediction distribution (not filtered by confidence_threshold)
    so it matches the legacy baseline's reported accuracy numbers.
    """
    predicted = (probs > 0.5).astype(int)
    return float(np.mean(predicted == y_true.astype(int)))


# ── Main ─────────────────────────────────────────────────────────────────────

def run_fold(full_db: pd.DataFrame, fold: dict, cfg: WalkForwardConfig) -> dict:
    train_end   = pd.Timestamp(fold["train_end"])
    val_start   = pd.Timestamp(fold["val_start"])
    val_end     = pd.Timestamp(fold["val_end"])

    train_db = full_db[full_db["Date"] <= train_end].copy()
    val_db   = full_db[(full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)].copy()

    # Drop rows missing the target (forward return labels unavailable near data edge)
    train_db = train_db.dropna(subset=[cfg.projection_horizon])
    val_db   = val_db.dropna(subset=[cfg.projection_horizon])

    if len(val_db) == 0:
        return {"label": fold["label"], "bss": float("nan"), "accuracy": float("nan"),
                "n_train": len(train_db), "n_val": 0, "avg_k": 0.0,
                "buy": 0, "sell": 0, "hold": 0}

    matcher = PatternMatcher(cfg)
    matcher.fit(train_db, FEATURE_COLS)

    probs, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)

    # Write decision journal for this fold (BUY/SELL signals only)
    _j_top_n = getattr(cfg, 'journal_top_n', 0)
    if _j_top_n > 0 and matcher.last_journal:
        from pattern_engine.journal import write_journal_parquet
        from pathlib import Path
        _jdir = REPO_ROOT / "results" / "journals"
        _jdir.mkdir(parents=True, exist_ok=True)
        _jpath = _jdir / f"journal_fold_{fold['label'].replace(' ', '_')}.parquet"
        write_journal_parquet(matcher.last_journal, _jpath)
        print(f"  Journal: {len(matcher.last_journal)} BUY/SELL entries -> {_jpath.name}")

    probs_arr = np.asarray(probs)
    y_true    = val_db[cfg.projection_horizon].values.astype(np.float64)

    return {
        "label":    fold["label"],
        "bss":      bss(probs_arr, y_true),
        "accuracy": accuracy(probs_arr, y_true),
        "n_train":  len(train_db),
        "n_val":    len(val_db),
        "avg_k":    float(np.mean(n_matches)),
        "buy":      int(np.sum(np.array(signals) == "BUY")),
        "sell":     int(np.sum(np.array(signals) == "SELL")),
        "hold":     int(np.sum(np.array(signals) == "HOLD")),
    }


def main() -> None:
    t0 = time.time()

    print("=" * 68)
    print("  FPPE Walk-Forward Validation — Phase 3Z Production PatternMatcher")
    print("=" * 68)
    feature_set = "vol_normalized_8" if FEATURE_COLS is VOL_NORM_COLS else "returns_only_8"
    print(f"\nConfig: max_d={1.1019}, top_k=50, thresh={0.65}, cal=platt")
    print(f"Features: {feature_set} -> {FEATURE_COLS}")
    print(f"\nLoading analogue database from {PROCESSED_DIR}...")

    full_db = load_full_db()
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    print(f"  {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers, "
          f"{full_db['Date'].min().date()} to {full_db['Date'].max().date()}")

    cfg = WalkForwardConfig()
    results = []

    print(f"\n{'Fold':<14} {'Train':>8} {'Val':>8} {'BSS':>9} {'Acc':>7} {'AvgK':>6} "
          f"{'BUY':>5} {'SELL':>5} {'HOLD':>6}")
    print("-" * 68)

    for i, fold in enumerate(FOLDS, 1):
        print(f"  [{i}/6] fitting {fold['label']}...", end=" ", flush=True)
        t_fold = time.time()
        r = run_fold(full_db, fold, cfg)
        elapsed = time.time() - t_fold
        results.append(r)

        bss_str = f"{r['bss']:+.5f}" if not np.isnan(r['bss']) else "   N/A "
        acc_str = f"{r['accuracy']:.1%}" if not np.isnan(r['accuracy']) else "  N/A "
        flag = " *" if r["bss"] > 0 else ""
        print(f"done ({elapsed:.0f}s)")
        print(f"{r['label']:<14} {r['n_train']:>8,} {r['n_val']:>8,} "
              f"{bss_str:>9} {acc_str:>7} {r['avg_k']:>6.1f} "
              f"{r['buy']:>5} {r['sell']:>5} {r['hold']:>6}{flag}")

    total = time.time() - t0
    print("-" * 68)

    pos_folds = sum(1 for r in results if r["bss"] > 0)
    mean_bss = np.nanmean([r["bss"] for r in results])
    print(f"\nPositive-BSS folds: {pos_folds}/6   Mean BSS: {mean_bss:+.5f}")
    print(f"Total runtime: {total:.0f}s")

    if pos_folds > 1:
        print("\nWARNING: >1 positive-BSS fold -- verify no data leakage.")
    if any(r["avg_k"] >= 49.9 for r in results):
        print("\nWARNING: AvgK ~= 50 in a fold -- regime filter may be broken.")

    print("\nBaseline (52-ticker):  Fold6 BSS=+0.00103  Mean BSS≈-0.00786")
    print("=" * 68)


if __name__ == "__main__":
    main()
