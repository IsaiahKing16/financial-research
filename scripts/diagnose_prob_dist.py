"""
diagnose_prob_dist.py — Platt probability distribution diagnostic for fold 6.

Runs the 2024-Val fold only and reports the full calibrated probability
distribution so we can choose a correct confidence_threshold for E1/E2/E3.

Usage:
    python scripts/diagnose_prob_dist.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import VOL_NORM_COLS
from scripts.run_walkforward import WalkForwardConfig, load_full_db

FOLD = {"label": "2024-Val", "train_end": "2023-12-31", "val_start": "2024-01-01", "val_end": "2024-12-31"}
FEATURE_COLS = VOL_NORM_COLS

PERCENTILES = [1, 5, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]


def main() -> None:
    print("=" * 60)
    print("  Platt Probability Distribution — Fold 6 (2024-Val)")
    print("=" * 60)

    full_db = load_full_db()
    full_db["Date"] = pd.to_datetime(full_db["Date"])

    cfg = WalkForwardConfig()  # threshold irrelevant — we capture raw probs

    train_db = full_db[full_db["Date"] <= pd.Timestamp(FOLD["train_end"])].copy()
    val_db   = full_db[
        (full_db["Date"] >= pd.Timestamp(FOLD["val_start"])) &
        (full_db["Date"] <= pd.Timestamp(FOLD["val_end"]))
    ].copy()
    train_db = train_db.dropna(subset=[cfg.projection_horizon])
    val_db   = val_db.dropna(subset=[cfg.projection_horizon])

    print(f"\nTrain rows: {len(train_db):,}  |  Val rows: {len(val_db):,}")
    print("Fitting PatternMatcher...", end=" ", flush=True)
    matcher = PatternMatcher(cfg)
    matcher.fit(train_db, FEATURE_COLS)
    print("done")

    print("Querying val set...", end=" ", flush=True)
    probs, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
    probs = np.asarray(probs)
    print("done\n")

    y_true = val_db[cfg.projection_horizon].values.astype(np.float64)
    base_rate = float(y_true.mean())

    print(f"Base rate (P(up)): {base_rate:.4f}")
    print(f"N predictions:     {len(probs):,}")
    print(f"Prob min/max:      [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"Prob mean/std:     {probs.mean():.4f} / {probs.std():.4f}")
    print()

    print("── Percentile table ──────────────────────────────")
    pcts = np.percentile(probs, PERCENTILES)
    for p, v in zip(PERCENTILES, pcts):
        bar = "#" * int((v - 0.3) * 200) if v > 0.3 else ""
        print(f"  p{p:>3d}: {v:.4f}  {bar}")

    print()
    print("── Signal counts at candidate thresholds ─────────")
    print(f"{'Threshold':>12}  {'BUY':>8}  {'SELL':>8}  {'HOLD':>8}  {'BUY+SELL%':>10}")
    for thresh in [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65]:
        buy  = int(np.sum(probs > thresh))
        sell = int(np.sum(probs < (1.0 - thresh)))
        hold = len(probs) - buy - sell
        pct  = (buy + sell) / len(probs) * 100
        print(f"  {thresh:.2f}        {buy:>8,}  {sell:>8,}  {hold:>8,}  {pct:>9.2f}%")

    print()
    print("── Accuracy at threshold (directional only) ──────")
    for thresh in [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65]:
        mask_buy  = probs > thresh
        mask_sell = probs < (1.0 - thresh)
        mask = mask_buy | mask_sell
        if mask.sum() == 0:
            print(f"  {thresh:.2f}        no signals")
            continue
        pred = np.where(mask_buy, 1, np.where(mask_sell, 0, -1))
        acc  = float(np.mean(pred[mask] == y_true[mask].astype(int)))
        brier = float(np.mean((probs[mask] - y_true[mask]) ** 2))
        brier_clim = float(np.var(y_true[mask])) if np.var(y_true[mask]) > 0 else 1.0
        bss_val = 1.0 - brier / brier_clim
        print(f"  {thresh:.2f}        acc={acc:.3f}  BSS(filtered)={bss_val:+.5f}  n={mask.sum():,}")

    print("=" * 60)


if __name__ == "__main__":
    main()
