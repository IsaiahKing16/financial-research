"""
diagnose_sector_conviction.py — Reveal actual lift distribution of SectorConvictionLayer
on the 585-ticker universe and recommend a calibrated min_sector_lift threshold.

Problem: SectorConvictionLayer(min_sector_lift=0.03) vetoes 100% of BUY/SELL signals
on the 585T universe. The 0.03 threshold was calibrated on a 52-ticker universe.
This script diagnoses what thresholds are appropriate for the 585T universe.

Usage:
    cd "C:\\Users\\Isaia\\.claude\\financial-research"
    PYTHONUTF8=1 py -3.12 scripts/diagnose_sector_conviction.py
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
from pattern_engine.sector_conviction import SectorConvictionLayer
from pattern_engine.sector import SECTOR_MAP
from scripts.run_walkforward import WalkForwardConfig, load_full_db, FOLDS


FEATURE_COLS = VOL_NORM_COLS

# Use the 2024-Val fold (index 5) — cheapest representative fold
FOLD = FOLDS[5]  # {"label": "2024-Val", "train_end": "2023-12-31", ...}

# 585T calibrated confidence threshold (max Platt prob = 0.6195; 0.65 never fires)
CONFIDENCE_THRESHOLD = 0.55

THRESHOLDS = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03]

TARGET_LOW  = 0.05   # 5% lower bound for recommendation
TARGET_HIGH = 0.15   # 15% upper bound for recommendation
TARGET_MID  = 0.10   # ~10% ideal target


def main() -> None:
    print("=" * 72)
    print("  FPPE — SectorConvictionLayer Lift Diagnostics")
    print(f"  Fold: {FOLD['label']}  |  threshold={CONFIDENCE_THRESHOLD}")
    print(f"  Universe: 585-ticker  |  features: VOL_NORM_COLS ({len(FEATURE_COLS)} cols)")
    print("=" * 72)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading full_db...")
    full_db = load_full_db()
    full_db["Date"] = pd.to_datetime(full_db["Date"])

    train_end  = pd.Timestamp(FOLD["train_end"])
    val_start  = pd.Timestamp(FOLD["val_start"])
    val_end    = pd.Timestamp(FOLD["val_end"])

    train_db = full_db[full_db["Date"] <= train_end].copy()
    val_db   = full_db[(full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)].copy()

    # ── 2. Drop NaN targets ───────────────────────────────────────────────────
    horizon = "fwd_7d_up"
    train_db = train_db.dropna(subset=[horizon])
    val_db   = val_db.dropna(subset=[horizon])

    print(f"     Train rows: {len(train_db):,}   Val rows: {len(val_db):,}")

    # ── 3. Fit PatternMatcher and query ───────────────────────────────────────
    print("\n[2/4] Fitting PatternMatcher and querying val_db...")
    cfg = WalkForwardConfig(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        use_hnsw=True,
    )
    matcher = PatternMatcher(cfg)
    matcher.fit(train_db, FEATURE_COLS)
    probs, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)

    probs_arr   = np.asarray(probs)
    signals_arr = np.asarray(signals)

    n_buy  = int(np.sum(signals_arr == "BUY"))
    n_sell = int(np.sum(signals_arr == "SELL"))
    n_hold = int(np.sum(signals_arr == "HOLD"))
    n_active = n_buy + n_sell

    print(f"     Signals — BUY: {n_buy:,}  SELL: {n_sell:,}  HOLD: {n_hold:,}")
    print(f"     Active (BUY+SELL): {n_active:,}  ({n_active/len(val_db):.1%} of val rows)")

    # ── 4. Fit SectorConvictionLayer ──────────────────────────────────────────
    print("\n[3/4] Fitting SectorConvictionLayer on train_db...")
    layer = SectorConvictionLayer(SECTOR_MAP)
    layer.fit(train_db, target_col=horizon)

    # Compute sector scores on val_db
    tickers = val_db["Ticker"].values
    sector_scores = layer.sector_scores(probs_arr, tickers)

    # ── 5. Compute per-sector lift table ──────────────────────────────────────
    print("\n[4/4] Computing sector lift distribution...\n")

    all_sectors = sorted(set(layer.sector_base_rates_.keys()) | set(sector_scores.keys()))
    rows = []
    for sector in all_sectors:
        base_rate  = layer.sector_base_rates_.get(sector, float("nan"))
        mean_prob  = sector_scores.get(sector, float("nan"))
        lift       = mean_prob - base_rate if not (np.isnan(mean_prob) or np.isnan(base_rate)) else float("nan")
        rows.append({"sector": sector, "base_rate": base_rate, "mean_prob": mean_prob, "lift": lift})

    lift_df = pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)

    # ── Print lift table ──────────────────────────────────────────────────────
    col_w = 18
    header = f"{'Sector':<{col_w}}  {'Base Rate':>10}  {'Mean Prob':>10}  {'Lift':>10}"
    sep    = "-" * len(header)
    print(header)
    print(sep)
    for _, row in lift_df.iterrows():
        br  = f"{row['base_rate']:.4f}"  if not np.isnan(row['base_rate'])  else "   N/A"
        mp  = f"{row['mean_prob']:.4f}"  if not np.isnan(row['mean_prob'])  else "   N/A"
        lft = f"{row['lift']:+.4f}"      if not np.isnan(row['lift'])       else "   N/A"
        print(f"{row['sector']:<{col_w}}  {br:>10}  {mp:>10}  {lft:>10}")
    print(sep)

    valid_lifts = lift_df["lift"].dropna().values
    if len(valid_lifts) > 0:
        print(f"\n  Lift summary: min={valid_lifts.min():+.4f}  "
              f"p25={np.percentile(valid_lifts, 25):+.4f}  "
              f"median={np.median(valid_lifts):+.4f}  "
              f"p75={np.percentile(valid_lifts, 75):+.4f}  "
              f"max={valid_lifts.max():+.4f}")

    # ── 6. Threshold sweep ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Threshold Sweep — BUY/SELL signals passing each min_sector_lift")
    print("=" * 72)

    # Build per-signal sector lookup for active signals only
    active_mask = (signals_arr == "BUY") | (signals_arr == "SELL")
    active_idx  = np.where(active_mask)[0]
    active_tickers = tickers[active_idx]

    # Precompute lift per active signal
    signal_lifts = np.array([
        (sector_scores.get(SECTOR_MAP.get(str(t), "Unknown"), float("nan"))
         - layer.sector_base_rates_.get(SECTOR_MAP.get(str(t), "Unknown"), float("nan")))
        for t in active_tickers
    ])

    sweep_header = f"  {'Threshold':>12}  {'Pass':>8}  {'Pass %':>8}  {'Veto':>8}  {'Veto %':>8}"
    sweep_sep    = "-" * len(sweep_header)
    print(sweep_header)
    print(sweep_sep)

    best_threshold = None
    best_dist = float("inf")

    for thresh in THRESHOLDS:
        if len(signal_lifts) > 0:
            pass_mask = ~np.isnan(signal_lifts) & (signal_lifts >= thresh)
            n_pass = int(pass_mask.sum())
        else:
            n_pass = 0

        n_veto  = n_active - n_pass
        pct     = n_pass / n_active if n_active > 0 else 0.0
        veto_pct = n_veto / n_active if n_active > 0 else 0.0

        marker = ""
        if TARGET_LOW <= pct <= TARGET_HIGH:
            marker = "  <-- in target range"
            dist = abs(pct - TARGET_MID)
            if dist < best_dist:
                best_dist = dist
                best_threshold = thresh

        print(f"  {thresh:>12.4f}  {n_pass:>8,}  {pct:>7.1%}  {n_veto:>8,}  {veto_pct:>7.1%}{marker}")

    print(sweep_sep)

    # ── 7. Recommendation ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RECOMMENDATION")
    print("=" * 72)

    if best_threshold is not None:
        # Recompute pass count for best threshold
        pass_mask = ~np.isnan(signal_lifts) & (signal_lifts >= best_threshold)
        n_pass = int(pass_mask.sum())
        pct = n_pass / n_active if n_active > 0 else 0.0
        print(f"\n  Recommended min_sector_lift = {best_threshold}")
        print(f"  → {n_pass:,} / {n_active:,} BUY/SELL signals pass ({pct:.1%})")
        print(f"  → This is the threshold closest to the 10% target within the 5–15% range.")
        print(f"\n  To apply: SectorConvictionLayer(SECTOR_MAP, min_sector_lift={best_threshold})")
        print(f"  In WalkForwardConfig: sector_conviction_lift={best_threshold}")
    else:
        # Fall back: find threshold closest to 10% even outside range
        closest_thresh = None
        closest_pct = None
        closest_dist = float("inf")
        for thresh in THRESHOLDS:
            if len(signal_lifts) > 0:
                pass_mask = ~np.isnan(signal_lifts) & (signal_lifts >= thresh)
                n_pass = int(pass_mask.sum())
            else:
                n_pass = 0
            pct = n_pass / n_active if n_active > 0 else 0.0
            dist = abs(pct - TARGET_MID)
            if dist < closest_dist:
                closest_dist = dist
                closest_thresh = thresh
                closest_pct = pct

        print(f"\n  WARNING: No threshold in swept range achieves 5–15% pass rate.")
        print(f"  Closest option: min_sector_lift = {closest_thresh}")
        if closest_pct is not None:
            print(f"  → {closest_pct:.1%} pass rate (target: ~10%)")
        print(f"  Consider extending the sweep range or re-examining the sector map.")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
