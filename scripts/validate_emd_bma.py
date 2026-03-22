"""
validate_emd_bma.py — Standalone fold comparison: EMDDistance + BMACalibrator vs baseline.

Loads the full analogue DB via DataLoader (same path as bss_regression_test.py),
splits into 2024 train/val fold, then runs:
  - EMDDistance (time_penalty=0.5, price_penalty=1.0) for neighbour retrieval
  - BMACalibrator (df=3, n_iter=30) for probability calibration

Calibration leakage prevention:
  - CAL_FRAC = 0.76 chronological split within X_train
  - X_fit  = X_train[:cal_split]  → KNN index built on this portion
  - X_cal  = X_train[cal_split:]  → calibration samples, each queries X_fit only
  This mirrors the temporal ordering constraint from production Platt setup.

Speed accelerations:
  - Euclidean pre-filter: per query, discard candidates with L2 > PREFILTER_RADIUS
    before running expensive EMD. Reduces 175k→~500 candidates per query (~350× speedup).
  - --max-val-samples N: stratified subsample of val set (default 500, 0 = full run).
  - --max-cal-samples N: cap BMA calibration rows (default 2000, 0 = all).

Logs result to data/results/experiments_emd_bma.tsv for provenance.
Every BSS number reported here MUST trace to this file or terminal output.

Usage:
    python scripts/validate_emd_bma.py                      # fast (~5 min)
    python scripts/validate_emd_bma.py --max-val-samples 0  # full run (hours)
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Validate EMDDistance + BMACalibrator")
parser.add_argument(
    "--max-val-samples", type=int, default=500,
    help="Stratified subsample of val set (0 = full run). Default: 500",
)
parser.add_argument(
    "--max-cal-samples", type=int, default=2000,
    help="Cap BMA calibration rows (0 = all). Default: 2000",
)
parser.add_argument(
    "--prefilter-k", type=int, default=500,
    help="Euclidean pre-filter: keep only top-K L2 candidates before EMD. Default: 500",
)
parser.add_argument(
    "--time-penalty", type=float, default=0.5,
    help="EMDDistance time_penalty param (0.0 = no temporal weighting). Default: 0.5",
)
parser.add_argument(
    "--price-penalty", type=float, default=1.0,
    help="EMDDistance price_penalty param. Default: 1.0",
)
parser.add_argument(
    "--no-bma", action="store_true",
    help="Skip BMA: use raw fraction of positive analogues instead.",
)
args = parser.parse_args()

# Project root on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pattern_engine.data import DataLoader
from pattern_engine.features import RETURN_COLS  # [ret_1d, ret_3d, ..., ret_90d]
from research.emd_distance import EMDDistance
from research.bma_calibrator import BMACalibrator


# ---------------------------------------------------------------------------
# Config — mirrors locked settings from CLAUDE.md
# ---------------------------------------------------------------------------
FOLD_TRAIN_END = "2023-12-31"
FOLD_VAL_START = "2024-01-01"
FOLD_VAL_END   = "2024-12-31"
TOP_K          = 50
MAX_DISTANCE   = 1.1019
FEATURE_COLS   = RETURN_COLS           # ret_1d, ret_3d, ret_7d, ..., ret_90d (8 cols)
TARGET_COL     = "fwd_7d_up"
CAL_FRAC       = 0.76                  # chronological split within training data
BASELINE_BSS   = 0.00103              # fold 6 (2024), walkforward terminal output — locked

# ---------------------------------------------------------------------------
# Load full DB via DataLoader (same pattern as bss_regression_test.py)
# ---------------------------------------------------------------------------
data_dir = project_root / "data"
raw_dir = data_dir / "raw"

# Main repo stores CSVs directly in data/ (not data/raw/)
if not raw_dir.exists() and list(data_dir.glob("*.csv")):
    import shutil
    raw_dir.mkdir(parents=True, exist_ok=True)
    for csv_file in data_dir.glob("*.csv"):
        dest = raw_dir / csv_file.name
        if not dest.exists():
            shutil.copy2(csv_file, dest)
    print(f"  Copied {len(list(raw_dir.glob('*.csv')))} CSVs to data/raw/")

print("Loading data...")
loader = DataLoader(data_dir=str(data_dir))
raw_data = loader.download(force_refresh=False)
print(f"  Loaded {len(raw_data)} tickers")

full_db = loader.compute_features(raw_data)
print(f"  Full DB: {len(full_db)} rows")

# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------
full_db["Date"] = full_db["Date"].astype("datetime64[ns]")
train_db = full_db[full_db["Date"] <= FOLD_TRAIN_END]
val_db   = full_db[
    (full_db["Date"] >= FOLD_VAL_START) & (full_db["Date"] <= FOLD_VAL_END)
]

print(f"\nFold split: train={len(train_db)} rows, val={len(val_db)} rows")

X_train = train_db[FEATURE_COLS].values.astype(float)  # (N_train, 8)
X_val   = val_db[FEATURE_COLS].values.astype(float)    # (N_val, 8)
y_train = train_db[TARGET_COL].values.astype(float)
y_val   = val_db[TARGET_COL].values.astype(float)

# ---------------------------------------------------------------------------
# Stratified subsample of validation set (if requested)
# ---------------------------------------------------------------------------
if args.max_val_samples > 0 and len(X_val) > args.max_val_samples:
    rng = np.random.default_rng(42)
    # Stratify by target label (0/1)
    pos_idx = np.where(y_val == 1)[0]
    neg_idx = np.where(y_val == 0)[0]
    n_pos = min(args.max_val_samples // 2, len(pos_idx))
    n_neg = args.max_val_samples - n_pos
    n_neg = min(n_neg, len(neg_idx))
    sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    sel_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    sel_idx = np.sort(np.concatenate([sel_pos, sel_neg]))
    X_val = X_val[sel_idx]
    y_val = y_val[sel_idx]
    print(f"  Subsampled val: {len(X_val)} rows (stratified, seed=42, "
          f"pos={n_pos}, neg={n_neg})")
else:
    print(f"  Using full val set: {len(X_val)} rows")

PREFILTER_K = args.prefilter_k  # Euclidean top-K before EMD

# ---------------------------------------------------------------------------
# EMDDistance: fit on training set
# ---------------------------------------------------------------------------
print(f"\nFitting EMDDistance(time_penalty={args.time_penalty}, price_penalty={args.price_penalty})...")
t0 = time.time()
metric = EMDDistance(time_penalty=args.time_penalty, price_penalty=args.price_penalty)
metric.fit(X_train)

# Retrieve top-K raw analogue outcomes for each validation sample
# Euclidean pre-filter: reduces per-query candidates from N_train → PREFILTER_K
print(f"Running EMD on {len(X_val)} validation samples "
      f"(Euclidean pre-filter top-{PREFILTER_K}, then EMD)...")
raw_probs = np.full((len(X_val), TOP_K), np.nan)
skipped = 0

for i, query in enumerate(X_val):
    # Step 1: fast Euclidean pre-filter → top PREFILTER_K candidates
    l2_dists = np.linalg.norm(X_train - query, axis=1)     # (N_train,)
    pre_idx  = np.argpartition(l2_dists, PREFILTER_K)[:PREFILTER_K]
    X_cands  = X_train[pre_idx]                            # (PREFILTER_K, 8)
    y_cands  = y_train[pre_idx]

    # Step 2: EMD only on PREFILTER_K candidates
    emd_dists    = metric.compute(query, X_cands)           # (PREFILTER_K,)
    within_max   = emd_dists <= MAX_DISTANCE
    n_within     = int(within_max.sum())
    if n_within < TOP_K:
        skipped += 1
        continue
    within_idx   = np.where(within_max)[0]
    top_idx      = np.argsort(emd_dists[within_idx])[:TOP_K]
    raw_probs[i] = y_cands[within_idx[top_idx]]

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta  = (len(X_val) - i - 1) / rate
        print(f"  {i + 1}/{len(X_val)} done ({elapsed:.0f}s, ETA {eta:.0f}s, {skipped} skipped)")

elapsed_emd = time.time() - t0
print(f"\nEMD complete in {elapsed_emd:.0f}s. Skipped {skipped}/{len(X_val)} samples (< {TOP_K} neighbours).")

# Drop rows with insufficient neighbours
valid_mask = ~np.isnan(raw_probs[:, 0])
raw_probs_valid = raw_probs[valid_mask]   # (N_valid, TOP_K)
y_val_valid     = y_val[valid_mask]       # (N_valid,)
print(f"Valid samples for calibration: {valid_mask.sum()}/{len(X_val)}")

# ---------------------------------------------------------------------------
# BMACalibrator — fit on CAL_FRAC chronological split of training data
# CRITICAL: Cal samples must query X_fit only (not X_train) — prevents leakage
# ---------------------------------------------------------------------------
print(f"\nFitting BMACalibrator (CAL_FRAC={CAL_FRAC})...")
cal_split = int(len(X_train) * CAL_FRAC)
X_fit = X_train[:cal_split]   # (N_fit, 8)
X_cal = X_train[cal_split:]   # (N_cal, 8)
y_fit = y_train[:cal_split]
y_cal = y_train[cal_split:]

print(f"  Fit portion: {len(X_fit)} rows, Cal portion: {len(X_cal)} rows")

# Optionally cap BMA calibration rows for speed
if args.max_cal_samples > 0 and len(X_cal) > args.max_cal_samples:
    rng_cal = np.random.default_rng(7)
    cal_sel = rng_cal.choice(len(X_cal), size=args.max_cal_samples, replace=False)
    cal_sel.sort()
    X_cal = X_cal[cal_sel]
    y_cal = y_cal[cal_sel]
    print(f"  Capped cal to {len(X_cal)} rows (seed=7)")

# Build cal raw_probs: each cal sample queries X_fit only (with Euclidean pre-filter)
cal_raw = np.full((len(X_cal), TOP_K), np.nan)
for i, query in enumerate(X_cal):
    # Pre-filter on X_fit using Euclidean
    l2_fit   = np.linalg.norm(X_fit - query, axis=1)
    pf_k     = min(PREFILTER_K, len(X_fit))
    pre_idx  = np.argpartition(l2_fit, pf_k)[:pf_k]
    X_fit_c  = X_fit[pre_idx]
    y_fit_c  = y_fit[pre_idx]

    distances = metric.compute(query, X_fit_c)   # EMD against pre-filtered fit set
    top_idx   = np.argsort(distances)[:TOP_K]
    cal_raw[i] = y_fit_c[top_idx]

# Drop cal rows with any NaN (shouldn't happen since no MAX_DISTANCE filter here)
cal_valid = ~np.isnan(cal_raw[:, 0])
cal_raw_valid = cal_raw[cal_valid]
y_cal_valid   = y_cal[cal_valid]
print(f"  Cal rows for BMA fit: {cal_valid.sum()}/{len(X_cal)}")

if args.no_bma:
    print("  --no-bma: using raw fraction of positives (no BMA).")
    bma = None
else:
    bma = BMACalibrator(n_iter=30, df=3.0)
    bma.fit(cal_raw_valid, y_cal_valid)
    print(f"  BMA fitted. Top 5 analogue weights: {bma.weights[:5].round(4)}")

# ---------------------------------------------------------------------------
# Calibrated probabilities on validation set
# ---------------------------------------------------------------------------
if bma is not None:
    cal_probs = np.array([float(bma.transform(row)) for row in raw_probs_valid])
else:
    cal_probs = raw_probs_valid.mean(axis=1)   # raw fraction of positive analogues

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
brier_emd_bma = brier_score_loss(y_val_valid, cal_probs)
brier_ref     = brier_score_loss(y_val_valid, np.full(len(y_val_valid), y_val_valid.mean()))
bss_emd_bma   = 1.0 - (brier_emd_bma / brier_ref)
delta         = bss_emd_bma - BASELINE_BSS
gate_cleared  = delta >= 0.02

print("\n" + "=" * 60)
print("  EMD + BMA vs Baseline (2024 fold)")
print("=" * 60)
print(f"  BSS (EMD+BMA):  {bss_emd_bma:+.5f}")
print(f"  BSS (baseline): {BASELINE_BSS:+.5f}")
print(f"  Delta:          {delta:+.5f}")
print(f"  Gate cleared:   {gate_cleared}  (need delta >= +0.02000)")
print("=" * 60)

if gate_cleared:
    print("\n  PROMOTION GATE CLEARED — Phase C work may proceed.")
else:
    print("\n  Gate NOT cleared. Try tuning time_penalty (0.0-2.0) or price_penalty.")

# ---------------------------------------------------------------------------
# Provenance log → data/results/experiments_emd_bma.tsv
# ---------------------------------------------------------------------------
results_dir = data_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)
tsv_path = results_dir / "experiments_emd_bma.tsv"

row = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "experiment_name": "validate_emd_bma_2024fold",
    "fold": "2024",
    "n_valid": int(valid_mask.sum()),
    "n_skipped": skipped,
    "bss_emd_bma": round(float(bss_emd_bma), 7),
    "bss_baseline": BASELINE_BSS,
    "delta": round(float(delta), 7),
    "gate_cleared": gate_cleared,
    "notes": (
        f"EMDDistance(tp={args.time_penalty},pp={args.price_penalty})"
        f"+{'raw_frac' if args.no_bma else 'BMA(df=3,n_iter=30)'} "
        f"CAL_FRAC=0.76 TOP_K=50 MAX_DIST=1.1019 "
        f"max_val={args.max_val_samples} max_cal={args.max_cal_samples} "
        f"prefilter_k={PREFILTER_K}"
    ),
}

write_header = not tsv_path.exists()
with open(tsv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
    if write_header:
        writer.writeheader()
    writer.writerow(row)

print(f"\n  Provenance logged → {tsv_path}")
